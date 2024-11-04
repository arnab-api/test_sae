import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, overload

import baukit
import torch
import transformers
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.env_utils import DEFAULT_MODELS_DIR
from src.utils.typing import TokenizerOutput
from src.utils.tokenization_utils import set_padding_side

logger = logging.getLogger(__name__)

CACHEABLE_FUNCS = [
    "forward",
    # "ssm", "selective_scan" , # specific to Mamba models
]


class ModelandTokenizer(LanguageModel):
    def __init__(
        self,
        base_lm: Optional[LanguageModel] = None,
        tokenizer: Optional[transformers.AutoTokenizer] = None,
        model_key: Optional[
            str
        ] = "EleutherAI/gpt-j-6B",  # if model is provided, this will be ignored and rewritten
        torch_dtype=torch.float16,
    ) -> None:
        assert (
            base_lm is not None or model_key is not None
        ), "Either the `base_lm` or `model_key` must be provided"
        if base_lm is not None:
            self.__dict__ = base_lm.__dict__
            self.name = base_lm._model.config._name_or_path.split("/")[-1]

        else:
            model_key = get_full_model_path(model_key)
            self.__dict__ = LanguageModel(
                model_key=model_key,
                torch_dtype=torch_dtype,
                device_map="auto",
                dispatch=True,
            ).__dict__
            self.name = model_key

        self._model.eval()
        self.tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        self.device = determine_device(self._model)
        self.parse_config()

        logger.info(
            f"loaded model <{model_key}> | size: {get_model_size(self._model)} | dtype: {determine_dtype(self._model)} | device: {self.device}"
        )
        self.cache_forwards()

    def parse_config(self) -> None:
        fields = {
            "n_layer": None,
            "n_embd": None,
            "layer_name_format": None,
            "layer_names": None,
            "embedder_name": None,
            "final_layer_norm_name": None,
            "lm_head_name": None,
        }

        if (
            is_gpt_variant(self)
            or is_llama_variant(self)
            or is_gemma_variant(self)
            or is_pythia_variant(self)
            or is_qwen_variant(self)
            or is_olmo_variant(self)
        ) == False:
            logger.error(
                f"Unknown model type: {type(unwrap_model(self)).__name__}. Parsing may fail."
            )

        fields["n_layer"] = len(determine_layers(self))
        fields["n_embd"] = determine_hidden_size(self)
        fields["embedder_name"] = determine_embedding_layer_path(self)
        fields["final_layer_norm_name"] = determine_final_layer_norm_path(self)
        fields["lm_head_name"] = determine_lm_head_path(self)
        fields["layer_name_format"] = determine_layer_name_format(self)

        fields["attn_module_name_format"] = None
        fields["mlp_module_name_format"] = None
        if (
            is_llama_variant(self)
            or is_gemma_variant(self)
            or is_qwen_variant(self)
            or is_olmo_variant(self)
        ):
            fields["mlp_module_name_format"] = "model.layers.{}.mlp"
            fields["attn_module_name_format"] = "model.layers.{}.self_attn"

        elif is_gpt_variant(self):
            # ! will be different for neox models. Ignoring for now
            fields["mlp_module_name_format"] = "transformer.h.{}.mlp"
            fields["attn_module_name_format"] = "transformer.h.{}.attn"

        elif is_pythia_variant(self):
            fields["mlp_module_name_format"] = "gpt_neox.layers.{}.mlp"
            fields["attn_module_name_format"] = "gpt_neox.layers.{}.attention"

        if fields["layer_name_format"] is not None and fields["n_layer"] is not None:
            fields["layer_names"] = [
                fields["layer_name_format"].format(i) for i in range(fields["n_layer"])
            ]

        for key, value in fields.items():
            if value is None:
                logger.error(
                    f"!!! Error ({type(unwrap_model(self)).__name__}): {key} could not be set !!!"
                )
            setattr(self, key, value)

    @property
    def lm_head(self) -> torch.nn.Sequential:
        lm_head = baukit.get_module(unwrap_model(self), self.lm_head_name)
        ln_f = baukit.get_module(unwrap_model(self), self.final_layer_norm_name)
        return LMHead(final_layer_norm=ln_f, lm_head=lm_head)

    def cache_forwards(self):
        """
        Caches the forward pass of all the modules.
        Usuful to reset the model to its original state after an overwrite.
        """
        self._module_forwards: dict = {}
        for name, module in self._model.named_modules():
            self._module_forwards[name] = {}
            for func_name in CACHEABLE_FUNCS:
                if hasattr(module, func_name):
                    self._module_forwards[name][func_name] = getattr(module, func_name)

    def reset_forward(self) -> None:
        """
        Resets the forward pass of all the modules to their original state.
        """
        for name, module in self._model.named_modules():
            # print(name, hasattr(module, "forward"))
            for func_name in CACHEABLE_FUNCS:
                if hasattr(module, func_name):
                    setattr(module, func_name, self._module_forwards[name][func_name])

    def __call__(self, *args, **kwargs) -> Any:
        """Call the model."""
        return self._model(*args, **kwargs)


class LMHead(torch.nn.Module):
    def __init__(self, final_layer_norm: torch.nn.Module, lm_head: torch.nn.Module):
        super().__init__()
        self.lm_head = lm_head
        self.final_layer_norm = final_layer_norm

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.final_layer_norm(x)
        return self.lm_head(x)


def get_model_size(
    model: torch.nn.Module, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all = param_size + buffer_size

    return bytes_to_human_readable(size_all, unit)


def bytes_to_human_readable(
    size: int, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return f"{size / denom:.3f} {unit}"


def get_full_model_path(model_name: str) -> str:
    full_path = os.path.join(DEFAULT_MODELS_DIR, model_name)
    if os.path.exists(full_path):
        return full_path
    else:
        logger.warning(
            f"""{model_name} not found in {DEFAULT_MODELS_DIR}
If not found in cache, model will be downloaded from HuggingFace to cache directory"""
        )
        return model_name


def unwrap_model(
    net: ModelandTokenizer | LanguageModel | torch.nn.Module,
) -> torch.nn.Module:
    if isinstance(net, LanguageModel):
        return net._model
    if isinstance(net, torch.nn.Module):
        return net
    raise ValueError("mt must be a ModelandTokenizer or a torch.nn.Module")


def unwrap_tokenizer(mt: ModelandTokenizer | AutoTokenizer) -> AutoTokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt


def untuple(object: Any):
    if isinstance(object, tuple):
        return object[0]
    return object


from src.utils.typing import Model, Tokenizer


def maybe_prefix_bos(tokenizer, prompt: str) -> str:
    """Prefix prompt with EOS token if model has no special start token."""
    tokenizer = unwrap_tokenizer(tokenizer)
    if hasattr(tokenizer, "bos_token"):
        prefix = tokenizer.bos_token
        if not prompt.startswith(prefix):
            prompt = prefix + " " + prompt
    return prompt


def is_pythia_variant(model: Model | ModelandTokenizer) -> bool:
    """Determine if model is pythia variant."""
    if isinstance(model, ModelandTokenizer) or isinstance(model, LanguageModel):
        model = unwrap_model(model)
    try:
        return "pythia" in model.config._name_or_path.lower()
    except:
        return False


def is_gpt_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model/tokenizer is GPT variant."""
    if isinstance(mt, ModelandTokenizer) or isinstance(mt, LanguageModel):
        mt = unwrap_model(mt)

    # pythia models also have GPTNeoXForCausalLM architecture, but they have slightly  different structure
    # so we need to check for them separately
    if is_pythia_variant(mt):
        return False
    return isinstance(
        mt,
        transformers.GPT2LMHeadModel
        | transformers.GPTJForCausalLM
        | transformers.GPTNeoForCausalLM
        | transformers.GPTNeoXForCausalLM
        | transformers.GPT2TokenizerFast
        | transformers.GPTNeoXTokenizerFast,
    )


def is_llama_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model/tokenizer is llama variant."""
    if isinstance(mt, ModelandTokenizer) or isinstance(mt, LanguageModel):
        mt = unwrap_model(mt)
    if isinstance(mt, transformers.LlamaForCausalLM):
        return True
    if hasattr(mt, "config"):
        config = mt.config
        if hasattr(config, "_name_or_path"):
            name = config._name_or_path
            return "llama" in name.lower() or "mistral" in name.lower()
    return False


def is_gemma_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model/tokenizer is gemma variant."""
    if isinstance(mt, ModelandTokenizer) or isinstance(mt, LanguageModel):
        mt = unwrap_model(mt)
    if isinstance(mt, transformers.GemmaForCausalLM | transformers.Gemma2ForCausalLM):
        return True
    if hasattr(mt, "config"):
        config = mt.config
        if hasattr(config, "_name_or_path"):
            name = config._name_or_path
            return "gemma" in name.lower()
    return False


def is_olmo_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model/tokenizer is gemma variant."""
    if isinstance(mt, ModelandTokenizer) or isinstance(mt, LanguageModel):
        mt = unwrap_model(mt)
    if isinstance(mt, transformers.OlmoForCausalLM):
        return True
    if hasattr(mt, "config"):
        config = mt.config
        if hasattr(config, "_name_or_path"):
            name = config._name_or_path
            return "gemma" in name.lower()
    return False


def is_qwen_variant(mt: Model | ModelandTokenizer) -> bool:
    """Determine if model/tokenizer is qwen variant."""
    if isinstance(mt, ModelandTokenizer) or isinstance(mt, LanguageModel):
        mt = unwrap_model(mt)
    if isinstance(mt, transformers.Qwen2ForCausalLM):
        return True
    if hasattr(mt, "config"):
        config = mt.config
        if hasattr(config, "_name_or_path"):
            name = config._name_or_path
            return "qwen" in name.lower()
    return False


def any_parameter(model: ModelandTokenizer | Model) -> torch.nn.Parameter | None:
    """Get any example parameter for the model."""
    model = unwrap_model(model)
    return next(iter(model.parameters()), None)


def determine_embedding_layer_path(model: ModelandTokenizer | Model) -> str:
    model = unwrap_model(model)
    if is_gpt_variant(model):
        return "transformer.wte"
    elif (
        is_llama_variant(model)
        or is_gemma_variant(model)
        or is_qwen_variant(model)
        or is_olmo_variant(model)
    ):
        return "model.embed_tokens"
    elif is_pythia_variant(model):
        return "gpt_neox.embed_in"
    else:
        raise ValueError(f"unknown model type: {type(model).__name__}")


def determine_final_layer_norm_path(model: ModelandTokenizer | Model) -> str:
    model = unwrap_model(model)
    if is_gpt_variant(model):
        return "transformer.ln_f"
    elif (
        is_llama_variant(model)
        or is_gemma_variant(model)
        or is_qwen_variant(model)
        or is_olmo_variant(model)
    ):
        return "model.norm"
    elif is_pythia_variant(model):
        return "gpt_neox.final_layer_norm"
    else:
        raise ValueError(f"unknown model type: {type(model).__name__}")


def determine_lm_head_path(model: ModelandTokenizer | Model) -> str:
    model = unwrap_model(model)
    if is_gpt_variant(model):
        return "lm_head"
    elif (
        is_llama_variant(model)
        or is_gemma_variant(model)
        or is_qwen_variant(model)
        or is_olmo_variant(model)
    ):
        return "lm_head"
    elif is_pythia_variant(model):
        return "embed_out"
    else:
        raise ValueError(f"unknown model type: {type(model).__name__}")


def determine_layers(model: ModelandTokenizer | Model) -> tuple[int, ...]:
    """Return all hidden layer names for the given model."""
    model = unwrap_model(model)
    assert isinstance(model, Model)

    if (
        is_gpt_variant(model)
        or is_llama_variant(model)
        or is_gemma_variant(model)
        or is_qwen_variant(model)
        or is_pythia_variant(model)
        or is_olmo_variant(model)
    ):
        n_layer = model.config.num_hidden_layers
    else:
        n_layer = model.config.n_layer

    return (*range(n_layer),)


from src.utils.typing import Layer, Sequence


def determine_layer_name_format(
    model: ModelandTokenizer | Model,
) -> str | None:
    """Determine the format of layer names."""
    model = unwrap_model(model)

    if is_gpt_variant(model):
        if isinstance(model, transformers.GPTNeoXForCausalLM):
            return "gpt_neox.layers.{}"
        return "transformer.h.{}"
    elif (
        is_llama_variant(model)
        or is_gemma_variant(model)
        or is_qwen_variant(model)
        or is_olmo_variant(model)
    ):
        return "model.layers.{}"
    elif is_pythia_variant(model):
        return "gpt_neox.layers.{}"


@overload
def determine_layer_paths(
    model: ModelandTokenizer | Model,
    layers: Optional[Sequence[Layer]] = ...,
    *,
    return_dict: Literal[False] = ...,
) -> Sequence[str]:
    """Determine layer path for each layer."""
    ...


@overload
def determine_layer_paths(
    model: ModelandTokenizer | Model,
    layers: Optional[Sequence[Layer]] = ...,
    *,
    return_dict: Literal[True],
) -> dict[Layer, str]:
    """Determine mapping from layer to layer path."""
    ...


def determine_layer_paths(
    model: ModelandTokenizer | Model,
    layers: Optional[Sequence[Layer]] = None,
    *,
    return_dict: bool = False,
) -> Sequence[str] | dict[Layer, str]:
    """Determine the absolute paths to the given layers in the model.

    Args:
        model: The model.
        layers: The specific layer (numbers/"emb") to look at. Defaults to all of them.
            Can be a negative number.
        return_dict: If True, return mapping from layer to layer path,
            otherwise just return list of layer paths in same order as `layers`.

    Returns:
        Mapping from layer number to layer path.

    """
    model = unwrap_model(model)

    if layers is None:
        layers = determine_layers(model)

    assert isinstance(model, Model), type(model)

    layer_paths: dict[Layer, str] = {}
    layer_name_format = determine_layer_name_format(model)
    for layer in layers:
        if layer == "emb":
            layer_paths[layer] = determine_embedding_layer_path(model)
            continue
        if layer == "ln_f":
            layer_paths[layer] = determine_final_layer_norm_path(model)
            continue

        layer_index = layer
        if layer_index < 0:
            layer_index = len(determine_layers(model)) + layer

        layer_paths[layer] = layer_name_format.format(layer_index)

    return layer_paths if return_dict else tuple(layer_paths[la] for la in layers)


def determine_hidden_size(model: ModelandTokenizer | Model) -> int:
    """Determine hidden rep size for the model."""
    model = unwrap_model(model)
    try:
        return model.config.hidden_size
    except AttributeError:
        embed = baukit.get_module(model, determine_embedding_layer_path(model))
        return embed.weight.shape[-1]


def determine_device(model: ModelandTokenizer | Model) -> torch.device | None:
    """Determine device model is running on."""
    parameter = any_parameter(model)
    return parameter.device if parameter is not None else None


def determine_dtype(model: ModelandTokenizer | Model) -> torch.dtype | None:
    """Determine dtype of model."""
    parameter = any_parameter(model)
    return parameter.dtype if parameter is not None else None


def prepare_offset_mapping(string, tokenized, special_tokens):
    """LLaMA3 tokenizer in Huggingface is buggy. This function is a workaround for the bug."""
    """
    <Test>
    
    prompts = ["The Eiffle Tower is located in", "The Space Needle is located in"]
    inp = prepare_input(
        prompts = prompts,
        tokenizer=mt,
        return_offsets_mapping=True,
        device="cuda"
    )

    i=1 # <to be changed>
    for token_id, offset in zip(inp["input_ids"][i], inp["offset_mapping"][i]):
        print(f"`{tokenizer.decode(token_id)}`, {offset=} | `{prompts[i][offset[0]:offset[1]]}`")

    """
    # logger.debug(f"{special_tokens}")
    offset_mapping = []
    end = 0
    for token in tokenized:
        if token in special_tokens:
            offset_mapping.append((end, end))
            continue
        # print(f"{string[end:].find(token)} | {end=}, {token=}, {string[end:]}")
        next_tok_idx = string[end:].find(token)
        assert next_tok_idx != -1, f"{token} not found in {string[end:]}"
        assert next_tok_idx in [
            0,
            1,
        ], f"{token} not found at the beginning of the string"

        start = end
        end = start + string[end:].find(token) + len(token)
        offset_mapping.append((start, end))
    return offset_mapping


def prepare_input(
    prompts: str | list[str],
    tokenizer: ModelandTokenizer | Tokenizer,
    n_gen_per_prompt: int = 1,
    device: torch.device = "cpu",
    add_bos_token: bool = False,
    return_offsets_mapping=False,
    padding_side: Optional[Literal["left", "right"]] = None,
    **kwargs,
) -> TokenizerOutput:
    """Prepare input for the model."""
    if isinstance(tokenizer, ModelandTokenizer):
        device = determine_device(
            tokenizer
        )  # if tokenizer type is ModelandTokenizer, get device and ignore the passed device
    # calculate_offsets = return_offsets_mapping and (
    #     isinstance(tokenizer, ModelandTokenizer) and "llama-3" in tokenizer.name.lower()
    # )
    calculate_offsets = False # updated tokenizer fixed offset mapping issue for llama tokenizers

    tokenizer = unwrap_tokenizer(tokenizer)
    prompts = [prompts] if isinstance(prompts, str) else prompts
    if add_bos_token:
        prompts = [maybe_prefix_bos(tokenizer, p) for p in prompts]
    prompts = [p for p in prompts for _ in range(n_gen_per_prompt)]

    padding_side = padding_side or tokenizer.padding_side

    with set_padding_side(tokenizer, padding_side):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            return_offsets_mapping=return_offsets_mapping,
            **kwargs,
        )

    if calculate_offsets:
        offsets = []
        for i in range(len(prompts)):
            tokenized = [tokenizer.decode(t) for t in inputs["input_ids"][i]]
            offsets.append(
                prepare_offset_mapping(
                    string=prompts[i],
                    tokenized=tokenized,
                    special_tokens=tokenizer.all_special_tokens,
                )
            )
        inputs["offset_mapping"] = torch.tensor(offsets)

    inputs = inputs.to(device)
    return inputs
