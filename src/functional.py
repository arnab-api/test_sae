import ctypes
import gc
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch
from nnsight import LanguageModel
from openai import OpenAI
from tqdm import tqdm

from src.dataset import BridgeDataset, BridgeSample, Relation
from src.models import ModelandTokenizer, is_llama_variant, prepare_input
from src.utils.env_utils import GPT_4O_CACHE_DIR
from src.utils.typing import PredictedToken, Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


@torch.inference_mode()
def interpret_logits(
    tokenizer: ModelandTokenizer | Tokenizer,
    logits: torch.Tensor,
    k: int = 10,
    get_proba: bool = False,
) -> list[tuple[str, float]]:
    tokenizer = unwrap_tokenizer(tokenizer)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    token_ids = logits.topk(dim=-1, k=k).indices.squeeze().tolist()
    logit_values = logits.topk(dim=-1, k=k).values.squeeze().tolist()
    return [(tokenizer.decode(t), round(v, 3)) for t, v in zip(token_ids, logit_values)]


@torch.inference_mode()
def logit_lens(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    after_layer_norm: bool = False,
    interested_tokens: list[int] = [],
    get_proba: bool = False,
    k: int = 10,
) -> tuple[list[tuple[str, float]], dict]:
    lm_head = mt.lm_head if not after_layer_norm else mt.lm_head.lm_head
    h = untuple(h) if after_layer_norm else h
    logits = lm_head(h)
    logits = torch.nn.functional.softmax(logits, dim=-1) if get_proba else logits
    # don't pass `get_proba` or softmax will be applied twice with `get_proba=True`
    candidates = interpret_logits(mt, logits, k=k)
    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        interested_logits = {
            t: {
                "p": logits[t].item(),
                "rank": rank_tokens.index(t) + 1,
                "token": mt.tokenizer.decode(t),
            }
            for t in interested_tokens
        }
        return candidates, interested_logits
    return candidates


def untuple(object: Any):
    if isinstance(object, tuple) or (
        "LanguageModelProxy" in str(type(object)) and len(object) > 1
    ):
        return object[0]
    return object


def unwrap_model(mt: ModelandTokenizer | torch.nn.Module) -> torch.nn.Module:
    if isinstance(mt, ModelandTokenizer):
        return mt.model
    if isinstance(mt, torch.nn.Module):
        return mt
    raise ValueError("mt must be a ModelandTokenizer or a torch.nn.Module")


def unwrap_tokenizer(mt: ModelandTokenizer | Tokenizer) -> Tokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt


# useful for logging
def bytes_to_human_readable(
    size: int, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return f"{size / denom:.3f} {unit}"


def any_is_nontrivial_prefix(predictions: list[str], target: str) -> bool:
    """Return true if any prediction is (case insensitive) prefix of the target."""
    return any(is_nontrivial_prefix(p, target) for p in predictions)


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def get_tick_marker(value: bool) -> str:
    """Returns a tick or cross marker depending on the value."""
    return "✓" if value else "✗"


def format_whitespace(s: str) -> str:
    """Format whitespace in a string for printing."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


@torch.inference_mode()
def filter_relation_samples_by_model_knowledge(
    mt: ModelandTokenizer,
    relation: Relation,
    limit: Optional[int] = None,
    modify_inplace: bool = False,
) -> Relation:
    """Filter samples by model knowledge."""
    logger.debug(f'"{relation.name}" | filtering with {mt.name}')

    filtered_samples = []
    for i in range(len(relation.samples)):
        question, answer = relation[i]
        predictions = predict_next_token(mt, question, k=5)[0]
        top_pred = predictions[0]
        is_known = is_nontrivial_prefix(prediction=top_pred.token, target=answer)
        sample = relation.samples[i]

        logger.debug(
            f"{sample.subject=} -> {answer=} | predicted = '{top_pred.token}'({top_pred.prob:.3f}) ==> ({get_tick_marker(is_known)})"
        )
        if is_known:
            filtered_samples.append(sample)
        if limit is not None and len(filtered_samples) >= limit:
            break

    logger.info(
        f'filtered relation "{relation.name}" to {len(filtered_samples)} samples (with {len(relation._few_shot_samples)}-shots)'
    )

    if modify_inplace == False:
        ret_relation = Relation.from_dict(relation.to_dict())
    else:
        ret_relation = relation

    ret_relation.samples = filtered_samples
    return ret_relation


def release_cache():
    gc.collect()
    if torch.cuda.is_available():
        # before = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        # after = torch.cuda.memory_allocated()
        # freed = before - after # ! the effect of empty_cache() is not immediate


@torch.inference_mode()
def predict_next_token(
    mt: ModelandTokenizer,
    inputs: Union[str, list[str]] | TokenizerOutput,
    k: int = 5,
    batch_size: int = 8,
    token_of_interest: Optional[Union[Union[str, int], list[Union[str, int]]]] = None,
):
    """Predict the next token(s) given the input."""
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            inputs.pop("offset_mapping")
    else:
        inputs = prepare_input(prompts=inputs, tokenizer=mt.tokenizer)
    if token_of_interest is not None:
        token_of_interest = (
            [token_of_interest]
            if not isinstance(token_of_interest, list)
            else token_of_interest
        )
    if token_of_interest is not None:
        assert len(token_of_interest) == len(inputs["input_ids"])
        track_interesting_tokens = []

    predictions = []
    for i in range(0, len(inputs["input_ids"]), batch_size):
        batch_inputs = {
            k: v[i : i + batch_size] if isinstance(v, list) else v
            for k, v in inputs.items()
        }

        with mt.trace(batch_inputs, scan=i == 0) as tr:
            batch_logits = mt.output.logits.save()

        batch_logits = batch_logits[:, -1, :]
        batch_probs = batch_logits.float().softmax(dim=-1)
        batch_topk = batch_probs.topk(k=k, dim=-1)

        for token_ids, token_probs in zip(batch_topk.indices, batch_topk.values):
            predictions.append(
                [
                    PredictedToken(
                        token=mt.tokenizer.decode(token_ids[j]),
                        prob=token_probs[j].item(),
                        token_id=token_ids[j],
                    )
                    for j in range(k)
                ]
            )

        if token_of_interest is not None:
            _t_idx = 1 if is_llama_variant(mt) else 0
            for j in range(i, i + batch_inputs["input_ids"].shape[0]):
                tok_id = (
                    mt.tokenizer(token_of_interest[j]).input_ids[_t_idx]
                    if type(token_of_interest[j]) == str
                    else token_of_interest[j]
                )
                probs = batch_probs[j]
                order = probs.topk(dim=-1, k=probs.shape[-1]).indices.squeeze()
                prob_tok = probs[tok_id]
                rank = int(torch.where(order == tok_id)[0].item() + 1)
                track_interesting_tokens.append(
                    (
                        rank,
                        PredictedToken(
                            token=mt.tokenizer.decode(tok_id),
                            prob=prob_tok.item(),
                            token_id=tok_id,
                        ),
                    )
                )
        if token_of_interest is not None:
            return predictions, track_interesting_tokens

        return predictions


def get_module_nnsight(model, layer_name):
    layer = model
    for name in layer_name.split("."):
        layer = layer[int(name)] if name.isdigit() else getattr(layer, name)
    return layer


@dataclass(frozen=False)
class PatchSpec:
    location: tuple[str, int]
    patch: torch.Tensor
    clean: Optional[torch.Tensor] = None


@torch.inference_mode()
def get_hs(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    locations: tuple[str, int] | list[tuple[str, int]],
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
    return_dict: bool = False,
) -> dict[tuple[str, int], torch.Tensor]:

    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    if isinstance(locations, tuple):
        locations = [locations]
    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]

    def is_an_attn_head(module_name) -> bool | tuple[int, int]:
        attn_id = mt.attn_module_name_format.split(".")[-1]
        if attn_id not in module_name:
            return False
        if module_name.endswith(attn_id):
            return False

        head_id = module_name.split(".")[-1]
        layer_id = ".".join(module_name.split(".")[:-1])

        return layer_id, int(head_id)

    layer_names = [layer_name for layer_name, _ in locations]
    layer_names = list(set(layer_names))
    layer_states = {layer_name: torch.empty(0) for layer_name in layer_names}
    with mt.trace(input, scan=True) as tracer:
        if patches is not None:
            for cur_patch in patches:
                module_name, index = cur_patch.location
                if is_an_attn_head(module_name) != False:
                    raise NotImplementedError(
                        "patching not supported yet for attn heads"
                    )
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output
                    if ("mlp" in module_name or module_name == mt.embedder_name)
                    else module.output[0].save()
                )
                current_state[0, index, :] = cur_patch.patch

        for layer_name in layer_names:
            if is_an_attn_head(layer_name) == False:
                module = get_module_nnsight(mt, layer_name)
                layer_states[layer_name] = module.output.save()
            else:
                attn_module_name, head_idx = is_an_attn_head(layer_name)
                o_proj_name = attn_module_name + ".o_proj"
                head_dim = mt.n_embd // mt.model.config.num_attention_heads
                o_proj = get_module_nnsight(mt, o_proj_name)
                layer_states[layer_name] = o_proj.input[0][0][
                    :, :, head_idx * head_dim : (head_idx + 1) * head_dim
                ].save()

    hs = {}

    for layer_name, index in locations:
        # print(layer_name, layer_states[layer_name].shape)
        hs[(layer_name, index)] = untuple(layer_states[layer_name])[
            :, index, :
        ].squeeze()

    # print(f"==========> {len(hs)=}")
    if len(hs) == 1 and not return_dict:
        return list(hs.values())[0]
    return hs


@torch.inference_mode
def get_all_module_states(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    kind: Literal["residual", "mlp", "attention"] = "residual",
) -> dict[tuple[str, int], torch.Tensor]:
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError(f"kind must be one of 'residual', 'mlp', 'attention'")

    layer_and_index = []
    for layer_idx in range(mt.n_layer):
        for token_idx in range(input.input_ids.shape[1]):
            layer_and_index.append((layer_name_format.format(layer_idx), token_idx))

    return get_hs(mt, input, layer_and_index)


def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[Tokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')

    # logger.debug(f"Found substring in string {string.count(substring)} times")

    if occurrence < 0:
        # If occurrence is negative, count from the right.
        char_start = string.rindex(substring)
        for _ in range(-1 - occurrence):
            try:
                char_start = string.rindex(substring, 0, char_start)
            except ValueError as error:
                raise ValueError(
                    f"could not find {-occurrence} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    else:
        char_start = string.index(substring)
        for _ in range(occurrence):
            try:
                char_start = string.index(substring, char_start + 1)
            except ValueError as error:
                raise ValueError(
                    f"could not find {occurrence + 1} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    char_end = char_start + len(substring)

    # logger.debug(
    #     f"char range: [{char_start}, {char_end}] => `{string[char_start:char_end]}`"
    # )

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = prepare_input(
            string, return_offsets_mapping=True, tokenizer=tokenizer, **kwargs
        )
        offset_mapping = tokens.offset_mapping

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        # logger.debug(f"{index=} | token range: [{token_char_start}, {token_char_end}]")
        if token_char_start == token_char_end:
            # Skip special tokens # ! Is this the proper way of doing this?
            continue
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert token_start is not None
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def predict_bridge_entity(
    mt: ModelandTokenizer,
    prompt: str,
    search_span: int = 100,
    separator: Literal["#", "-"] = "#",
) -> str:
    inputs = prepare_input(prompts=prompt, tokenizer=mt, add_bos_token=False)

    bridge_entity = ""
    while search_span > 0:
        predicted_token = predict_next_token(mt=mt, inputs=inputs, k=1)[0][0]
        if predicted_token.token.strip() == separator:
            break

        bridge_entity += predicted_token.token
        # print(torch.tensor([predicted_token.token_id]).to(mt.device))
        # print(inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat(
            [
                inputs["input_ids"],
                torch.tensor([predicted_token.token_id])[None].to(mt.device),
            ],
            dim=1,
        )
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.tensor([1])[None].to(mt.device)], dim=1
        )
        search_span -= 1

        if search_span == 0:
            logger.error(f"search span exceeded - found: {bridge_entity}")
            return bridge_entity

    return bridge_entity


def ask_gpt4o(
    query_sample: BridgeSample,
    predicted_answer: str,
):

    ##################################################
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
    )
    MODEL_NAME = "gpt-4o"
    ##################################################

    prompt = f"""
A smaller language model was asked the following question:
"What is a common link between {query_sample.entity_pair[0]} and {query_sample.entity_pair[1]}?"
And the model gave the following answer:
"{predicted_answer.strip()}"
Is it correct? Your answer should start with "Yes" or "No". If the answer is "Yes", don't say anything else. If the answer is "No", give explanation why.
"""
    hash_val = hashlib.md5(prompt.encode()).hexdigest()
    if f"{hash_val}.json" in os.listdir(GPT_4O_CACHE_DIR):
        logger.debug(f"found cached gpt4o response for {hash_val} - loading")
        with open(os.path.join(GPT_4O_CACHE_DIR, f"{hash_val}.json"), "r") as f:
            json_data = json.load(f)
            return json_data["response"]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=4000,
    )
    response = response.choices[0].message.content

    with open(os.path.join(GPT_4O_CACHE_DIR, f"{hash_val}.json"), "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "response": response,
                "model": MODEL_NAME,
                "hash": hash_val,
                "tempraure": 0,
            },
            f,
        )

    return response


@torch.inference_mode()
def filter_bridge_samples_by_model_knowledge(
    mt: ModelandTokenizer, dataset: BridgeDataset, limit: Optional[int] = None
) -> BridgeDataset:
    filtered_samples = []
    for i in tqdm(range(len(dataset))):
        prompt, answer = dataset[i]
        sample = dataset.examples[i]
        predicted_bridge = predict_bridge_entity(mt, prompt)
        # is_correct = is_nontrivial_prefix(sample.bridge.lower(), predicted_bridge) or is_nontrivial_prefix(predicted_bridge, sample.bridge)
        is_correct = ask_gpt4o(sample, predicted_bridge).lower().startswith("yes")

        logger.info(
            f"{sample.entity_pair} <> {sample.bridge} | predicted: {predicted_bridge} => ({get_tick_marker(is_correct)})"
        )
        if is_correct:
            filtered_samples.append(sample)
        if limit is not None and len(filtered_samples) >= limit:
            break

    logger.info(
        f"filtered {len(filtered_samples)} samples out of {len(dataset)} with {len(dataset.icl_examples)} icl examples"
    )

    dataset.examples = filtered_samples
    return dataset


def free_gpu_cache():
    before = torch.cuda.memory_allocated()
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    freed = before - after

    # logger.debug(
    #     f"freed {models.bytes_to_human_readable(freed)} | before={models.bytes_to_human_readable(before)} -> after={models.bytes_to_human_readable(after)}"
    # )
