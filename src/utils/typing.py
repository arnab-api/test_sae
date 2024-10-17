"""Some useful type aliases relevant to this project."""

import pathlib
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy
import torch
import transformers
import transformers.modeling_outputs
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel

ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = str | pathlib.Path
Device = str | torch.device

# Throughout this codebase, we use HuggingFace model implementations.
Model = (
    LanguageModel
    | transformers.GPT2LMHeadModel
    | transformers.GPTJForCausalLM
    | transformers.GPTNeoXForCausalLM
    | transformers.LlamaForCausalLM
    | transformers.Gemma2ForCausalLM
    | transformers.GemmaForCausalLM
    | transformers.Qwen2ForCausalLM
    | transformers.OlmoForCausalLM
)
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping = Sequence[tuple[int, int]]
TokenizerOutput = transformers.tokenization_utils_base.BatchEncoding

ModelInput = transformers.BatchEncoding
ModelOutput = transformers.modeling_outputs.CausalLMOutput
ModelGenerateOutput = transformers.generation.utils.GenerateOutput | torch.LongTensor

Layer = int | Literal["emb"] | Literal["ln_f"]

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = list[str] | tuple[str, ...]


@dataclass(frozen=True)
class PredictedToken(DataClassJsonMixin):
    """A predicted token and its probability."""

    token: str
    prob: float
    token_id: Optional[int] = None

    def __str__(self) -> str:
        return f'"{self.token}" (p={self.prob:.3f})'
