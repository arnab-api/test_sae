import gc
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel
from tqdm.auto import tqdm

from src.dataset import InContextQuery, Relation
from src.functional import (
    find_token_range,
    get_all_module_states,
    get_module_nnsight,
    guess_subject,
    predict_next_token,
)
from src.models import ModelandTokenizer, is_llama_variant, prepare_input
from src.utils.typing import PredictedToken, Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


def insert_padding_before_subj(
    inp: TokenizerOutput,
    subj_range: tuple[int, int],
    subj_ends: int,
    pad_id: int,
    fill_attn_mask: bool = False,
):
    """

    Inserts padding tokens before the subject in the query to balance the input tensor.

    TEST:

    for idx, (tok_id, attn_mask) in enumerate(zip(clean_inputs.input_ids[0], clean_inputs.attention_mask[0])):
        print(f"{idx=} [{attn_mask}] | {mt.tokenizer.decode(tok_id)}")

    """
    pad_len = subj_ends - subj_range[1]
    inp["input_ids"] = torch.cat(
        [
            inp.input_ids[:, : subj_range[0]],
            torch.full(
                (1, pad_len),
                pad_id,
                dtype=inp.input_ids.dtype,
                device=inp.input_ids.device,
            ),
            inp.input_ids[:, subj_range[0] :],
        ],
        dim=1,
    )

    inp["attention_mask"] = torch.cat(
        [
            inp.attention_mask[:, : subj_range[0]],
            torch.full(
                (1, pad_len),
                fill_attn_mask,
                dtype=inp.attention_mask.dtype,
                device=inp.attention_mask.device,
            ),
            inp.attention_mask[:, subj_range[0] :],
        ],
        dim=1,
    )
    return inp


@torch.inference_mode()
def patched_run(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    states: dict[tuple[str, int], torch.Tensor],
    scan: bool = False,
    kind: Literal["residual", "mlp", "attention"] = "residual",
) -> torch.Tensor:
    with mt.trace(inputs, scan=scan) as trace:
        for location in states:
            layer_name, token_idx = location
            module = get_module_nnsight(mt, layer_name)
            current_states = module.output if kind == "mlp" else module.output[0]
            current_states[0, token_idx, :] = states[location]
        logits = mt.output.logits[0][-1].save()
    return logits


def get_window(layer_name_format, idx, window_size, n_layer):
    return [
        layer_name_format.format(i)
        for i in range(
            max(0, idx - window_size // 2), min(n_layer - 1, idx + window_size // 2) + 1
        )
    ]


@torch.inference_mode()
def calculate_indirect_effects(
    mt: ModelandTokenizer,
    locations: list[tuple[int, int]],  # layer_idx, token_idx
    corrupted_input: TokenizerOutput,
    clean_states: dict[
        tuple[str, int], torch.Tensor
    ],  # expects the states to be in clean_states
    clean_ans_t: int,
    layer_name_format: str,
    window_size: int = 1,
    kind: Literal["residual", "mlp", "attention"] = "residual",
) -> dict[tuple[str, int], float]:
    is_first = True
    indirect_effects = {loc: -1 for loc in locations}
    for loc in tqdm(locations):
        layer_names = get_window(layer_name_format, loc[0], window_size, mt.n_layer)
        token_idx = loc[1]
        states = {(l, token_idx): clean_states[(l, token_idx)] for l in layer_names}
        affected_logits = patched_run(
            mt=mt,
            inputs=corrupted_input,
            states=states,
            scan=is_first,
            kind=kind,
        )
        prob = affected_logits.softmax(dim=-1)[clean_ans_t].item()
        indirect_effects[loc] = prob
        is_first = False
    return indirect_effects


@dataclass
class CausalTracingResult(DataClassJsonMixin):
    clean_input_toks: list[str]
    corrupt_input_toks: list[str]
    trace_start_idx: int
    answer: PredictedToken
    low_score: float
    indirect_effects: torch.Tensor
    normalized: bool
    kind: Literal["residual", "mlp", "attention"] = "residual"
    window: int = 1


@torch.inference_mode()
def trace_important_states(
    mt: ModelandTokenizer,
    prompt_template: str,
    clean_subj: str,
    patched_subj: str,
    clean_input: Optional[TokenizerOutput] = None,
    patched_input: Optional[TokenizerOutput] = None,
    kind: Literal["residual", "mlp", "attention"] = "residual",
    window_size: int = 1,
    normalize=True,
) -> CausalTracingResult:

    if clean_input is None:
        clean_input = prepare_input(
            prompts=prompt_template.format(clean_subj),
            tokenizer=mt,
            return_offsets_mapping=True,
        )
    if patched_input is None:
        patched_input = prepare_input(
            prompts=prompt_template.format(patched_subj),
            tokenizer=mt,
            return_offsets_mapping=True,
        )

    clean_subj_range = find_token_range(
        string=prompt_template.format(clean_subj),
        substring=clean_subj,
        tokenizer=mt.tokenizer,
        occurrence=-1,
        offset_mapping=clean_input["offset_mapping"][0],
    )
    patched_subj_range = find_token_range(
        string=prompt_template.format(patched_subj),
        substring=patched_subj,
        tokenizer=mt.tokenizer,
        occurrence=-1,
        offset_mapping=patched_input["offset_mapping"][0],
    )

    if clean_subj_range == patched_subj_range:
        subj_start, subj_end = clean_subj_range
    else:
        subj_end = max(clean_subj_range[1], patched_subj_range[1])
        clean_input = insert_padding_before_subj(
            inp=clean_input,
            subj_range=clean_subj_range,
            subj_ends=subj_end,
            pad_id=mt.tokenizer.pad_token_id,
            fill_attn_mask=True,
        )
        patched_input = insert_padding_before_subj(
            inp=patched_input,
            subj_range=patched_subj_range,
            subj_ends=subj_end,
            pad_id=mt.tokenizer.pad_token_id,
            fill_attn_mask=True,
        )

        clean_subj_shift = subj_end - clean_subj_range[1]
        clean_subj_range = (clean_subj_range[0] + clean_subj_shift, subj_end)
        patched_subj_shift = subj_end - patched_subj_range[1]
        patched_subj_range = (patched_subj_range[0] + patched_subj_shift, subj_end)
        subj_start = min(clean_subj_range[0], patched_subj_range[0])

    trace_start_idx = 0
    if (
        clean_input.input_ids[0][0]
        == patched_input.input_ids[0][0]
        == mt.tokenizer.pad_token_id
    ):
        trace_start_idx = 1

    # base run with the patched subject
    patched_states = get_all_module_states(mt=mt, input=patched_input, kind=kind)
    answer = predict_next_token(mt=mt, inputs=patched_input, k=1)[0][0]
    base_probability = answer.prob
    logger.debug(f"{answer=}")

    # clean run
    clean_answer, track_ans = predict_next_token(
        mt=mt, inputs=clean_input, k=1, token_of_interest=answer.token
    )
    clean_answer = clean_answer[0][0]
    low_probability = track_ans[0][1].prob
    logger.debug(f"{clean_answer=}")
    logger.debug(f"{track_ans=}")

    logger.debug("---------- tracing important states ----------")

    assert (
        answer.token != clean_answer.token
    ), "Answers in the clean and corrupt runs are the same"

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError(f"kind must be one of 'residual', 'mlp', 'attention'")

    # calculate indirect effects in the patched run
    locations = [
        (layer_idx, token_idx)
        for layer_idx in range(mt.n_layer)
        for token_idx in range(trace_start_idx, clean_input.input_ids.size(1))
    ]
    indirect_effects = calculate_indirect_effects(
        mt=mt,
        locations=locations,
        corrupted_input=clean_input,
        clean_states=patched_states,
        clean_ans_t=answer.token_id,
        layer_name_format=layer_name_format,
        window_size=window_size,
        kind=kind,
    )

    indirect_effect_matrix = []
    for token_idx in range(trace_start_idx, clean_input.input_ids.size(1)):
        indirect_effect_matrix.append(
            [
                indirect_effects[(layer_idx, token_idx)]
                for layer_idx in range(mt.n_layer)
            ]
        )

    indirect_effect_matrix = torch.tensor(indirect_effect_matrix)
    if normalize:
        indirect_effect_matrix = (indirect_effect_matrix - low_probability) / (
            base_probability - low_probability
        )

    return CausalTracingResult(
        clean_input_toks=[
            mt.tokenizer.decode(tok) for tok in patched_input.input_ids[0]
        ],
        corrupt_input_toks=[
            mt.tokenizer.decode(tok) for tok in clean_input.input_ids[0]
        ],
        trace_start_idx=trace_start_idx,
        answer=answer,
        low_score=low_probability,
        indirect_effects=indirect_effect_matrix,
        normalized=normalize,
        kind=kind,
        window=window_size,
    )


@torch.inference_mode()
def trace_important_states_ICQ(
    mt: ModelandTokenizer,
    clean_query: InContextQuery,
    corrupt_query: InContextQuery,
    kind: Literal["residual", "mlp", "attention"] = "residual",
    trace_token_strategy: Literal["subj_query", "all"] = "subj_query",
    window_size: int = 1,
    normalize=True,
) -> CausalTracingResult:
    assert (
        clean_query.template == corrupt_query.template
    ), "Queries do not have the same template"

    clean_inputs = prepare_input(
        prompts=clean_query.query, tokenizer=mt, return_offsets_mapping=True
    )
    corrupt_inputs = prepare_input(
        prompts=corrupt_query.query, tokenizer=mt, return_offsets_mapping=True
    )

    if trace_token_strategy == "subj_query":
        clean_subj_range = find_token_range(
            string=clean_query.query,
            substring=clean_query.subject,
            tokenizer=mt.tokenizer,
            occurrence=-1,
            offset_mapping=clean_inputs["offset_mapping"][0],
        )
        corrupt_subj_range = find_token_range(
            string=corrupt_query.query,
            substring=corrupt_query.subject,
            tokenizer=mt.tokenizer,
            occurrence=-1,
            offset_mapping=corrupt_inputs["offset_mapping"][0],
        )
        logger.debug(f"{clean_subj_range=} | {corrupt_subj_range=}")

        # always insert 1 padding token
        subj_end = max(clean_subj_range[1], corrupt_subj_range[1]) + 1
        logger.debug(f"setting {subj_end=}")

        clean_inputs = insert_padding_before_subj(
            clean_inputs,
            clean_subj_range,
            subj_end,
            pad_id=mt.tokenizer.pad_token_id,
        )
        corrupt_inputs = insert_padding_before_subj(
            corrupt_inputs,
            corrupt_subj_range,
            subj_end,
            pad_id=mt.tokenizer.pad_token_id,
        )

        clean_shift = subj_end - clean_subj_range[1]
        clean_subj_range = (clean_subj_range[0] + clean_shift, subj_end)

        corrupt_shift = subj_end - corrupt_subj_range[1]
        corrupt_subj_range = (corrupt_subj_range[0] + corrupt_shift, subj_end)

        logger.debug(f"<shifted> {clean_subj_range=} | {corrupt_subj_range=}")

    elif trace_token_strategy == "all":
        clean_subj_ranges = [
            find_token_range(
                string=clean_query.query,
                substring=clean_query.subject,
                tokenizer=mt.tokenizer,
                occurrence=order,
                offset_mapping=clean_inputs["offset_mapping"][0],
            )
            for order in [0, -1]
        ]

        corrupt_subj_ranges = [
            find_token_range(
                string=corrupt_query.query,
                substring=corrupt_query.subject,
                tokenizer=mt.tokenizer,
                occurrence=order,
                offset_mapping=corrupt_inputs["offset_mapping"][0],
            )
            for order in [0, -1]
        ]

        clean_cofa_range = find_token_range(
            string=clean_query.query,
            substring=guess_subject(clean_query.cf_description),
            tokenizer=mt.tokenizer,
            occurrence=-1,
            offset_mapping=clean_inputs["offset_mapping"][0],
        )

        corrupt_cofa_range = find_token_range(
            string=corrupt_query.query,
            substring=guess_subject(corrupt_query.cf_description),
            tokenizer=mt.tokenizer,
            occurrence=-1,
            offset_mapping=corrupt_inputs["offset_mapping"][0],
        )

        # align the subjects in the context
        subj_end_in_context = (
            max(clean_subj_ranges[0][1], corrupt_subj_ranges[0][1]) + 1
        )
        clean_inputs = insert_padding_before_subj(
            clean_inputs,
            clean_subj_ranges[0],
            subj_end_in_context,
            pad_id=mt.tokenizer.pad_token_id,
        )
        corrupt_inputs = insert_padding_before_subj(
            corrupt_inputs,
            corrupt_subj_ranges[0],
            subj_end_in_context,
            pad_id=mt.tokenizer.pad_token_id,
        )

        n_clean_pads = subj_end_in_context - clean_subj_ranges[0][1]
        clean_subj_ranges[1] = (
            clean_subj_ranges[1][0] + n_clean_pads,
            clean_subj_ranges[1][1] + n_clean_pads,
        )
        clean_cofa_range = (
            clean_cofa_range[0] + n_clean_pads,
            clean_cofa_range[1] + n_clean_pads,
        )

        n_corrupt_pads = subj_end_in_context - corrupt_subj_ranges[0][1]
        corrupt_subj_ranges[1] = (
            corrupt_subj_ranges[1][0] + n_corrupt_pads,
            corrupt_subj_ranges[1][1] + n_corrupt_pads,
        )
        corrupt_cofa_range = (
            corrupt_cofa_range[0] + n_corrupt_pads,
            corrupt_cofa_range[1] + n_corrupt_pads,
        )

        # align the counterfactuals in the context
        cofa_ends_in_context = max(clean_cofa_range[1], corrupt_cofa_range[1]) + 1
        clean_inputs = insert_padding_before_subj(
            clean_inputs,
            clean_cofa_range,
            cofa_ends_in_context,
            pad_id=mt.tokenizer.pad_token_id,
        )
        corrupt_inputs = insert_padding_before_subj(
            corrupt_inputs,
            corrupt_cofa_range,
            cofa_ends_in_context,
            pad_id=mt.tokenizer.pad_token_id,
        )

        n_clean_pads = cofa_ends_in_context - clean_cofa_range[1]
        clean_subj_ranges[1] = (
            clean_subj_ranges[1][0] + n_clean_pads,
            clean_subj_ranges[1][1] + n_clean_pads,
        )
        n_corrupt_pads = cofa_ends_in_context - corrupt_cofa_range[1]
        corrupt_subj_ranges[1] = (
            corrupt_subj_ranges[1][0] + n_corrupt_pads,
            corrupt_subj_ranges[1][1] + n_corrupt_pads,
        )

        # align the subjects in the query
        subj_ends_in_query = max(clean_subj_ranges[1][1], corrupt_subj_ranges[1][1]) + 1
        clean_inputs = insert_padding_before_subj(
            clean_inputs,
            clean_subj_ranges[1],
            subj_ends_in_query,
            pad_id=mt.tokenizer.pad_token_id,
        )
        corrupt_inputs = insert_padding_before_subj(
            corrupt_inputs,
            corrupt_subj_ranges[1],
            subj_ends_in_query,
            pad_id=mt.tokenizer.pad_token_id,
        )

    else:
        raise ValueError("trace_token_strategy must be one of 'subj_query', 'all'")

    for idx, (t1, a1, t2, a2) in enumerate(
        zip(
            clean_inputs.input_ids[0],
            clean_inputs.attention_mask[0],
            corrupt_inputs.input_ids[0],
            corrupt_inputs.attention_mask[0],
        )
    ):
        logger.debug(
            f"{idx=} =>  [{a1}] {mt.tokenizer.decode(t1)} || [{a2}] {mt.tokenizer.decode(t2)}"
        )

    # trace start idx
    if trace_token_strategy == "subj_query":
        trace_start_idx = min(clean_subj_range[0], corrupt_subj_range[0])
    elif trace_token_strategy == "all":
        trace_start_idx = 1

    # clean run
    clean_states = get_all_module_states(mt=mt, input=clean_inputs, kind=kind)
    answer = predict_next_token(mt=mt, inputs=clean_inputs, k=1)[0][0]
    base_probability = answer.prob
    logger.debug(f"{answer=}")

    # corrupted run
    # corrupt_states = get_all_module_states(mt=mt, input=corrupt_inputs, kind=kind)
    corrupt_answer, track_ans = predict_next_token(
        mt=mt, inputs=corrupt_inputs, k=1, token_of_interest=answer.token
    )
    corrupt_answer = corrupt_answer[0][0]
    corrupt_probability = track_ans[0][1].prob
    logger.debug(f"{corrupt_answer=}")
    logger.debug(f"{track_ans=}")

    logger.debug("---------- tracing important states ----------")

    assert (
        answer.token != corrupt_answer.token
    ), "Answers in the clean and corrupt runs are the same"

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError(f"kind must be one of 'residual', 'mlp', 'attention'")

    # calculate indirect effects in the patched run
    locations = [
        (layer_idx, token_idx)
        for layer_idx in range(mt.n_layer)
        for token_idx in range(trace_start_idx, clean_inputs.input_ids.size(1))
    ]
    indirect_effects = calculate_indirect_effects(
        mt=mt,
        locations=locations,
        corrupted_input=corrupt_inputs,
        clean_states=clean_states,
        clean_ans_t=answer.token_id,
        layer_name_format=layer_name_format,
        window_size=window_size,
        kind=kind,
    )

    indirect_effect_matrix = []
    for token_idx in range(trace_start_idx, clean_inputs.input_ids.size(1)):
        indirect_effect_matrix.append(
            [
                indirect_effects[(layer_idx, token_idx)]
                for layer_idx in range(mt.n_layer)
            ]
        )

    indirect_effect_matrix = torch.tensor(indirect_effect_matrix)
    if normalize:
        indirect_effect_matrix = (indirect_effect_matrix - corrupt_probability) / (
            base_probability - corrupt_probability
        )

    return CausalTracingResult(
        clean_input_toks=[
            mt.tokenizer.decode(tok) for tok in clean_inputs.input_ids[0]
        ],
        corrupt_input_toks=[
            mt.tokenizer.decode(tok) for tok in corrupt_inputs.input_ids[0]
        ],
        trace_start_idx=trace_start_idx,
        answer=answer,
        low_score=corrupt_probability,
        indirect_effects=indirect_effect_matrix,
        normalized=normalize,
        kind=kind,
        window=window_size,
    )
