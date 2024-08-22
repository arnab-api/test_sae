import logging
import os
from typing import Literal, Optional

import matplotlib.pyplot as plt
import torch

from src.trace import CausalTracingResult

logger = logging.getLogger(__name__)


def get_color_map(kind: Literal["residual", "mlp", "attention"] = "residual"):
    if kind == "residual":
        return "Purples"
    if kind == "mlp":
        return "Greens"
    if kind == "attention":
        return "Reds"
    return "Greys"


def replace_special_tokens(token_list, pad_token="[PAD]"):
    for i, token in enumerate(token_list):
        if token.startswith("<|") and token.endswith("|>"):
            token_list[i] = pad_token
    return token_list


def plot_trace_heatmap(
    result: CausalTracingResult,
    savepdf: Optional[str] = None,
    model_name: Optional[str] = None,
    scale_range: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
):
    scores = result.indirect_effects
    clean_tokens = replace_special_tokens(result.clean_input_toks)
    corrupt_tokens = replace_special_tokens(result.corrupt_input_toks)

    tokens = []
    for clean_tok, corrupt_tok in zip(
        clean_tokens[result.trace_start_idx :], corrupt_tokens[result.trace_start_idx :]
    ):
        tokens.append(
            f"{clean_tok}/{corrupt_tok}" if clean_tok != corrupt_tok else clean_tok
        )

    with plt.rc_context(
        rc={
            "font.family": "Times New Roman",
            "font.size": 6,
        }
    ):
        fig, ax = plt.subplots(figsize=(3.5, len(tokens) * 0.08 + 1.8), dpi=200)
        scale_kwargs = dict(
            vmin=result.low_score if scale_range is None else scale_range[0],
        )
        if scale_range is not None:
            scale_kwargs["vmax"] = scale_range[1]

        heatmap = ax.pcolor(
            scores,
            cmap=get_color_map(result.kind),
            **scale_kwargs,
        )

        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(scores))])
        ax.set_xticks([0.5 + i for i in range(0, scores.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, scores.shape[1] - 6, 5)))
        # print(len(tokens))
        ax.set_yticklabels(tokens)

        if title is None:
            title = f"Indirect Effects of {result.kind.upper()} Layers"
        ax.set_title(title)

        if result.window == 1:
            ax.set_xlabel(f"single restored layer within {model_name}")
        else:
            ax.set_xlabel(
                f"center of interval of {result.window} restored {result.kind.upper()} layers"
            )

        color_scale = plt.colorbar(heatmap)
        color_scale.ax.set_title(
            f"p({result.answer.token.strip()})", y=-0.12, fontsize=10
        )

        if savepdf is not None:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight", dpi=300)
        plt.show()


import matplotlib.pyplot as plt


def visualize_attn_matrix(
    attn_matrix: torch.Tensor,
    tokens: list[str],
    remove_eos: Optional[str] = None,
    title: str | None = None,
    color_scheme: str = "Blues",
    savepdf: str | None = None,
    start_idx: int = 0,
):
    assert (
        attn_matrix.shape[0] == attn_matrix.shape[1]
    ), "Attention matrix must be square"
    assert (
        len(tokens) == attn_matrix.shape[-1]
    ), "Tokens and attention matrix must have the same length"

    if remove_eos and start_idx == 0:
        start_idx = 1 if tokens[0] == remove_eos else 0

    plt.rcParams["figure.dpi"] = 300
    with plt.rc_context(
        rc={
            "font.family": "Times New Roman",
            # "font.size": 2,
        }
    ):

        img = plt.imshow(
            attn_matrix[start_idx:, start_idx:],
            cmap=color_scheme,
            interpolation="nearest",
            vmin=0,
            vmax=attn_matrix[start_idx:, start_idx:].max().item(),
        )
        plt.colorbar(img, orientation="vertical")

        plt.xticks(
            range(len(tokens) - start_idx),
            [f'" {t}"' for t in tokens[start_idx:]],
            rotation=90,
        )
        plt.yticks(
            range(len(tokens) - start_idx),
            [f'" {t}"' for t in tokens[start_idx:]],
        )

        plt.ylabel("Query Token")
        plt.xlabel("Key Token")

        if title:
            plt.title(title)

        if savepdf is not None:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight", dpi=300)

        plt.show()


from src.utils.typing import ArrayLike, PathLike


def matrix_heatmap(
    matrix: ArrayLike,
    limit_dim: int = 100,
    canvas: plt = plt,
    save_path: PathLike | None = None,
    title: str | None = None,
    tick_gap: int | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> None:
    """Plot cross section of matrix as a heatmap."""

    limit_dim = min(limit_dim, matrix.shape[0])

    matrix = torch.stack([w[:limit_dim] for w in matrix[:limit_dim]]).cpu()
    limit = max(abs(matrix.min().item()), abs(matrix.max().item()))
    img = plt.imshow(matrix, cmap="RdBu", interpolation="nearest", vmin=0, vmax=limit)
    canvas.colorbar(img, orientation="vertical")

    if x_label is not None:
        canvas.xlabel(x_label)
    if y_label is not None:
        canvas.ylabel(y_label)
    if tick_gap is not None:
        canvas.xticks(range(0, limit_dim, tick_gap), range(0, limit_dim, tick_gap))
        canvas.yticks(range(0, limit_dim, tick_gap), range(0, limit_dim, tick_gap))

    if title is not None:
        canvas.title(title)
    if save_path is not None:
        canvas.savefig(str(save_path))
    canvas.show()
