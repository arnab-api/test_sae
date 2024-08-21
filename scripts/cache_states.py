import argparse
import logging
import os

import numpy as np
import torch
import transformers
from datasets import load_dataset

from src.functional import free_gpu_cache, get_module_nnsight
from src.models import ModelandTokenizer, prepare_input
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)


logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


def cache_activations(
    model_name: str,
    limit: int = 20000,
    context_limit: int = 1024,
    save_dir: str = "cache_states",
):
    mt = ModelandTokenizer(
        model_key=model_name,
        torch_dtype=torch.float16,
    )

    cache_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        save_dir,
        model_name.split("/")[-1],
    )
    os.makedirs(cache_dir, exist_ok=True)

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    counter = 0

    for doc_idx in np.random.permutation(len(ds["train"])):
        doc = ds["train"][int(doc_idx)]["text"]
        inputs = prepare_input(prompts=doc, tokenizer=mt)

        if inputs["input_ids"].shape[1] < 30:  # ignore too short documents
            continue
        elif inputs["input_ids"].shape[1] > context_limit:
            inputs["input_ids"] = inputs["input_ids"][:, :context_limit]
            inputs["attention_mask"] = inputs["attention_mask"][:, :context_limit]

        doc_cache: dict[int, torch.Tensor] = {}

        with mt.trace(inputs, scan=False, validate=False) as trace:
            for layer in mt.layer_names:
                module = get_module_nnsight(mt, layer)
                doc_cache[layer] = module.output[0].save()

        for layer in mt.layer_names:
            doc_cache[layer] = (
                doc_cache[layer].detach().cpu().numpy().astype(np.float32)
            )

        cache_path = os.path.join(cache_dir, f"{doc_idx}")
        np.savez_compressed(cache_path, **doc_cache)

        free_gpu_cache()
        counter += inputs["input_ids"].shape[1]
        if counter > limit:
            break

        logger.info(f"Processed {counter}/{limit} tokens ({counter/limit:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13B-hf",
            "EleutherAI/gpt-j-6b",
            "openai-community/gpt2",
            "openai-community/gpt2-xl",
        ],
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="cache_states",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(args)

    cache_activations(model_name=args.model, limit=args.limit, save_dir=args.save_dir)
