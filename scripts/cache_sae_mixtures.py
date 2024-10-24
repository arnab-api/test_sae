import argparse
import logging
import os

import numpy as np
import torch
import transformers
from datasets import load_dataset
from tqdm.auto import tqdm

from dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder
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
    sae_data_name: str,
    eval_dataset_name: str,
    sae_data_checkpoint: int = 2000000,
    limit: int = 20000,
    context_limit: int = 1024,
    save_dir: str = "cache_sae_mixtures",
):
    mt = ModelandTokenizer(
        model_key=model_name,
        torch_dtype=torch.float16,
    )

    model_data_dir = os.path.join(
        model_name.split("/")[-1],
        sae_data_name.split("/")[-1],
    )

    cache_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        save_dir,
        eval_dataset_name.split("/")[-1],
        model_data_dir,
        str(sae_data_checkpoint),
    )
    os.makedirs(cache_dir, exist_ok=True)

    sae_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        "train_sae",
        model_data_dir,
        str(sae_data_checkpoint),
        "trainer_0/ae.pt",
    )
    sae = GatedAutoEncoder.from_pretrained(path=sae_dir, device=mt.device).to(mt.dtype)

    dataset = load_dataset(eval_dataset_name)
    context_limit = 1024
    limit = min(limit, len(dataset["train"]))

    sae_layer_name = mt.layer_name_format.format(mt.n_layer // 2)
    relu = torch.nn.ReLU()

    for doc_index, doc in tqdm(enumerate(dataset["train"][:limit]["text"])):
        inputs = prepare_input(prompts=doc, tokenizer=mt)
        if inputs["input_ids"].shape[1] > context_limit:
            inputs["input_ids"] = inputs["input_ids"][:, :context_limit]
            inputs["attention_mask"] = inputs["attention_mask"][:, :context_limit]

        # print(f"{doc=}")
        # logger.info(inputs["input_ids"].shape)

        with mt.trace(inputs, scan=False, validate=False) as trace:
            module = get_module_nnsight(mt, sae_layer_name)
            sae_input = module.output[0].save()

        sae_mixture = sae.encode(sae_input)
        # logger.info(f"{sae_input.shape=} | {sae_mixture.shape=}")

        cache = {
            "layer": sae_layer_name,
            "doc": doc,
            "sae_input": sae_input.detach().cpu().numpy().astype(np.float32),
            "sae_mixture": sae_mixture.detach().cpu().numpy().astype(np.float32),
        }

        cache_path = os.path.join(cache_dir, f"{doc_index}")
        np.savez_compressed(cache_path, **cache)

        free_gpu_cache()

        logger.info(
            f"Processed {doc_index+1}/{limit} tokens ({(doc_index+1)/limit:.2%})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            # "EleutherAI/pythia-160m",
            # "EleutherAI/pythia-410m",
            # "openai-community/gpt2",
            # "openai-community/gpt2-xl",
            "meta-llama/Llama-3.2-1B",
            "Qwen/Qwen2.5-1.5B",
            "allenai/OLMo-1B-0724-hf",
        ],
        default="meta-llama/Llama-3.2-1B",
    )

    parser.add_argument(
        "--sae-data",
        type=str,
        choices=["wikimedia/wikipedia", "roneneldan/TinyStories"],
        default="wikimedia/wikipedia",
    )

    parser.add_argument(
        "--eval-data",
        type=str,
        choices=[
            "mickume/harry_potter_tiny",
            "jahjinx/IMDb_movie_reviews",
        ],
        default="jahjinx/IMDb_movie_reviews",
    )

    parser.add_argument(
        "--sae-checkpoint",
        type=int,
        default=2000000,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="cache_sae_mixtures",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(args)

    cache_activations(
        model_name=args.model,
        sae_data_name=args.sae_data,
        eval_dataset_name=args.eval_data,
        limit=args.limit,
        save_dir=args.save_dir,
        sae_data_checkpoint=args.sae_checkpoint,
    )
