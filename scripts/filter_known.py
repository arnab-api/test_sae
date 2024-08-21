import argparse
import json
import logging
import os
from typing import Optional

import torch
import transformers
from tqdm.auto import tqdm

from src.dataset import load_relation_dataset
from src.functional import filter_relation_samples_by_model_knowledge, free_gpu_cache
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)


logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


def filter_known(
    model_name: str,
    limit: Optional[int] = None,
    save_dir: str = "relation_known",
):
    mt = ModelandTokenizer(
        model_key=model_name,
        torch_dtype=torch.float16,
    )

    dataset = load_relation_dataset().filter(relation_type=["factual"])
    for relation in tqdm(dataset):
        filter_relation_samples_by_model_knowledge(
            mt=mt, relation=relation, modify_inplace=True, limit=limit
        )
        free_gpu_cache()

    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        save_dir,
    )

    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f'{mt.name.split("/")[-1]}.json'), "w") as f:
        json.dump(dataset.to_json(), f)


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
        default="relation_known",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(args)

    filter_known(model_name=args.model, limit=args.limit, save_dir=args.save_dir)
