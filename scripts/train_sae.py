import argparse
import logging
import os
import time
from typing import Optional

import torch
import transformers
from datasets import load_dataset

from dictionary_learning import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder
from dictionary_learning.trainers import (
    TrainerTopK,
    GatedSAETrainer,
    GatedAnnealTrainer,
    StandardTrainer,
)
from dictionary_learning.training import trainSAE
from src.functional import get_module_nnsight
from src.models import ModelandTokenizer
from src.utils import env_utils, experiment_utils, logging_utils

logger = logging.getLogger(__name__)
logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


def train_SAE(
    model_name: str,
    dataset_name: str,
    limit_docs: int = 1000000,
    save_dir: str = "train_sae",
    dictionary_dim: int = 16384,
    text_batch_size: int = 32,
    context_len: int = 256,
    checkpoint_interval: int = 1000,
    store_checkpoints_on_steps: list = [0, 10, 50, 100, 500],
    log_steps: int = 25,
    dictionary_dim_scale_factor: Optional[int] = None,
):

    assert dictionary_dim is not None or dictionary_dim_scale_factor is not None
    assert dictionary_dim is None or dictionary_dim_scale_factor is None

    mt = ModelandTokenizer(
        model_key=model_name,
        torch_dtype=torch.float32,
    )

    cache_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        save_dir,
        model_name.split("/")[-1],
        dataset_name.split("/")[-1],
        str(limit_docs),
    )
    os.makedirs(cache_dir, exist_ok=True)

    if dataset_name == "wikimedia/wikipedia":
        ds = load_dataset(dataset_name, "20231101.en")
    else:
        ds = load_dataset(dataset_name)

    submodule = get_module_nnsight(mt, mt.layer_name_format.format(mt.n_layer // 2))
    activation_dim = mt.n_embd
    dictionary_dim = (
        dictionary_dim
        if dictionary_dim is not None
        else int(activation_dim * dictionary_dim_scale_factor)
    )

    data_iter = iter(ds["train"][:limit_docs]["text"])

    data_buffer = ActivationBuffer(
        data_iter,
        mt,
        submodule,
        d_submodule=activation_dim,
        n_ctxs=text_batch_size,
        ctx_len=context_len,
        refresh_batch_size=text_batch_size,
        device=mt.device,
    )

    # train the sparse autoencoder (SAE)
    n_steps_approx = limit_docs // (text_batch_size)
    warmup_steps = min(max(n_steps_approx // 10, 5), 3000)
    logger.info(f"expected steps: {n_steps_approx}, warmup steps: {warmup_steps}")
    ae = trainSAE(
        data=data_buffer,
        trainer_configs=[
            {
                # "trainer": StandardTrainer,
                # "dict_class": AutoEncoder,
                "trainer": GatedSAETrainer,
                "dict_class": GatedAutoEncoder,
                "activation_dim": activation_dim,
                "dict_size": dictionary_dim,
                "lr": 1e-5,
                "l1_penalty": 3 * 1e-1,  # it seems to need a bit of a push
                "warmup_steps": warmup_steps,
                "resample_steps": None,
                "seed": None,
                "wandb_name": f"DISENT_{mt.name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{str(limit_docs)}",
                "lm_name": mt.name,
                "layer": submodule,
                "submodule_name": mt.layer_name_format.format(mt.n_layer // 2),
            }
        ],
        save_dir=cache_dir,
        save_steps=checkpoint_interval,
        store_checkpoints_on_steps=store_checkpoints_on_steps,
        use_wandb=True,
        wandb_entity="dl-homeworks",
        wandb_project="test_sae",
        log_steps=log_steps,
        wandb_name=f"{mt.name.split('/')[-1]}_{dataset_name.split('/')[-1]}___{str(time.ctime()).replace(' ', '_')}",
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
            "openai-community/gpt2",
            # "openai-community/gpt2-xl",
            "google/gemma-2-2b",
            "meta-llama/Llama-3.2-1B",
            "Qwen/Qwen2.5-1.5B",
            "allenai/OLMo-1B-0724-hf",
        ],
        default="openai-community/gpt2",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikimedia/wikipedia", "roneneldan/TinyStories"],
        default="wikimedia/wikipedia",
    )

    parser.add_argument(
        "--doc-limit",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="train_sae",
    )

    parser.add_argument(
        "--dictionary-dim",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--context-len",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--log-steps",
        type=int,
        default=10,
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(args)

    train_SAE(
        model_name=args.model,
        dataset_name=args.dataset,
        limit_docs=args.doc_limit,
        save_dir=args.save_dir,
        dictionary_dim=args.dictionary_dim,
        text_batch_size=args.batch_size,
        context_len=args.context_len,
        checkpoint_interval=args.save_steps,
        log_steps=args.log_steps,
    )
