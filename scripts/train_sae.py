import argparse
import logging
import os
import time
from typing import Optional

import torch
import transformers
from datasets import load_dataset

from dictionary_learning import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.trainers.standard import StandardTrainer
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
    save_dir: str = "trained_sae",
    dictionary_dim: int = 4096,
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
        d_submodule=activation_dim,  # output dimension of the model component
        n_ctxs=500,  # you can set this higher or lower dependong on your available memory
        device=mt.device,  # doesn't have to be the same device that you train your autoencoder on
    )  # buffer will return batches of tensors of dimension = submodule's output dimension

    # train the sparse autoencoder (SAE)
    ae = trainSAE(
        data=data_buffer,
        trainer_configs=[
            {
                "trainer": StandardTrainer,
                "dict_class": AutoEncoder,
                "activation_dim": activation_dim,
                "dict_size": dictionary_dim,
                "lr": 1e-3,
                "l1_penalty": 1e-1,
                "warmup_steps": 10000,
                "resample_steps": None,
                "seed": None,
                "wandb_name": f"{mt.name.split('/')[-1]}_{dataset_name.split('/')[-1]}",
                "lm_name": mt.name,
                "layer": submodule,
                "submodule_name": "residual",
            }
        ],
        save_dir=cache_dir,
        save_steps=1000,
        use_wandb=True,
        wandb_entity="dl-homeworks",
        wandb_project="test_sae",
        log_steps=50,
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
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "openai-community/gpt2",
            "openai-community/gpt2-xl",
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
        "--save_dir",
        type=str,
        default="trained_saes",
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
    )
