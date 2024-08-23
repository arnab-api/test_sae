## Do SAEs merely learn spurious correlations of the data they were trained on? 

In this short (~2 workdays) project I set out to investigate the question in the title. I explained my experiment setup and summarized (very) preliminary findings in this [<ins>**google doc**</ins>](https://docs.google.com/document/d/1a4Xj2rCcdnmSixCe-sA-7OPDiMamEAiT3PDyCPaSjhI/edit). I would suggest reading the doc as a sketch or inspiration. Do comment/suggest if you have any ideas or questions. And if you are interested in collaborating, please reach out to me.

### Setup

Import the conda environment from the `conda_env.yml` with

```bash
conda env create -f conda_env.yml
```

The conda environment probably has more packages than necessary. You will also need to create `env.yml` file in the root directory - checkout `env.yml.example`.

I also used the [`dictionary_learning`](https://github.com/saprmarks/dictionary_learning) package. You will need to replace the `dictionary_learning/training.py` file with [this](https://pastebin.com/wa8A0GBF) - which is minimally modified to address some wandb logging issues and removes a few lines to avoid saving a config that results in an `nnsight` error (`Envoy is not serializable`).