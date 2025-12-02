# Music style transfer

This repository contains the implementation of a music style transfer model using Variational Autoencoders (VAE). The model is designed to transfer the style of one piece of music to another while preserving the content.

## Project Structure

- `model/`: Training scripts and model definitions.
- `preprocessing/`: Data preprocessing utilities.
- `data/`: Datasets and analysis results.
- `evaluation/`: Evaluation scripts and metrics.
- `utils/`: Helper functions for audio, files, and plotting.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `docs/`: Documentation, thesis, and diagrams.
- `pipeline_tests/`: Unit and integration tests.

## Methodology & Results

- Master's thesis (Spanish): [docs/Tesis.pdf](https://github.com/LucasSom/style-transfer/blob/main/docs/Tesis.pdf)
- Paper (English): [Zenodo](https://zenodo.org/records/14908040)

The VAE implementation is based on [Rui Guo's work](https://github.com/ruiguo-bio/colab_tension_vae.git).

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
sh dependencies.sh
```

## Usage

The code is based on modular scripts for training, inference, and evaluation.
These task can be executed with the [pydoit](https://pydoit.org/) task runner.
Refer to the `dodo.py` file for available tasks and their configurations.
