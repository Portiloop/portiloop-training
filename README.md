# The Portiloop

![Prototype](https://github.com/nicolasvalenchon/Portiloop/blob/main/images/photo_portiloop.jpg)

Your training curves can be visualized in the Portiloop [wandb project](https://wandb.ai/portiloop).

## Quick start guide

- clone the repo
- cd to the root of the repo where `setup.py` is
- pip install with the -e option:
```terminal
pip install -e .
```
- download the datasets and the experiments zip files
- unzip the `datasets.zip` file and paste its content under `Portiloop>Software>dataset`
- unzip the `experiments.zip` file and paste its content under `Portiloop>Software>experiments`

### Inference / Portiloop simulation:
The `simulate_Portiloop_1_input_classification.ipynb` [notebook](https://github.com/nicolasvalenchon/Portiloop/blob/main/notebooks/simulate_Portiloop_1_input_classification.ipynb) enables stimulating the Portiloop system with and perform inference.
This notebook can be executed with `jupyter notebook`.

### Training:
We provide the bash scripts examples for `slurm` to train the model on HPC systems.
Adapt these scripts to your configuration.
