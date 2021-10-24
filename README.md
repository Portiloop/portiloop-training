# The Portiloop

![Prototype](https://github.com/nicolasvalenchon/Portiloop/blob/main/images/photo_portiloop.jpg)

## Quick start guide

- clone the repo
- cd to the root of the repo (i.e., the folder where `setup.py` is)
- pip install with the -e option:
```terminal
pip install -e .
```
- download the [datasets.zip](https://github.com/nicolasvalenchon/Portiloop/releases/download/v0.0.1/dataset.zip) and the [experiments.zip](https://github.com/nicolasvalenchon/Portiloop/releases/download/v0.0.1/experiments.zip) files
- unzip the `datasets.zip` file and paste its content under `Portiloop>Software>dataset`
- unzip the `experiments.zip` file and paste its content under `Portiloop>Software>experiments`

### Offline inference / simulation:
The `simulate_Portiloop_1_input_classification.ipynb` [notebook](https://github.com/nicolasvalenchon/Portiloop/blob/main/notebooks/simulate_Portiloop_1_input_classification.ipynb) enables stimulating the Portiloop system and perform inference.
This notebook can be executed with `jupyter notebook`.

### Training:
Functions used for training are defined in python under the `Software` folder.
We provide [bash scripts examples](https://github.com/nicolasvalenchon/Portiloop/releases/download/v0.0.1/scripts.zip) for `SLURM` to train the model on HPC systems.
Adapt these scripts to your configuration.
Your training curves can be visualized in real time easily using [wandb](https://wandb.ai/portiloop) (the code is ready, you may adapt it to your project name and entity).

### Hardware implementation:
The current hardware implementation (pynq FPGA with Vivado / Vivado HLS) is provided under the `Hardware` folder.
