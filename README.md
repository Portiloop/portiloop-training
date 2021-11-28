# The Portiloop

![Prototype](https://github.com/nicolasvalenchon/Portiloop/blob/main/images/photo_portiloop.jpg)

## Quick start guide

- clone the repo
- cd to the root of the repo (i.e., the folder where `setup.py` is)
- pip install with the -e option:
```terminal
pip install -e .
```
- download the [datasets.zip](https://github.com/nicolasvalenchon/Portiloop/releases/download/v0.0.2/dataset.zip) and the [experiments.zip](https://github.com/nicolasvalenchon/Portiloop/releases/download/v0.0.2/experiments.zip) files
- unzip the `datasets.zip` file and paste its content under `Portiloop>portiloop_software>dataset`
- unzip the `experiments.zip` file and paste its content under `Portiloop>portiloop_software>experiments`

### Simulation:
The `simulate_Portiloop_1_input_classification.ipynb` [notebook](https://github.com/nicolasvalenchon/Portiloop/blob/main/notebooks/simulate_Portiloop_1_input_classification.ipynb) enables stimulating the Portiloop system and perform inference.
This notebook can be executed with `jupyter notebook`.

### Offline inference:
We enable easily using out trained artificial neural network on your own data to detect sleep spindles (note that the data must be collected in the same experimental setting as MODA for this to work, see [our paper](https://arxiv.org/abs/2107.13473)).

This is easily done by writing your signal in a simple text file, on the model of the `example_data_not_annotated.txt` file provided in the `datasets.zip` file.
Your file can then be directly used for inference in our `offline_inference` [notebook](https://github.com/nicolasvalenchon/Portiloop/blob/main/notebooks/offline_inference.ipynb).

### Training:
Functions used for training are defined in python under the `portiloop_software` folder.
We provide [bash scripts examples](https://github.com/nicolasvalenchon/Portiloop/releases/download/v0.0.2/scripts.zip) for `SLURM` to train the model on HPC systems.
Adapt these scripts to your configuration.
Your training curves can be visualized in real time easily using [wandb](https://wandb.ai/portiloop) (the code is ready, you may adapt it to your project name and entity).

### Hardware implementation:
The current hardware implementation (pynq FPGA with Vivado / Vivado HLS) is provided under the `portiloop_hardware` folder.
