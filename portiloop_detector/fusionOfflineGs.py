import os
from pathlib import Path

import pandas as pd

path_dataset = Path(__file__).absolute().parent.parent / 'dataset'
os.chdir(path_dataset)

offline_vector = pd.read_csv("detVect.txt", header=None)[1]
dataset = pd.read_csv("dataset_big_envelope_matlab.txt", header=None)

dataset[2][(offline_vector >= 0.8) & (dataset[2] == 0)] = 0.8
dataset.to_csv("dataset_big_envelope_fusion.txt", header=False, index=False)
