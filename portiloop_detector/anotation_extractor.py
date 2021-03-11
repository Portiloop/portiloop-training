# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:00:21 2021

@author: Nicolas
"""
import glob
import pandas as pd
import pyedflib
import numpy as np
from sklearn.preprocessing import scale
from time import time
from pathlib import Path
import os

t_start = time()
path_dataset = Path(".").absolute().parent / 'dataset'
os.chdir(path_dataset)
annotation_files = glob.glob("*E1.edf")
fe = 256
new_fe = 250
signal_list = []
pre_sequence_length_s = 10
file_to_remove = []
size_dataset = 'small'
if len(annotation_files) > 15:
    size_dataset = 'big'
for filename in annotation_files:
    try:
        with pyedflib.EdfAnnotation(filename) as edf_file:
            my_file = edf_file
    except OSError:
        print("File " + filename[:-8] + " ignored because of corruption")
