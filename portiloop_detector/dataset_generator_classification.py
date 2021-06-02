# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:00:21 2021

@author: Nicolas
"""
import glob
import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import pyedflib

t_start = time()
path_dataset = Path(__file__).absolute().parent.parent / 'dataset'
os.chdir(path_dataset/'classification_annot')
annotation_files = glob.glob("*MODA_GS.txt")
os.chdir(path_dataset)
annotation_files = [files for files in annotation_files]
signal_files = [(subject[0:-12] + " PSG.edf") for subject in annotation_files]
fe = 256
new_fe = 250
signal_list = []
pre_sequence_length_s = 15
post_sequence_length_s = 15
file_to_remove = []
reference_list = pd.read_csv("8_MODA_primChan_180sjt.txt", delim_whitespace=True)

size_dataset = 'small'
if len(annotation_files) > 15:
    size_dataset = 'big'
for filename in signal_files:
    try:
        with pyedflib.EdfReader(filename) as edf_file:
            # print(edf_file.getSignalLabels())
            indices_C3 = [i for i, s in enumerate(edf_file.getSignalLabels()) if 'C3' in s]
            signal = edf_file.readSignal(indices_C3[0])
            assert edf_file.getSampleFrequency(indices_C3[0]) == fe
            if 'C3-A2' in reference_list[reference_list["subject"] == filename[:-8] + ".edf"]["channel"].values:
                indices_A2_CLE = [i for i, s in enumerate(edf_file.getSignalLabels()) if 'A2' in s]
                ref = edf_file.readSignal(indices_A2_CLE[0])
                assert edf_file.getSampleFrequency(indices_A2_CLE[0]) == fe
                assert len(ref) == len(signal)
                signal -= ref
            signal_list.append(signal)
    except OSError:
        file_to_remove.append(filename)
        print("File " + filename + " ignored because of corruption")

for filename in file_to_remove:
    signal_files.remove(filename)
    annotation_files.remove(filename[:-8] + "_MODA_GS.txt")

annotation_list = [pd.read_csv(file, delim_whitespace=True) for file in annotation_files]
sequence_list = [annotation[annotation['eventName'] == "segmentViewed"] for annotation in annotation_list]
spindle_list = [annotation[annotation['eventName'] == "spindle"] for annotation in annotation_list]

signal_seq_list = np.empty((len(sequence_list)), dtype=object)
spindle_seq_list = np.empty((len(sequence_list)), dtype=object)
subject_seq_list = []

pre_sequence_length = pre_sequence_length_s * new_fe
post_sequence_length = pre_sequence_length_s * new_fe
cnt = 0
for i, seq in enumerate(sequence_list):
    signal_seq_list[i] = np.empty((len(seq)), dtype=object)
    spindle_seq_list[i] = np.empty((len(seq)), dtype=object)
    subject_seq_list.append([annotation_files[i][:-12], cnt, cnt])  # name, first (include), last (excluded)
    for index, row in seq.iterrows():
        cnt += 1
        subject_seq_list[i][2] = cnt
        startSeq = row["startSec"]
        endSeq = startSeq + row["durationSec"]
        spindle_seq = spindle_list[i][(startSeq < spindle_list[i]["startSec"]) & (spindle_list[i]["startSec"] < endSeq)]
        startIdx = int(startSeq * new_fe) - pre_sequence_length
        if startIdx < 0:
            print("sequence too early")
            continue
        endIdx = int(endSeq * new_fe) + post_sequence_length
        lenSignal = endIdx - startIdx
        signal_seq_list[i][index] = signal_list[i][int(startIdx * fe / new_fe):int(endIdx * fe / new_fe)]
        spindle_seq_list[i][index] = np.zeros((lenSignal,), dtype=float)
        spindle_seq_list[i][index][:pre_sequence_length] = -1
        spindle_seq_list[i][index][-post_sequence_length:] = -2
        for temp, spindleRow in spindle_seq.iterrows():
            startSpin = int(spindleRow["startSec"] * new_fe) - startIdx
            endSpin = int((spindleRow["startSec"] + spindleRow["durationSec"]) * new_fe) - startIdx
            spindle_seq_list[i][index][startSpin:endSpin] = 1

signal = np.hstack(np.hstack(signal_seq_list))
spindle = np.hstack(np.hstack(spindle_seq_list))

if fe == new_fe:
    np.savetxt(f"dataset_full_{size_dataset}.txt", np.transpose((signal, spindle)), fmt='%e,%f')
else:
    np.savetxt(f"dataset_full_{size_dataset}_at_{fe}_to_resample.txt", np.transpose(signal), fmt='%e')
    np.savetxt(f"spindles_annotations_full_{size_dataset}_at_{new_fe}hz.txt", np.transpose(spindle), fmt='%f')

np.savetxt(f"subject_sequence_full_{size_dataset}.txt", subject_seq_list, fmt="%s")
print("tot_time = ", str(time() - t_start))
