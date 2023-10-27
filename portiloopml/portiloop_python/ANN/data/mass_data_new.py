import time
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import os


class MassDataset(Dataset):
    def __init__(self, data_path, subjects=None):
        super(MassDataset, self).__init__()
        self.data_path = data_path
        self.subjects = subjects

        # Start by finding the necessary subsets to load based on the names of the subjects required
        if self.subjects is not None:
            self.subsets = list(set([subject[3:5] for subject in self.subjects]))
        else:
            self.subsets = ['01', '02', '03', '05']

        self.data = {}

        # Load the necessary data
        for subset in self.subsets:
            data = self.read_data(os.path.join(self.data_path, f'mass_spindles_ss{subset[1]}.npz'))
            for key in data.keys():
                start = time.time()
                self.data[key] = data[key].item()
                end = time.time()
                print(f"Time taken to load {key}: {end - start}")

    @staticmethod
    def read_data(path):
        data = np.load(path, allow_pickle=True)
        return data
    
    @staticmethod
    def onsets_2_labelvector(spindles, length):
        label_vector = torch.zeros(length)
        for spindle in spindles:
            onset = spindle[0]
            offset = spindle[1]
            label_vector[onset:offset] = 1
        return label_vector
    

if __name__ == "__main__":
    start = time.time()
    test = MassDataset('/project/MASS/mass_spindles_dataset')
    end = time.time()

    print(f"Time taken: {end - start}")
