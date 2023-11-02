import time
from typing import Iterator
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import os
import pandas as pd


class SubjectLoader:
    def __init__(self, csv_file):
        '''
        A class which loads a subject info CSV and allows to retrieve subject lists which fit in different criteria.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file containing the subject info.
        '''
        self.subject_info = pd.read_csv(csv_file)

    def select_subset(self, subset, num_subjects=-1, seed=None, exclude=None):
        '''
        Return a list of subjects which are in the specified subset.

        Parameters
        ----------
        subset : int
            Subset to select subjects from.
        num_subjects : int, optional
            Number of subjects to return. If -1, all subjects in the subset will be returned. The default is -1.
        seed : int, optional
            Seed for the random selection. The default is None.
        exclude : list, optional
            List of subjects to exclude from the random selection. The default is None.
        '''
        assert subset in [1, 2, 3, 5]
        subset = f"01-0{subset}"

        if exclude is not None:
            select_from = self.subject_info[~self.subject_info['SubjectID'].isin(
                exclude)]
        else:
            select_from = self.subject_info

        if num_subjects != -1:
            if seed is not None:
                selected_subjects = list(select_from[select_from['SubjectID'].str.startswith(
                    subset)].sample(num_subjects, random_state=seed)['SubjectID'])
            else:
                selected_subjects = list(select_from[select_from['SubjectID'].str.startswith(
                    subset)].sample(num_subjects)['SubjectID'])
        else:
            selected_subjects = list(select_from[select_from['SubjectID'].str.startswith(
                subset)]['SubjectID'])

        return selected_subjects

    def select_random_subjects(self, num_subjects, exclude=None, seed=None):
        '''
        Return a list of random subjects.

        Parameters
        ----------
        num_subjects : int
            Number of subjects to return.
        exclude : list, optional
            List of subjects to exclude from the random selection. The default is None.
        seed : int, optional
            Seed for the random selection. The default is None.
        '''
        if exclude is not None:
            select_from = self.subject_info[~self.subject_info['SubjectID'].isin(
                exclude)]
        else:
            select_from = self.subject_info

        if seed is not None:
            sampled_subjects = list(select_from.sample(
                num_subjects, random_state=seed)['SubjectID'])
        else:
            sampled_subjects = list(
                select_from.sample(num_subjects)['SubjectID'])

        return sampled_subjects

    def select_subjects_age(self, min_age, max_age, num_subjects=-1, seed=None, exclude=None):
        '''
        Return a list of subjects which are in the age range specified.

        Parameters
        ----------
        min_age : int
            Minimum age of the subjects to return.
        max_age : int
            Maximum age of the subjects to return.
        num_subjects : int, optional
            Number of subjects to return. If -1, all subjects in the age range will be returned. The default is -1.
        seed : int, optional
            Seed for the random selection. The default is None.
        exclude : list, optional
            List of subjects to exclude from the random selection. The default is None.
        '''
        if exclude is not None:
            select_from = self.subject_info[~self.subject_info['SubjectID'].isin(
                exclude)]
        else:
            select_from = self.subject_info

        if num_subjects != -1:
            if seed is not None:
                selected_subjects = list(select_from[(select_from['Age'] >= min_age) & (
                    select_from['Age'] <= max_age)].sample(num_subjects, random_state=seed)['SubjectID'])
            else:
                selected_subjects = list(select_from[(select_from['Age'] >= min_age) & (
                    select_from['Age'] <= max_age)].sample(num_subjects)['SubjectID'])
        else:
            selected_subjects = list(select_from[(select_from['Age'] >= min_age) & (
                select_from['Age'] <= max_age)]['SubjectID'])

        return selected_subjects

    def select_subjects_gender(self, gender, num_subjects=-1, seed=None, exclude=None):
        '''
        Return a list of subjects which are in the age range specified.

        Parameters
        ----------
        gender : int
            Desired gender of the subjects to return.
        num_subjects : int, optional
            Number of subjects to return. If -1, all subjects in the age range will be returned. The default is -1.
        seed : int, optional
            Seed for the random selection. The default is None.
        exclude : list, optional
            List of subjects to exclude from the random selection. The default is None.
        '''
        if exclude is not None:
            select_from = self.subject_info[~self.subject_info['SubjectID'].isin(
                exclude)]
        else:
            select_from = self.subject_info

        if num_subjects != -1:
            if seed is not None:
                selected_subjects = list(select_from[(select_from['Age'] == gender)].sample(
                    num_subjects, random_state=seed)['SubjectID'])
            else:
                selected_subjects = list(
                    select_from[(select_from['Age'] == gender)].sample(num_subjects)['SubjectID'])
        else:
            selected_subjects = list(
                select_from[(select_from['Age'] == gender)]['SubjectID'])

        return selected_subjects


class MassSampler(Sampler):
    def __init__(self, dataset, option='spindle_mass', seed=None):
        assert option in ['spindle_mass', 'spindle_filt', 'staging_eq', 'staging_all'], "Option must be either spindle or staging"
        self.dataset = dataset
        self.option = option
        self.seed = seed

        if self.option == 'spindle_mass':
            # Get the same number of non spindles as we have spindles
            non_spindles = np.random.choice(np.arange(len(self.dataset)), len(self.dataset.labels_indexes['spindle_label_mass']), replace=False)
            self.indexes = np.concatenate((self.dataset.labels_indexes['spindle_label_mass'], non_spindles))
            # Shuffle the indexes
            np.random.shuffle(self.indexes)
        elif self.option == 'spindle_filt':
            # Get the same number of non spindles as we have spindles
            non_spindles = np.random.choice(np.arange(len(self.dataset)), len(self.dataset.labels_indexes['spindle_label_filt']), replace=False)
            self.indexes = np.concatenate((self.dataset.labels_indexes['spindle_label_filt'], non_spindles))
            # Shuffle the indexes
            np.random.shuffle(self.indexes)
        elif self.option == 'staging_eq':
            # Find the sleep stage with the least amount of samples
            min_samples = np.min([
                len(self.dataset.labels_indexes['staging_label_N1']), 
                len(self.dataset.labels_indexes['staging_label_N2']), 
                len(self.dataset.labels_indexes['staging_label_N3']), 
                len(self.dataset.labels_indexes['staging_label_R']), 
                len(self.dataset.labels_indexes['staging_label_W'])])
            
            # Get the same number of samples from each sleep stage
            N1 = np.random.choice(self.dataset.labels_indexes['staging_label_N1'], min_samples, replace=False)
            N2 = np.random.choice(self.dataset.labels_indexes['staging_label_N2'], min_samples, replace=False)
            N3 = np.random.choice(self.dataset.labels_indexes['staging_label_N3'], min_samples, replace=False)
            R = np.random.choice(self.dataset.labels_indexes['staging_label_R'], min_samples, replace=False)
            W = np.random.choice(self.dataset.labels_indexes['staging_label_W'], min_samples, replace=False)
            self.indexes = np.concatenate((N1, N2, N3, R, W))
            # Shuffle the indexes
            np.random.shuffle(self.indexes)
        elif self.option == 'staging_all':
            # Get all the indexes
            N1 = self.dataset.labels_indexes['staging_label_N1']
            N2 = self.dataset.labels_indexes['staging_label_N2']
            N3 = self.dataset.labels_indexes['staging_label_N3']
            R = self.dataset.labels_indexes['staging_label_R']
            W = self.dataset.labels_indexes['staging_label_W']
            self.indexes = np.concatenate((N1, N2, N3, R, W))
            # Shuffle the indexes
            np.random.shuffle(self.indexes)

    def __iter__(self):
        return iter(self.indexes)

        

class MassDataset(Dataset):
    def __init__(self, data_path, window_size, seq_stride, seq_len, subjects=None):
        super(MassDataset, self).__init__()
        self.data_path = data_path
        self.subjects = subjects
        self.window_size = window_size
        self.seq_stride = seq_stride
        self.seq_len = seq_len
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride

        # Start by finding the necessary subsets to load based on the names of the subjects required
        if self.subjects is not None:
            self.subsets = list(set([subject[3:5]
                                for subject in self.subjects]))
        else:
            self.subsets = ['01', '02', '03', '05']

        self.data_unloaded = {}

        all_subjects = []

        # Open the necessary files and store them in a dictionary
        for subset in self.subsets:
            data = self.read_data(os.path.join(
                self.data_path, f'mass_spindles_ss{subset[1]}.npz'))
            self.data_unloaded[subset] = data
            all_subjects += list(data.keys())

        self.subjects = all_subjects if self.subjects is None else self.subjects

        # Actually load the data of each subject into a dictionary
        self.data = {}
        for key in self.subjects:
            start = time.time()
            subset = key[3:5]
            self.data[key] = self.data_unloaded[subset][key].item()
            end = time.time()
            print(f"Time taken to load {key}: {end - start}")

        # Convert the spindle labels to vector to make lookup faster
        for key in self.data:
            self.data[key]['spindle_label_filt'] = self.onsets_2_labelvector(
                self.data[key]['spindle_label_filt'][key], len(self.data[key]['signal_filt']))
            self.data[key]['spindle_label_mass'] = self.onsets_2_labelvector(
                self.data[key]['spindle_label_mass'][key], len(self.data[key]['signal_filt']))

        # Get a lookup table to match all possible sampleable signals to a (subject, index) pair
        self.lookup_table = []
        self.labels_indexes = {
            'spindle_label_filt': [],
            'spindle_label_mass': [],
            'staging_label_N1': [],
            'staging_label_N2': [],
            'staging_label_N3': [],
            'staging_label_R': [],
            'staging_label_W': [],
        }

        start = time.time()
        for subject in self.data:
            indices = np.arange(len(self.data[subject]['signal_filt']))
            valid_indices = indices[(indices >= self.past_signal_len) & (
                indices <= len(self.data[subject]['signal_filt']) - self.window_size)]

            # Get the labels of the label indices and keep track of them
            self.labels_indexes['spindle_label_filt'] = valid_indices[self.data[subject]['spindle_label_filt'][valid_indices] == 1] + len(self.lookup_table)
            self.labels_indexes['spindle_label_mass'] = valid_indices[self.data[subject]['spindle_label_mass'][valid_indices] == 1] + len(self.lookup_table)
            self.labels_indexes['staging_label_N1'] = valid_indices[self.data[subject]['ss_label'][valid_indices] == 0] + len(self.lookup_table)
            self.labels_indexes['staging_label_N2'] = valid_indices[self.data[subject]['ss_label'][valid_indices] == 1] + len(self.lookup_table)
            self.labels_indexes['staging_label_N3'] = valid_indices[self.data[subject]['ss_label'][valid_indices] == 2] + len(self.lookup_table)
            self.labels_indexes['staging_label_R'] = valid_indices[self.data[subject]['ss_label'][valid_indices] == 3] + len(self.lookup_table)
            self.labels_indexes['staging_label_W'] = valid_indices[self.data[subject]['ss_label'][valid_indices] == 4] + len(self.lookup_table)

            self.lookup_table += list(zip([subject]
                                      * len(valid_indices), valid_indices))
        end = time.time()
        print(f"Time taken to create lookup table: {end - start}")

        print(f"Number of sampleable indices: {len(self.lookup_table)}")
        print(f"Number of filtered spindles: {len(self.labels_indexes['spindle_label_filt'])}")
        print(f"Number of mass spindles: {len(self.labels_indexes['spindle_label_mass'])}")
        print(f"Number of N1: {len(self.labels_indexes['staging_label_N1'])}")
        print(f"Number of N2: {len(self.labels_indexes['staging_label_N2'])}")
        print(f"Number of N3: {len(self.labels_indexes['staging_label_N3'])}")
        print(f"Number of R: {len(self.labels_indexes['staging_label_R'])}")
        print(f"Number of W: {len(self.labels_indexes['staging_label_W'])}")

    def get_labels(self, subject, signal_idx):
        '''
        Return the labels of a subject and signal.

        Parameters
        ----------
        subject : str
            Subject ID.
        signal_idx : int
            Index of the signal to return the labels from.
        '''
        labels = {
            'spindle_filt': self.data[subject]['spindle_label_filt'][signal_idx],
            'spindle_mass': self.data[subject]['spindle_label_mass'][signal_idx],
            'age': self.data[subject]['age'],
            'gender': self.data[subject]['gender'],
            'subject': subject,
            'sleep_stage': self.data[subject]['ss_label'][signal_idx],
        }
        return labels

    def get_signal(self, subject, signal_idx, filtered_signal=False):
        '''
        Return the signal of a subject and signal.

        Parameters
        ----------
        subject : str
            Subject ID.
        signal_idx : int
            Index of the signal to return.
        filtered_signal : bool, optional
            Whether to return the filtered signal or the mass signal. The default is False.
        '''
        signal_use = 'signal_filt' if filtered_signal else 'signal_mass'
        # Make sure this works
        signal = self.data[subject][signal_use][signal_idx - self.past_signal_len:signal_idx +
                                                self.window_size]
        signal = torch.tensor(signal).unfold(
            0, self.window_size, self.seq_stride)
        signal = signal.unsqueeze(0)
        return signal

    def __getitem__(self, index):
        '''
        Return the signal and labels of a subject and signal.

        Parameters
        ----------
        index : int
            Index of the signal to return.
        '''
        subject, signal_idx = self.lookup_table[index]
        labels = self.get_labels(subject, signal_idx)
        signal = self.get_signal(subject, signal_idx)
        return signal, labels

    def __len__(self):
        return len(self.lookup_table)

    @staticmethod
    def read_data(path):
        data = np.load(path, allow_pickle=True)
        return data

    @staticmethod
    def onsets_2_labelvector(spindles, length):
        label_vector = torch.zeros(length)
        spindles = list(zip(spindles['onsets'], spindles['offsets']))
        for spindle in spindles:
            onset = spindle[0]
            offset = spindle[1]
            label_vector[onset:offset] = 1
        return label_vector
    
    @staticmethod
    def get_ss_labels():
        return ['1', '2', '3', 'R', 'W', '?']

if __name__ == "__main__":

    window_size = 54
    seq_stride = 42
    seq_len = 50

    loader = SubjectLoader(
        '/project/MASS/mass_spindles_dataset/subject_info.csv')

    subjects = loader.select_subjects_age(18, 30, num_subjects=3, seed=42)

    start = time.time()
    test = MassDataset(
        '/project/MASS/mass_spindles_dataset', 
        subjects=subjects,
        window_size=window_size,
        seq_stride=seq_stride,
        seq_len=seq_len)
    end = time.time()

    print(f"Time taken: {end - start}")

    print(test[0])
    print(test[0][0].shape)
