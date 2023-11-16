import time
from typing import Iterator
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import os
import pandas as pd
import sys


def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


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

    def select_all_subjects(self):
        '''
        Return a list of all subjects.
        '''
        return list(self.subject_info['SubjectID'])

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
    def __init__(self, dataset, option='spindles', seed=None, num_samples=None):
        assert option in ['spindles', 'random', 'staging_eq',
                          'staging_all'], "Option must be either spindle or staging"
        self.dataset = dataset
        self.option = option
        self.seed = seed

        if self.option == 'spindles':
            # Get the same number of non spindles as we have spindles
            non_spindles = np.random.choice(np.arange(len(self.dataset)), len(
                self.dataset.labels_indexes['spindle_label']), replace=False)
            self.indexes = np.concatenate(
                (self.dataset.labels_indexes['spindle_label'], non_spindles))
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
            N1 = np.random.choice(
                self.dataset.labels_indexes['staging_label_N1'], min_samples, replace=False)
            N2 = np.random.choice(
                self.dataset.labels_indexes['staging_label_N2'], min_samples, replace=False)
            N3 = np.random.choice(
                self.dataset.labels_indexes['staging_label_N3'], min_samples, replace=False)
            R = np.random.choice(
                self.dataset.labels_indexes['staging_label_R'], min_samples, replace=False)
            W = np.random.choice(
                self.dataset.labels_indexes['staging_label_W'], min_samples, replace=False)
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
        elif self.option == 'random':
            self.indexes = np.arange(len(self.dataset))
            np.random.shuffle(self.indexes)

        # Limit to a number of samples per epoch
        self.num_samples = num_samples

    def __iter__(self):
        if self.num_samples is not None:
            for _ in range(self.num_samples):
                random_index = np.random.randint(len(self.indexes))
                yield self.indexes[random_index]
        else:
            for index in self.indexes:
                yield index

    def __len__(self):
        return len(self.indexes) if self.num_samples is None else self.num_samples


class MassDataset(Dataset):
    def __init__(self, data_path, window_size, seq_stride, seq_len, subjects=None, use_filtered=True, sampleable='both'):
        '''
        A Class which loads the MASS dataset and returns the signals and labels of the subjects.

        Parameters
        ----------
        data_path : str
            Path to the MASS dataset.
        window_size : int
            Size of the window to return.
        seq_stride : int
            Stride of the sliding window.
        seq_len : int
            Length of the sequence to return.
        subjects : list, optional
            List of subjects to load. If None, all subjects will be loaded. The default is None.
        use_filtered : bool, optional
            Whether to use the filtered signal or the mass signal. The default is True.
        sampleable : str, optional
            Whether to prepare for sampling spindles, staging, both or none. The default is 'both'.
        '''
        super(MassDataset, self).__init__()

        assert sampleable in ['both', 'spindles', 'staging',
                              'none'], "Sampleable must be either both, spindles, staging or none."

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

        # Remove the unloaded data to free up memory
        del self.data_unloaded

        # Remove the unfiltered/filtered signal depending on the option
        if use_filtered:
            for key in self.data:
                del self.data[key]['signal_mass']
                # Rename the filtered signal to signal
                self.data[key]['signal'] = self.data[key]['signal_filt']
        else:
            for key in self.data:
                del self.data[key]['signal_filt']
                # Rename the mass signal to signal
                self.data[key]['signal'] = self.data[key]['signal_mass']

        # Convert the spindle labels to vector to make lookup faster
        for key in self.data:
            if use_filtered:
                self.data[key]['spindle_label'] = self.onsets_2_labelvector(
                    self.data[key]['spindle_label_filt'][key], len(self.data[key]['signal']))
            else:
                self.data[key]['spindle_label'] = self.onsets_2_labelvector(
                    self.data[key]['spindle_label_mass'][key], len(self.data[key]['signal']))

        # Get a lookup table to match all possible sampleable signals to a (subject, index) pair
        self.lookup_table = []
        self.subjects_table = {}
        self.labels_indexes = {
            'spindle_label': np.array([], dtype=np.int16),
            'staging_label_N1': np.array([], dtype=np.int16),
            'staging_label_N2': np.array([], dtype=np.int16),
            'staging_label_N3': np.array([], dtype=np.int16),
            'staging_label_R': np.array([], dtype=np.int16),
            'staging_label_W': np.array([], dtype=np.int16),
        }

        start = time.time()
        for subject in self.data:
            indices = np.arange(len(self.data[subject]['signal']))
            valid_indices = indices[(indices >= self.past_signal_len) & (
                indices <= len(self.data[subject]['signal']) - self.window_size)]

            if sampleable == 'spindles' or sampleable == 'both':
                # Get the labels of the label indices and keep track of them
                self.labels_indexes['spindle_label'] = np.concatenate((self.labels_indexes['spindle_label'], np.where(self.data[subject]
                                                                                                                      ['spindle_label'][valid_indices] == 1)[0] + len(self.lookup_table)))

            if sampleable == 'staging' or sampleable == 'both':
                self.labels_indexes['staging_label_N1'] = np.concatenate((self.labels_indexes['staging_label_N1'], np.where(self.data[subject]
                                                                                                                            ['ss_label'][valid_indices] == 0)[0] + len(self.lookup_table)))
                self.labels_indexes['staging_label_N2'] = np.concatenate((self.labels_indexes['staging_label_N2'], np.where(self.data[subject]
                                                                                                                            ['ss_label'][valid_indices] == 1)[0] + len(self.lookup_table)))
                self.labels_indexes['staging_label_N3'] = np.concatenate((self.labels_indexes['staging_label_N3'], np.where(self.data[subject]
                                                                                                                            ['ss_label'][valid_indices] == 2)[0] + len(self.lookup_table)))
                self.labels_indexes['staging_label_R'] = np.concatenate((self.labels_indexes['staging_label_R'], np.where(self.data[subject]
                                                                                                                          ['ss_label'][valid_indices] == 3)[0] + len(self.lookup_table)))
                self.labels_indexes['staging_label_W'] = np.concatenate((self.labels_indexes['staging_label_W'], np.where(self.data[subject]
                                                                                                                          ['ss_label'][valid_indices] == 4)[0] + len(self.lookup_table)))

            self.subjects_table[subject] = (len(self.lookup_table), len(
                self.lookup_table) + len(valid_indices) - 1)
            self.lookup_table += list(valid_indices)

        end = time.time()
        print(f"Time taken to create lookup table: {end - start}")

        print(f"Number of sampleable indices: {len(self.lookup_table)}")
        print(
            f"Number of spindles: {len(self.labels_indexes['spindle_label'])}")
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
            'spindle_label': self.data[subject]['spindle_label'][signal_idx],
            'age': self.data[subject]['age'],
            'gender': self.data[subject]['gender'],
            'subject': subject,
            'sleep_stage': self.data[subject]['ss_label'][signal_idx],
        }
        return labels

    def get_signal(self, subject, signal_idx):
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
        # Make sure this works
        signal = self.data[subject]['signal'][signal_idx - self.past_signal_len:signal_idx +
                                              self.window_size]
        signal = torch.tensor(signal, dtype=torch.float).unfold(
            0, self.window_size, self.seq_stride)
        signal = signal.unsqueeze(1)
        return signal

    def get_subject_by_index(self, index):
        '''
        Return the subject of a lookup table index.
        '''
        for subject in self.subjects_table:
            start, end = self.subjects_table[subject]
            if index >= start and index <= end:
                return subject

    def __getitem__(self, index):
        '''
        Return the signal and labels of a subject and signal.

        Parameters
        ----------
        index : int
            Index of the signal to return.
        '''
        signal_idx = self.lookup_table[index]
        subject = self.get_subject_by_index(index)
        labels = self.get_labels(subject, signal_idx)
        signal = self.get_signal(subject, signal_idx)
        return signal, labels

    def __len__(self):
        return len(self.lookup_table)

    def get_memory_usage(self):
        return get_size(self)

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


class CombinedDataLoader:
    '''
    A class which combines two dataloaders into a single one.
    It takes the length of the shortest dataloader and returns the data from both dataloaders in the same order.
    '''

    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2

    def __iter__(self):
        self.iter1 = iter(self.loader1)
        self.iter2 = iter(self.loader2)
        return self

    def __next__(self):
        try:
            data1 = next(self.iter1)
            data2 = next(self.iter2)
            return data1, data2
        except StopIteration:
            raise StopIteration

    def __len__(self):
        len1 = len(self.loader1)
        len2 = len(self.loader2)
        return min(len1, len2)


if __name__ == "__main__":

    window_size = 54
    seq_stride = 42
    seq_len = 50

    loader = SubjectLoader(
        '/project/MASS/mass_spindles_dataset/subject_info.csv')

    subjects = loader.select_subjects_age(18, 30, num_subjects=40, seed=42)
    # subjects = loader.select_all_subjects()

    start = time.time()
    test = MassDataset(
        '/project/MASS/mass_spindles_dataset',
        subjects=subjects,
        window_size=window_size,
        seq_stride=seq_stride,
        seq_len=seq_len,
        use_filtered=True,
        sampleable='staging')
    end = time.time()

    print(f"Time taken: {end - start}")
