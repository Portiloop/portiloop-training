import csv
import json
import logging
import os
import pathlib
import random
import time

import numpy as np
import pandas as pd
import pyedflib
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from portiloop_software.portiloop_python.ANN.utils import get_configs


def read_patient_info(dataset_path):
    """
    Read the patient info from a patient_info file and initialize it in a dictionary
    """
    patient_info_file = os.path.join(dataset_path, 'patient_info.csv')
    with open(patient_info_file, 'r') as patient_info_f:
        # Skip the header if present
        has_header = csv.Sniffer().has_header(patient_info_f.read(1024))
        patient_info_f.seek(0)  # Rewind.
        reader = csv.reader(patient_info_f)
        if has_header:
            next(reader)  # Skip header row.

        patient_info = {
            line[0]: {
                'age': int(line[1]),
                'gender': line[2]
            } for line in reader
        }
    return patient_info


def read_pretraining_dataset(dataset_path, patients_to_keep=None):
    """
    Load all dataset files into a dictionary to be ready for a Pytorch Dataset.
    Note that this will only read the first signal even if the EDF file contains more.
    """
    patient_info = read_patient_info(dataset_path)

    for patient_id in patient_info.keys():
        if patients_to_keep is not None and patient_id not in patients_to_keep:
            continue
        filename = os.path.join(dataset_path, patient_id + ".edf")
        try:
            with pyedflib.EdfReader(filename) as edf_file:
                patient_info[patient_id]['signal'] = edf_file.readSignal(0)
        except FileNotFoundError:
            logging.debug(f"Skipping file {filename} as it is not in dataset.")

    # Remove all patients whose signal is not in dataset
    dataset = {patient_id: patient_details for (patient_id, patient_details) in patient_info.items()
        if 'signal' in patient_info[patient_id].keys()}

    return dataset


class PretrainingDataset(Dataset):
    def __init__(self, dataset_path, config, device=None):
        self.device = device
        self.window_size = config['window_size']

        data = read_pretraining_dataset(dataset_path)

        def sort_by_gender_and_age(subject):
            res = 0
            assert data[subject]['age'] < 255, f"{data[subject]['age']} years is a bit old."
            if data[subject]['gender'] == 'M':
                res += 1000
            res += data[subject]['age']
            return res

        self.subjects = sorted(data.keys(), key=sort_by_gender_and_age)
        self.nb_subjects = len(self.subjects)

        logging.debug(f"DEBUG: {self.nb_subjects} subjects:")
        for subject in self.subjects:
            logging.debug(
                f"DEBUG: {subject}, {data[subject]['gender']}, {data[subject]['age']} yo")

        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        # signal needed before the last window
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride
        self.min_signal_len = self.past_signal_len + \
            self.window_size  # signal needed for one sample

        self.full_signal = []
        self.genders = []
        self.ages = []

        for subject in self.subjects:
            assert self.min_signal_len <= len(
                data[subject]['signal']), f"Signal {subject} is too short."
            data[subject]['signal'] = torch.tensor(
                data[subject]['signal'], dtype=torch.float)
            self.full_signal.append(data[subject]['signal'])
            gender = 1 if data[subject]['gender'] == 'M' else 0
            age = data[subject]['age']
            ones = torch.ones_like(data[subject]['signal'], dtype=torch.uint8)
            gender_tensor = ones * gender
            age_tensor = ones * age
            self.genders.append(gender_tensor)
            self.ages.append(age_tensor)
            del data[subject]  # we delete this as it is not needed anymore

        # all signals concatenated (float32)
        self.full_signal = torch.cat(self.full_signal)
        # all corresponding genders (uint8)
        self.genders = torch.cat(self.genders)
        self.ages = torch.cat(self.ages)  # all corresponding ages (uint8)
        assert len(self.full_signal) == len(self.genders) == len(self.ages)

        self.samplable_len = len(self.full_signal) - self.min_signal_len + 1

        # Masking probabilities
        prob_not_masked = 1 - config['ratio_masked']
        prob_masked = config['ratio_masked'] * (1 - (config['ratio_replaced'] + config['ratio_kept']))
        prob_replaced = config['ratio_masked'] * config['ratio_replaced']
        prob_kept = config['ratio_masked'] * config['ratio_kept']
        self.mask_probs = torch.tensor([prob_not_masked, prob_masked, prob_replaced, prob_kept])
        self.mask_cum_probs = self.mask_probs.cumsum(0)

    def __len__(self):
        return self.samplable_len

    def __getitem__(self, idx):
        assert 0 <= idx < len(self), f"Index out of range ({idx}/{len(self)})."

        idx += self.past_signal_len

        # Get data
        x_data = self.full_signal[idx - self.past_signal_len:idx + self.window_size].unfold(
            0, self.window_size, self.seq_stride)  # TODO: double-check
        # TODO: double-check
        x_gender = self.genders[idx + self.window_size - 1]
        x_age = self.ages[idx + self.window_size - 1]  # TODO: double-check
        
        # Get random mask from given probabilities:
        mask = torch.searchsorted(self.mask_cum_probs, torch.rand(self.seq_len))

        # Get the sequence for masked sequence modeling
        masked_seq = x_data.clone()
        for seq_idx, mask_token in enumerate(mask):
            # No mask or skip mask or MASK token (which is done later)
            if mask_token in [0, 1, 3]: 
                continue
            elif mask_token == 2:
                # Replace token with replacement
                random_idx = int(torch.randint(high=len(self)-self.window_size, size=(1, )))
                masked_seq[seq_idx] = self.full_signal[random_idx: random_idx+self.window_size]
            else:
                raise RuntimeError("Issue with masks, shouldn't get a value not in {0, 1, 2, 3}")

        return x_data, x_gender, x_age, mask, masked_seq


class ValidationSampler(Sampler):
    def __init__(self, data_source, dividing_factor):
        self.len_max = len(data_source)
        self.data = data_source
        self.dividing_factor = dividing_factor

    def __iter__(self):
        for idx in range(0, self.len_max, self.dividing_factor):
            yield random.randint(0, self.len_max - 1)

    def __len__(self):
        return self.len_max // self.dividing_factor



def get_dataloaders_sleep_stage(MASS_dir, ds_dir, config):
    """
    Get the dataloaders for the MASS dataset
    - Start by dividing the available subjects into train and test sets
    - Create the train and test datasets and dataloaders
    """
    # Read all the subjects available in the dataset
    labels = read_sleep_staging_labels(ds_dir) 

    # Divide the subjects into train and test sets
    subjects = list(labels.keys())
    random.shuffle(subjects)
    train_subjects = subjects[:int(len(subjects) * 0.8)]
    test_subjects = subjects[int(len(subjects) * 0.8):]

    # Read the pretraining dataset
    data = read_pretraining_dataset(MASS_dir)

    # Create the train and test datasets
    train_dataset = SleepStageDataset(train_subjects, data, labels, config)
    test_dataset = SleepStageDataset(test_subjects, data, labels, config)

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=SleepStageSampler(train_dataset, config),
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size_validation'],
        sampler=SleepStageSampler(test_dataset, config),
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, test_dataloader


def read_sleep_staging_labels(MASS_dir):
    '''
    Read the sleep_staging.csv file in the given directory and stores info in a dictionary
    '''
    # Read the sleep_staging.csv file 
    sleep_staging_file = os.path.join(MASS_dir, 'sleep_staging.csv')
    with open(sleep_staging_file, 'r') as f:
        reader = csv.reader(f)
        # Remove the header line from the information
        sleep_staging = list(reader)[1:]

    sleep_stages = {}
    for i in range(len(sleep_staging)):
        subject = sleep_staging[i][0]
        sleep_stages[subject] = [stage for stage in sleep_staging[i][1:] if stage != '']

    return sleep_stages
    

class SleepStageSampler(Sampler):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.window_size = config['window_size']
        self.max_len = len(dataset) - self.dataset.past_signal_len - self.window_size

    def __iter__(self):
        while True:
            index = random.randint(0, self.max_len - 1)
            # Make sure that the label at the end of the window is not '?'
            label = self.dataset.full_labels[index + self.dataset.past_signal_len + self.window_size - 1]
            if label != SleepStageDataset.get_labels().index('?'):
                yield index

    def __len__(self):
        return len(self.indices)


class SleepStageDataset(Dataset):
    def __init__(self, subjects, data, labels, config):
        '''
        This class takes in a list of subject, a path to the MASS directory 
        and reads the files associated with the given subjects as well as the sleep stage annotations
        '''
        super().__init__()

        self.config = config
        self.window_size = config['window_size']
        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        # signal needed before the last window
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride

        # Get the sleep stage labels
        self.full_signal = []
        self.full_labels = []

        for subject in subjects:
            if subject not in data.keys():
                print(f"Subject {subject} not found in the pretraining dataset")
                continue
            # assert subject in data.keys(), f"Subject {subject} not found in the pretraining dataset" 
            signal = torch.tensor(
                data[subject]['signal'], dtype=torch.float)
            # Get all the labels for the given subject
            label = []
            for lab in labels[subject]:
                label += [SleepStageDataset.get_labels().index(lab)] * self.config['fe']
            
            # Add some '?' padding at the end to make sure the length of signal and label match
            label += [SleepStageDataset.get_labels().index('?')] * (len(signal) - len(label))

            # Make sure that the signal and the labels are the same length
            assert len(signal) == len(label)

            # Add to full signal and full label
            self.full_labels.append(torch.tensor(label, dtype=torch.uint8))
            self.full_signal.append(signal)
            del data[subject], signal, label
        
        self.full_signal = torch.cat(self.full_signal)
        self.full_labels = torch.cat(self.full_labels)

    @staticmethod
    def get_labels():
        return ['1', '2', '3', 'R', 'W', '?']

    def __getitem__(self, index):
        # Get the signal and label at the given index
        index += self.past_signal_len

        # Get data
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)  # TODO: double-check
        label = self.full_labels[index + self.window_size - 1]

        assert label != 5, "Label is '?'"

        return signal, label.type(torch.LongTensor)

    def __len__(self):
        return len(self.full_signal)



def generate_spindle_trains_dataset(raw_dataset_path, output_file, electrode='Cz'):
    "Constructs a dataset of spindle trains from the MASS dataset"
    data = {}

    spindle_infos = os.listdir(os.path.join(raw_dataset_path, "spindle_info"))

    # List all files in the subject directory
    for subject_dir in os.listdir(os.path.join(raw_dataset_path, "subject_info")):
        subset = subject_dir[5:8]
        # Get the spindle info file where the subset is in the filename
        spindle_info_file = [f for f in spindle_infos if subset in f and electrode in f][0]

        # Read the spindle info file
        train_ds_ss = read_spindle_train_info(\
            os.path.join(raw_dataset_path, "subject_info", subject_dir), \
                os.path.join(raw_dataset_path, "spindle_info", spindle_info_file))
        
        # Append the data
        data.update(train_ds_ss)

    # Write the data to a json file
    with open(output_file, 'w') as f:
        json.dump(data, f)


def read_spindle_train_info(subject_dir, spindle_info_file):
    """
    Read the spindle train info from the given subject directory and spindle info file
    """
    subject_names = pd.read_csv(subject_dir, header=None).to_numpy()[:, 0]
    spindle_info = pd.read_csv(spindle_info_file)
    headers = list(spindle_info.columns)[:-1]

    data = {}
    for subj in subject_names:
        data[subj] = {
            headers[0]: [],
            headers[1]: [],
            headers[2]: [],
        }
    subject_counter = 1
    for index, row in spindle_info.iterrows():
        if index != 0 and row['onsets'] < spindle_info.iloc[index-1]['onsets']:
            subject_counter += 1
    
    assert subject_counter == len(subject_names), \
        f"The number of subjects in the subject_info file and the spindle_info file should be the same, \
            found {len(subject_names)} and {subject_counter} respectively"

    def convert_row_to_250hz(row):
        "Convert the row to 250hz"
        row['onsets'] = int((row['onsets'] / 256) * 250)
        row['offsets'] = int((row['offsets'] / 256) * 250)
        if row['onsets'] == row['offsets']:
            return None
        assert row['onsets'] < row['offsets'], "The onset should be smaller than the offset"
        return row

    subject_names_counter = 0
    for index, row in spindle_info.iterrows():
        if index != 0 and row['onsets'] < spindle_info.iloc[index-1]['onsets']:
            subject_names_counter += 1
        row = convert_row_to_250hz(row)
        if row is None:
            continue
        for h in headers:
            data[subject_names[subject_names_counter]][h].append(row[h])

    for subj in subject_names:
        assert len(data[subj][headers[0]]) == len(data[subj][headers[1]]) == len(data[subj][headers[2]]), "The number of onsets, offsets and labels should be the same"

    return data

def get_dataloaders_spindle_trains(config):
    """
    Get the dataloaders for the MASS dataset
    - Start by dividing the available subjects into train and test sets
    - Create the train and test datasets and dataloaders
    """
    # Read all the subjects available in the dataset
    labels = read_spindle_trains_labels(config['old_dataset']) 

    # Divide the subjects into train and test sets
    subjects = list(labels.keys())
    random.shuffle(subjects)
    train_subjects = subjects[:int(len(subjects) * 0.8)]
    test_subjects = subjects[int(len(subjects) * 0.8):]

    # Read the pretraining dataset
    data = read_pretraining_dataset(config['MASS_dir'])

    # Create the train and test datasets
    train_dataset = SpindleTrainDataset(train_subjects, data, labels, config)
    test_dataset = SpindleTrainDataset(test_subjects, data, labels, config)

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=SpindleTrainRandomSampler(train_dataset, sample_list=[1, 2]),
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size_validation'],
        sampler=SpindleTrainRandomSampler(test_dataset, sample_list=[1, 2]),
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, test_dataloader

def get_dataloaders_mass(config):
    """
    Get the dataloaders for the MASS dataset
    - Start by dividing the available subjects into train and test sets
    - Create the train and test datasets and dataloaders
    """
    # Read all the subjects available in the dataset
    labels = read_spindle_trains_labels(config['old_dataset']) 

    # Divide the subjects into train and test sets
    subjects = list(labels.keys())
    random.shuffle(subjects)
    train_subjects = subjects[:int(len(subjects) * 0.8)]
    test_subjects = subjects[int(len(subjects) * 0.8):]

    # Read the pretraining dataset
    data = read_pretraining_dataset(config['MASS_dir'], patients_to_keep=train_subjects + test_subjects)

    # Create the train and test datasets
    train_dataset = MassDataset(train_subjects, data, labels, config)
    # test_dataset = MassDataset(test_subjects, data, labels, config)
    test_dataset = SingleSubjectDataset(test_subjects[0], data, labels, config)

    # Create the train and test dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=MassRandomSampler(train_dataset, config['batch_size'], nb_batch=config['nb_batch_per_epoch']),
        pin_memory=True,
        drop_last=True
    )

    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=config['batch_size_validation'],
    #     sampler=MassRandomSampler(test_dataset, config['batch_size_validation'], nb_batch=config['nb_batch_per_epoch']),
    #     pin_memory=True,
    #     drop_last=True
    # )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=SingleSubjectSampler(len(test_dataset), config['seq_stride']),
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, test_dataloader


class MassRandomSampler(Sampler):
    def __init__(self, dataset, batch_size, nb_batch=1000):
        self.dataset = dataset

        self.all_spindles = self.dataset.spindle_labels_iso + self.dataset.spindle_labels_first + self.dataset.spindle_labels_train
        self.all_spindles = np.array(self.all_spindles)
        self.max_spindle_index = len(self.all_spindles)
        self.max_index = len(self.dataset) - self.dataset.past_signal_len - self.dataset.window_size + 1
        self.spindle_index = 0
        self.length = nb_batch * batch_size

    def __iter__(self):
        self.batch_nb = 0
        while self.batch_nb < self.length:
            self.batch_nb += 1
            # choose if sample a spindle or not
            sample_spindle = np.random.choice([0, 1])
            if sample_spindle == 0:
                # Sample from the rest of the signal
                yield random.randint(0, self.max_index)
            else:
                index = self.all_spindles[self.spindle_index]
                self.spindle_index += 1
                if self.spindle_index >= self.max_spindle_index:
                    self.spindle_index = 0
                index_to_yield = index - self.dataset.past_signal_len - self.dataset.window_size + 1
                assert self.dataset.full_labels[index] != 0, "The label should not be 0"
                yield index_to_yield
    
    def __len__(self):
        return self.length


class SpindleTrainRandomSampler(Sampler):
    def __init__(self, dataset, sample_list=[0, 1, 2, 3]):
        """ 
        ratio: list of ratios for each class [non-spindle, isolated, first, train]
        """
        self.sample_list = sample_list
        self.dataset = dataset

        self.isolated_index = 0
        self.first_index = 0
        self.train_index = 0

        self.max_isolated_index = len(dataset.spindle_labels_iso)
        self.max_first_index = len(dataset.spindle_labels_first)
        self.max_train_index = len(dataset.spindle_labels_train)

    def __iter__(self):
        while True:
            # Select a random class
            class_choice = np.random.choice(self.sample_list)
            if class_choice == 0:
                # Sample from the rest of the signal
                yield random.randint(0, len(self.dataset.full_signal) - self.dataset.min_signal_len - self.dataset.window_size)
            elif class_choice == 1:
                index = self.dataset.spindle_labels_iso[self.isolated_index]
                self.isolated_index += 1
                if self.isolated_index >= self.max_isolated_index:
                    self.isolated_index = 0
                assert index in self.dataset.spindle_labels_iso, "Spindle index not found in list"
                yield index - self.dataset.past_signal_len - self.dataset.window_size + 1
            elif class_choice == 2:
                index = self.dataset.spindle_labels_first[self.first_index]
                self.first_index += 1
                if self.first_index >= self.max_first_index:
                    self.first_index = 0
                assert index in self.dataset.spindle_labels_first, "Spindle index not found in list"
                yield index - self.dataset.past_signal_len - self.dataset.window_size + 1
            elif class_choice == 3:
                index = self.dataset.spindle_labels_train[self.train_index]
                self.train_index += 1
                if self.train_index >= self.max_train_index:
                    self.train_index = 0
                assert index in self.dataset.spindle_labels_train, "Spindle index not found in list"
                yield index - self.dataset.past_signal_len - self.dataset.window_size + 1
            
    def __len__(self):
        return len(self.dataset.full_signal)


def read_spindle_trains_labels(ds_dir):
    '''
    Read the sleep_staging.csv file in the given directory and stores info in a dictionary
    '''
    spindle_trains_file = os.path.join(ds_dir, 'spindle_trains_annots.json')
    # Read the json file
    with open(spindle_trains_file, 'r') as f:
        labels = json.load(f)
    return labels


class SpindleTrainDataset(Dataset):
    def __init__(self, subjects, data, labels, config):
        '''
        This class takes in a list of subject, a path to the MASS directory 
        and reads the files associated with the given subjects as well as the sleep stage annotations
        '''
        super().__init__()

        self.config = config
        self.window_size = config['window_size']
        self.seq_len = config['seq_len']
        self.seq_stride = config['seq_stride']
        
        # signal needed before the last window
        self.past_signal_len = (self.seq_len - 1) * self.seq_stride
        self.min_signal_len = self.past_signal_len + self.window_size

        # Get the sleep stage labels
        self.full_signal = []
        self.full_labels = []
        self.spindle_labels_iso = []
        self.spindle_labels_first = []
        self.spindle_labels_train = []

        accumulator = 0
        # List to keep track of where each subject data starts and ends
        self.subject_list = []
        for subject in subjects:
            if subject not in data.keys():
                print(f"Subject {subject} not found in the pretraining dataset")
                continue

            # Keeps track of the first index of the signal for the given subject
            self.subject_list.append(len(self.full_signal))

            # Get the signal for the given subject
            signal = torch.tensor(
                data[subject]['signal'], dtype=torch.float)


            # Get all the labels for the given subject
            label = torch.zeros_like(signal, dtype=torch.uint8)
            for index, (onset, offset, l) in enumerate(zip(labels[subject]['onsets'], labels[subject]['offsets'], labels[subject]['labels_num'])):
                
                # Some of the spindles in the dataset overlap with the previous spindle
                # If that is the case, we need to make sure that the onset is at least the offset of the previous spindle
                if onset < labels[subject]['offsets'][index - 1]:
                    onset = labels[subject]['offsets'][index - 1]

                label[onset:offset] = l
                # Make a separate list with the indexes of all the spindle labels so that sampling is easier
                to_add = list(range(accumulator + onset, accumulator + offset))
                assert offset < len(signal), f"Offset {offset} is greater than the length of the signal {len(signal)} for subject {subject}"
                if l == 1:
                    self.spindle_labels_iso += to_add
                elif l == 2:
                    self.spindle_labels_first += to_add
                elif l == 3:
                    self.spindle_labels_train += to_add
                else:
                    raise ValueError(f"Unknown label {l} for subject {subject}")
            # increment the accumulator
            accumulator += len(signal)

            # Make sure that the signal and the labels are the same length
            assert len(signal) == len(label)

            # Add to full signal and full label
            self.full_labels.append(label)
            self.full_signal.append(signal)
            del data[subject], signal, label
        
        # Concatenate the full signal and the full labels into one continuous tensor
        self.full_signal = torch.cat(self.full_signal)
        self.full_labels = torch.cat(self.full_labels)

        # Shuffle the spindle labels
        start = time.time()
        random.shuffle(self.spindle_labels_iso)
        random.shuffle(self.spindle_labels_first)
        random.shuffle(self.spindle_labels_train)
        end = time.time()
        print(f"Shuffling took {end - start} seconds")
        print(f"Number of spindle labels: {len(self.spindle_labels_iso) + len(self.spindle_labels_first) + len(self.spindle_labels_train)}")


    @staticmethod
    def get_labels():
        return ['non-spindle', 'isolated', 'first', 'train']

    def __getitem__(self, index):

        # Get data
        index = index + self.past_signal_len
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)
        label = self.full_labels[index + self.window_size - 1]

        # Make sure that the last index of the signal is the same as the label
        # assert signal[-1, -1] == self.full_signal[index + self.window_size - 1], "Issue with the data and the labels"
        label = label.type(torch.LongTensor)
        print(f"super getitem {label}")
        return signal, label

    def __len__(self):
        return len(self.full_signal) - self.window_size


class MassDataset(SpindleTrainDataset):
    def __getitem__(self, index):
        # Get data
        index = index + self.past_signal_len
        signal = self.full_signal[index - self.past_signal_len:index + self.window_size].unfold(
            0, self.window_size, self.seq_stride)
        label = self.full_labels[index + self.window_size - 1]

        # Make sure that the last index of the signal is the same as the label
        # assert signal[-1, -1] == self.full_signal[index + self.window_size - 1], "Issue with the data and the labels"
        label = label.type(torch.LongTensor)

        label = 0 if label == 0 else 1
        return signal, signal, signal, label


class MassValidationSampler(Sampler):
    def __init__(self, subject_list, seq_stride, window_size, total_len, past_signal_len, max_batch_size, max_segment_len):
        """
        Samples in order from a dataset. This is used for validation.
        To accelerate validation, each batch will be a collation of each point in the seq_stride for each subject, times the dividing factor.
        This means that the sequence for each subject will be divided according to the dividing factor to speed things up.
        Args:
        subject_list: list of the first index of each subject in the dataset,
        seq_stride: the stride between each window of the sequence,
        total_len: the total length of the dataset,
        past_signal_len: the length of the past signal, how much of past signal we need to sample a sequence.
        max_batch_size: the maximum batch size. This is used to select the number of segments to sample from each subject. 
        max_segment_len: the maximum length of each segment of the sequence.
        """
        self.subject_indices = subject_list
        self.seq_stride = seq_stride
        self.window_size = window_size
        self.total_len = total_len-1
        self.past_signal_len = past_signal_len
        self.max_batch_size = max_batch_size
        self.max_segment_len = max_segment_len

        # Check that the max_segment_len is smaller than the smallest sequence size
        self.subject_lengths = [self.subject_indices[i+1] - (self.subject_indices[i] + self.past_signal_len) for i in range(len(self.subject_indices)-1)]\
                                + [self.total_len - (self.subject_indices[-1] + self.past_signal_len)]
        smallest_subject = min(self.subject_lengths)
        assert max_segment_len < smallest_subject, f"Max segment length {max_segment_len} is greater than the smallest subject length {smallest_subject}"
        
        self.max_length = self.max_segment_len // self.seq_stride
        self.segment_starts = self.select_sequences()
        self.indexes = list(self.get_iterator())

    def select_sequences(self):
        """
        Select some random sequences from each subject.
        We want to have at most batch_size sequences in total.
        """

        # Get a list of the length of each subject
        seg_starts = []
        subject_index = 0
        while len(seg_starts) < self.max_batch_size:
            subject_index += 1
            # If we have gone through all the subjects, start again
            if subject_index >= len(self.subject_indices):
                subject_index = 0

            # Get the starting point for this subject
            start_index = self.subject_indices[subject_index]
            
            # Get a random valid point for this subject
            start = random.randint(start_index + self.past_signal_len, start_index + (self.subject_lengths[subject_index] - self.max_segment_len))
            seg_starts.append(start)

            assert start + self.max_segment_len < self.total_len, "The selected sequence is too long"

        # This is now a batch_size length list of random starting points for sampling sequences spread around the subjects
        return seg_starts

    def get_validation_batch_size(self): 
        return len(self.segment_starts)
            
    def __iter__(self):
        """
        Send an index in the right order.
        We start by iterating through the seq_stride, then through the subdivisions, then through the subjects.
        """
        return iter(self.indexes)

    def get_iterator(self):
        for i in range(self.max_length):
            # Each batch is a collation of all the prechosen segments accross all subjects
            for start in self.segment_starts:
                yield start + i * self.seq_stride

    def __len__(self):
        return self.max_length


class SingleSubjectSampler(Sampler):
    def __init__(self, dataset_len, seq_stride):
        self.dataset_len = dataset_len
        self.seq_stride = seq_stride
        self.indices = torch.arange(0, self.dataset_len, self.seq_stride)
        print(f"Length of sampler: {len(self)}")

    def __iter__(self):
        # Get the indices of the first window of each sequence
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class SingleSubjectDataset(Dataset):
    def __init__(self, subject_id, data, labels, config):
        '''
        This class takes in a list of subject, a path to the MASS directory 
        and reads the files associated with the given subjects as well as the sleep stage annotations
        '''
        super().__init__()

        self.config = config
        self.window_size = config['window_size']
        self.seq_len = 1
        self.seq_stride = config['seq_stride']
        
        # Get the sleep stage labels
        self.full_signal = []
        self.full_labels = []
        self.spindle_labels = []

        assert subject_id in data.keys(), f"Subject {subject_id} not found in the pretraining dataset"

        # Get the signal for the given subject
        signal = torch.tensor(
            data[subject_id]['signal'], dtype=torch.float)

        # Get all the labels for the given subject
        label = torch.zeros_like(signal, dtype=torch.uint8)
        for index, (onset, offset, l) in enumerate(zip(labels[subject_id]['onsets'], labels[subject_id]['offsets'], labels[subject_id]['labels_num'])):
            
            # Some of the spindles in the dataset overlap with the previous spindle
            # If that is the case, we need to make sure that the onset is at least the offset of the previous spindle
            if onset < labels[subject_id]['offsets'][index - 1]:
                onset = labels[subject_id]['offsets'][index - 1]

            # We always want to set spindles to 1
            label[onset:offset] = 1

            # Make a separate list with the indexes of all the spindle labels so that sampling is easier
            to_add = list(range(onset, offset))
            assert offset < len(signal), f"Offset {offset} is greater than the length of the signal {len(signal)} for subject {subject_id}"
            if l in [1, 2, 3]:
                self.spindle_labels += to_add
            else:
                raise ValueError(f"Unknown label {l} for subject {subject_id}")

        # Make sure that the signal and the labels are the same length
        assert len(signal) == len(label)

        # Add to full signal and full label
        self.full_labels = label
        self.full_signal = signal
        del data[subject_id], signal, label
        
        print(f"Number of spindle labels: {len(self.spindle_labels)}")
        print(f"len of full signal: {len(self.full_signal)}")

    @staticmethod
    def get_labels():
        return ['non-spindle', 'spindle']

    def __getitem__(self, index):
        """
        Get the signal and the label for the given index given the seq_stride and the network stride
        """
        # Get data
        signal = self.full_signal[index:index + self.window_size].unsqueeze(0)

        label = self.full_labels[index + self.window_size - 1]
        label = label.type(torch.LongTensor)

        return signal, signal, signal, label

    def __len__(self):
        return len(self.full_signal) - self.window_size



if __name__ == "__main__":
    # Get the path to the dataset directory
    # dataset_path = pathlib.Path(__file__).parents[2].resolve() / 'dataset'
    # generate_spindle_trains_dataset(dataset_path / 'SpindleTrains_raw_data', dataset_path / 'spindle_trains_annots.json')
    config = get_configs("Test", False, 0)
    config['batch_size_validation'] = 64
    subjects = ['01-01-0001']
    data = read_pretraining_dataset(config['MASS_dir'], patients_to_keep=subjects)
    labels = read_spindle_trains_labels(config['old_dataset']) 
    dataset = MassDataset(subjects, data, labels, config)


    # Get the dataloader
    sampler = MassValidationSampler(
        subject_list=dataset.subject_list, 
        seq_stride=config['seq_stride'], 
        window_size=config['window_size'],
        total_len=len(dataset), 
        past_signal_len=dataset.past_signal_len,
        max_batch_size=4000,
        max_segment_len=15000)
    
    batch_size = sampler.get_validation_batch_size()
    print(f"Number of batches {len(sampler)}")
    print(f"batch_size: {batch_size}")
    config['validation_batch_size'] = batch_size
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4,
    )

    start = time.time()
    for index, i in enumerate(train_dataloader):
        if index % 100 == 0:
            print(index)
    print(time.time() - start)

    batches = torch.tensor(list(sampler)).reshape(-1, batch_size)
    print(batches[-1, :])

