import copy
import os
import re
import numpy as np
import math
import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler


class SleepStageSampler(Sampler):
    def __init__(self, dataset, nb_batch_per_epoch, batch_size):
        self.dataset = dataset
        self.max_len = len(dataset)
        self.limit = nb_batch_per_epoch * batch_size
        self.nb_batch_per_epoch = nb_batch_per_epoch

    def __iter__(self):
        for i in range(self.limit):
            # Get a random class between 0 and 4
            sample_class = random.randint(0, 4)

            # Get a random index from the list of indexes for the given class
            while True:
                sample_index = random.choice(
                    self.dataset.label_indexes[sample_class])
                if sample_index - (self.dataset.seq_len * self.dataset.window_size) > 0:
                    break

            yield sample_index

    def __len__(self):
        return self.limit


class SSValidationSampler(Sampler):
    def __init__(self, dataset, nb_batch_per_epoch, batch_size):
        self.dataset = dataset
        self.max_len = len(dataset)
        self.limit = nb_batch_per_epoch * batch_size
        self.nb_batch_per_epoch = nb_batch_per_epoch

    def __iter__(self):
        for i in range(self.limit):
            # GEt a class depending on the weight of that class:
            sample_class = random.choices(
                list(range(0, 5)), self.dataset.sampleable_weights)[0]

            # Get a random index from the list of indexes for the given class
            while True:
                sample_index = random.choice(
                    self.dataset.label_indexes[sample_class])
                if sample_index - (self.dataset.seq_len * self.dataset.window_size) > 0:
                    break

            yield sample_index

    def __len__(self):
        return self.limit


def pytorch_generator_to_keras(generator):
    generator = iter(generator)
    while True:
        batch = next(generator)
        new_batch = []
        for b in batch:
            new_batch.append(b.numpy())
        new_batch[0] = new_batch[0].reshape(-1, 3000, 1)
        new_batch = tuple(new_batch)
        yield tuple(new_batch)


def get_sleepedf_loaders(num_subjects, config):
    assert num_subjects <= 82, "Only 82 subjects in the dataset."
    assert num_subjects > 1, "Need more than one subject."

    num_train_subjects = int(math.ceil(num_subjects * 0.8))
    num_val_subjects = num_subjects - num_train_subjects
    if num_val_subjects == 0:
        num_train_subjects -= 1

    train_subjects = list(range(num_train_subjects))
    test_subjects = list(range(num_train_subjects, num_subjects))

    path = '/home/ubuntu/portiloop-training/portiloopml/dataset/eeg_fpz_cz'

    # train_dataset = SeqDataset(
    #     train_subjects, path, config['seq_len'])

    # test_dataset = SeqDataset(
    #     test_subjects, path, config['seq_len'])

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['batch_size'],
    #     sampler=RandomSampler(train_dataset),
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True
    # )

    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=config['batch_size'],
    #     sampler=RandomSampler(test_dataset),
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True
    # )

    train_dataset = SleepEDFDataset(
        train_subjects, path, config['seq_len'], config['window_size'])

    test_dataset = SleepEDFDataset(
        test_subjects, path, config['seq_len'], config['window_size'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=SleepStageSampler(
            train_dataset, config['batches_per_epoch'], config['batch_size']),
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        sampler=SSValidationSampler(
            test_dataset, config['batches_per_epoch'], config['batch_size']),
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    return train_loader, test_loader, train_dataset.sampleable_weights


def get_sleepedf_loaders_keras(num_subjects, config):
    assert num_subjects <= 82, "Only 82 subjects in the dataset."
    assert num_subjects > 1, "Need more than one subject."

    num_train_subjects = int(math.ceil(num_subjects * 0.8))
    num_val_subjects = num_subjects - num_train_subjects
    if num_val_subjects == 0:
        num_train_subjects -= 1

    train_subjects = list(range(num_train_subjects))
    test_subjects = list(range(num_train_subjects, num_subjects))

    path = '/home/ubuntu/portiloop-training/portiloopml/dataset/eeg_fpz_cz'

    train_dataset = SeqDataset(
        train_subjects, path, config['seq_len'])

    test_dataset = SeqDataset(
        test_subjects, path, config['seq_len'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=RandomSampler(train_dataset),
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        # sampler=RandomSampler(test_dataset, num_samples=64 * 1000),
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # train_dataset = SleepEDFDataset(
    #     train_subjects, path, config['seq_len'], config['window_size'])

    # test_dataset = SleepEDFDataset(
    #     test_subjects, path, config['seq_len'], config['window_size'])

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     sampler=SleepStageSampler(
    #         train_dataset, 10000000000000, 1),
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True
    # )

    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     sampler=SSValidationSampler(
    #         test_dataset, 64 * 1000, 1),
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True
    # )

    train_loader = pytorch_generator_to_keras(train_loader)
    test_loader = pytorch_generator_to_keras(test_loader)

    return train_loader, test_loader


class SeqDataset(Dataset):
    def __init__(self, subject_id_list, data_path, seq_length):

        files = os.listdir(data_path)

        subject_files = []
        for i in subject_id_list:
            subject_files += get_subject_files('sleepedf', files, i)

        subject_files = [os.path.join(data_path, subject_file)
                         for subject_file in subject_files]

        self.inputs, self.targets, _ = load_data(subject_files)
        self.seq_length = seq_length

        # Create a map of each sampleable integer (0 to sampleable_len) to the subject and index in the sequence
        self.sampleable_map = []
        for i in range(len(self.inputs)):
            for j in range(self.seq_length - 1, len(self.inputs[i])):
                self.sampleable_map.append((i, j))

        # Get the weight of each class
        self.sampleable_weights = []
        for i in range(5):
            self.sampleable_weights.append(
                sum([self.targets[j].tolist().count(i) for j in range(len(self.targets))]) / len(self))

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError("Index out of range")

        subject_idx, sequence_idx = self.sampleable_map[index]

        # Get the sequence
        sequence = self.inputs[subject_idx][sequence_idx -
                                            self.seq_length+1:sequence_idx+1]
        sequence = torch.from_numpy(copy.deepcopy(sequence)).unsqueeze(1)
        target = self.targets[subject_idx][sequence_idx]
        target = torch.Tensor([target]).type(torch.LongTensor).squeeze(0)

        assert sequence.shape == (
            self.seq_length, 1, 3000), f"Sequence shape is {sequence.shape}"

        return sequence, target

    def __len__(self):
        return len(self.sampleable_map)


class SleepEDFDataset(Dataset):
    def __init__(self, subject_id_list, data_path, seq_len, window_size):
        super().__init__()
        self.seq_len = seq_len
        self.window_size = window_size

        files = os.listdir(data_path)

        subject_files = []
        for i in subject_id_list:
            subject_files += get_subject_files('sleepedf', files, i)

        subject_files = [os.path.join(data_path, subject_file)
                         for subject_file in subject_files]

        x, y, _ = load_data(subject_files)
        repeats = x[0].shape[1]
        self.full_signal = torch.from_numpy(
            np.reshape(np.concatenate(x, axis=0), (-1)))
        self.full_labels = torch.from_numpy(
            np.repeat(np.concatenate(y, axis=0), repeats))

        assert len(self.full_signal) == len(
            self.full_labels), f"Data and label length mismatch. {len(self.full_signal)} != {len(self.full_labels)}"

        # Make a dictionary of all the indexes of each class label
        self.label_indexes = {}
        for label in self.full_labels.unique():
            self.label_indexes[int(label)] = (
                self.full_labels == label).nonzero(as_tuple=False).reshape(-1)

        total_samp_labels = sum([len(self.label_indexes[i]) for i in range(5)])
        self.sampleable_weights = [
            len(self.label_indexes[i]) / total_samp_labels for i in range(5)]

    def __getitem__(self, index):
        """
        Get the sample at index `index`.
        """
        signal = self.full_signal[index -
                                  (self.seq_len * self.window_size):index]
        signal = signal.unfold(0, self.window_size, self.window_size)
        signal = signal.unsqueeze(1)
        label = self.full_labels[index]

        return signal, label.type(torch.LongTensor)

    def __len__(self):
        """
        Returns:
            The total number of samples in the dataset.
        """
        return len(self.full_signal)


def get_subject_files(dataset, files, sid):
    """Get a list of files storing each subject data."""

    # Pattern of the subject files from different datasets
    if "mass" in dataset:
        reg_exp = f".*-00{str(sid+1).zfill(2)} PSG.npz"
        # reg_exp = "SS3_00{}\.npz$".format(str(sid+1).zfill(2))
    elif "sleepedf" in dataset:
        reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
        # reg_exp = "[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(str(sid).zfill(2))
    elif "isruc" in dataset:
        reg_exp = f"subject{sid+1}.npz"
    else:
        raise Exception("Invalid datasets.")

    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)

    return subject_files


def load_data(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            x = f['x']
            y = f['y']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate


if __name__ == "__main__":
    path = '/project/tinysleepnet/data/sleepedf/sleep-cassette/eeg_fpz_cz'
    subjects = [0]
    seq_len = 1
    window_size = 30 * 100
    dataset = SeqDataset(subjects, path, seq_len)

    print(len(dataset))
    # test = dataset[100000]

    test = dataset[0]

    for i in range(len(dataset)):
        test = dataset[i]
        assert test[0].shape == (seq_len, 1, window_size)

    train_loader, test_loader, weights = get_sleepedf_loaders_keras(
        2, {'batch_size': 32, 'seq_len': 50, 'window_size': 30 * 100})

    test = next(iter(train_loader))

    print(test[0].shape)
