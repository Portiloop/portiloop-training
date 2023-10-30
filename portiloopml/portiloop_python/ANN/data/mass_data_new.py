import time
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


class MassDataset(Dataset):
    def __init__(self, data_path, subjects=None):
        super(MassDataset, self).__init__()
        self.data_path = data_path
        self.subjects = subjects

        # Start by finding the necessary subsets to load based on the names of the subjects required
        if self.subjects is not None:
            self.subsets = list(set([subject[3:5]
                                for subject in self.subjects]))
        else:
            self.subsets = ['01', '02', '03', '05']

        self.data = {}

        # Load the necessary data
        for subset in self.subsets:
            data = self.read_data(os.path.join(
                self.data_path, f'mass_spindles_ss{subset[1]}.npz'))
            for key in data.keys():
                start = time.time()
                # self.data[key] = data[key].item()
                self.data[key] = None
                end = time.time()
                print(f"Time taken to load {key}: {end - start}")

        # Write all the

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
    # start = time.time()
    # test = MassDataset('/project/MASS/mass_spindles_dataset')
    # end = time.time()

    # print(f"Time taken: {end - start}")

    test = SubjectLoader(
        '/project/MASS/mass_spindles_dataset/subject_info.csv')

    random = test.random_subjects(10)

    print()
