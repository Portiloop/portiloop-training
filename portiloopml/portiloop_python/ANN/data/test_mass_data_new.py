import unittest
import pandas as pd
from io import StringIO

import torch
from mass_data_new import SubjectLoader
from portiloopml.portiloop_python.ANN.data.mass_data_new import MassDataset


class TestSubjectLoader(unittest.TestCase):
    def setUp(self):
        self.loader = SubjectLoader(
            '/project/MASS/mass_spindles_dataset/subject_info.csv')

    def test_random_subjects(self):
        subjects = self.loader.select_random_subjects(5)
        self.assertEqual(len(subjects), 5)

    def test_random_subjects_with_exclude(self):
        subjects = self.loader.select_random_subjects(
            5, exclude=['01-01-0001'])
        self.assertEqual(len(subjects), 5)
        self.assertTrue(
            all(subject not in ['01-01-0001'] for subject in subjects))

    def test_random_subjects_with_seed(self):
        subjects1 = self.loader.select_random_subjects(5, seed=1)
        subjects2 = self.loader.select_random_subjects(5, seed=1)
        self.assertEqual(subjects1, subjects2)

    def test_select_subjects_age(self):
        subjects = self.loader.select_subjects_age(30, 40)
        self.assertTrue(all(
            30 <= self.loader.subject_info[
                self.loader.subject_info['SubjectID'] == subject]['Age'].values[0] <= 40
            for subject in subjects))

    def test_select_subjects_gender(self):
        subjects = self.loader.select_subjects_gender('F')
        self.assertTrue(all(self.loader.subject_info[self.loader.subject_info['SubjectID']
                        == subject]['Sexe'].values[0] == 'F' for subject in subjects))

    def test_select_subset(self):
        subjects = self.loader.select_subset(1)
        self.assertTrue(all(subject[3:].startswith('01')
                        for subject in subjects))


class TestMassDataset(unittest.TestCase):
    def setUp(self):
        subjects = ['01-01-0001']
        self.dataset = MassDataset('/project/MASS/mass_spindles_dataset',
                                   subjects=subjects,)

    def test_get_labels(self):
        subject, signal_idx = self.dataset.lookup_table[0]
        labels = self.dataset.get_labels(subject, signal_idx)
        self.assertIsInstance(labels, dict)
        self.assertIn('spindle_filt', labels)
        self.assertIn('spindle_mass', labels)
        self.assertIn('age', labels)
        self.assertIn('gender', labels)
        self.assertIn('subject', labels)
        self.assertIn('sleep_stage', labels)

    def test_get_signal(self):
        subject, signal_idx = self.dataset.lookup_table[0]
        signal = self.dataset.get_signal(subject, signal_idx)
        self.assertIsInstance(signal, torch.Tensor)

    def test_get_item(self):
        signal, labels = self.dataset[0]
        self.assertIsInstance(signal, torch.Tensor)
        self.assertIsInstance(labels, dict)

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.dataset.lookup_table))

    def test_onsets_2_labelvector(self):
        spindles = {'onsets': [0, 10, 20], 'offsets': [5, 15, 25]}
        label_vector = self.dataset.onsets_2_labelvector(spindles, 30)
        self.assertIsInstance(label_vector, torch.Tensor)
        self.assertEqual(label_vector.sum(), 15)


if __name__ == '__main__':
    unittest.main()
