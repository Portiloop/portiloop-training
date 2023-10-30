import unittest
import pandas as pd
from io import StringIO
from mass_data_new import SubjectLoader


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


if __name__ == '__main__':
    unittest.main()
