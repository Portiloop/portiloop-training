import numpy as np
import os

from portiloopml.portiloop_python.ANN.wamsley_utils import detect_lacourse

subsets = ['01', '02', '03', '05']
data_path = '/project/MASS/mass_spindles_dataset'
new_data_path = '/project/MASS/mass_spindles_dataset_lacourse'

all_subjects = []


def read_data(path):
    data = np.load(path, allow_pickle=True)
    return data


# Open the necessary files and store them in a dictionary
for subset in subsets:
    subset = int(subset)
    data = read_data(os.path.join(
        data_path, f'mass_spindles_ss{subset}.npz'))
    data_unloaded[subset] = data
    data_out = {}

    for subject in data:
        item = data[subject].item()
        # Load the signal and the annotations
        signal_raw = item['signal_mass']
        ss_labels = item['ss_label']

        spindle_mass_lacourse = {
            subject: {
                'onsets': [],
                'offsets': [],
                'labels_num': []
            }
        }

        # Annotate using the fixed Wamsley
        mask = (ss_labels == 1) | (ss_labels == 2)

        spindles_mass = detect_lacourse(signal_raw, mask)

        for spindle in spindles_mass:
            spindle_mass_lacourse[subject]['onsets'].append(spindle[0])
            spindle_mass_lacourse[subject]['offsets'].append(spindle[2])
            spindle_mass_lacourse[subject]['labels_num'].append(1)

        print(
            f"Subject {subject}: Length of Lacourse Spindles: {len(spindles_mass)}")

        item['spindle_mass_lacourse'] = spindle_mass_lacourse

        data_out[subject] = item

    np.savez_compressed(os.path.join(
        new_data_path, f"mass_spindles_ss{subset}.npz"), **data_out)
