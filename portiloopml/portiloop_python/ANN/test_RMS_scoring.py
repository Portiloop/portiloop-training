# Test the RMS scoring function on compute canada

# Parse arguments from command line
import argparse

import numpy as np
from matplotlib import pyplot as plt

from portiloopml.portiloop_python.ANN.data.mass_data_new import MassDataset
from portiloopml.portiloop_python.ANN.wamsley_utils import RMS_score_all, filter_signal_for_RMS

parser = argparse.ArgumentParser(
    description='Test the RMS scoring function on compute canada')
parser.add_argument('--data_path', type=str, help='Path to the data folder',
                    default='/project/portinight-dataset/')

args = parser.parse_args()
data_path = args.data_path

subject = 'PN_08_AC_Night5'
 
# Load the signal using dataset
dataset = MassDataset(data_path, 30, 30, 30, subjects=[
                      subject], use_filtered=False, sampleable='spindles', compute_spindle_labels=False)

signal = dataset.data[subject]['signal']

# Get a random number of indexes from the signal
indexes = np.random.randint(0, len(signal), 100000)

# orders = list(range(10))
orders = range(300)
nans = []
means = []
stds = []

# Count the number of NaNs in the input signal
# nans.append(np.sum(np.isnan(signal)))
# print(f'Number of NaNs in the signal: {nans[0]}')

# Get the indexes of all the NaNs
nan_indexes = np.where(np.isnan(signal))[0]

# Count the number of infs
infs = np.sum(np.isinf(signal))
print(f'Number of infs in the signal: {infs}')

# Remove the Values above 500 in the signal 
# signal[np.abs(signal) > 500] = 0

plt.plot(signal[2872800:])
plt.savefig('test_signal.png')

filtered_signal = filter_signal_for_RMS(signal, filter_order=249)

# Print the number of NaNs
print(f'Number of NaNs in the filtered signal: {np.sum(np.isnan(filtered_signal))}')

all_scores = np.array(RMS_score_all(filtered_signal, indexes, filter_order=249))

# Print the number of NaNs
print(f'Number of NaNs: {np.sum(np.isnan(all_scores))}')

# for order in tqdm(orders):

#     # Print order of the filter
#     # print(f'Order: {order}')
#     # print(f"Order: {order}")
#     # print(f'Number of NaNs in the filter: {np.sum(np.isnan(sos))}')

#     # Get the RMS of the signal
#     all_scores = np.array(RMS_score_all(signal, indexes, filter_order=order))

#     # Print the number of NaNs
#     # print(f'Number of NaNs: {np.sum(np.isnan(all_scores))}')
#     nans.append(np.sum(np.isnan(all_scores)))

#     # Remove the NaNs
#     all_scores = all_scores[~np.isnan(all_scores)]

#     # Get the mean and std of the scores
#     mean = np.mean(all_scores)
#     means.append(mean)
#     std = np.std(all_scores)
#     stds.append(std)

#     print(f'Mean: {mean}, Std: {std}')

# Print the minimum value of std and it's index
# min_std = np.min(stds)
# min_std_index = np.argmin(stds)
# print(f'Minimum std: {min_std}, order: {orders[min_std_index]}')

# plt.clf()
# plt.plot(orders, nans, label='Number of NaNs')
# plt.savefig('test_rms_nans.png')

# plt.clf()
# plt.plot(orders, means, label='Mean')
# plt.savefig('test_rms_means.png')

# plt.clf()
# plt.plot(orders, stds, label='Std')
# plt.savefig('test_rms_stds.png')

