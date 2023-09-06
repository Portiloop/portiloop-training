# Test the sleep staging performance of TSN on a real dataset from start to finish
import random
from scipy import signal
import torch
import numpy as np
from portiloopml.portiloop_python.ANN.adaptation_training import parse_config, resample_signal
from portiloopml.portiloop_python.ANN.data.mass_data import SingleSubjectDataset, SingleSubjectSampler, read_pretraining_dataset, read_sleep_staging_labels, read_spindle_trains_labels
from portiloopml.portiloop_python.ANN.lightning_tests import load_model
from portiloopml.portiloop_python.ANN.utils import get_configs


def run_ss(ss_net, subject_id, labels):
    config['window_size'] = 30 * 250
    config['seq_len'] = 1
    config['subject_id'] = subject_id
    data = read_pretraining_dataset(
        config['MASS_dir'], patients_to_keep=[subject_id])

    assert subject_id in data.keys(), 'Subject not in the dataset'
    assert subject_id in labels.keys(), 'Subject not in the dataset'

    dataset = SingleSubjectDataset(
        config['subject_id'], data=data, labels=labels, config=config, ss_labels=labels)
    sampler = SingleSubjectSampler(len(dataset), config['window_size'])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0)

    h = None
    preds = []
    for i in dataloader:
        input = i[0]
        signal = torch.Tensor(resample_signal(input, 250, 100)).cuda()
        output, h = ss_net(signal, h)
        print(h)
        prediction = output.argmax(dim=1)
        preds.append(prediction)

    preds = torch.tensor(preds)
    print(preds)


def resample_signal(input_signal, start_freq, target_freq):

    # Extract all but the last dimension
    size = input_signal.shape[:-1]

    input_signal = input_signal.view(input_signal.size(-1))
    # Calculate the resampling factor
    resampling_factor = target_freq / start_freq

    # Create the time axis for the original signal
    original_time = np.arange(0, len(input_signal)) / start_freq

    # Create the time axis for the resampled signal
    resampled_time = np.arange(
        0, len(input_signal) * resampling_factor) / target_freq

    # Use scipy's resample function to perform resampling
    resampled_signal = signal.resample(input_signal, len(resampled_time))

    desired_size = size + (len(resampled_signal),)
    resampled_signal = resampled_signal.reshape(desired_size)

    return resampled_signal


if __name__ == "__main__":
    # Parse config dict important for the adapatation
    args = parse_config()
    if args.seed == -1:
        seed = random.randint(0, 100000)
    else:
        seed = args.seed

    config = get_configs(args.experiment_name, False, seed)
    # config['nb_conv_layers'] = 4
    # config['hidden_size'] = 64
    # config['nb_rnn_layers'] = 4

    # Load the model
    ss_net = load_model('Original_params_1692894764.8012033')
    ss_net.cuda().eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run some testing on subject 1
    # Load the data
    labels = read_spindle_trains_labels(config['old_dataset'])
    ss_labels = read_sleep_staging_labels(config['path_dataset'])

    run_ss(ss_net, '01-02-0019', ss_labels)
