import argparse
import torch
from torch import nn
from torch import optim
import copy

from portiloop_software.portiloop_python.ANN.data.mass_data import SingleSubjectDataset, read_pretraining_dataset, read_spindle_trains_labels
from portiloop_software.portiloop_python.ANN.models.lstm import get_trained_model
from portiloop_software.portiloop_python.ANN.utils import get_configs, get_metrics


def run_adaptation(dataloader, net, device, config):
    """
    Goes over the dataset and learns at every step.
    Returns the accuracy and loss as well as fp, fn, tp, tn count for spindles
    Also returns the updated model
    """

    # Initialize optimizer and criterion
    optimizer = optim.AdamW(net.parameters(), lr=config['lr_adam'], weight_decay=config['adam_w'])
    criterion = nn.BCELoss(reduction='none')

    # Initialize All the necessary variables
    net_copy = copy.deepcopy(net)
    net_copy = net_copy.to(device)
    net_copy = net_copy.train()
    loss = 0
    n = 0
    window_labels_total = torch.tensor([], device=device)
    output_total = torch.tensor([], device=device)

    # Initialize the hidden state of the GRU to Zero. We always have a batch size of 1 in this case
    h1 = torch.zeros((config['nb_rnn_layers'], 1, config['hidden_size']), device=device)

    # Run through the dataloader
    out_grus = []
    out_loss = 0
    for index, info in enumerate(dataloader):
        # Get the data and labels
        window_data, window_labels = info
        window_data = window_data.to(device)
        window_labels = window_labels.to(device)

        optimizer.zero_grad()

        # Get the output of the network
        output, h1, out_gru = net_copy(window_data, h1, torch.tensor(out_grus))
        out_grus.append(out_gru)

        if len(out_grus) > config['max_h_length']:
            out_grus.pop(0)

        # Compute the loss
        loss = criterion(output, window_labels)

        if index % 1 == 0:
            # Update the model
            loss.backward()
            optimizer.step()

        # Update the loss
        out_loss += loss.item()
        n += 1

        # Get the predictions
        output = (output >= 0.5)

        # Concatenate the predictions
        window_labels_total = torch.cat([window_labels_total, window_labels])
        output_total = torch.cat([output_total, output])
    
    # Compute metrics
    loss /= n
    acc = (output_total == window_labels_total).float().mean()
    output_total = output_total.float()
    window_labels_total = window_labels_total.float()
    # Get the true positives, true negatives, false positives and false negatives
    tp = (window_labels_total * output_total)
    tn = ((1 - window_labels_total) * (1 - output_total))
    fp = ((1 - window_labels_total) * output_total)
    fn = (window_labels_total * (1 - output_total))

    return output_total, window_labels_total, loss, acc, tp, tn, fp, fn, net_copy


def parse_config():
    """
    Parses the config file
    """
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--subject_id', type=str, default='01-01-0001', help='Subject on which to run the experiment')
    parser.add_argument('--model_path', type=str, default='testing_att_after_gru', help='Model for the starting point of the model')
    parser.add_argument('--experiment_name', type=str, default='test', help='Name of the model')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    seed = 42
    # Parse config dict important for the adapatation
    args = parse_config()
    config = get_configs(args.experiment_name, False, seed)
    config['subject_id'] = args.subject_id

    # Load the data
    labels = read_spindle_trains_labels(config['old_dataset'])
    data = read_pretraining_dataset(config['MASS_dir'])
    dataset = SingleSubjectDataset(config['subject_id']) 
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0)       

    # Load the model
    net = get_trained_model(config, config['path_models'] / args.model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run the adaptation
    output_total, window_labels_total, loss, acc, tp, tn, fp, fn, net_copy = run_adaptation(dataloader, net, device, config)

    # Get the metrics
    f1, precision, recall = get_metrics(tp, fp, fn)

    # Print the results
    print('Loss: ', loss)
    print('Accuracy: ', acc)
    print('F1: ', f1)
    print('Precision: ', precision)
    print('Recall: ', recall)
