import argparse
import random
import time
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
    out_grus = None
    out_loss = 0
    for index, info in enumerate(dataloader):

        # print(f"Batch {index}")
        # Get the data and labels
        window_data, window_labels = info
        window_data = window_data.to(device)
        window_labels = window_labels.to(device)

        optimizer.zero_grad()

        # Get the output of the network
        output, h1, out_gru = net_copy(window_data, h1, past_x=out_grus)
    
        # Update the past embeddings
        if out_grus is None:
            out_grus = out_gru
        else:
            out_grus = torch.cat([out_grus, out_gru], dim=1)

        if len(out_grus) > config['max_h_length']:
            out_grus = out_grus[:, 1:, :]

        # Compute the loss
        output = output.squeeze(-1)
        window_labels = window_labels.float()
        loss = criterion(output, window_labels)

        if index % 1 == 0 and False:
            # Update the model
            loss.backward()
            optimizer.step()

        h1 = h1.detach()
        out_gru = out_gru.detach()
        out_grus = out_grus.detach()
        output = output.detach()
        
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
    parser.add_argument('--seed', type=int, default=-1, help='Seed for the experiment')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Parse config dict important for the adapatation
    args = parse_config()
    if args.seed == -1:
        seed = random.randint(0, 100000)
    else:
        seed = args.seed

    config = get_configs(args.experiment_name, False, seed)
    config['subject_id'] = args.subject_id

    # Load the data
    labels = read_spindle_trains_labels(config['old_dataset'])
    data = read_pretraining_dataset(config['MASS_dir'], patients_to_keep=[args.subject_id])

    assert args.subject_id in data.keys(), 'Subject not in the dataset'
    assert args.subject_id in labels.keys(), 'Subject not in the dataset'

    dataset = SingleSubjectDataset(config['subject_id'], data=data, labels=labels, config=config) 
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0)       

    # Load the model
    net = get_trained_model(config, config['path_models'] / args.model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run the adaptation
    start = time.time()
    output_total, window_labels_total, loss, acc, tp, tn, fp, fn, net_copy = run_adaptation(dataloader, net, device, config)
    end = time.time()
    print('Time: ', end - start)

    # Get the metrics
    f1, precision, recall = get_metrics(tp, fp, fn)

    # Print the results
    print('Loss: ', loss)
    print('Accuracy: ', acc)
    print('F1: ', f1)
    print('Precision: ', precision)
    print('Recall: ', recall)
