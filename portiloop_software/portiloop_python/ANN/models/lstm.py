import torch.nn as nn
import torch
import logging
from torch.nn import functional as F
import numpy as np
from math import floor
import copy

from portiloop_software.portiloop_python.ANN.models.model_blocks import AttentionLayer, FullAttention, TransformerEncoderLayer


ABLATION = 0


def get_final_model_config_dict(index=0, split_i=0):
    """
    Configuration dictionary of the final 1-input pre-trained model presented in the Portiloop paper.

    Args:
        index: last number in the name of the pre-trained model (several are provided)
        split_i: index of the random train/validation/test split (you can ignore this for inference)

    Returns:
        configuration dictionary of the pre-trained model
    """
    c_dict = {'experiment_name': f'sanity_check_final_model_1',
              'device_train': 'cuda',
              'device_val': 'cuda',
              'device_inference': 'cpu',
              'nb_epoch_max': 150,
              'max_duration': 257400,
              'nb_epoch_early_stopping_stop': 100,
              'early_stopping_smoothing_factor': 0.1,
              'fe': 250,
              'nb_batch_per_epoch': 1000,
              'first_layer_dropout': False,
              'power_features_input': False,
              'dropout': 0.0,
              'adam_w': 0.01,
              'distribution_mode': 0,
              'classification': True,
              'reg_balancing': 'none',
              'split_idx': split_i,
              'validation_network_stride': 1,
              'nb_conv_layers': 3,
              'seq_len': 50,
              'nb_channel': 31,
              'hidden_size': 7,
              'seq_stride_s': 0.170,
              'nb_rnn_layers': 1,
              'RNN': True,
              'envelope_input': False,
              'lr_adam': 0.0005,
              'batch_size': 256,
              'window_size_s': 0.218,
              'stride_pool': 1,
              'stride_conv': 1,
              'kernel_conv': 7,
              'kernel_pool': 7,
              'dilation_conv': 1,
              'dilation_pool': 1,
              'nb_out': 18,
              'time_in_past': 8.5,
              'estimator_size_memory': 188006400}
    return c_dict


class ConvPoolModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channel,
                 kernel_conv,
                 stride_conv,
                 conv_padding,
                 dilation_conv,
                 kernel_pool,
                 stride_pool,
                 pool_padding,
                 dilation_pool,
                 dropout_p):
        super(ConvPoolModule, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channel,
                              kernel_size=kernel_conv,
                              stride=stride_conv,
                              padding=conv_padding,
                              dilation=dilation_conv)
        self.pool = nn.MaxPool1d(kernel_size=kernel_pool,
                                 stride=stride_pool,
                                 padding=pool_padding,
                                 dilation=dilation_pool)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return self.dropout(x)


class FcModule(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 dropout_p):
        super(FcModule, self).__init__()

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.dropout(x)


class PortiloopNetwork(nn.Module):
    def __init__(self, c_dict):
        super(PortiloopNetwork, self).__init__()

        RNN = c_dict["RNN"]
        stride_pool = c_dict["stride_pool"]
        stride_conv = c_dict["stride_conv"]
        kernel_conv = c_dict["kernel_conv"]
        kernel_pool = c_dict["kernel_pool"]
        nb_channel = c_dict["nb_channel"]
        hidden_size = c_dict["hidden_size"]
        window_size_s = c_dict["window_size_s"]
        dropout_p = c_dict["dropout"]
        dilation_conv = c_dict["dilation_conv"]
        dilation_pool = c_dict["dilation_pool"]
        fe = c_dict["fe"]
        nb_conv_layers = c_dict["nb_conv_layers"]
        nb_rnn_layers = c_dict["nb_rnn_layers"]
        first_layer_dropout = c_dict["first_layer_dropout"]
        self.envelope_input = c_dict["envelope_input"]
        self.power_features_input = c_dict["power_features_input"]
        self.classification = c_dict["classification"]

        self.after = "attention"
        # self.after = "cnn"
        # self.after = None

        conv_padding = 0  # int(kernel_conv // 2)
        pool_padding = 0  # int(kernel_pool // 2)
        window_size = int(window_size_s * fe)
        nb_out = window_size

        for _ in range(nb_conv_layers):
            nb_out = out_dim(nb_out, conv_padding, dilation_conv, kernel_conv, stride_conv)
            nb_out = out_dim(nb_out, pool_padding, dilation_pool, kernel_pool, stride_pool)

        output_cnn_size = int(nb_channel * nb_out)

        self.RNN = RNN
        self.cnn = ConvPoolModule(in_channels=1,
                                                 out_channel=nb_channel,
                                                 kernel_conv=kernel_conv,
                                                 stride_conv=stride_conv,
                                                 conv_padding=conv_padding,
                                                 dilation_conv=dilation_conv,
                                                 kernel_pool=kernel_pool,
                                                 stride_pool=stride_pool,
                                                 pool_padding=pool_padding,
                                                 dilation_pool=dilation_pool,
                                                 dropout_p=dropout_p if first_layer_dropout else 0)

        self.cnn2 = nn.Sequential(*(ConvPoolModule(in_channels=nb_channel,
                                                         out_channel=nb_channel,
                                                         kernel_conv=kernel_conv,
                                                         stride_conv=stride_conv,
                                                         conv_padding=conv_padding,
                                                         dilation_conv=dilation_conv,
                                                         kernel_pool=kernel_pool,
                                                         stride_pool=stride_pool,
                                                         pool_padding=pool_padding,
                                                         dilation_pool=dilation_pool,
                                                         dropout_p=dropout_p) for _ in range(nb_conv_layers - 1)))
        self.gru = nn.GRU(input_size=output_cnn_size,
                                    hidden_size=hidden_size,
                                    num_layers=nb_rnn_layers,
                                    dropout=0,
                                    batch_first=True)

        fc_features = 0
        fc_features += hidden_size
        out_features = 1

        n_heads = 4

        self.attention_layer = TransformerEncoderLayer(
            attention=AttentionLayer(
                FullAttention(),
                fc_features,
                n_heads
            ),
            d_model=fc_features,
            dropout=dropout_p,
            norm_layer=nn.LayerNorm(fc_features)
        )
        self.cls = nn.Parameter(torch.randn(1, 1, fc_features))

        self.fc = nn.Linear(in_features=fc_features * 2 if self.after == "attention" else fc_features,  # enveloppe and signal + power features ratio
                            out_features=out_features)  # probability of being a spindle


    def forward(self, x, h, past_x=None, max_value=np.inf):
        # x1 : input 1 : cleaned signal
        # x2 : input 2 : envelope
        # x3 : power features ratio
        # h1 : gru 1 hidden size
        # h2 : gru 2 hidden size
        # max_value (optional) : print the maximal value reach during inference (used to verify if the FPGA implementation precision is enough)
        (batch_size, sequence_len, features) = x.shape

        x = x.view(-1, 1, features)
        x = self.cnn(x)
        x = self.cnn2(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = x.view(batch_size, sequence_len, -1)
        x, h = self.gru(x, h)

        out_gru = x[:, -1, :]  # output size: 1

        if self.after == "attention":
            
            # Use the accumulated past embeddings for attention (for validation)
            if past_x is not None:
                x = torch.cat((past_x, x), dim=1)

            # Append the cls token to the output of the GRU
            cls = self.cls.expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)

            out_attention = self.attention_layer(x)

            # Concatenate the output of the GRU and the output of the attention layer
            out = torch.cat((out_gru, out_attention[:, 0, :]), dim=1)
        elif self.after == "cnn":
            out = out_gru
            # Add CNN experiment
        else:
            out = out_gru

        out = self.fc(out)

        x = torch.sigmoid(out)

        return x, h, out_gru


def out_dim(window_size, padding, dilation, kernel, stride):
    return floor((window_size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


if __name__ == "__main__":
    config = get_final_model_config_dict()
    model = PortiloopNetwork(config)
    window_size = int(config['window_size_s'] * config['fe'])
    x = torch.randn(config['batch_size'], config['seq_len'], window_size)
    h = torch.zeros(config['nb_rnn_layers'], config['batch_size'], config['hidden_size'])

    res_x, res_h = model(x, h)
