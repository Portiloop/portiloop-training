# %% [markdown]
# ## Data Loading

# %%
from torchsummary import summary
from einops import rearrange
import torch.nn as nn
import time

import numpy as np

from portiloop_ml.portiloop_python.ANN.utils import get_configs

from portiloop_ml.portiloop_python.ANN.data.mass_data import read_pretraining_dataset, read_sleep_staging_labels, read_spindle_trains_labels
import torch


experiment_name = 'test_sleep_staging'
seed = 42

# config = get_configs(experiment_name, False, seed)
# config['nb_conv_layers'] = 4
# config['hidden_size'] = 64
# config['nb_rnn_layers'] = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run some testing on subject 1
# Load the data
# labels = read_spindle_trains_labels(config['old_dataset'])
# ss_labels = read_sleep_staging_labels(config['path_dataset'])
# for index, patient_id in enumerate(ss_labels.keys()):


# data = read_pretraining_dataset(config['MASS_dir'])

# %% [markdown]
# ## Model stuffs

# %%

# %%

class TransformerEncoderWithCLS(nn.Module):
    def __init__(self, embedder, embedding_size, num_heads, num_layers, num_classes):
        super(TransformerEncoderWithCLS, self).__init__()

        self.embedder = embedder

        self.positional_embedding = nn.Embedding(
            10 * 250, embedding_size).to(device)  # Positional embedding

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_size, num_heads),
            num_layers
        )
        self.cls_token = nn.Parameter(torch.randn(
            1, 1, embedding_size))  # Learnable <cls> token
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(
            embedding_size, num_classes)  # Classification layer

    def forward(self, x):
        x = self.embedder(x)  # Embed the input sequence
        batch_size, seq_len, embedding_size = x.size()

        # Add positional embedding
        positions = torch.arange(0, seq_len).expand(
            batch_size, seq_len).to(device)
        positions = self.positional_embedding(positions).expand(
            batch_size, seq_len, embedding_size)
        x = x + positions

        # Shape: (batch_size, 1, embedding_size)
        cls_tokens = self.cls_token.expand(batch_size, 1, embedding_size)
        # Shape: (batch_size, seq_length + 1, embedding_size)
        x_cls = torch.cat([cls_tokens, x], dim=1)

        output = self.transformer_encoder(x_cls)  # Apply TransformerEncoder

        out_cls = output[:, 0]  # Return the representation of the <cls> token

        out_cls = self.activation(out_cls)  # Apply activation function

        # Classify the <cls> token representation
        return self.classifier(out_cls)


# %%


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv1d(in_channels, out_channels[0], kernel_size=1)

        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[1], kernel_size=1),
            nn.Conv1d(out_channels[1], out_channels[2],
                      kernel_size=3, padding=1)
        )

        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[3], kernel_size=1),
            nn.Conv1d(out_channels[3], out_channels[4],
                      kernel_size=5, padding=2)
        )

        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels[5], kernel_size=1)
        )

    def forward(self, x):
        # x = rearrange(x, 'b s c -> b c s')  # Convert from (batch_size, seq_len, channels) to (batch_size, channels, seq_len)
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        # Concatenate the branch outputs along the channel dimension
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        outputs = torch.cat(outputs, dim=1)
        # Convert back to (batch_size, seq_len, channels)
        outputs = rearrange(outputs, 'b c s -> b s c')

        return outputs


# %%
incept = InceptionBlock(1, [16, 8, 16, 8, 16, 16])
batch_size = 64
freq = 250
seq_len = 10 * freq
in_size = 1
x = torch.randn(batch_size, in_size, seq_len)
embedding_size = 64

transformer = TransformerEncoderWithCLS(incept, embedding_size, 8, 2, 5)


# %%

summary(transformer, (1, seq_len))

# %%
transformer = transformer.to(device)
x = x.to(device)
out = transformer(x)
out.shape
