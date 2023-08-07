import math
import torch.nn as nn
import torch
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TSNConv(nn.Module):
    def __init__(self, fs):
        super(TSNConv, self).__init__()
        self.feature_1 = nn.Sequential(
            nn.Conv1d(1, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(0.5),
        )
        self.feature_2 = nn.Sequential(
            nn.Conv1d(128, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.feature_3 = nn.Sequential(
            nn.Conv1d(128, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.feature_4 = nn.Sequential(
            nn.Conv1d(128, 128, 8, 2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        """
        :param x: [bs, seq_len, 1, 7500]
        """
        batch_size = x.size(0)
        x = rearrange(x, 'b s c w -> (b s) c w')
        x = self.feature_1(x)
        # x = self.feature_2(x)
        x = self.feature_3(x)
        x = self.feature_4(x)
        x = rearrange(x, '(b s) e -> b s e', b=batch_size)
        return x


class TransformerEncoderWithCLS(nn.Module):
    def __init__(self, embedder, embedding_size, num_heads, num_layers, num_classes, cls=False, dropout=0.1):
        super(TransformerEncoderWithCLS, self).__init__()

        self.embedder = embedder

        self.positional_embedding = PositionalEncoding(
            embedding_size, dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embedding_size, num_heads, dropout=dropout, batch_first=True),
            num_layers
        )
        self.cls = cls
        if cls:
            self.cls_token = nn.Parameter(torch.randn(
                1, 1, embedding_size))  # Learnable <cls> token
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(
            embedding_size, num_classes)  # Classification layer
        self.final_activation = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.from_30_window = True

    def forward(self, x):

        # Split the input into windows
        if self.from_30_window:
            assert x.size(1) == 1
            x = x.unfold(-1, 200, 20)
            x = rearrange(x, 'b s_old ch s_new w -> b (s_old s_new) ch w')

        x = self.embedder(x)  # Embed the input sequence
        batch_size, seq_len, embedding_size = x.size()

        # Add positional embedding
        # x = rearrange(x, 'b e s -> s b e')
        # x = self.positional_embedding(x)
        # x = rearrange(x, 's b e -> b e s')

        if self.cls:
            # Shape: (batch_size, 1, embedding_size)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # Shape: (batch_size, seq_length + 1, embedding_size)
            x = torch.cat([cls_tokens, x], dim=1)

        # x = rearrange(x, 'b e s -> b s e')

        output = self.transformer_encoder(x)  # Apply TransformerEncoder
        output = x
        if self.cls:
            # Return the representation of the <cls> token
            output = output[:, 0, :]
        else:
            output = torch.mean(output, dim=1)

        # Classify the <cls> token representation
        out = self.dropout(self.classifier(output))

        return out


class SimpleModel(nn.Module):
    def __init__(self, seq_len, hidden_dim, num_classes):
        super(SimpleModel, self).__init__()

        # self.embedding = nn.Embedding(seq_len, hidden_dim)
        self.hidden_layer = nn.Linear(seq_len, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


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

        # # Max pooling branch
        # self.branch_pool = nn.Sequential(
        #     nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        #     nn.Conv1d(in_channels, out_channels[5], kernel_size=1)
        # )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        # branch_pool = self.branch_pool(x)

        # Concatenate the branch outputs along the channel dimension
        outputs = [branch1x1, branch3x3, branch5x5]
        outputs = torch.cat(outputs, dim=1)
        # Convert back to (batch_size, seq_len, channels)
        outputs = rearrange(outputs, 'b c s -> b s c')

        return outputs


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=5)

        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)

        return x


if __name__ == "__main__":
    config = {
        'embedding_size': 128,
        'num_heads': 8,
        'num_layers': 1,
        'dropout': 0.1,
        'cls': True,
    }

    x = torch.randn(32, 1, 1, 3000)
    emb = TSNConv(100)
    model = TransformerEncoderWithCLS(
        emb,
        config['embedding_size'],
        config['num_heads'],
        config['num_layers'],
        5,
        dropout=config['dropout'],
        cls=config['cls'])
    # print(emb(x).shape)
    # x = x.unfold(-1, 200, 20)
    # x = rearrange(x, 'b s_old ch s_new w -> b (s_old s_new) ch w')
    # print(emb(x).shape)
    print(model(x).shape)
