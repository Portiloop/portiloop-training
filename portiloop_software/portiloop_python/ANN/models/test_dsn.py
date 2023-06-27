import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from einops import rearrange

Fs = 250


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.5, bidirectional=True)

    def forward(self, x):
        # set initial hidden and cell states
        # RuntimeError: Input and hidden tensors are not at the same device
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        return out


class TinySleepNet(nn.Module):
    def __init__(self, fs):
        super(TinySleepNet, self).__init__()
        self.feature_1 = nn.Sequential(
            nn.Conv1d(1, 128, fs // 2, fs // 8),
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
            nn.Conv1d(128, 128, 8, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(0.5),
        )

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True,
        )

        self.classifier = nn.Linear(128, 5)

    def forward(self, x):
        """
        :param x: [bs, seq_len, 1, 7500]
        """
        batch_size = x.size(0)
        x = rearrange(x, 'b s c w -> (b s) c w')
        x = self.feature_1(x)
        x = self.feature_2(x)
        x = self.feature_3(x)
        x = self.feature_4(x)
        x = rearrange(x, '(b s) e -> b s e', b=batch_size)
        x, _ = self.rnn(x)
        x = F.relu(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


class DeepSleepNet(nn.Module):

    def __init__(self, ch=1):
        super(DeepSleepNet, self).__init__()
        self.features_s = nn.Sequential(
            nn.Conv1d(ch, 64, 50, 6),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(),
            nn.Conv1d(64, 128, 6),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 6),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.features_l = nn.Sequential(
            nn.Conv1d(ch, 64, 400, 50),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(),
            nn.Conv1d(64, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.features_seq = nn.Sequential(
            BiLSTM(9856, 512, 2),
        )
        self.res = nn.Linear(9856, 1024)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 5),
        )

    def forward(self, x):
        x_s = self.features_s(x)
        x_l = self.features_l(x)
        x_s = x_s.flatten(1, 2)
        x_l = x_l.flatten(1, 2)
        x = torch.cat((x_s, x_l), 1)  # [bs, 7296]
        x_seq = x.unsqueeze(1)
        x_blstm = self.features_seq(x_seq)  # [bs, 1, 1024]
        x_blstm = torch.squeeze(x_blstm, 1)
        x_res = self.res(x)
        x = torch.mul(x_res, x_blstm)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    ch_num = 1
    batch_size = 3
    seq_len = 20
    net = TinySleepNet(fs=Fs)
    summary(net, (seq_len, 1, int(Fs * 30)))
    net = net.cuda()
    inputs = torch.rand(batch_size, seq_len, 1, int(Fs * 30))
    inputs = inputs.cuda()
    outputs = net(inputs)
    print(outputs.size())
