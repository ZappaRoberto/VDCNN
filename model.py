import torch
from torch import nn
import random
from torchinfo import summary


class LookUpTable(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super(LookUpTable, self).__init__()
        self.embeddings = nn.Embedding(num_embedding, embedding_dim)

    def forward(self, x):
        return self.embeddings(x).transpose(1, 2)


class FirstConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, skip_connection):
        super(FirstConvLayer, self).__init__()
        '''
        We first apply one layer of 64 convolutions of size 3
        '''
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size),
        )
        if skip_connection is not None:
            self.sequential.append(skip_connection)

    def forward(self, x):
        return self.sequential(x)


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection):
        super(ConvolutionalBlock, self).__init__()
        if skip_connection is not None:
            self.skip_connection = skip_connection
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels * 2, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.skip_connection is not None:
            short = self.skip.get()
            out = self.sequential(x)
            if out.shape != short.shape:
                short = self.shortcut(short)
            out = torch.add(short, out)
            self.skip_connection(out)
            return out
        else:
            return self.sequential(x)


class SkipConnection(nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()
        self.x = []

    def forward(self, x):
        self.x.append(x)
        return x

    def get(self):
        return self.x.pop()

    def clean(self):
        self.x = []


class FullyConnectedBlock(nn.Module):
    def __init__(self, n_class):
        super(FullyConnectedBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_class),
            nn.Softmax(n_class)
        )

    def forward(self, x):
        return self.sequential(x)


class VDCNN(nn.Module):
    def __init__(self, depth, want_shortcut=True):
        super(VDCNN, self).__init__()
        channels = [64, 128, 256, 512]
        if depth == 9:
            num_conv_block = [1, 1, 1, 1]
        elif depth == 17:
            num_conv_block = [2, 2, 2, 2]
        elif depth == 29:
            num_conv_block = [5, 5, 2, 2]
        else:
            num_conv_block = [8, 8, 5, 3]

        if want_shortcut:
            self.skip_connection = SkipConnection()
        else:
            self.skip_connection = None

        self.sequential = nn.Sequential(
            LookUpTable(num_embedding=69, embedding_dim=16),
            FirstConvLayer(in_channels=16, out_channels=64, kernel_size=3, skip_connection=self.skip_connection)
        )
        for x in range(len(num_conv_block)):
            for _ in range(num_conv_block[x]):
                self.sequential.append(ConvolutionalBlock(channels[x], channels[x], self.skip_connection))
            self.sequential.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.sequential = self.sequential[:-1].append(nn.AdaptiveMaxPool1d(8))

        self.fc = FullyConnectedBlock(10)

    def forward(self, x):
        out = self.sequential(x)
        out = out.view(out.size(0), -1)
        if self.skip_connection is not None:
            self.skip_connection.clean()
        return self.fc(out)


if __name__ == "__main__":
    device = torch.device("cuda")
    model = VDCNN(depth=9)
    model.eval().to(device)
    summary(model)
