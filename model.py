import torch
from torch import nn
from torchinfo import summary


class LookUpTable(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super(LookUpTable, self).__init__()
        self.embeddings = nn.Embedding(num_embedding, embedding_dim)

    def forward(self, x):
        return self.embeddings(x).transpose(1, 2)


class FirstConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FirstConvLayer, self).__init__()
        self.sequential = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size))

    def forward(self, x):
        return self.sequential(x)


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, want_shortcut, downsample, last_layer, pool_type):
        super(ConvolutionalBlock, self).__init__()

        self.want_shortcut = want_shortcut
        if self.want_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.sequential = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)

        if downsample:
            if last_layer:
                self.want_shortcut = False
                self.sequential.append(nn.AdaptiveMaxPool1d(8))
            else:
                if pool_type == 'convolution':
                    self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                           kernel_size=3, stride=2, padding=1, bias=False)
                elif pool_type == 'kmax':
                    channels = [64, 128, 256, 512]
                    dimension = [511, 256, 128]
                    index = channels.index(in_channels)
                    self.sequential.append(nn.AdaptiveMaxPool1d(dimension[index]))
                else:
                    self.sequential.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.want_shortcut:
            short = x
            out = self.conv1(x)
            out = self.sequential(out)
            if out.shape != short.shape:
                short = self.shortcut(short)
            out = self.relu(short + out)
            return out
        else:
            out = self.conv1(x)
            return self.sequential(out)


class FullyConnectedBlock(nn.Module):
    def __init__(self, n_class):
        super(FullyConnectedBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.sequential(x)


class VDCNN(nn.Module):
    def __init__(self, depth, n_classes, want_shortcut, pool_type):
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

        self.sequential = nn.Sequential(
            LookUpTable(num_embedding=69, embedding_dim=16),
            FirstConvLayer(in_channels=16, out_channels=64, kernel_size=3)
        )
        last_layer = False
        for x in range(len(num_conv_block)):
            for i in range(num_conv_block[x]):
                if num_conv_block[x] - 1 == i:
                    if len(num_conv_block) - 1 == x:
                        last_layer = True
                        self.sequential.append(
                            ConvolutionalBlock(channels[x], channels[x], want_shortcut, True, last_layer, pool_type))
                    else:
                        self.sequential.append(ConvolutionalBlock(channels[x], channels[x] * 2,
                                                                  want_shortcut, True, last_layer, pool_type))
                else:
                    self.sequential.append(ConvolutionalBlock(channels[x], channels[x],
                                                              want_shortcut, False, last_layer, pool_type))

        self.fc = FullyConnectedBlock(n_classes)

    def forward(self, x):
        out = self.sequential(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VDCNN(depth=9, n_classes=10)
    model.eval().to(device)
    summary(model)
