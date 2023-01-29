import torch
from torch import nn
from torchinfo import summary
# TODO: understand embedding dimension. Is not clear


class LookUpTable(nn.Module):
    def __init__(self, f0=69, s=16):
        super(LookUpTable, self).__init__()
        '''
        Our model begins with a look-up table that generates a 2D tensor of size (f0, s) that contain the embeddings
        of the s characters. s is fixed to 1024, and f0 can be seen as the ”RGB” dimension of the input text.
        '''
        self.embeddings = nn.Embedding(f0, s)

    def forward(self, x):
        return self.embeddings(x)


class FirstConvLayer(nn.Module):
    def __init__(self, skip_connection, want_shortcut):
        super(FirstConvLayer, self).__init__()
        '''
        We first apply one layer of 64 convolutions of size 3
        '''
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3),
        )
        self.want_shortcut = want_shortcut
        if self.want_shortcut:
            self.skip = skip_connection
            self.sequential.append(self.skip)

    def forward(self, x):
        if self.want_shortcut:
            self.skip.clean()
        return self.sequential(x)


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection, want_shortcut):
        super(ConvolutionalBlock, self).__init__()
        '''
         Each convolutional block is a sequence of two convolutional layers, 
         each one followed by a temporal BatchNorm layer and an ReLU activation.
         The kernel size of all the temporal convolutions is 3, with padding such that the temporal resolution is
         preserved (or halved in the case of the convolutional pooling with stride 2)
        '''
        self.want_shortcut = want_shortcut
        if self.want_shortcut:
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
        if self.want_shortcut:
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
    def __init__(self, depth, want_shortcut=False):
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

        self.skip_connection = SkipConnection()
        self.sequential = nn.Sequential(
            LookUpTable(),
            FirstConvLayer(self.skip_connection, want_shortcut)
        )
        for x in range(len(num_conv_block)):
            for _ in range(num_conv_block[x]):
                self.sequential.append(ConvolutionalBlock(channels[x], channels[x], self.skip_connection, want_shortcut))
            self.sequential.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.sequential.append(FullyConnectedBlock(10))

    def forward(self, x):
        return self.sequential(x)


if __name__ == "__main__":
    device = torch.device("cuda")
    model = VDCNN(depth=9)
    model.eval().to(device)
    summary(model)
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))