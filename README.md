<a href="https://aimeos.org/">
    <img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" alt="Pytorch logo" title="Pytorch" align="right" height="90" />
</a>

# Very Deep Convolutional Networks for Text Classification

[![Total Downloads](https://img.shields.io/github/downloads/ZappaRoberto/VDCNN/total.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

VDCNN is a neural network that use deep architectures of many convolutional layers to approach text classification and sentiment analysis using up to 29 layers.
You could read the original paper at the following [link](https://aclanthology.org/E17-1104/).

This repository is a personal implementation of this paper using PyTorch 1.13. 


## Table Of Content

- [Architecture Analysis](#Architecture-Analysis)
- [Dataset](#Dataset)
    - [Yahoo! Answer](#Yahoo!-Answer)
    - [Amazon Reviews](#Amazon-Reviews)
- [Training](#Training)
    - [Yahoo! Answer](#Yahoo!-Answer)
    - [Amazon Reviews](#Amazon-Reviews)
- [Result Analysis](#Result-Analysis)
    - [Text Classification](#Text-Classification)
    - [Sentiment Analysis](#Sentiment-Analysis)
- [Installation Guide](#Installation-Guide)


## Architecture Analysis

The overall architecture of this network is shown in the following figure:
<p align="center">
  <img src="https://github.com/ZappaRoberto/VDCNN/blob/main/img/architecture.png" />
</p>

The first block is a **`lookup table`** that generates a 2D tensor  of size (f0, s) that contain the embeddings of the s characters. The output dimension of the nn.Embedding layer is (s, f0), so we need to do the transpose in order to have the right output dimension. 

```bash
class LookUpTable(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super(LookUpTable, self).__init__()
        self.embeddings = nn.Embedding(num_embedding, embedding_dim)

    def forward(self, x):
        return self.embeddings(x).transpose(1, 2)
```

The second layer is a **`convolutional layer`** with in_channel dimension of 64 and kernel dimension of size 3.

```bash
class FirstConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FirstConvLayer, self).__init__()
        self.sequential = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size))
```

The third layer is a **`convolutional block layer`** structured as shown in the following figure:
<p align="center">
  <img src="https://github.com/ZappaRoberto/VDCNN/blob/main/img/conv_block.png" />
</p>
We have also the possibility to add short-cut and in some layer we have to half the resolution with pooling. We can choose between three different pooling method: resnet like, VGG like or with k-max pooling.

```bash
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, want_shortcut, downsample, last_layer, pool_type='vgg'):
        super(ConvolutionalBlock, self).__init__()

        self.want_shortcut = want_shortcut
        if self.want_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(out_channels)
            )
```

with the variable **`want_shortcut`** we can choose if we want add shortcut to our net.

```bash

        self.sequential = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
```

in this piece of code we build the core part of the convolutional block, as shown in the previously figure. self.conv1 can't be added in self.sequential because its stride depends on the type of pooling we want to use.

```bash

        if downsample:
            if last_layer:
                self.want_shortcut = False
                self.sequential.append(nn.AdaptiveMaxPool1d(8))
            else:
                if pool_type == 'resnet':
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
```

the final part of this layer manage the type of pooling that we want to use. We can select the pooling type with the variable **`pool_type`**. The last layer use always k-max pooling with dimension 8 and for this reason we manage this difference between previously layer with the variable **`last_layer`**.

```bash
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
```

After the sequence of convolutional blocks we have 3 fully connected layer where we have to choose the output number of classes. Different task require different number of classes. We choose the number of classes with the variable **`n_class`**. Since we want to have the probability of each class given a text we use the softmax.

```bash

class VDCNN(nn.Module):
    def __init__(self, depth, n_classes, want_shortcut=True, pool_type='VGG'):
```
The last class named VDCNN build all the layer in the right way and with the variable **`depth`** we can choose how many layer to add to our net. The paper present 4 different level of depth: 9, 17, 29, 49.


## Dataset

The dataset used for the training part are the yahoo
