<a href="https://aimeos.org/">
    <img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" alt="Pytorch logo" title="Pytorch" align="right" height="90" />
</a>

# Very Deep Convolutional Networks for Text Classification

[![Total Downloads](https://img.shields.io/github/downloads/ZappaRoberto/VDCNN/total.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

VDCNN is a neural network that use deep architectures of many convolutional layers to approach text classification and sentiment analysis using up to 29 layers.
You could read the original paper at the following [link](https://aclanthology.org/E17-1104/).

This repository is a personal implementation of this paper using pytorch. 


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

    def forward(self, x):
        return self.sequential(x)
```

The third layer is a **`convolutional block layer`** structured as shown in the following figure:
<p align="center">
  <img src="https://github.com/ZappaRoberto/VDCNN/blob/main/img/conv_block.png" />
</p>


