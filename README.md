<a href="https://pytorch.org/">
    <img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" alt="Pytorch logo" title="Pytorch" align="right" height="80" />
</a>

# Very Deep Convolutional Networks for Text Classification

[![Total Downloads](https://img.shields.io/github/downloads/ZappaRoberto/VDCNN/total.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Release](https://img.shields.io/github/v/release/ZappaRoberto/VDCNN?color=purple)


VDCNN is a neural network that use deep architectures of many convolutional layers to approach text classification and sentiment analysis using up to 49 layers.
You could read the original paper at the following [link](https://aclanthology.org/E17-1104/).
This repository is a personal implementation of this paper using PyTorch 1.13. 


## Table Of Content

- [Architecture Analysis](#Architecture-Analysis)
- [Dataset](#Dataset)
    - [Yahoo! Answer topic classification](#Yahoo!-Answer-topic-classification)
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

The first block is a **`lookup table`** that generates a 2D tensor  of size (f0, s) that contain the embeddings of the s characters.

```python
class LookUpTable(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super(LookUpTable, self).__init__()
        self.embeddings = nn.Embedding(num_embedding, embedding_dim)

    def forward(self, x):
        return self.embeddings(x).transpose(1, 2)
```
> **Note**
> 
> The output dimension of the nn.Embedding layer is (s, f0). Use **`.transpose`** in order to have the right output dimension.

The second layer is a **`convolutional layer`** with in_channel dimension of 64 and kernel dimension of size 3.

```python
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

```python
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

```python

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

```python

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

```python
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

```python

class VDCNN(nn.Module):
    def __init__(self, depth, n_classes, want_shortcut=True, pool_type='VGG'):
```
The last class named VDCNN build all the layer in the right way and with the variable **`depth`** we can choose how many layer to add to our net. The paper present 4 different level of depth: 9, 17, 29, 49. You can find all theese piece of code inside the **model.py** file.

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>


## Dataset

The dataset used for the training part are the [Yahoo! Answers Topic Classification](https://www.kaggle.com/datasets/b78db332b73c8b0caf9bd02e2f390bdffc75460ea6aaaee90d9c4bd6af30cad2) and a subset of [Amazon review data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) that can be downloaded [here](https://drive.google.com/file/d/0Bz8a_Dbh9QhbZVhsUnRWRDhETzA/view?usp=share_link&resourcekey=0-Rp0ynafmZGZ5MflGmvwLGg). All this datasets are maneged by **`Dataset class`** inside dataset.py file. 


### Yahoo! Answer topic classification

The Yahoo! Answers topic classification dataset is constructed using the 10 largest main categories. Each class contains 140000 training samples and 6000 testing samples. Therefore, the total number of training samples is 1400000, and testing samples are 60000. The categories are:

* Society & Culture
* Science & Mathematics
* Health
* Education & Reference
* Computers & Internet
* Sports
* Business & Finance
* Entertainment & Music
* Family & Relationships
* Politics & Government


### Amazon Reviews

The Amazon Reviews dataset is constructed using 5 categories (star ratings).

<div align="right">[ <a href="#Table-Of-Content">↑ to top ↑</a> ]</div>

## Training

> **Warning**
> 
> Even if it can be choosen the device between cpu or GPU, I used and tested the training part only with GPU.

First things first, at the beginning of train.py file there are a some useful global variable that manage the key settings of the training.

```python

LEARNING_RATE = 0.01
MOMENTUM = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
MAX_LENGTH = 1024
NUM_EPOCHS = 1
PATIENCE = 40
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = "dataset/amazon/train.csv"
TEST_DIR = "dataset/amazon/test.csv"
```

> **Note**
> 
> Change **`TRAIN_DIR`** and **`TEST_DIR`** with your datasets local position.

The train_fn function is build to run one epoch and return the average loss and accuracy of the epoch.

```python

def train_fn(epoch, loader, model, optimizer, loss_fn, scaler):
    # a bunch of code
    return train_loss, train_accuracy
```

The main function is build to inizialize and manage the training part until the end.

```python

def main():
    model = VDCNN(depth=9, n_classes=5, want_shortcut=True, pool_type='vgg').to(DEVICE)
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.1, patience=int(PATIENCE / 2), threshold=0.0001,
                                                     threshold_mode='rel')
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, test_loader = get_loaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE, MAX_LENGTH, NUM_WORKERS, PIN_MEMORY)
    train_l, train_a, test_l, test_a = [], [], [], []

    patience = PATIENCE
    min_test_loss = 1000
    for epoch in range(NUM_EPOCHS):
        # run 1 epoch
        # check accuracy
        # save model if test_loss < min_test_loss
        # mange patience for early stopping
    save_plot(train_l, train_a, test_l, test_a)
    sys.exit()
```

**`get_loaders`**, **`save_checkpoint`**, **`load_checkpoint`**, **`check_accuracy`**, **`save_plot`** that are use inside tran.py file are function that can be find inside utils.py.


















> **Warning**
> Be careful running this with elevated privileges. Code execution can be achieved with write permissions on the config file.

> **Note**
> If you're using Linux Bash for Windows, [see this guide](https://www.howtogeek.com/261575/how-to-run-graphical-linux-desktop-applications-from-windows-10s-bash-shell/) or use `node` from the command prompt.

## Support 🌟

<a href="https://www.buymeacoffee.com/5Zn8Xh3l9" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/purple_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

:bulb:

|                            | 🔰 ArminC AutoExec  | ◾ Other Configs |
| -------------------------- | :----------------: | :-------------: |
| Optimized values           |         ✔️         |        ❌        |
| Useful scripts             |         ✔️         |        ❌        |
| Documented commands        |         ✔️         |        ❌        |
| Enabled in-game advantages |         ✔️         |        ❌        |
| No misconcepted commands   |         ✔️         |        ❌        |
| Professional info sources  |         ✔️         |        ❌        |
| Clean sheet/template       |         ✔️         |        ❌        |
| Easy to customize          |         ✔️         |        ❌        |
| Categorized by functions   |         ✔️         |        ❌        |
| New commands/values        |         ✔️         |        ❌        |
| No old command leftovers   |         ✔️         |        ❌        |
