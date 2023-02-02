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
- [License](#license)
- [Links](#links)


## Architecture Analysis

The overall architecture of this network is shown in the following Figure:
![VDCNN Architecture](https://user-images.githubusercontent.com/213803/211545083-d0820b63-26f2-453e-877f-ecd5ec128713.jpg)

The latest TYPO3 version can be installed via composer. This is especially useful, if you want to create new TYPO3 installations automatically or play with the latest code. You need to install the composer package first, if it isn't already available:

```bash
php -r "readfile('https://getcomposer.org/installer');" | php -- --filename=composer
```

To install the TYPO3 base distribution first, execute this command:

```bash
composer create-project typo3/cms-base-distribution myshop
# or install a specific TYPO3 version:
composer create-project "typo3/cms-base-distribution:^11" myshop
```

It will install TYPO3 into the `./myshop/` directory. Change into the directory and install TYPO3 as usual:
