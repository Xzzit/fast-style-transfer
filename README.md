# Fast Neural Style Transfer in Pytorch :art: :rocket:

A Pytorch implementation of paper [**Perceptual Losses for Real-Time Style Transfer and Super-Resolution**](https://arxiv.org/abs/1603.08155) by *Justin Johnson, Alexandre Alahi, and Fei-Fei Li*. **Note that** the original paper proposes the algorithm to conduct 1) neural style transfer task and 2) image super-resolution task. This implementation can only be used to stylize images with arbitrary artistic style.

The idea 'neural style transfer' is proposed by *Leon A. Gatys, Alexander S. Ecker, Matthias Bethge* in paper [**Image Style Transfer Using Convolutional Neural Networks**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), where the content features are represented as outputs of some selected VGG-16 layers and style features are represented as their Gram matrix.

This repo is based on the code [**fast-neural-style-pytorch**](https://github.com/rrmina/fast-neural-style-pytorch) by *rrmina* and [**fast_neural_style**](https://github.com/pytorch/examples/tree/main/fast_neural_style) by *Pytorch Official*.

## Dependencies
Tested With:
* Windows 10
* Python 3.7.15
* Pytorch 1.10

```
conda create -n fst python=3.7
conda activate fst
pip install -r requirements.txt
```

## Example Output

## Hardware Requirements

## Usage

## What's New
