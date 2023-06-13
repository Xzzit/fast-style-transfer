# Fast Neural Style Transfer in Pytorch :art: :rocket:

A Pytorch implementation of paper [**Perceptual Losses for Real-Time Style Transfer and Super-Resolution**](https://arxiv.org/abs/1603.08155) by *Justin Johnson, Alexandre Alahi, and Fei-Fei Li*. ***Note that*** the original paper proposes the algorithm to conduct 1) neural style transfer task and 2) image super-resolution task. This implementation can only be used to 1) stylize images with arbitrary artistic style.

The idea 'neural style transfer' is proposed by *Leon A. Gatys, Alexander S. Ecker, Matthias Bethge* in paper [**Image Style Transfer Using Convolutional Neural Networks**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), where the content features are represented as outputs of some selected VGG-16 layers and style features are represented as their Gram matrix.

This repo is based on the code [**fast-neural-style-pytorch**](https://github.com/rrmina/fast-neural-style-pytorch) by *rrmina* and [**fast_neural_style**](https://github.com/pytorch/examples/tree/main/fast_neural_style) by *Pytorch Official*.

## Dependencies
Tested With:
* Windows 10/11 || Mac M1 chip || Ubuntu 22.04 (Reconmended)
* Python 3.10
* Pytorch 2.0.1

```
conda create -n fst python=3.10
conda activate fst
pip install -r requirements.txt
```
Then download the latest [**PyTorch**](https://pytorch.org/).

## Example Output
```
python stylize_arg.py --c ./pretrained_models/bear.jpg --m ./pretrained_models/Fauvism_André-Derain_Pier.pth
```

<div style="display: flex; justify-content: center;">
    <img src="pretrained_models/bear.jpg" height="300px" title="content image">
    <img src="pretrained_models/Fauvism_André-Derain_Pier.jpg" height="300px" title="style image">
    <img src="pretrained_models/stylized.jpg" height="300px" title="generated image">
</div>

## Usage
***Train the model*** :hammer_and_wrench:

```
python train_arg.py --d <path/to/content/images/folder> --i <path/to/style/image/file>
```

- `--d`: path to training content images folder, I use Train images [118K/18GB] in [COCO 2017](https://cocodataset.org/#download).
- `--i`: path to style-image.
- `--mps`: add it for running on macOS GPU
- `--save-model-dir`: path to folder where trained model will be saved.
- `--c`: weight for content-loss, default is 1e5.
- `--s`: weight for style-loss, default is 1e10.
- `--cs`: weight for consistency-loss, default is 1e0.
- `-tv`: weight for total variance-loss, default is 1e0.

To learn about additional command line arguments, please refer to `train_arg.py`. For more information on the neural network architecture, please see the `models` folder.
If you're training new models, you may need to adjust the values of `--c`, `--s`, `--cs`, and `--tv`.

***Stylize the image*** :paintbrush:

```
python stylize_arg.py --c <path/to/content/image/file> --m <path/to/saved/model>
```

- `--c`: path to content image you want to stylize.
- `--m`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
- `--mps`: add it for running on macOS GPU
- `--output-path`: path for saving the output image, default is current path.
- `--output-name`: name of output image, default format is `stylized.jpg`
- `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)

Make sure that stylizaiton neural network has same `model-type` with pre-trained model.
