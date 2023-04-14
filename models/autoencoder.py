import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    conv -> Ins/Batch norm
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type='instance'):
        super().__init__()

        # Convolution Layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size//2, padding_mode='reflect')

        # Normalization Layers
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels, affine=True)

        self.norm_type = norm_type

    def forward(self, x):
        y = self.conv(x)

        if self.norm_type == 'None':
            pass
        else:
            y = self.norm(y)
        return y


class ResidualBlock(nn.Module):
    """
    ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)

    def forward(self, x):
        residual = x
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        y = y + residual
        y = F.relu(y, inplace=True)
        return y


class UpsampleConvLayer(nn.Module):
    """
    UpsampleConvLayer
    Upsamples the input and then does a convolution. This method prevent checkboard artifacts
    thus achieve better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm_type='instance'):
        super().__init__()

        self.upsample = upsample

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, padding_mode='reflect')

        # Normalization Layers
        self.norm_type = norm_type
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):

        if self.upsample:
            x = nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)

        out = self.conv2d(x)

        if self.norm_type == 'None':
            pass
        else:
            out = self.norm(out)

        
        return out


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial convolution block
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),  # (b, 32, h, w)
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),  # (b, 64, h//2, w//2)
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),  # (b, 128, h//4, w//4)
            nn.ReLU()
        )

        # Residual block
        self.ResidualBlock = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        # Deconvolution block
        self.UpSampleBlock = nn.Sequential(
            UpsampleConvLayer(128, 64, 3, 1, 2),  # (b, 64, h//4, w//4)
            nn.ReLU(),
            UpsampleConvLayer(64, 32, 3, 1, 2),  # (b, 32, h//2, w//2)
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm_type='None')  # (b, 3, h, w)
        )

    def forward(self, x):
        y = self.ConvBlock(x)
        y = self.ResidualBlock(y)
        y = self.UpSampleBlock(y)
        return y


if __name__ == "__main__":
    test_data = torch.randn(5, 3, 256, 256)
    print('Before: ', test_data.shape)

    trans = Autoencoder()
    print('# of parameters: ', sum(p.numel() for p in trans.parameters()))

    print('After: ', trans(test_data).shape)
