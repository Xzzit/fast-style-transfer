import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    conv -> Ins/Batch norm
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type='instance', bias=True):
        super().__init__()

        # Convolution Layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size//2, padding_mode='reflect', bias=bias)

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


class NormReluConv(nn.Module):
    """
    Ins/Batch norm -> ReLU -> conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type="instance", bias=True):
        super().__init__()

        # Normalization Layers
        if norm_type == "instance":
            self.norm_layer = nn.InstanceNorm2d(in_channels, affine=True)
        elif norm_type == "batch":
            self.norm_layer = nn.BatchNorm2d(in_channels, affine=True)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                    padding=kernel_size//2, padding_mode='reflect', bias=bias)

    def forward(self, x):
        x = self.norm_layer(x)
        x = F.relu(x, inplace=True)
        x = self.conv_layer(x)
        return x


class DenseLayerBottleNeck(nn.Module):
    """
    NORM - RELU - CONV1 -> NORM - RELU - CONV3
    out_channels = Growth Rate
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = NormReluConv(in_channels, 4*out_channels, 1, 1, bias=False)
        self.conv3 = NormReluConv(4*out_channels, out_channels, 3, 1, bias=False)

    def forward(self, x):
        out = self.conv3(self.conv1(x))
        out = torch.cat((x, out), 1)
        return out


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


class DenseNet(nn.Module):
    """
    Feedforward Transformer Network using DenseNet Block instead of Residual Block
    """
    def __init__(self):
        super().__init__()

        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1, bias=False),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2, bias=False),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2, bias=False),
            nn.ReLU()
        )

        self.DenseBlock = nn.Sequential(
            NormReluConv(128, 64, 1, 1),
            DenseLayerBottleNeck(64, 16),
            DenseLayerBottleNeck(80, 16),
            DenseLayerBottleNeck(96, 16),
            DenseLayerBottleNeck(112, 16)
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
        x = self.ConvBlock(x)
        x = self.DenseBlock(x)
        out = self.UpSampleBlock(x)
        return out


if __name__ == "__main__":
    test_data = torch.randn(5, 3, 256, 256)
    print('Before: ', test_data.shape)

    trans = DenseNet()
    print('# of parameters: ', sum(p.numel() for p in trans.parameters()))

    print('After: ', trans(test_data).shape)