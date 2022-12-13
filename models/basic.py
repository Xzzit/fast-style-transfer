import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type='instance'):
        super().__init__()
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels, affine=True)

        self.norm_type = norm_type

    def forward(self, x):
        y = self.reflection_pad(x)
        y = self.conv(y)

        if self.norm_type == 'None':
            pass
        else:
            y = self.norm(y)
        return y


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        y = y + residual
        return y


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        # Normalization Layers
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out


class SelfAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.query = self.conv(channels, channels // 8)
        self.key = self.conv(channels, channels // 8)
        self.value = self.conv(channels, channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.sm = nn.Softmax(dim=1)

    def conv(self, n_in, n_out):
        return nn.Conv1d(n_in, n_out, 1, bias=False)

    def forward(self, x):
        shape = x.size()
        x = x.view(*shape[:2], -1)
        f = self.query(x)
        g = self.key(x)
        h = self.value(x)
        beta = self.sm(torch.bmm(f.transpose(1, 2), g))
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*shape).contiguous()