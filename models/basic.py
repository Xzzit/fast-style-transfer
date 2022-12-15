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


class ConvLayerNB(nn.Module):
    """
    conv w/o bias -> Ins/Batch norm
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, norm_type='instance'):
        super().__init__()

        # Convolution Layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size//2, padding_mode='reflect', bias=False, groups=groups)

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


class BottleNetLayer(nn.Module):
    """
    Aggregated Residual Transformations for Deep Neural Networks
    Equal to better performance with 10x less parameters
    https://arxiv.org/abs/1611.05431
    """
    def __init__(self, in_ch=128, channels=[64, 64, 128], kernel_size=3):
        super().__init__()
        ch1, ch2, ch3 = channels
        self.conv1 = ConvLayer(in_ch, ch1, kernel_size=1, stride=1)
        self.conv2 = ConvLayer(ch1, ch2, kernel_size=kernel_size, stride=1)
        self.conv3 = ConvLayer(ch2, ch3, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = self.conv3(out)
        out = out + identity
        return out


class NormReluConv(nn.Module):
    """
    Ins/Batch norm -> ReLU -> conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type="instance"):
        super().__init__()

        # Normalization Layers
        if norm_type == "instance":
            self.norm_layer = nn.InstanceNorm2d(in_channels, affine=True)
        elif norm_type == "batch":
            self.norm_layer = nn.BatchNorm2d(in_channels, affine=True)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                    padding=kernel_size//2, padding_mode='reflect')

    def forward(self, x):
        x = self.norm_layer(x)
        x = F.relu(x, inplace=True)
        x = self.conv_layer(x)
        return x


class NormReluConvNB(nn.Module):
    """
    Ins/Batch norm -> ReLU -> conv w/o bias
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type="instance"):
        super().__init__()

        # Normalization Layers
        if norm_type == "instance":
            self.norm_layer = nn.InstanceNorm2d(in_channels, affine=True)
        elif norm_type == "batch":
            self.norm_layer = nn.BatchNorm2d(in_channels, affine=True)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                    padding=kernel_size//2, padding_mode='reflect', bias=False)

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

        self.conv1 = NormReluConvNB(in_channels, 4*out_channels, 1, 1)
        self.conv3 = NormReluConvNB(4*out_channels, out_channels, 3, 1)

    def forward(self, x):
        out = self.conv3(self.conv1(x))
        out = torch.cat((x, out), 1)
        return out


class ResNextBottleneck(nn.Module):
    """
    paper: https://arxiv.org/abs/1611.05431
    """

    def __init__(self, channels, cardinality):
        super().__init__()

        D = channels // 2

        self.conv1 = ConvLayerNB(channels, D, 1, 1, norm_type='batch')
        self.conv2 = ConvLayerNB(D, D, 3, 1, groups=cardinality, norm_type='batch')
        self.conv3 = ConvLayerNB(D, channels, 1, 1, norm_type='batch')

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y, inplace=True)
        y = self.conv2(y)
        y = F.relu(y, inplace=True)
        y = self.conv3(y)
        return F.relu(x + y, inplace=True)
