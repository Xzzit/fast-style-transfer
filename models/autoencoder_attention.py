import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Note that attention mechanism consume huge amount of computational resources
256 x 256 image is ok for 32gb RAM
1080 x 1080 is not work
"""


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


class AutoencoderAttention(nn.Module):
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

        # Self Attention Layers
        self.sa = SelfAttention(128)

    def forward(self, x):
        y = self.ConvBlock(x)
        y = self.sa(y)
        y = self.ResidualBlock(y)
        y = self.UpSampleBlock(y)
        return y


if __name__ == "__main__":
    test_data = torch.randn(5, 3, 256, 256)
    print('Before: ', test_data.shape)

    trans = AutoencoderAttention()
    print('# of parameters: ', sum(p.numel() for p in trans.parameters()))

    print('After: ', trans(test_data).shape)
