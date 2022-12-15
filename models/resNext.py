import torch
import torch.nn as nn
from models.basic import ConvLayer, ResNextBottleneck, DeconvLayer


class ResNext(nn.Module):
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

        # ResNext block
        self.ResNextBottleneck = nn.Sequential(
            *[ResNextBottleneck(128, 32) for _ in range(5)]
        )

        # Deconvolution block
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),  # (b, 64, h//4, w//4)
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),  # (b, 32, h//2, w//2)
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm_type='None')  # (b, 3, h, w)
        )

    def forward(self, x):
        y = self.ConvBlock(x)
        y = self.ResNextBottleneck(y)
        y = self.DeconvBlock(y)
        return y


if __name__ == "__main__":
    test_data = torch.randn(5, 3, 256, 256)
    print('Before: ', test_data.shape)

    trans = ResNext()
    print('# of parameters: ', sum(p.numel() for p in trans.parameters()))

    print('After: ', trans(test_data).shape)
