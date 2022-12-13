import torch
import torch.nn as nn
from models.basic import ConvLayer, DeconvLayer, ConvLayerNB, NormReluConv, DenseLayerBottleNeck


class DenseNet(nn.Module):
    """
    Feedforward Transformer Network using DenseNet Block instead of Residual Block
    """
    def __init__(self):
        super().__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayerNB(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayerNB(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayerNB(64, 128, 3, 2),
            nn.ReLU()
        )
        self.DenseBlock = nn.Sequential(
            NormReluConv(128, 64, 1, 1),
            DenseLayerBottleNeck(64, 16),
            DenseLayerBottleNeck(80, 16),
            DenseLayerBottleNeck(96, 16),
            DenseLayerBottleNeck(112, 16)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm_type="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.DenseBlock(x)
        out = self.DeconvBlock(x)
        return out


if __name__ == "__main__":
    test_data = torch.randn(5, 3, 256, 256)
    print('Before: ', test_data.shape)

    trans = DenseNet()
    print('# of parameters: ', sum(p.numel() for p in trans.parameters()))

    print('After: ', trans(test_data).shape)