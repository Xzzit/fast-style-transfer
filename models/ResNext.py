import torch.nn as nn
from basic import ConvLayer, DeconvLayer, ResNextLayer


class TransformerResNextNetwork(nn.Module):
    """
    Feedforward Transformation Network - ResNeXt

        - No Tanh
        + ResNeXt Layer
    """

    def __init__(self):
        super(TransformerResNextNetwork, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResNextLayer(128, [64, 64, 128], kernel_size=3),
            ResNextLayer(128, [64, 64, 128], kernel_size=3),
            ResNextLayer(128, [64, 64, 128], kernel_size=3),
            ResNextLayer(128, [64, 64, 128], kernel_size=3),
            ResNextLayer(128, [64, 64, 128], kernel_size=3)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out