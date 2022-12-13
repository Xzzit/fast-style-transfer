import torch
from basic import ConvLayer, ResidualBlock, UpsampleConvLayer, SelfAttention


class AutoencoderAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Up-sampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearity
        self.relu = torch.nn.ReLU()
        # Self Attention Layers
        self.sa1 = SelfAttention(32)
        self.sa2 = SelfAttention(64)
        self.sa3 = SelfAttention(128)

    def forward(self, x):  # (b, 3, h, w)
        y = self.relu(self.in1(self.conv1(x)))  # (b, 32, h, w)
        # y = self.sa1(y)
        y = self.relu(self.in2(self.conv2(y)))  # (b, 64, h//2, w//2)
        # y = self.sa2(y)
        y = self.relu(self.in3(self.conv3(y)))  # (b, 128, h//4, w//4)
        y = self.sa3(y)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))  # (b, 64, h//4, w//4)
        y = self.relu(self.in5(self.deconv2(y)))  # (b, 32, h//2, w//2)
        y = self.deconv3(y)  # (b, 3, h, w)
        return y


if __name__ == "__main__":
    test_data = torch.randn(5, 3, 256, 256)

    trans = AutoencoderAttention()
    print(sum(p.numel() for p in trans.parameters()))

    trans_attention = AutoencoderAttention()
    print(sum(p.numel() for p in trans_attention.parameters()))

    trans_attention(test_data)