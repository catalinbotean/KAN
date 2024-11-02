import torch
import torch.nn as nn

class DoubleKANConvolutionalBlock(nn.Module):
    """
    - KAN Convolutional Layer
    - Batch Normalization
    - ReLU Activation
    - KAN Convolutional Layer
    - Batch Normalization
    - ReLU Activation
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation = dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)