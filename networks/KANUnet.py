import torch
import torch.nn as nn
from layers.DoubleKANConvolutionalBlock import DoubleKANConvolutionalBlock


class KANUpBlock(nn.Module):
    """
    Upsampling block with KAN-based convolutional block.
    - Upsample
    - KAN Convolutional Layer
    - Batch Normalization
    - ReLU Activation
    - KAN Convolutional Layer
    - Batch Normalization
    - ReLU Activation
    """
    def __init__(self, in_channels, out_channels):
        super(KANUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.kan = DoubleKANConvolutionalBlock(in_channels, out_channels)

    def forward(self, x):
        return self.kan(self.up(x))


class KANUNet(nn.Module):
    """
    U-Net architecture with KAN-based convolutional blocks.
    """
    def __init__(self, in_channels, out_channels):
        super(KANUNet, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = DoubleKANConvolutionalBlock(in_channels, 64)
        self.enc2 = DoubleKANConvolutionalBlock(64, 128)
        self.enc3 = DoubleKANConvolutionalBlock(128, 256)
        self.enc4 = DoubleKANConvolutionalBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleKANConvolutionalBlock(512, 1024)

        # Decoder (Expanding Path) with KAN-based upsampling
        self.up4 = KANUpBlock(1024, 512)
        self.dec4 = DoubleKANConvolutionalBlock(1024, 512)  # 1024 due to skip connection

        self.up3 = KANUpBlock(512, 256)
        self.dec3 = DoubleKANConvolutionalBlock(512, 256)

        self.up2 = KANUpBlock(256, 128)
        self.dec2 = DoubleKANConvolutionalBlock(256, 128)

        self.up1 = KANUpBlock(128, 64)
        self.dec1 = DoubleKANConvolutionalBlock(128, 64)

        # Final output layer
        self.final_conv = DoubleKANConvolutionalBlock(64, out_channels)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Final output
        output = self.final_conv(dec1)
        return output


# Example usage:
if __name__ == "__main__":
    # Create model instance
    model = KANUNet(in_channels=1, out_channels=1)

    # Test with random input
    x = torch.randn(1, 1, 512, 512)
    output = model(x)
    print(f"Output shape: {output.shape}")
