# UNet structure with single conv and stride 2
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, type, activation="RELU", use_dropout=True, alpha=0.2):
        super(UNetBlock, self).__init__()

        assert type == "UP" or type == "DOWN"
        assert activation == "RELU" or activation == "LEAKY RELU"

        # Conv
        self.conv = None
        if type == "UP":
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect")

        # BatchNorm
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)
        # Relu
        self.relu = nn.ReLU() if activation == "RELU" else nn.LeakyReLU(negative_slope=alpha)
        # Dropout
        self.dropout = nn.Dropout(p=0.5) if use_dropout else None

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.relu(x)
        x = self.dropout(x) if self.dropout != None else x

        return x


class Generator(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(Generator, self).__init__()
        
        self.down0 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, padding=1, padding_mode="reflect"), # 64, ~=128, ~=128  (With padding=1, slightly different from 128)
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down1 = UNetBlock(base_channels, base_channels*2, type="DOWN", activation="LEAKY RELU", use_dropout=False)   # 128, ~=64, ~=64
        self.down2 = UNetBlock(base_channels*2, base_channels*4, type="DOWN", activation="LEAKY RELU", use_dropout=False) # 256, ~=32, ~=32
        self.down3 = UNetBlock(base_channels*4, base_channels*8, type="DOWN", activation="LEAKY RELU", use_dropout=False) # 512, ~=16, ~=16
        self.down4 = UNetBlock(base_channels*8, base_channels*8, type="DOWN", activation="LEAKY RELU", use_dropout=False) # 512, ~=8, ~=8
        self.down5 = UNetBlock(base_channels*8, base_channels*8, type="DOWN", activation="LEAKY RELU", use_dropout=False) # 512, ~=4, ~=4
        self.down6 = UNetBlock(base_channels*8, base_channels*8, type="DOWN", activation="LEAKY RELU", use_dropout=False) # 512, ~=2, ~=2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*8, 4, 2, padding=1, padding_mode="reflect"),   # 512, 1, 1
            nn.ReLU()
        )

        self.up1 = UNetBlock(base_channels*8, base_channels*8, type="UP", activation="RELU", use_dropout=True)
        self.up2 = UNetBlock(base_channels*8*2, base_channels*8, type="UP", activation="RELU", use_dropout=True)
        self.up3 = UNetBlock(base_channels*8*2, base_channels*8, type="UP", activation="RELU", use_dropout=True)
        self.up4 = UNetBlock(base_channels*8*2, base_channels*8, type="UP", activation="RELU", use_dropout=False)
        self.up5 = UNetBlock(base_channels*8*2, base_channels*4, type="UP", activation="RELU", use_dropout=False)
        self.up6 = UNetBlock(base_channels*4*2, base_channels*2, type="UP", activation="RELU", use_dropout=False)
        self.up7 = UNetBlock(base_channels*2*2, base_channels, type="UP", activation="RELU", use_dropout=False)
        
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, in_channels, 4, 2, padding=1),
            nn.Tanh()    
        )

    def forward(self, x):
        encode1 = self.down0(x)
        encode2 = self.down1(encode1)
        encode3 = self.down2(encode2)
        encode4 = self.down3(encode3)
        encode5 = self.down4(encode4)
        encode6 = self.down5(encode5)
        encode7 = self.down6(encode6)

        bottleneck = self.bottleneck(encode7)

        decode7 = self.up1(bottleneck)
        decode6 = self.up2(torch.cat([encode7, decode7], 1))
        decode5 = self.up3(torch.cat([encode6, decode6], 1))
        decode4 = self.up4(torch.cat([encode5, decode5], 1))
        decode3 = self.up5(torch.cat([encode4, decode4], 1))
        decode2 = self.up6(torch.cat([encode3, decode3], 1))
        decode1 = self.up7(torch.cat([encode2, decode2], 1))
        decode0 = self.up8(torch.cat([encode1, decode1], 1))

        return decode0
