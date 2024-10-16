import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F


class ContractingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        conv = self.conv(x)
        out = self.downsample(conv)
        return conv, out


class ExpandingPath(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, scale),
            nn.ReLU(),
            nn.ReflectionPad2d((2, 2, 0, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 0, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 0, 0)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        crop = T.CenterCrop(x.shape[2:])
        x = torch.concat((x, crop(skip)), dim=1)
        out = self.conv(x)
        return out


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ls1 = ContractingPath(1, 64)
        self.ls2 = ContractingPath(64, 128)
        self.ls3 = ContractingPath(128, 256)
        self.ls4 = ContractingPath(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )

        self.rs4 = ExpandingPath(1024, 512, 3)
        self.rs3 = ExpandingPath(512, 256)
        self.rs2 = ExpandingPath(256, 128, 3)
        self.rs1 = ExpandingPath(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 2, 2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(2),
        )

    def forward(self, x):
        skip1, x = self.ls1(x)
        skip2, x = self.ls2(x)
        skip3, x = self.ls3(x)
        skip4, x = self.ls4(x)
        x = self.bottleneck(x)
        x = self.rs4(x, skip4)
        x = self.rs3(x, skip3)
        x = self.rs2(x, skip2)
        x = self.rs1(x, skip1)
        out = self.final(x)
        return out
