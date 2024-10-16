import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, skip):
        super().__init__()
        self.skip = skip

        self.layer = nn.Sequential(
            nn.Conv2d(in_features, in_features, 1, 1 if skip else 2),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features, out_features, 3),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
        )

        self.activate = nn.ReLU()

    def forward(self, x):
        if self.skip is True:
            out = x + self.layer(x)
            out = self.activate(out)
        else:
            out = self.layer(x)
        return out


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(3, 1000),
            nn.ReLU(),
            nn.Linear(1000, 7),
            nn.ReLU(),
        )

        cfg = [3, 4, 6, 3]
        in_features = [64, 128, 256, 512]
        out_features = [256, 512, 1024, 2048]

        for i, blocks in enumerate(cfg):
            layer = nn.ModuleList(
                [
                    ResidualBlock(
                        in_features=in_features[i],
                        out_features=out_features[i],
                        skip=False if i > 0 and j == 0 else True,
                    )
                    for j in range(blocks)
                ]
            )
            self.layers.append(nn.Sequential(*layer))

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        out = self.fc(x)
        return out
