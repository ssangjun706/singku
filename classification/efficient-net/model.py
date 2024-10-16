import torch.nn as nn
from torchvision.models import efficientnet_b4


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = efficientnet_b4()
        self.linear = nn.Sequential(
            nn.LayerNorm(1000),
            nn.ReLU6(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        conv = self.conv(x)
        return self.linear(conv)
