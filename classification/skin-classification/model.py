import torch.nn as nn
from transformers import AutoModelForImageClassification


class AutoTransformerModel(nn.Module):
    def __init__(self, model_path: str, dropout: float = 0.2):
        super().__init__()
        h_dim = 1000
        self.model = AutoModelForImageClassification.from_pretrained(model_path)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True


        self.linear = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        for layer in self.linear.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        features = self.model(x)["logits"]
        return self.linear(features).squeeze(1)