import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification


class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft")
        self.swinv2 = base_model.swinv2
        self.classifier = base_model.classifier

        for param in self.swinv2.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True


    @classmethod
    def from_pretrained(cls, model_path: str):
        self = cls()
        state_dict = torch.load(model_path, weights_only=True)
        self.classifier.load_state_dict(state_dict)
        for param in self.classifier.parameters():
            param.requires_grad = False
        return self


    def forward(self, x):
        pooler_output = self.swinv2(x)['pooler_output']
        features = self.classifier(pooler_output)
        return features


class SwinTransformerFFN(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1000, 64),
            nn.LayerNorm(64),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        state_dict = torch.load(model_path, weights_only=True)
        model_linear_keys = [(key, value) for key, value in state_dict.items() if key.startswith('linear')]
        self.load_state_dict(dict(model_linear_keys))

    def forward(self, x):
        return self.linear(x).squeeze(1)