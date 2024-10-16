import torch.nn as nn
import torch


class CLIP(nn.Module):
    def __init__(self, model):
        super(CLIP, self).__init__()
        self.model = model

    def forward(self, image, text):
        text_model = self.model.encode_text
        image_model = self.model.encode_image
        return image_model(image), text_model(text)


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_norm = torch.norm(x)
        y_norm = torch.norm(y)
        eps = 1e-8
        parent = max(x_norm * y_norm, eps)
        z = torch.matmul(x, y.T)
        return z / parent
