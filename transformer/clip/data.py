import os
import json
from torch.utils.data import Dataset
import clip
from PIL import Image
import torch
import torchvision.transforms as T


class CustomDataset(Dataset):
    def __init__(self, data_path, image_dir, process):
        super().__init__()
        self.process = process
        self.image_dir = image_dir
        self.images = []
        self.captions = []

        with open(data_path, "r") as json_file:
            data = json.load(json_file)
            for name, caption in data.items():
                self.images.append(os.path.join(self.image_dir, name))
                self.captions.append(caption)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        cap = clip.tokenize(self.captions[idx]).squeeze()
        processed_img = self.process(img)
        return processed_img, cap


def train_test_split(dataset):
    t = len(dataset)
    train_size = int(0.9 * t)
    test_size = t - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_set, val_set


class CustomLabelDataset(Dataset):
    def __init__(self, image_dir, label_path, process, original=False):
        super().__init__()
        self.process = process
        self.image_dir = image_dir
        self.images = []
        self.labels = []
        self.original = original
        with open(label_path, "r") as json_file:
            data = json.load(json_file)
            for name, label in data.items():
                self.images.append(os.path.join(self.image_dir, name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        transform = T.ToTensor()
        img = Image.open(self.images[idx])
        processed_img = self.process(img).float()
        label = torch.tensor(self.labels[idx] - 1, dtype=torch.float)
        if self.original:
            return transform(img), processed_img, label
        return processed_img, label
