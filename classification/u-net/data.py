import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image


def train_test_split(dataset, ratio):
    dataset_size = len(dataset)
    test_size = int(ratio * dataset_size)
    train_size = len(dataset) - test_size
    train, test = random_split(dataset, [train_size, test_size])
    return train, test


class HAM10000(Dataset):
    def __init__(self, image_dir, mask_dir, train=True, ratio=0.8, resize=(450, 450)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.to_tensor = T.ToTensor()
        self.transform = T.Compose([T.Grayscale(), T.Resize(resize), T.ToTensor()])

        image_list = sorted(os.listdir(image_dir))
        mask_list = sorted(os.listdir(mask_dir))
        t = int(ratio * len(image_list))

        self.image_list = image_list[:t] if train else image_list[t:]
        self.mask_list = mask_list[:t] if train else mask_list[t:]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image_path = os.path.join(self.image_dir, self.image_list[i])
        mask_path = os.path.join(self.mask_dir, self.mask_list[i])

        image_data = Image.open(image_path)
        mask_data = Image.open(mask_path)

        image = self.transform(image_data) / 255
        mask = self.to_tensor(mask_data).squeeze().type(torch.int64)
        mask = F.one_hot(mask, num_classes=2)
        mask = mask.permute(2, 0, 1).type(torch.float32)

        image_data.close()
        mask_data.close()

        return image, mask
