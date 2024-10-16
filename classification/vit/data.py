import os
from torch.utils.data import Dataset, random_split
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T


class HAM10000(Dataset):
    def __init__(
        self,
        image_dir,
        label_path,
        resize,
        train=True,
        ratio=0.9,
    ):
        self.image_dir = image_dir
        self.labels = dict()

        _images = sorted(os.listdir(self.image_dir))
        _df = pd.read_csv(label_path).sort_index(axis=0).to_numpy()
        _t = int(len(_images) * ratio)

        if train is True:
            _df = _df[:_t, :]
            self.images = _images[:_t]
        else:
            _df = _df[_t:, :]
            self.images = _images[_t:]

        for key, label in zip(_df[:, 0], _df[:, 1:]):
            self.labels[key] = label.astype(float)

        self.transform = T.Compose(
            [
                T.Resize((resize, resize)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = self.transform(Image.open(img_path))
        label = self.labels[self.images[idx].split(".")[0]]
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


def train_test_split(dataset, ratio=0.1):
    t = len(dataset)
    len_test = int(t * ratio)
    len_train = t - len_test
    train, test = random_split(dataset, [len_train, len_test])
    return train, test
