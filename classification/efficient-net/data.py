import os
from torch.utils.data import Dataset
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
    ):
        self.image_dir = image_dir
        self.labels = dict()

        self.images = os.listdir(self.image_dir)
        df = pd.read_csv(label_path).sort_index(axis=0).to_numpy()

        for key, label in zip(df[:, 0], df[:, 1:]):
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
        label = torch.tensor(label, dtype=torch.float)
        return image, label
