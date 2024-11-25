import os
import torch

from PIL import Image
from torch.utils.data import Dataset, random_split
from transformers import AutoImageProcessor

# class 0
# psoriasis/eczema
# vasculitis

# class 1
# pigmented purpuric dermatosis (ppd)
# stasis dermatitis

class BinaryMedicalDataset(Dataset):
    def __init__(
        self,
        dir: str,
        model_path: str,
    ):
        self.cache = {}
        self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        self.data, self.label = [], []
        for dir_name in os.listdir(dir):
            label = 0 if dir_name == "class0" else 1
            path = os.path.join(dir, dir_name)
            for filename in os.listdir(path):
                self.data.append(os.path.join(path, filename))
                self.label.append(label)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx not in self.cache:
            image = Image.open(self.data[idx])
            self.cache[idx] = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
            image.close()
        label = torch.tensor(self.label[idx], dtype=torch.float)
        return self.cache[idx], label


# def train_test_split(dataset: Dataset, ratio: float):
#     train_size = int(len(dataset) * ratio)
#     test_size = len(dataset) - train_size
#     train_data, test_data = random_split(
#         dataset, 
#         lengths=[train_size, test_size], 
#         generator=torch.Generator().manual_seed(42)
#     )
#     return train_data, test_data