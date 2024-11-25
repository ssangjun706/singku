import os
import torch

import clip

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from swinv2_large import SwinTransformer
from transformers import AutoImageProcessor
import pandas as pd

LABEL_FILES = {
    "skin": "./data/skin_classes.txt",
    "ham10000": "./data/ham10000.txt"
}


class SkinDataset(Dataset):
    def __init__(self, dir, processor):
        self.cache = {}
        self.processor = processor
        self.data, self.targets = [], []
        for dir_name in os.listdir(dir):
            label = 0 if dir_name == "class0" else 1
            path = os.path.join(dir, dir_name)
            for filename in os.listdir(path):
                self.data.append(os.path.join(path, filename))
                self.targets.append(label)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx not in self.cache:
            image = Image.open(self.data[idx])
            if self.processor is None:
                pass
            elif isinstance(self.processor, Compose):
                processed_image = self.processor(image)
            else:
                processed_image = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

            self.cache[idx] = processed_image
            image.close()
        label = torch.tensor(self.targets[idx], dtype=torch.float)
        return self.cache[idx], label


class HAM10000_Dataset(Dataset):
    
    def __init__(self, dir, processor):
        self.cache = {}
        self.processor = processor
        self.data, self.targets = [], []
        
        self.class_mapping = {
            "MEL": 0,  # Melanoma
            "NV": 1,   # Melanocytic Nevi
            "BCC": 2,  # Basal Cell Carcinoma
            "AKIEC": 3,  # Actinic Keratoses and Intraepithelial Carcinoma
            "BKL": 4,  # Benign Keratosis-like Lesions
            "DF": 5,   # Dermatofibroma
            "VASC": 6  # Vascular Lesions
        }

        # 데이터와 레이블 읽기
        for dir_name in os.listdir(dir):
            if dir_name not in self.class_mapping:
                print(f"'{dir_name}'는 매핑되지 않은 클래스입니다. 무시합니다.")
                continue
            
            label = self.class_mapping[dir_name]
            path = os.path.join(dir, dir_name)
            
            for filename in os.listdir(path):
                self.data.append(os.path.join(path, filename))
                self.targets.append(label)
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx not in self.cache:
            image = Image.open(self.data[idx])
            if self.processor is None:
                pass
            elif isinstance(self.processor, Compose):
                processed_image = self.processor(image)
            else:
                processed_image = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

            self.cache[idx] = processed_image
            image.close()
        label = torch.tensor(self.targets[idx], dtype=torch.float)
        return self.cache[idx], label

def get_data(dir, processor = None):
    return HAM10000_Dataset(dir, processor)



def get_targets_only(dir):
    pil_data = get_data(dir)
    return pil_data.targets


def get_target_model(target_name, device):
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    elif target_name == 'swinv2-large': 
        target_model = SwinTransformer.from_pretrained("./pretrained/swinv2-large-classifier.pt").to(device)
        preprocess = AutoImageProcessor.from_pretrained(
            "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft", 
            use_fast=True
        )
        target_model.eval()
    else:
        assert False, "Unknown model name" 
    
    return target_model, preprocess