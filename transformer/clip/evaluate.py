import os
from model import ImageCLIP, TextCLIP, CosineSimilarity
import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import CustomLabelDataset
from tqdm import tqdm


def tokenize_class_label(texts):
    tokens = [clip.tokenize(f"A Photo of a {text}.") for text in texts]
    tokens = torch.stack(tokens).squeeze()
    return tokens


image_dir = "../../data/coco/train2014"
label_path = "data/labels.json"
checkpoint_path = "checkpoint/checkpoint_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with open("data/class_labels.txt", "r") as fp:
    texts = [label.strip() for label in fp.readlines()]
    class_label = dict(enumerate(texts))

model, process = clip.load("ViT-B/32", device=device, jit=False)
checkpoint = torch.load(checkpoint_path, map_location=device)

dataset = CustomLabelDataset(image_dir, label_path, process=process)
test_loader = DataLoader(
    dataset, batch_size=32, num_workers=4, shuffle=True, drop_last=True
)

image_model = ImageCLIP(model)
text_model = TextCLIP(model)

image_model.load_state_dict(checkpoint["image_model_state_dict"])
text_model.load_state_dict(checkpoint["text_model_state_dict"])

sim = CosineSimilarity()
text_tokens = tokenize_class_label(texts).to(device)

with torch.no_grad():
    accr = 0
    text_embedding = text_model(text_tokens)
    for X, y in tqdm(test_loader):
        X, y = X.to(device), y.to(device)
        image_embedding = image_model(X)

        similarity = sim(image_embedding, text_embedding)
        accr += (similarity.argmax(1) == y).float().mean().item()

    accr /= len(test_loader)
    print(f"Accruacy: {accr :.4f}")
