import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from model import CLIP, convert_models_to_fp32
import clip
from data import CustomDataset, train_test_split

import wandb


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    return logits_per_x1, logits_per_x2


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 4
lr = 5e-5

annotation_path = "data/annotations.json"
image_dir = "../../data/coco/images"


# checkpoint_path = "checkpoint/checkpoint_model.pt"

# checkpoint = torch.load(checkpoint_path, map_location=device)
clip_model, preprocess = clip.load("ViT-B/16", device=device)
model = CLIP(clip_model)
model = DataParallel(model)

data = CustomDataset(annotation_path, image_dir, process=preprocess)
train_data, val_data = train_test_split(data)
train_loader = DataLoader(train_data, num_workers=4, batch_size=1024)
val_loader = DataLoader(val_data, drop_last=True, num_workers=4, batch_size=512)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optim = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

# wandb.init(project="clip_finetuning")

for epoch in range(1, 1 + num_epochs):
    with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}") as pbar:
        model.train()
        train_loss = 0
        for images, texts in pbar:
            optim.zero_grad()
            images, texts = images.to(device), texts.to(device)

            image_embedding, text_embedding = model(images, texts)
            logit_scale = clip_model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(
                image_embedding, text_embedding, logit_scale
            )

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2
            loss.backward()
            train_loss += loss.item()
            convert_models_to_fp32(model)
            optim.step()
            clip.model.convert_weights(model)
        breakpoint()
        print(f"Training Loss: { train_loss / len(train_loader) :.4f}")

    with tqdm(val_loader, desc=f"Validation") as pbar:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, texts in pbar:
                images, texts = images.to(device), texts.to(device)
                image_embedding, text_embedding = model(images, texts)
                logit_scale = clip_model.logit_scale.exp()
                logits_per_image, logits_per_text = create_logits(
                    image_embedding, text_embedding, logit_scale
                )

                ground_truth = torch.arange(
                    len(images), dtype=torch.long, device=device
                )
                val_loss += (
                    0.5
                    * (
                        loss_img(logits_per_image, ground_truth)
                        + loss_txt(logits_per_text, ground_truth)
                    )
                ).item()

            print(f"Validation Loss: { val_loss / len(val_loader) :.4f}")

    # wandb.log(
    #     {
    #         "train loss": total_loss / len(train_loader),
    #         "lr": optim.param_groups[-1]["lr"],
    #     }
    # )

    # if epoch % 4 == 0:
    #     torch.save(
    #         {
    #             "epoch": epoch,
    #             "model_state_dict": model.module.state_dict(),
    #             "optim_state_dict": optim.state_dict(),
    #             "scheduler_state_dict": scheduler.state_dict(),
    #         },
    #         checkpoint_path,
    #     )
