import os

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from test import test
from data import HAM10000, train_test_split

from model.vit import ViT

from model.resnet import ResNet50
from utils import init_weight

from tqdm import tqdm
import wandb
import constants as args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def train(model, train_loader, val_loader, device):
    loss_fn = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7)

    best_loss, best_model = 1e8, model
    for epoch in range(args.epochs):
        train_loss, val_loss = 0, 0
        model.train()
        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.epochs}") as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                train_loss += loss.item()
                loss.backward()
                optim.step()
                optim.zero_grad()

        scheduler.step()
        train_loss /= len(train_loader)

        val_loss, accuracy = test(model, val_loader, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

        if args.use_wandb:
            wandb.log(
                {
                    "train loss": train_loss,
                    "val loss": val_loss,
                    "accuracy": accuracy,
                }
            )

    return best_model


def main():
    if args.use_wandb:
        wandb.init(project="ham_classification")

    dataset = HAM10000(
        image_dir=args.image_dir,
        label_path=args.label_path,
        resize=args.resize,
    )

    train_data, val_data = train_test_split(dataset)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    model = (
        ViT(
            image_size=args.image_size,
            patch_size=args.patch_size,
            h_dim=args.h_dim,
            mlp_dim=args.mlp_dim,
            num_classes=args.num_classes,
        )
        if args.model == "vit"
        else ResNet50()
    )

    if args.use_checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model = nn.DataParallel(model)
    model.to(device)
    model.apply(init_weight)

    best_model = train(model, train_loader, val_loader, device)
    torch.save(best_model.module.state_dict(), args.checkpoint)


if __name__ == "__main__":
    main()
