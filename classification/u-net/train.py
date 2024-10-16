import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import HAM10000, train_test_split
from model import UNet
from utils import iou_metric, normalize

from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_workers", default=32, type=int)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--lr", default=1e-4, type=float)

parser.add_argument("--interpolate", default="bilinear", type=str)

parser.add_argument("--image_dir", default="dataset/imgs", type=str)
parser.add_argument("--mask_dir", default="dataset/masks", type=str)

parser.add_argument("--gpu_num", default="0,1,2,3,4,5,6,7", type=str)
parser.add_argument("--use_wandb", default=True, type=bool)


def init_weight(model):
    cname = model.__class__.__name__
    if cname.find("Linear") != -1:
        nn.init.kaiming_normal_(model.weight)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Count of using GPUs:", torch.cuda.device_count())

    if args.use_wandb:
        wandb.init(project="ham_segmentation")

    train = HAM10000(image_dir=args.image_dir, mask_dir=args.mask_dir)
    train, val = train_test_split(dataset=train, ratio=0.1)

    train_loader = DataLoader(
        train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True
    )

    model = UNet().to(device)
    model.apply(init_weight)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    CHECKPOINT_PATH = "model/best_model.pt"
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

    model = nn.DataParallel(model).to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, 1e-8)

    best_model = model
    best_loss = 1e8

    for epoch in range(args.epochs):
        train_loss, train_iou = 0, 0
        val_loss, val_iou = 0, 0
        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.epochs}") as pbar:
            for X, y in pbar:
                X, y = normalize(X.to(device)), y.to(device)
                pred = model(X)
                pred = F.interpolate(pred, size=(450, 600), mode=args.interpolate)
                loss = loss_fn(pred, y)
                train_loss += loss.item()
                train_iou += iou_metric(pred, y)
                loss.backward()
                optim.step()
                optim.zero_grad()

        scheduler.step()
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for X, y in pbar:
                    X, y = normalize(X.to(device)), y.to(device)
                    pred = model(X)
                    pred = F.interpolate(pred, size=(450, 600), mode=args.interpolate)
                    val_loss += loss_fn(pred, y).item()
                    val_iou += iou_metric(pred, y)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

        if args.use_wandb:
            wandb.log(
                {
                    "train loss": train_loss,
                    "train iou": train_iou,
                    "val loss": val_loss,
                    "val iou": val_iou,
                }
            )

        if (epoch + 1) % 10 == 0:
            torch.save(
                best_model.module.state_dict(),
                f"model/best_model.pt",
            )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
