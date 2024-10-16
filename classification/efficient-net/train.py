import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model import EfficientNet
from data import HAM10000
import constants as args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def validate(model, data_loader, device):
    loss, accuracy = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader, desc="Validation") as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss += loss_fn(pred, y).item()
                accuracy += (pred.argmax(1) == y.argmax(1)).float().mean().item()

    loss /= len(data_loader)
    accuracy /= len(data_loader)
    return loss, accuracy


def train(model, train_loader, val_loader, device):
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), weight_decay=0.25, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7)
    base_epoch = 0

    # if args.use_checkpoint and os.path.exists(args.checkpoint_path):
    #     checkpoint = torch.load(args.checkpoint_path, map_location=device)
    #     base_epoch = checkpoint["epoch"]
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     optim.load_state_dict(checkpoint["optim_state_dict"])
    #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    #     best_loss = checkpoint["loss"]
    #     best_model = model

    model = nn.DataParallel(model).to(device)

    best_model, best_loss = model, 1e4
    num_epochs = base_epoch + args.epochs

    for epoch in range(base_epoch, num_epochs):
        train_loss = 0
        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{num_epochs}") as pbar:
            model.train()
            for X, y in pbar:
                optim.zero_grad()
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                train_loss += loss.item()
                loss.backward()
                optim.step()
            scheduler.step()

        train_loss /= len(train_loader)
        val_loss, accuracy = validate(model, val_loader, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train loss": train_loss,
                    "val loss": val_loss,
                    "accuracy": accuracy,
                    "lr": scheduler.get_last_lr()[-1],
                }
            )

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "loss": best_loss,
                    "model_state_dict": model.module.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                args.checkpoint_path,
            )

    torch.save(
        best_model.module.state_dict(),
        args.model_path,
    )


def main():
    if args.use_wandb:
        wandb.init(project="efficient_net")

    train_dataset = HAM10000(
        image_dir=args.train_dir,
        label_path=args.label_path,
        resize=args.resize,
    )

    val_dataset = HAM10000(
        image_dir=args.val_dir,
        label_path=args.label_path,
        resize=args.resize,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        drop_last=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = EfficientNet(args.num_classes)
    train(model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
