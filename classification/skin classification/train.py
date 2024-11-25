import os
import torch
import torch.optim as optim
from tqdm import tqdm
from distributed import DistributedDataLoader, DistributedParallel, DistributedTrainer
from model import AutoTransformerModel
from data import BinaryMedicalDataset
from utils import plot, evaluate, create_metadata, BinaryFocalLoss
from constants import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='new_train09')
parser.add_argument("--model", type=str, default='swin', choices=['vit-b', 'vit-l', 'swin'])
parser.add_argument("--num_epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--checkpoint_epoch", type=int, default=100)
args = parser.parse_args()

train_data = BinaryMedicalDataset(dir=TRAIN_DIR, model_path=MODEL_PATH[args.model])
val_data = BinaryMedicalDataset(dir=VAL_DIR, model_path=MODEL_PATH[args.model])


def train(rank):
    model = AutoTransformerModel(model_path=MODEL_PATH[args.model])
    loss_fn = BinaryFocalLoss(args.beta)
    val_loss_fn = BinaryFocalLoss(args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        eta_min=1e-8,
        T_mult=1,
        T_0=25,
    )

    model = DistributedParallel(model, rank, name=args.name)
    best_model, best_loss = model.module, 1e4

    train_loader = DistributedDataLoader(train_data, args.batch_size, True, True)
    val_loader = DistributedDataLoader(val_data, args.batch_size)

    for epoch in range(args.num_epoch):
        train_loss = 0
        model.train()
        with tqdm(train_loader, desc=f"Epoch: {epoch+1}/{args.num_epoch}") as pbar:
            for X, y in pbar:
                X, y = X.to(rank), y.to(rank)
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, y)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
        scheduler.step()
        train_loss /= len(train_loader)
        val_loss, accuracy = evaluate(model, val_loader, val_loss_fn, rank)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.module

        if rank == 0 and (epoch + 1) % args.checkpoint_epoch == 0:
            torch.save(
                best_model.state_dict(),
                os.path.join(WEIGHT_PATH, f"{args.name}_checkpoint_{epoch+1}.pt"),
            )
        yield train_loss, val_loss, accuracy

    if rank == 0:
        torch.save(
            best_model.state_dict(),
            os.path.join(WEIGHT_PATH, f"{args.name}.pt"),
        )

if __name__ == "__main__":
    train_losses, val_losses, accuracies = [], [], []
    create_metadata(args, dir=METADATA_PATH, name=args.name)
    trainer = DistributedTrainer(train, port=9990, gather=True, device_ids=list(range(8)))

    for epoch, (train_loss, val_loss, accuracy) in enumerate(trainer):        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        plot(train_losses, val_losses, accuracies, PLOT_PATH, args.name)
            
        
