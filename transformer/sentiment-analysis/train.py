import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import SentimentDataset
from model import TransformerEncoder
from parallel import DistributedDataLoader, DistributedParallel, DistributedTrainer
import wandb


def learning_rate(t):
    return (h_dim ** (-0.5)) * min(t ** (-0.5), t * (4000 ** (-1.5)))


def schedule(t, optim):
    for g in optim.param_groups:
        g["lr"] = learning_rate(t)


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

epoch = 1
num_sequences = 12
h_dim = 768
batch_size = 1024

checkpoint_path = "./checkpoint_model.pt"
# checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

# wandb.init(project="sentiment-analysis")


def train(rank):
    train_data = SentimentDataset("data/amazon_train.csv", num_sequences)
    train_loader = DistributedDataLoader(
        train_data, batch_size, shuffle=True, num_workers=16
    )

    val_data = SentimentDataset("data/amazon_val.csv", num_sequences)
    val_loader = DistributedDataLoader(
        val_data, batch_size, drop_last=True, num_workers=16
    )

    model = TransformerEncoder(h_dim=h_dim, num_sequences=num_sequences)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # model.load_state_dict(checkpoint)
    model = DistributedParallel(model, rank, False)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, eps=1e-9, betas=(0.9, 0.98))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_model, best_loss = model, 1e9
    for t in range(1, 1 + epoch):
        model.train()
        train_loss, val_loss, accuracy = 0, 0, 0
        for tokens, masks, label in tqdm(
            train_loader,
            desc=f"Epoch: {t}/{epoch}",
            disable=rank != 0,
        ):
            tokens, masks, label = tokens.to(rank), masks.to(rank), label.to(rank)
            optimizer.zero_grad()
            pred = model(tokens, masks)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        schedule(t, optimizer)
        model.eval()
        with torch.no_grad():
            for tokens, masks, label in tqdm(
                val_loader,
                desc="Validation",
                disable=rank != 0,
            ):
                tokens, masks, label = (
                    tokens.to(rank),
                    masks.to(rank),
                    label.to(rank),
                )
                pred = model(tokens, masks)
                val_loss += loss_fn(pred, label).item()
                accuracy += (
                    (pred.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
                )

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            accuracy /= len(val_loader.dataset)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model

            # if epoch % 5 == 0:
            #     torch.save(best_model.module.state_dict(), checkpoint_path)

            print(f"train loss: {train_loss:.4f}")
            print(f"val loss: {val_loss:.4f}")
            print(f"accuracy: {accuracy * 100:.2f}")
            # wandb.log(
            #     {
            #         "train loss": round(train_loss, 4),
            #         "val loss": round(val_loss, 4),
            #         "accuracy": round(accuracy * 100, 2),
            #     }
            # )


if __name__ == "__main__":
    trainer = DistributedTrainer(train)
    trainer()
