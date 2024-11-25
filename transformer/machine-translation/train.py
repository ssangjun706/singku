import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_

from data import metrics
from data import TranslationDataset
from model import Transformer
from constants import args

from distributed import DistributedParallel, DistributedDataLoader, DistributedTrainer
from tqdm import tqdm
import wandb

def evaluate(model, data_loader, rank):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=args.ignore_index)
    model.eval()
    with torch.no_grad():
        loss, bleu = 0, 0
        for src_token, _, tgt_out_token, target in tqdm(data_loader, desc="Evaluate"):
            batch_size, num_sequences = src_token.size(0), src_token.size(1)
            tgt_token = torch.ones(batch_size, 1).long().fill_(101)
            src_token, tgt_token, tgt_out_token = (
                src_token.to(rank),
                tgt_token.to(rank),
                tgt_out_token.to(rank),
            )
            preds = None
            memory = model.module.encode(src_token)
            for _ in range(num_sequences):
                logits = model.module.decode(tgt_token, memory)[:, -1, :]
                logits = logits.unsqueeze(1)
                next_token = logits.softmax(-1).argmax(-1)
                tgt_token = torch.cat([tgt_token, next_token], -1)
                if preds is None:
                    preds = logits
                else:
                    preds = torch.cat([preds, logits], 1)
            loss += loss_fn(
                preds.reshape(-1, preds.size(-1)), tgt_out_token.reshape(-1)
            ).item()
            bleu += metrics(target, tgt_token.detach().cpu())

    loss /= len(data_loader)
    bleu /= len(data_loader)
    return loss, bleu



def train(rank):
    train_data = TranslationDataset("data/train", args.seq_len)
    val_data = TranslationDataset("data/val", args.seq_len)

    train_loader = DistributedDataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DistributedDataLoader(
        val_data,
        args.batch_size,
        drop_last=True,
        num_workers=4,
    )

    model = Transformer(args.h_dim, args.seq_len, args.vocab_size, args.ignore_index)

    model = DistributedParallel(model, rank)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.98),
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        eta_min=1e-8,
        T_0=args.warmup,
        T_mult=2,
    )

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=0.1,
        ignore_index=args.ignore_index,
    )

    for epoch in range(args.epoch):
        model.train()
        train_loss, val_loss, bleu = 0, 0, 0
        with tqdm(train_loader, desc=f"Epoch: {epoch+1}/{args.epoch}") as pbar:
            for src_token, tgt_token, tgt_out_token, _ in pbar:
                src_token = src_token.to(rank)
                tgt_token = tgt_token.to(rank)
                tgt_out_token = tgt_out_token.to(rank).reshape(-1)
                optimizer.zero_grad()
                logits = model(src_token, tgt_token)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out_token)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        if (epoch + 1) % 5 == 0:
            val_loss, bleu = evaluate(model, val_loader, rank)

        yield train_loss, val_loss, bleu, optimizer.param_groups[0]["lr"]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    wandb.init(project="machine-translation")
    trainer = DistributedTrainer(
        train,
        gather=True,
    )
    for x, y, z, w in trainer:
        # print(x, y, z, w)
        wandb.log(
            {
                "train loss": round(x, 4),
                "val loss": round(y, 4),
                "bleu": round(z, 3),
                "lr": w,
            }
        )
