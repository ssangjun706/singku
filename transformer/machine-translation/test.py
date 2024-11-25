import torch
from tqdm import tqdm
from data import metrics
import torch.nn as nn
from constants import args


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
            bleu += metrics(target, tgt_token.detach())

    loss /= len(data_loader)
    bleu /= len(data_loader)
    return loss, bleu
