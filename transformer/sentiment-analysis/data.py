import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    clean_up_tokenization_spaces=True,
)


class SentimentDataset(Dataset):
    def __init__(self, path, max_token_length):
        self.max_token_length = max_token_length
        fp = open(path)
        lines = fp.readlines()[:10000]
        self.items = [tuple(x.strip().split(",")) for x in lines]
        fp.close()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        X, y = self.items[idx]
        label = torch.zeros(2, dtype=torch.float)
        label[0 if y == "positive" else 1] = 1
        tokens = tokenizer.tokenize(X)
        tokens = tokenizer.encode(tokens[:510])
        pad_tokens = torch.tensor(
            tokens + [0] * (self.max_token_length - len(tokens)), dtype=torch.long
        )
        mask_tokens = torch.tensor(
            [1] * len(tokens) + [0] * (self.max_token_length - len(tokens)),
            dtype=torch.float,
        )

        return (
            pad_tokens[: self.max_token_length],
            mask_tokens[: self.max_token_length],
            label,
        )


def tokenize(text, max_token_length):
    tokens = tokenizer.tokenize(text)
    tokens = tokenizer.encode(tokens[:510])
    pad_tokens = torch.tensor(
        tokens + [0] * (max_token_length - len(tokens)), dtype=torch.long
    )
    mask_tokens = torch.tensor(
        [1] * len(tokens) + [0] * (max_token_length - len(tokens)),
        dtype=torch.float,
    )

    return (
        pad_tokens[:max_token_length],
        mask_tokens[:max_token_length],
    )
