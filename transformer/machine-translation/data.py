import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import evaluate
import random

tokenizer = AutoTokenizer.from_pretrained(
    "/home/sangjun/model/llama-3.2-11b-vision/",
    clean_up_tokenization_spaces=True,
)

class TranslationDataset(Dataset):
    def __init__(self, dir, max_length):
        self.max_length = max_length
        self.items = self.load_data(dir)

    @staticmethod
    def load_data(dir):
        items = []
        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename), "r") as fp:
                items.extend(tuple(line.strip().split(",")) for line in fp.readlines())
        return items
    
    
    def tokenize(self, text, add_special_tokens=False):
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        return tokenized["input_ids"].squeeze()


    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        ko, en = self.items[idx]
        src_token = self.tokenize(en, True)
        tgt_token = self.tokenize(ko, True)
        tgt_out_token = self.tokenize(f"{ko}{tokenizer.eos_token}")
        return src_token, tgt_token, tgt_out_token, ko


def batch_decode(tokens):
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)


def metrics(labels, logits):
    metric = evaluate.load("google_bleu")
    preds = [[pred] for pred in batch_decode(logits)]
    bleu_score = metric.compute(predictions=list(labels), references=preds)
    return bleu_score["google_bleu"]
