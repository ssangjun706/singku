import torch
import torch.nn as nn
import math
from transformers import BertModel


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, h_dim):
        super().__init__()

        self.seq_len = seq_len
        self.h_dim = h_dim

        assert self.seq_len % 2 == 0

        _seq = torch.arange(self.seq_len).reshape((-1, 1))
        _seq = _seq.expand((self.seq_len, self.h_dim))

        _freq = 2 * torch.floor(torch.arange(self.seq_len) * 0.5) / self.h_dim
        _freq = torch.pow(10000, (_freq)).reshape((-1, 1))
        _freq = _freq.expand((self.seq_len, self.h_dim))

        _pe = _seq / _freq
        _pe[:, ::2] = torch.sin(_pe[:, ::2])
        _pe[:, 1::2] = torch.cos(_pe[:, 1::2])
        self.encoding = _pe
        self.encoding.requires_grad = False

    def forward(self, x):
        encoding = self.encoding.unsqueeze(0).repeat_interleave(x.shape[0], dim=0)
        encoding = encoding.to(x.device)
        return x + encoding


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        h_dim,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.wQ = nn.Linear(h_dim, h_dim)
        self.wK = nn.Linear(h_dim, h_dim)
        self.wV = nn.Linear(h_dim, h_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v):
        q = self.wQ(q)
        k = self.wK(k)
        v = self.wV(v)
        score = torch.matmul(q, k.permute(0, 1, 3, 2))
        out = torch.matmul(self.softmax(score / math.sqrt(self.h_dim)), v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        h_dim,
    ):
        super().__init__()
        assert h_dim % n_heads == 0

        self.dim_per_head = h_dim // n_heads
        self.h_dim = h_dim
        self.attention = ScaledDotProductAttention(self.dim_per_head)
        self.ffn = nn.Linear(h_dim, h_dim)

    def forward(self, q, k, v):
        batch_size, num_sequences, _ = q.shape
        q = q.reshape(batch_size, -1, num_sequences, self.dim_per_head)
        k = k.reshape(batch_size, -1, num_sequences, self.dim_per_head)
        v = v.reshape(batch_size, -1, num_sequences, self.dim_per_head)
        out = self.attention(q, k, v).reshape(batch_size, num_sequences, -1)
        return self.ffn(out)


class WordEmbedding(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            model = BertModel.from_pretrained(
                "bert-base-uncased", torch_dtype=torch.float, attn_implementation="sdpa"
            )
            self.embedding = model.embeddings
            for params in self.embedding.parameters():
                params.requires_grad_ = False
        else:
            self.embedding = nn.Embedding(30522, 768)

    def forward(self, x):
        return self.embedding(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        num_sequences: int,
        n_heads=8,
        num_layers=6,
        dim_feedforward=2048,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Sequential(
            WordEmbedding(),
            PositionalEmbedding(num_sequences, h_dim),
            nn.Dropout(0.1),
        )
        self.mhattn = MultiHeadAttention(
            h_dim=h_dim,
            n_heads=n_heads,
        )
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, h_dim),
            nn.Dropout(0.1),
        )
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(num_sequences * h_dim),
            nn.ReLU6(),
            nn.Linear(num_sequences * h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU6(),
            nn.Linear(h_dim, 2),
        )

    def forward(self, tokens, masks):
        x = self.embedding(tokens) * masks.unsqueeze(-1)
        for _ in range(self.num_layers):
            mhattn = self.mhattn(x, x, x)
            x = self.norm(x + self.dropout(mhattn))
            x = self.norm(x + self.ffn(x))
        out = self.final_layer(x)
        return out
