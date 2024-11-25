import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, h_dim, seq_len):
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
        self.encoding.requires_grad_ = False

    def forward(self, x):
        encoding = self.encoding[: x.size(1)].unsqueeze(0)
        encoding = encoding.repeat_interleave(x.size(0), 0)
        return x + encoding.to(x.device)


class Embedding(nn.Module):
    def __init__(self, h_dim, vocab_size, seq_len, padding_idx):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, h_dim, padding_idx)
        self.pos_embed = PositionalEmbedding(h_dim, seq_len)
        self.norm = nn.LayerNorm(h_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, tokens):
        embed = self.token_embed(tokens)
        embed = self.pos_embed(embed)
        return self.dropout(self.norm(embed))
        # positions = torch.arange(
        #     seq_len,
        #     dtype=torch.long,
        #     device=tokens.device,
        # ).unsqueeze(0)
        # positions = positions.masked_fill(tokens == 0, 0)
        # position_embed = self.position_embedding(positions)
        # return self.dropout(self.norm(embed))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, h_dim, masked=False, dropout=0.1):
        super().__init__()
        self.masked = masked
        self.h_dim = h_dim
        self.wQ = nn.Linear(h_dim, h_dim)
        self.wK = nn.Linear(h_dim, h_dim)
        self.wV = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v):
        q = self.wQ(q)
        k = self.wK(k)
        v = self.wV(v)
        score = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.h_dim)
        if self.masked:
            seq_len = score.size(-1)
            mask = (
                torch.tril(torch.ones(seq_len, seq_len))
                .reshape(1, 1, seq_len, seq_len)
                .to(score.device)
            )
            score = score.masked_fill(mask == 0, float("-inf"))
        attention = self.softmax(score)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, h_dim, masked=False):
        super().__init__()
        assert h_dim % n_heads == 0
        self.masked = masked
        self.dim_per_head = h_dim // n_heads
        self.h_dim = h_dim
        self.attention = ScaledDotProductAttention(self.dim_per_head, self.masked)
        self.ffn = nn.Linear(h_dim, h_dim)

    def forward(self, query, key, value):
        batch_size, q_num_seq, _ = query.shape
        _, k_num_seq, _ = key.shape
        _, v_num_seq, _ = value.shape
        q = query.reshape(batch_size, -1, q_num_seq, self.dim_per_head)
        k = key.reshape(batch_size, -1, k_num_seq, self.dim_per_head)
        v = value.reshape(batch_size, -1, v_num_seq, self.dim_per_head)
        out = self.attention(q, k, v).reshape(batch_size, q_num_seq, -1)
        return self.ffn(out)
    

def init_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

class Transformer(nn.Module):
    def __init__(
        self,
        h_dim: int,
        seq_len: int,
        vocab_size: int,
        padding_idx: int,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
    ):
        super().__init__()
        self.embedding = Embedding(h_dim, vocab_size, seq_len, padding_idx)

        self.encoder = TransformerEncoder(
            h_dim=h_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )
        self.decoder = TransformerDecoder(
            h_dim=h_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )

        self.generator = nn.Sequential(
            nn.Linear(h_dim, vocab_size),
            nn.Tanh(),
        )
    
        self.apply(init_weights)


    def forward(self, src_token, tgt_token):
        encoder_output = self.encode(src_token)
        out = self.decode(tgt_token, encoder_output)
        return out

    def encode(self, token):
        query = self.embedding(token)
        return self.encoder(query)

    def decode(self, token, key):
        query = self.embedding(token)
        states = self.decoder(query, key)
        return self.generator(states)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mhattn = MultiHeadAttention(
            h_dim=h_dim,
            n_heads=n_heads,
        )
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(h_dim, eps=1e-12)
        self.norm2 = nn.LayerNorm(h_dim, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, h_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        for _ in range(self.num_layers):
            n1 = self.norm1(x)
            x = x + self.dropout(self.mhattn(n1, n1, n1))
            x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mmhattn = MultiHeadAttention(
            h_dim=h_dim,
            n_heads=n_heads,
            masked=True,
        )
        self.mhattn = MultiHeadAttention(
            h_dim=h_dim,
            n_heads=n_heads,
        )
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(h_dim, eps=1e-12)
        self.norm2 = nn.LayerNorm(h_dim, eps=1e-12)
        self.norm3 = nn.LayerNorm(h_dim, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, h_dim),
            nn.Dropout(0.1),
        )

    def forward(self, query, key):
        for _ in range(self.num_layers):
            q1 = self.norm1(query)
            query = query + self.dropout(self.mmhattn(q1, q1, q1))
            query = query + self.dropout(self.mhattn(self.norm2(query), key, key))
            query = query + self.ffn(self.norm3(query))
        return query
