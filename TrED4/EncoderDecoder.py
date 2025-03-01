import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False) # Revoir diff entre BatchNorm et LayerNorm
        return self.gamma * ((x - mean)/(std + 1e-8)) + self.beta

class Attention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.dropout = nn.Dropout(0.2)

    def forward(self, q, k, v, attn_mask=None, pad_mask=None):
        scores = q @ k.transpose(-2, -1)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0, float('-inf'))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores/sqrt(self.head_size), dim=-1)
        return self.dropout(attn_weights) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        assert embedding_size % num_heads == 0

        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.head_size = embedding_size // num_heads

        self.q_proj = nn.Linear(embedding_size, embedding_size)
        self.k_proj = nn.Linear(embedding_size, embedding_size)
        self.v_proj = nn.Linear(embedding_size, embedding_size)

        self.attention = Attention(self.head_size)
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = LayerNorm(embedding_size)

    def forward(self, q, k, v, attn_mask=None, pad_mask=None):
        x = q.clone().detach()
        seq_length = q.shape[1]

        q = self.q_proj(q).contiguous().view(-1, self.num_heads, seq_length, self.head_size)
        k = self.k_proj(k).contiguous().view(-1, self.num_heads, seq_length, self.head_size)
        v = self.v_proj(v).contiguous().view(-1, self.num_heads, seq_length, self.head_size)

        output = self.attention(q, k, v, attn_mask=attn_mask, pad_mask=pad_mask).reshape(-1, seq_length, self.embedding_size)
        return self.layer_norm(x + self.projection(self.dropout(output)))

class FeedForward(nn.Module):
    def __init__(self, embedding_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, output_size)
        self.linear2 = nn.Linear(output_size, embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = LayerNorm(embedding_size)

    def forward(self, x):
        return self.layer_norm(x + self.dropout(self.linear2(self.relu(self.linear1(x)))))

class PositionalEncoding(nn.Module):
    # TODO Optimize memory
    def __init__(self):
        super().__init__()

    def forward(self, x):
        seq_length, d_model = x.shape[-2:]
        pe = torch.zeros_like(x)
        numerator = torch.arange(seq_length).unsqueeze(-1)
        denominator = torch.pow(10000, torch.arange(0, d_model, 2)/d_model)
        pe[:, :, ::2] = torch.sin(numerator/denominator)
        pe[:, :, 1::2] = torch.cos(numerator/denominator)
        return x + pe

class Encoder(nn.Module):
    def __init__(self, vocab_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=512)
        self.positional_encoding = PositionalEncoding()
        self.layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(embedding_size=512, num_heads=8),
                FeedForward(embedding_size=512, output_size=2048)
            ) for _ in range(num_layers)
        ])

    def forward(self, input, attn_mask=None, pad_mask=None):
        input = self.embedding(input)
        input = self.positional_encoding(input)
        for layer in self.layers:
            mha, ffn = layer
            input = mha(input, input, input, attn_mask=attn_mask, pad_mask=pad_mask)
            input = ffn(input)
        return input

class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=512)
        self.positional_encoding = PositionalEncoding()
        self.linear = nn.Linear(512, vocab_size)
        self.layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(embedding_size=512, num_heads=8),
                MultiHeadAttention(embedding_size=512, num_heads=8),
                FeedForward(embedding_size=512, output_size=2048)
            ) for _ in range(num_layers)
        ])

    def forward(self, input, encoder_output, attn_mask=None, enc_pad_mask=None, dec_pad_mask=None):
        input = self.embedding(input)
        input = self.positional_encoding(input)
        for layer in self.layers:
            mha1, mha2, ffn = layer
            input = mha1(input, input, input, attn_mask=attn_mask, pad_mask=dec_pad_mask)
            input = mha2(input, encoder_output, encoder_output, pad_mask=enc_pad_mask)
            input = ffn(input)
        input = self.linear(input)
        return input

class EncoderDecoder(nn.Module):
    pass