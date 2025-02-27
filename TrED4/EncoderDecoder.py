import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.2)

    def forward(self, q, k, v, mask=None):
        scores = q @ k.transpose(-2, -1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores/sqrt(self.hidden_size), dim=-1)
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

        self.attention = Attention(self.head_size, embedding_size)
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, q, k, v, mask=None):
        seq_length = q.shape[1]

        q = self.q_proj(q).contiguous().view(-1, self.num_heads, seq_length, self.head_size)
        k = self.k_proj(k).contiguous().view(-1, self.num_heads, seq_length, self.head_size)
        v = self.v_proj(v).contiguous().view(-1, self.num_heads, seq_length, self.head_size)

        output = self.attention(q, k, v, mask=mask).reshape(-1, seq_length, self.embedding_size)
        return self.projection(self.dropout(output))

class FeedForward(nn.Module):
    def __init__(self, embedding_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, output_size)
        self.linear2 = nn.Linear(output_size, embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.linear2(self.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        _, seq_length, d_model = x.shape
        output = torch.zeros_like(x)
        numerator = torch.arange(seq_length).unsqueeze(-1)
        denominator = torch.pow(10000, torch.arange(0, d_model, 2)/d_model)
        output[:, :, ::2] = torch.sin(numerator/denominator)
        output[:, :, 1::2] = torch.cos(numerator/denominator)
        return output

class Encoder(nn.Module):
    pass

class Decoder(nn.Module):
    pass

class EncoderDecoder(nn.Module):
    pass