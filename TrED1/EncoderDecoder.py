import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_token):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, batch_first=True)
        self.pad_token = pad_token

    def forward(self, x, lengths):
        x = self.embedding.forward(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        return self.lstm.forward(x)[1]

class Decoder(nn.Module):
    def __init__(self, vocab_size, pad_token):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, batch_first=True)
        self.linear = nn.Linear(512, vocab_size)
        self.pad_token = pad_token

    def forward(self, x, h_0, c_0, lengths):
        seq_len = x.size(1)
        x = self.embedding.forward(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, (h_t, c_t) = self.lstm.forward(x, (h_0, c_0))
        x = pad_packed_sequence(x, batch_first=True, total_length=seq_len)[0]
        return self.linear(x), (h_t, c_t)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths):
        h_0, c_0 = self.encoder(encoder_inputs, encoder_lengths)
        return self.decoder(decoder_inputs, h_0, c_0, decoder_lengths)[0]