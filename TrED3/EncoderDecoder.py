import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.v = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_size, 1)))
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_size, hidden_size)))
        self.U = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_size, hidden_size)))

    def forward(self, s, h, mask=None):
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1) == 0, 0)
        s = torch.transpose(s, 0, 1)
        proj_s = s @ self.W
        proj_h = h @ self.U
        energy = torch.tanh(proj_s + proj_h) @ self.v
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        score = nn.functional.softmax(energy, dim=1)
        context = torch.sum(score * h, dim=1)
        return context

class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_token):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 512, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=512, hidden_size=1024, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.pad_token = pad_token

    def forward(self, x, lenghts):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = pack_padded_sequence(x, lenghts, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(x)
        output = pad_packed_sequence(output, batch_first=True, total_length=seq_len)[0]

        h_t = torch.cat([hidden[0][0], hidden[0][1]], dim=1)
        c_t = torch.cat([hidden[1][0], hidden[1][1]], dim=1)

        return output, (h_t[None, :, :], c_t[None, :, :]) # LSTM default states is always batch_first False

class Decoder(nn.Module):
    def __init__(self, vocab_size, pad_token):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 512, padding_idx=pad_token)
        self.attention = Attention(2048, 512)
        self.lstm = nn.LSTM(input_size=2560, hidden_size=2048, batch_first=True)
        self.linear = nn.Linear(2048, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.pad_token = pad_token

    def forward(self, x, encoder_output, hidden, lenghts, encoder_mask=None, decoder_mask=None):
        seq_len = x.size(1)
        x = self.embedding(x)
        for t in range(seq_len):
            h_t, c_t = hidden
            if decoder_mask is not None:
                mask_t = decoder_mask[:, t]
                h_t = h_t.masked_fill(mask_t.view(-1, 1, 1).transpose(0,1) == 0, 0)
            context = self.attention(h_t, encoder_output, mask=encoder_mask)
            input = torch.cat((x[: ,t].unsqueeze(1), context.unsqueeze(1)), dim=2)
            output, hidden = self.lstm(input, hidden)
            lstm_output = output if t == 0 else torch.cat([lstm_output, output], dim=1)
        if decoder_mask is not None:
            lstm_output = lstm_output.masked_fill(decoder_mask.unsqueeze(-1) == 0, 0)
        x = self.dropout(lstm_output)
        return  self.linear(x), hidden

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths, encoder_mask=None, decoder_mask=None):
        output, hidden = self.encoder(encoder_inputs, encoder_lengths)
        return self.decoder(decoder_inputs, output, hidden, decoder_lengths, encoder_mask=encoder_mask, decoder_mask=decoder_mask)[0]