# model.py
import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, nhid, nlayers, nclasses, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, nclasses)

    def forward(self, src):
        # src tensor shape: (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        # src tensor shape after positional encoding: (seq_len, batch_size, d_model)
        output = self.transformer_encoder(src)
        # output tensor shape: (seq_len, batch_size, d_model)
        output = self.decoder(output[-1]) # We only want the last output for classification
        # output tensor shape after decoder: (batch_size, nclasses)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x tensor shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        # x tensor shape after positional encoding: (seq_len, batch_size, d_model)
        return self.dropout(x)
