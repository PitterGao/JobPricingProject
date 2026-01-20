from __future__ import annotations
import math
import torch
import torch.nn as nn

class HealthLSTM(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # last time step
        y = self.head(last).squeeze(-1)
        return y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class HealthTransformer(nn.Module):
    def __init__(self, feat_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        z = self.proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        last = z[:, -1, :]
        y = self.head(last).squeeze(-1)
        return y
