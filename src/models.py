import torch
from torch import nn


class GreenhouseMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden=(128, 64, 32), dropout=0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GreenhouseLSTM(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=64, num_layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


class DynamicsMLP(nn.Module):
    def __init__(self, input_dim, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)
