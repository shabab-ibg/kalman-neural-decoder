import torch
import torch.nn as nn

class SequenceGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, x):
        outputs, _ = self.gru(self.drop(x))
        return self.out(outputs)

class WindowedGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, x):
        _, hidden = self.gru(self.drop(x))
        return self.out(hidden[-1])  # Extract last recurrent vector only
