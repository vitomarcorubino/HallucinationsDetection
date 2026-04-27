import torch.nn as nn
import torch

class LogisticRegressionProbe(nn.Module):
    def __init__(self, input_dim: int, use_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=use_bias)

    def forward(self, x):
        # We return logits for numerical stability with BCEWithLogitsLoss
        return self.linear(x).squeeze(-1)

class FFNProbe(nn.Module):
    """
    A simple Feed-Forward Neural Network (MLP) with one hidden layer.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)