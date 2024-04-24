import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
