import torch
import torch.nn as nn
import torch.nn.functional as F


class BahadanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.w = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.tensor, keys: torch.tensor) -> tuple[torch.tensor]:
        scores = self.v(torch.tanh(self.w(query) + self.u(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights
