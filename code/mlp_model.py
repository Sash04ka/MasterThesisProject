# mlp_model.py

import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.output_bias = nn.Parameter(torch.zeros(1))
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = self.model(x)
        return out * self.output_scale + self.output_bias
