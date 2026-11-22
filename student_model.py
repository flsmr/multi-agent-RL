"""Deterministic local-observation actor for classical student distillation."""
import torch
from torch import nn


class StudentActor(nn.Module):
    """Local-observation student, trained to predict bounded teacher actions."""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(24, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, states):
        return torch.tanh(self.actor(states))

