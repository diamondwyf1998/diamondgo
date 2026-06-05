from __future__ import annotations

import torch
from torch import nn

from diamondgo.config import ModelConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class PolicyValueNet(nn.Module):
    """Small AlphaZero-style network with one policy head and one value head."""

    def __init__(self, board_size: int, config: ModelConfig, input_planes: int = 4) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size + 1

        self.trunk = nn.Sequential(
            nn.Conv2d(input_planes, config.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.channels),
            nn.ReLU(inplace=True),
            *[ResidualBlock(config.channels) for _ in range(config.residual_blocks)],
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(config.channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, config.channels),
            nn.ReLU(inplace=True),
            nn.Linear(config.channels, 1),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.trunk(features)
        policy_logits = self.policy_head(trunk)
        value = self.value_head(trunk).squeeze(-1)
        return policy_logits, value
