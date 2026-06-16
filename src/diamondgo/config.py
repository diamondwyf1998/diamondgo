from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_MAX_MOVES, DEFAULT_9X9_SCORE_KOMI


@dataclass(frozen=True)
class BoardConfig:
    size: int = 9
    komi: float = DEFAULT_9X9_KOMI
    score_komi: float = DEFAULT_9X9_SCORE_KOMI
    history_moves: int = 0
    terminal_dead_stone_cleanup: bool = False
    score_margin_reward_scale: float = 0.0
    rules_backend: str = "sgfmill"


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 64
    residual_blocks: int = 4


@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    root_noise_fraction: float = 0.25
    temperature: float = 1.0


@dataclass(frozen=True)
class SelfPlayConfig:
    games: int = 16
    max_moves: int = DEFAULT_9X9_MAX_MOVES


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    steps: int = 1_000
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "baby-zero-9x9"
    seed: int = 1
    board: BoardConfig = field(default_factory=BoardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def input_plane_count(input_komi: bool, history_moves: int = 0) -> int:
    return 3 + max(0, int(history_moves)) + (1 if input_komi else 0)
