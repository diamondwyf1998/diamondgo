from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diamondgo.batched_demo import BatchedConfig, make_model, play_batched_games
from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_MAX_MOVES, DEFAULT_9X9_SCORE_KOMI
from diamondgo.demo_cpu import build_trace, write_json, write_sgf


@dataclass(frozen=True)
class OvernightConfig:
    board_size: int = 9
    komi: float = DEFAULT_9X9_KOMI
    score_komi: float = DEFAULT_9X9_SCORE_KOMI
    input_komi: bool = True
    terminal_dead_stone_cleanup: bool = False
    score_margin_reward_scale: float = 0.0
    channels: int = 32
    residual_blocks: int = 2
    simulations: int = 64
    games_per_cycle: int = 16
    max_moves: int = DEFAULT_9X9_MAX_MOVES
    train_steps_per_cycle: int = 64
    batch_size: int = 256
    replay_size: int = 50_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    c_puct: float = 1.5
    temperature: float = 1.0
    temperature_moves: int = 0
    late_temperature: float = 1.0
    root_dirichlet_alpha: float = 0.0
    root_noise_fraction: float = 0.0
    root_policy_temperature: float = 1.0
    augment_dihedral: bool = False
    seed: int = 1
    device: str = "cuda"
    rules_backend: str = "sgfmill"
    cycles: int = 10_000
    time_limit_minutes: float = 0.0
    checkpoint_every: int = 10
    early_checkpoint_cycles: int = 50
    early_checkpoint_every: int = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a stable overnight 9x9 baby-zero training loop.")
    parser.add_argument("--out-dir", default="artifacts/overnight-9x9")
    parser.add_argument("--cycles", type=int, default=OvernightConfig.cycles)
    parser.add_argument("--time-limit-minutes", type=float, default=OvernightConfig.time_limit_minutes)
    parser.add_argument("--games-per-cycle", type=int, default=OvernightConfig.games_per_cycle)
    parser.add_argument("--komi", type=float, default=OvernightConfig.komi)
    parser.add_argument("--score-komi", type=float, default=OvernightConfig.score_komi)
    parser.add_argument("--input-komi", action=argparse.BooleanOptionalAction, default=OvernightConfig.input_komi)
    parser.add_argument(
        "--terminal-dead-stone-cleanup",
        action=argparse.BooleanOptionalAction,
        default=OvernightConfig.terminal_dead_stone_cleanup,
        help="At terminal scoring, remove conservatively detected obvious dead groups.",
    )
    parser.add_argument(
        "--score-margin-reward-scale",
        type=float,
        default=OvernightConfig.score_margin_reward_scale,
        help="Scale for the capped +/-0.6 score-margin component; enabled targets use +/-0.4 win/loss base.",
    )
    parser.add_argument("--max-moves", type=int, default=OvernightConfig.max_moves)
    parser.add_argument("--simulations", type=int, default=OvernightConfig.simulations)
    parser.add_argument("--train-steps-per-cycle", type=int, default=OvernightConfig.train_steps_per_cycle)
    parser.add_argument("--batch-size", type=int, default=OvernightConfig.batch_size)
    parser.add_argument("--replay-size", type=int, default=OvernightConfig.replay_size)
    parser.add_argument("--channels", type=int, default=OvernightConfig.channels)
    parser.add_argument("--residual-blocks", type=int, default=OvernightConfig.residual_blocks)
    parser.add_argument("--learning-rate", type=float, default=OvernightConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=OvernightConfig.weight_decay)
    parser.add_argument("--c-puct", type=float, default=OvernightConfig.c_puct)
    parser.add_argument("--temperature", type=float, default=OvernightConfig.temperature)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=OvernightConfig.root_dirichlet_alpha)
    parser.add_argument("--root-noise-fraction", type=float, default=OvernightConfig.root_noise_fraction)
    parser.add_argument("--root-policy-temperature", type=float, default=OvernightConfig.root_policy_temperature)
    parser.add_argument("--temperature-moves", type=int, default=OvernightConfig.temperature_moves)
    parser.add_argument("--late-temperature", type=float, default=OvernightConfig.late_temperature)
    parser.add_argument("--augment-dihedral", action="store_true", default=OvernightConfig.augment_dihedral)
    parser.add_argument("--seed", type=int, default=OvernightConfig.seed)
    parser.add_argument("--device", default=OvernightConfig.device)
    parser.add_argument("--rules", choices=["simple", "sgfmill"], default=OvernightConfig.rules_backend)
    parser.add_argument("--checkpoint-every", type=int, default=OvernightConfig.checkpoint_every)
    parser.add_argument("--early-checkpoint-cycles", type=int, default=OvernightConfig.early_checkpoint_cycles)
    parser.add_argument("--early-checkpoint-every", type=int, default=OvernightConfig.early_checkpoint_every)
    parser.add_argument("--resume", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def should_save_cycle_checkpoint(
    cycle: int,
    checkpoint_every: int,
    early_checkpoint_cycles: int = 50,
    early_checkpoint_every: int = 5,
) -> bool:
    """Use denser snapshots early, then a coarser steady-state cadence."""
    if cycle <= 0:
        return False
    early_every = max(1, early_checkpoint_every)
    late_every = max(1, checkpoint_every)
    if cycle <= max(0, early_checkpoint_cycles):
        return cycle % early_every == 0
    return cycle % late_every == 0


def make_selfplay_config(config: OvernightConfig, seed: int) -> BatchedConfig:
    return BatchedConfig(
        board_size=config.board_size,
        komi=config.komi,
        score_komi=config.score_komi,
        input_komi=config.input_komi,
        terminal_dead_stone_cleanup=config.terminal_dead_stone_cleanup,
        score_margin_reward_scale=config.score_margin_reward_scale,
        channels=config.channels,
        residual_blocks=config.residual_blocks,
        simulations=config.simulations,
        max_moves=config.max_moves,
        games=config.games_per_cycle,
        train_steps=0,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        c_puct=config.c_puct,
        temperature=config.temperature,
        temperature_moves=config.temperature_moves,
        late_temperature=config.late_temperature,
        root_dirichlet_alpha=config.root_dirichlet_alpha,
        root_noise_fraction=config.root_noise_fraction,
        root_policy_temperature=config.root_policy_temperature,
        seed=seed,
        device=config.device,
        rules_backend=config.rules_backend,
    )


def train_from_replay(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: list[dict[str, object]],
    steps: int,
    batch_size: int,
    augment_dihedral: bool = False,
) -> list[dict[str, float]]:
    model.train()
    device = next(model.parameters()).device
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        batch = [random.choice(replay) for _ in range(min(batch_size, len(replay)))]
        features_np, policies_np = prepare_training_batch(batch, augment_dihedral)
        features = torch.tensor(features_np, dtype=torch.float32).to(device)
        policy_targets = torch.tensor(policies_np, dtype=torch.float32).to(device)
        value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.float32).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, values = model(features)
        policy_loss = -(policy_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": round(float(loss.item()), 6),
                "policy_loss": round(float(policy_loss.item()), 6),
                "value_loss": round(float(value_loss.item()), 6),
            }
        )
    model.eval()
    return history


def prepare_training_batch(
    batch: list[dict[str, object]],
    augment_dihedral: bool,
) -> tuple[np.ndarray, np.ndarray]:
    features_rows = []
    policy_rows = []
    for item in batch:
        features = np.asarray(item["features"], dtype=np.float32)
        policy = np.asarray(item["policy"], dtype=np.float32)
        if augment_dihedral:
            transform = random.randrange(8)
            features, policy = apply_dihedral_transform(features, policy, transform)
        features_rows.append(features)
        policy_rows.append(policy)
    return np.stack(features_rows), np.stack(policy_rows)


def apply_dihedral_transform(
    features: np.ndarray,
    policy: np.ndarray,
    transform: int,
) -> tuple[np.ndarray, np.ndarray]:
    board_size = int(features.shape[-1])
    rotations = transform % 4
    flip = transform >= 4

    transformed_features = np.rot90(features, k=rotations, axes=(-2, -1))
    board_policy = policy[:-1].reshape(board_size, board_size)
    transformed_policy_board = np.rot90(board_policy, k=rotations, axes=(0, 1))
    if flip:
        transformed_features = np.flip(transformed_features, axis=-1)
        transformed_policy_board = np.flip(transformed_policy_board, axis=-1)

    transformed_policy = np.concatenate(
        [transformed_policy_board.reshape(-1), policy[-1:]],
    ).astype(np.float32)
    return np.ascontiguousarray(transformed_features), np.ascontiguousarray(transformed_policy)


def save_checkpoint(
    path: Path,
    config: OvernightConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cycle: int,
    total_positions: int,
    total_train_steps: int,
    latest_metrics: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cycle": cycle,
            "total_positions": total_positions,
            "total_train_steps": total_train_steps,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latest_metrics": latest_metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint.get("cycle", 0)),
        int(checkpoint.get("total_positions", 0)),
        int(checkpoint.get("total_train_steps", 0)),
    )


def summarize_cycle(
    cycle: int,
    started_at: float,
    examples: list[dict[str, object]],
    train_history: list[dict[str, float]],
    replay_size: int,
    total_positions: int,
    total_train_steps: int,
    selfplay_stats: dict[str, object],
) -> dict[str, object]:
    elapsed = time.perf_counter() - started_at
    policies = [np.asarray(item["policy"], dtype=np.float32) for item in examples]
    entropies = [float(-(policy * np.log(np.clip(policy, 1e-9, 1.0))).sum()) for policy in policies]
    value_targets = [float(item["value_target"]) for item in examples]
    compact_selfplay_stats = {
        key: value for key, value in selfplay_stats.items() if key != "batch_sizes"
    }
    return {
        "cycle": cycle,
        "cycle_seconds": round(elapsed, 3),
        "positions": len(examples),
        "positions_per_second": round(len(examples) / max(elapsed, 1e-9), 3),
        "replay_size": replay_size,
        "total_positions": total_positions,
        "total_train_steps": total_train_steps,
        "latest_loss": train_history[-1] if train_history else {},
        "policy_entropy_mean": round(float(np.mean(entropies)), 4) if entropies else 0.0,
        "value_target_mean": round(float(np.mean(value_targets)), 4) if value_targets else 0.0,
        "value_target_min": round(min(value_targets), 4) if value_targets else 0.0,
        "value_target_max": round(max(value_targets), 4) if value_targets else 0.0,
        "selfplay": compact_selfplay_stats,
    }


def run(config: OvernightConfig, out_dir: Path, resume: str = "") -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but CUDA is unavailable")

    model = make_model(make_selfplay_config(config, config.seed))
    model.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start_cycle = 0
    total_positions = 0
    total_train_steps = 0
    if resume:
        start_cycle, total_positions, total_train_steps = load_checkpoint(Path(resume), model, optimizer)

    replay: list[dict[str, object]] = []
    params = sum(parameter.numel() for parameter in model.parameters())
    metrics_path = out_dir / "metrics.jsonl"
    started = time.perf_counter()
    last_metrics: dict[str, object] = {}

    for cycle in range(start_cycle + 1, config.cycles + 1):
        if config.time_limit_minutes > 0 and (time.perf_counter() - started) >= config.time_limit_minutes * 60:
            break

        cycle_start = time.perf_counter()
        selfplay_config = make_selfplay_config(config, config.seed + cycle)
        examples, selfplay_stats = play_batched_games(selfplay_config, model)
        replay.extend(examples)
        if len(replay) > config.replay_size:
            replay = replay[-config.replay_size :]
        total_positions += len(examples)

        train_history = train_from_replay(
            model=model,
            optimizer=optimizer,
            replay=replay,
            steps=config.train_steps_per_cycle,
            batch_size=config.batch_size,
            augment_dihedral=config.augment_dihedral,
        )
        total_train_steps += len(train_history)

        metrics = summarize_cycle(
            cycle=cycle,
            started_at=cycle_start,
            examples=examples,
            train_history=train_history,
            replay_size=len(replay),
            total_positions=total_positions,
            total_train_steps=total_train_steps,
            selfplay_stats=selfplay_stats,
        )
        last_metrics = metrics
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
            handle.flush()

        cycle_trace = build_trace(selfplay_config, examples)
        write_sgf(out_dir / "latest-cycle.sgf", selfplay_config, examples)
        write_json(out_dir / "latest-cycle-trace.json", cycle_trace)
        if cycle % 10 == 0:
            records_dir = out_dir / "cycle-records"
            write_sgf(records_dir / f"cycle-{cycle:05d}.sgf", selfplay_config, examples)
            write_json(records_dir / f"cycle-{cycle:05d}-trace.json", cycle_trace)
        save_checkpoint(
            out_dir / "latest.pt",
            config,
            model,
            optimizer,
            cycle,
            total_positions,
            total_train_steps,
            metrics,
        )
        if should_save_cycle_checkpoint(
            cycle,
            config.checkpoint_every,
            config.early_checkpoint_cycles,
            config.early_checkpoint_every,
        ):
            save_checkpoint(
                out_dir / "checkpoints" / f"cycle-{cycle:05d}.pt",
                config,
                model,
                optimizer,
                cycle,
                total_positions,
                total_train_steps,
                metrics,
            )
        print(json.dumps(metrics), flush=True)

    return {
        "out_dir": str(out_dir),
        "parameters": params,
        "total_positions": total_positions,
        "total_train_steps": total_train_steps,
        "latest_metrics": last_metrics,
        "checkpoint": str(out_dir / "latest.pt"),
        "metrics_path": str(metrics_path),
        "cycle_records_dir": str(out_dir / "cycle-records"),
    }


def main() -> None:
    args = build_parser().parse_args()
    config = OvernightConfig(
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        simulations=args.simulations,
        komi=args.komi,
        score_komi=args.score_komi,
        input_komi=args.input_komi,
        terminal_dead_stone_cleanup=args.terminal_dead_stone_cleanup,
        score_margin_reward_scale=args.score_margin_reward_scale,
        games_per_cycle=args.games_per_cycle,
        max_moves=args.max_moves,
        train_steps_per_cycle=args.train_steps_per_cycle,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temperature_moves=args.temperature_moves,
        late_temperature=args.late_temperature,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_noise_fraction=args.root_noise_fraction,
        root_policy_temperature=args.root_policy_temperature,
        augment_dihedral=args.augment_dihedral,
        seed=args.seed,
        device=args.device,
        rules_backend=args.rules,
        cycles=args.cycles,
        time_limit_minutes=args.time_limit_minutes,
        checkpoint_every=args.checkpoint_every,
        early_checkpoint_cycles=args.early_checkpoint_cycles,
        early_checkpoint_every=args.early_checkpoint_every,
    )
    summary = run(config, Path(args.out_dir), args.resume)
    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
