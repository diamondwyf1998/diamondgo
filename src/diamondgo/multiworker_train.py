from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diamondgo.batched_demo import BatchedConfig, make_model, play_batched_games
from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_MAX_MOVES, DEFAULT_9X9_SCORE_KOMI
from diamondgo.demo_cpu import build_trace, write_json, write_sgf
from diamondgo.overnight_train import (
    OvernightConfig,
    load_checkpoint,
    save_checkpoint,
    should_save_cycle_checkpoint,
    train_from_replay,
)


@dataclass(frozen=True)
class MultiWorkerConfig:
    board_size: int = 9
    komi: float = DEFAULT_9X9_KOMI
    score_komi: float = DEFAULT_9X9_SCORE_KOMI
    input_komi: bool = True
    terminal_dead_stone_cleanup: bool = False
    score_margin_reward_scale: float = 0.0
    channels: int = 32
    residual_blocks: int = 2
    simulations: int = 100
    workers: int = 8
    games_per_worker: int = 4
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
    parser = argparse.ArgumentParser(description="Run multi-worker 9x9 self-play with one trainer.")
    parser.add_argument("--out-dir", default="artifacts/multiworker-9x9")
    parser.add_argument("--resume", default="")
    parser.add_argument("--cycles", type=int, default=MultiWorkerConfig.cycles)
    parser.add_argument("--time-limit-minutes", type=float, default=MultiWorkerConfig.time_limit_minutes)
    parser.add_argument("--workers", type=int, default=MultiWorkerConfig.workers)
    parser.add_argument("--games-per-worker", type=int, default=MultiWorkerConfig.games_per_worker)
    parser.add_argument("--komi", type=float, default=MultiWorkerConfig.komi)
    parser.add_argument("--score-komi", type=float, default=MultiWorkerConfig.score_komi)
    parser.add_argument("--input-komi", action=argparse.BooleanOptionalAction, default=MultiWorkerConfig.input_komi)
    parser.add_argument(
        "--terminal-dead-stone-cleanup",
        action=argparse.BooleanOptionalAction,
        default=MultiWorkerConfig.terminal_dead_stone_cleanup,
        help="At terminal scoring, remove conservatively detected obvious dead groups.",
    )
    parser.add_argument(
        "--score-margin-reward-scale",
        type=float,
        default=MultiWorkerConfig.score_margin_reward_scale,
        help="Scale for the capped +/-0.6 score-margin component; enabled targets use +/-0.4 win/loss base.",
    )
    parser.add_argument("--max-moves", type=int, default=MultiWorkerConfig.max_moves)
    parser.add_argument("--simulations", type=int, default=MultiWorkerConfig.simulations)
    parser.add_argument("--train-steps-per-cycle", type=int, default=MultiWorkerConfig.train_steps_per_cycle)
    parser.add_argument("--batch-size", type=int, default=MultiWorkerConfig.batch_size)
    parser.add_argument("--replay-size", type=int, default=MultiWorkerConfig.replay_size)
    parser.add_argument("--channels", type=int, default=MultiWorkerConfig.channels)
    parser.add_argument("--residual-blocks", type=int, default=MultiWorkerConfig.residual_blocks)
    parser.add_argument("--learning-rate", type=float, default=MultiWorkerConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=MultiWorkerConfig.weight_decay)
    parser.add_argument("--c-puct", type=float, default=MultiWorkerConfig.c_puct)
    parser.add_argument("--temperature", type=float, default=MultiWorkerConfig.temperature)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=MultiWorkerConfig.root_dirichlet_alpha)
    parser.add_argument("--root-noise-fraction", type=float, default=MultiWorkerConfig.root_noise_fraction)
    parser.add_argument("--root-policy-temperature", type=float, default=MultiWorkerConfig.root_policy_temperature)
    parser.add_argument("--temperature-moves", type=int, default=MultiWorkerConfig.temperature_moves)
    parser.add_argument("--late-temperature", type=float, default=MultiWorkerConfig.late_temperature)
    parser.add_argument("--augment-dihedral", action="store_true", default=MultiWorkerConfig.augment_dihedral)
    parser.add_argument("--seed", type=int, default=MultiWorkerConfig.seed)
    parser.add_argument("--device", default=MultiWorkerConfig.device)
    parser.add_argument("--rules", choices=["simple", "sgfmill"], default=MultiWorkerConfig.rules_backend)
    parser.add_argument("--checkpoint-every", type=int, default=MultiWorkerConfig.checkpoint_every)
    parser.add_argument("--early-checkpoint-cycles", type=int, default=MultiWorkerConfig.early_checkpoint_cycles)
    parser.add_argument("--early-checkpoint-every", type=int, default=MultiWorkerConfig.early_checkpoint_every)
    parser.add_argument("--json", action="store_true")
    return parser


def to_overnight_config(config: MultiWorkerConfig) -> OvernightConfig:
    return OvernightConfig(
        board_size=config.board_size,
        komi=config.komi,
        score_komi=config.score_komi,
        input_komi=config.input_komi,
        terminal_dead_stone_cleanup=config.terminal_dead_stone_cleanup,
        score_margin_reward_scale=config.score_margin_reward_scale,
        channels=config.channels,
        residual_blocks=config.residual_blocks,
        simulations=config.simulations,
        games_per_cycle=config.workers * config.games_per_worker,
        max_moves=config.max_moves,
        train_steps_per_cycle=config.train_steps_per_cycle,
        batch_size=config.batch_size,
        replay_size=config.replay_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        c_puct=config.c_puct,
        temperature=config.temperature,
        temperature_moves=config.temperature_moves,
        late_temperature=config.late_temperature,
        root_dirichlet_alpha=config.root_dirichlet_alpha,
        root_noise_fraction=config.root_noise_fraction,
        root_policy_temperature=config.root_policy_temperature,
        augment_dihedral=config.augment_dihedral,
        seed=config.seed,
        device=config.device,
        rules_backend=config.rules_backend,
        cycles=config.cycles,
        time_limit_minutes=config.time_limit_minutes,
        checkpoint_every=config.checkpoint_every,
        early_checkpoint_cycles=config.early_checkpoint_cycles,
        early_checkpoint_every=config.early_checkpoint_every,
    )


def make_selfplay_config(config: MultiWorkerConfig, seed: int) -> BatchedConfig:
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
        games=config.games_per_worker,
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


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def worker_selfplay(
    worker_id: int,
    config_dict: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    config = MultiWorkerConfig(**config_dict)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    selfplay_config = make_selfplay_config(config, seed)
    model = make_model(selfplay_config)
    model.load_state_dict(state_dict)
    model.eval()
    started = time.perf_counter()
    examples, stats = play_batched_games(selfplay_config, model)
    elapsed = time.perf_counter() - started
    return {
        "worker_id": worker_id,
        "seed": seed,
        "positions": len(examples),
        "seconds": round(elapsed, 3),
        "positions_per_second": round(len(examples) / max(elapsed, 1e-9), 3),
        "examples": examples,
        "selfplay": {
            key: value for key, value in stats.items() if key != "batch_sizes"
        },
    }


def summarize_cycle(
    cycle: int,
    total_seconds: float,
    selfplay_seconds: float,
    train_seconds: float,
    write_seconds: float,
    examples: list[dict[str, object]],
    train_history: list[dict[str, float]],
    replay_size: int,
    total_positions: int,
    total_train_steps: int,
    worker_summaries: list[dict[str, Any]],
) -> dict[str, object]:
    policies = [np.asarray(item["policy"], dtype=np.float32) for item in examples]
    entropies = [float(-(policy * np.log(np.clip(policy, 1e-9, 1.0))).sum()) for policy in policies]
    value_targets = [float(item["value_target"]) for item in examples]
    game_summaries = [
        game
        for worker in worker_summaries
        for game in list(worker["selfplay"].get("game_summaries", []))
    ]
    network_seconds = sum(float(item["selfplay"].get("network_seconds", 0.0)) for item in worker_summaries)
    network_calls = sum(int(item["selfplay"].get("network_calls", 0)) for item in worker_summaries)
    timing_keys = [
        "encode_seconds",
        "tensor_seconds",
        "legal_actions_seconds",
        "state_copy_seconds",
        "select_seconds",
        "play_search_seconds",
        "sample_encode_seconds",
        "stone_count_seconds",
    ]
    selfplay_timing = {
        key: round(sum(float(item["selfplay"].get(key, 0.0)) for item in worker_summaries), 3)
        for key in timing_keys
    }
    weighted_batch_total = sum(
        float(item["selfplay"].get("average_batch_size", 0.0))
        * int(item["selfplay"].get("network_calls", 0))
        for item in worker_summaries
    )
    average_batch = weighted_batch_total / max(network_calls, 1)
    return {
        "cycle": cycle,
        "cycle_seconds": round(total_seconds, 3),
        "selfplay_seconds": round(selfplay_seconds, 3),
        "train_seconds": round(train_seconds, 3),
        "write_seconds": round(write_seconds, 3),
        "positions": len(examples),
        "positions_per_second": round(len(examples) / max(selfplay_seconds, 1e-9), 3),
        "replay_size": replay_size,
        "total_positions": total_positions,
        "total_train_steps": total_train_steps,
        "latest_loss": train_history[-1] if train_history else {},
        "policy_entropy_mean": round(float(np.mean(entropies)), 4) if entropies else 0.0,
        "value_target_mean": round(float(np.mean(value_targets)), 4) if value_targets else 0.0,
        "value_target_min": round(min(value_targets), 4) if value_targets else 0.0,
        "value_target_max": round(max(value_targets), 4) if value_targets else 0.0,
        "game_behavior": summarize_game_behavior(game_summaries, examples),
        "selfplay_timing": selfplay_timing,
        "selfplay": {
            "summed_network_seconds": round(network_seconds, 3),
            "summed_network_calls": network_calls,
            "average_network_batch": round(average_batch, 3),
            "max_network_batch": max(
                [int(item["selfplay"].get("max_batch_size", 0)) for item in worker_summaries] + [0]
            ),
        },
        "workers": [
            {
                "worker_id": item["worker_id"],
                "positions": item["positions"],
                "seconds": item["seconds"],
                "positions_per_second": item["positions_per_second"],
                "average_network_batch": item["selfplay"].get("average_batch_size", 0),
                "max_network_batch": item["selfplay"].get("max_batch_size", 0),
                "legal_actions_seconds": item["selfplay"].get("legal_actions_seconds", 0),
                "state_copy_seconds": item["selfplay"].get("state_copy_seconds", 0),
                "encode_seconds": item["selfplay"].get("encode_seconds", 0),
            }
            for item in worker_summaries
        ],
    }


def summarize_game_behavior(
    game_summaries: list[dict[str, object]], examples: list[dict[str, object]]
) -> dict[str, object]:
    moves = [int(item["moves"]) for item in game_summaries]
    pass_moves = sum(int(item.get("passes", 0)) for item in game_summaries)
    pass_moves_black = sum(1 for item in examples if item.get("is_pass") and item.get("player") == "b")
    pass_moves_white = sum(1 for item in examples if item.get("is_pass") and item.get("player") == "w")
    first_pass_moves = [
        int(item["first_pass_move"])
        for item in game_summaries
        if item.get("first_pass_move") is not None
    ]
    second_pass_moves = [
        int(item["second_pass_move"])
        for item in game_summaries
        if item.get("second_pass_move") is not None
    ]
    terminal_double_pass_moves = [
        int(item["terminal_double_pass_move"])
        for item in game_summaries
        if item.get("terminal_double_pass_move") is not None
    ]
    capture_moves = sum(int(item.get("capture_moves", 0)) for item in game_summaries)
    captured_stones = sum(int(item.get("captured_stones", 0)) for item in game_summaries)
    terminal_cleanup_black_stones = sum(
        int(item.get("terminal_cleanup_black_stones", 0)) for item in game_summaries
    )
    terminal_cleanup_white_stones = sum(
        int(item.get("terminal_cleanup_white_stones", 0)) for item in game_summaries
    )
    signed_margins = [float(item.get("black_score_margin", 0.0)) for item in game_summaries]
    margins = [abs(margin) for margin in signed_margins]
    black_win_margins = [margin for margin in signed_margins if margin > 0]
    white_win_margins = [-margin for margin in signed_margins if margin < 0]
    black_wins = sum(1 for item in game_summaries if item.get("winner") == "b")
    games = len(game_summaries)
    white_wins = games - black_wins
    black_win_rate = black_wins / max(games, 1)
    white_win_rate = white_wins / max(games, 1)
    color_bias_alert_threshold = 0.70
    early_pass_alert_threshold = 0.50
    early_first_pass_games_40 = sum(1 for move in first_pass_moves if move <= 40)
    early_first_pass_rate_40 = early_first_pass_games_40 / max(games, 1)
    return {
        "games": games,
        "ended_by_pass": sum(1 for item in game_summaries if item.get("ended_by") == "pass"),
        "ended_by_max_moves": sum(1 for item in game_summaries if item.get("ended_by") == "max_moves"),
        "moves_mean": round(float(np.mean(moves)), 3) if moves else 0.0,
        "moves_max": max(moves) if moves else 0,
        "pass_moves": pass_moves,
        "pass_moves_black": pass_moves_black,
        "pass_moves_white": pass_moves_white,
        "pass_move_fraction": round(pass_moves / max(len(examples), 1), 4),
        "first_pass_games": len(first_pass_moves),
        "first_pass_move_min": min(first_pass_moves) if first_pass_moves else 0,
        "first_pass_move_mean": round(float(np.mean(first_pass_moves)), 3) if first_pass_moves else 0.0,
        "first_pass_move_median": round(float(np.median(first_pass_moves)), 3) if first_pass_moves else 0.0,
        "second_pass_games": len(second_pass_moves),
        "second_pass_move_min": min(second_pass_moves) if second_pass_moves else 0,
        "second_pass_move_mean": round(float(np.mean(second_pass_moves)), 3) if second_pass_moves else 0.0,
        "second_pass_move_median": round(float(np.median(second_pass_moves)), 3) if second_pass_moves else 0.0,
        "terminal_double_pass_games": len(terminal_double_pass_moves),
        "terminal_double_pass_move_min": min(terminal_double_pass_moves) if terminal_double_pass_moves else 0,
        "terminal_double_pass_move_mean": round(float(np.mean(terminal_double_pass_moves)), 3)
        if terminal_double_pass_moves
        else 0.0,
        "terminal_double_pass_move_median": round(float(np.median(terminal_double_pass_moves)), 3)
        if terminal_double_pass_moves
        else 0.0,
        "early_first_pass_games_20": sum(1 for move in first_pass_moves if move <= 20),
        "early_first_pass_games_40": early_first_pass_games_40,
        "early_first_pass_games_60": sum(1 for move in first_pass_moves if move <= 60),
        "early_first_pass_rate_40": round(early_first_pass_rate_40, 4),
        "early_pass_alert_threshold": early_pass_alert_threshold,
        "early_pass_alert": early_first_pass_rate_40 >= early_pass_alert_threshold,
        "capture_moves": capture_moves,
        "captured_stones": captured_stones,
        "capture_move_fraction": round(capture_moves / max(len(examples), 1), 4),
        "terminal_cleanup_black_stones": terminal_cleanup_black_stones,
        "terminal_cleanup_white_stones": terminal_cleanup_white_stones,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "black_win_rate": round(black_win_rate, 4),
        "white_win_rate": round(white_win_rate, 4),
        "black_win_rate_alert": black_win_rate >= color_bias_alert_threshold,
        "white_win_rate_alert": white_win_rate >= color_bias_alert_threshold,
        "color_bias_alert_threshold": color_bias_alert_threshold,
        "color_bias_alert": (
            "black" if black_win_rate >= color_bias_alert_threshold
            else "white" if white_win_rate >= color_bias_alert_threshold
            else ""
        ),
        "black_score_margin_mean": round(float(np.mean(signed_margins)), 3)
        if signed_margins
        else 0.0,
        "abs_score_margin_mean": round(float(np.mean(margins)), 3) if margins else 0.0,
        "black_win_margin_mean": round(float(np.mean(black_win_margins)), 3)
        if black_win_margins
        else 0.0,
        "white_win_margin_mean": round(float(np.mean(white_win_margins)), 3)
        if white_win_margins
        else 0.0,
        "per_game": game_summaries,
    }


def run(config: MultiWorkerConfig, out_dir: Path, resume: str = "") -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
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
        state_dict = cpu_state_dict(model)
        selfplay_start = time.perf_counter()
        config_dict = asdict(config)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    worker_selfplay,
                    worker_id,
                    config_dict,
                    state_dict,
                    config.seed + cycle * 10_000 + worker_id,
                )
                for worker_id in range(1, config.workers + 1)
            ]
            worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        worker_results.sort(key=lambda item: item["worker_id"])
        selfplay_seconds = time.perf_counter() - selfplay_start

        examples: list[dict[str, object]] = []
        for worker_result in worker_results:
            worker_id = int(worker_result["worker_id"])
            for summary in worker_result["selfplay"].get("game_summaries", []):
                summary["worker_id"] = worker_id
                summary["local_game"] = int(summary["game"])
                summary["game"] = (worker_id - 1) * config.games_per_worker + int(summary["local_game"])
            for example in worker_result["examples"]:
                example["worker_id"] = worker_id
                example["local_game"] = example["game"]
                example["game"] = (
                    (worker_id - 1) * config.games_per_worker
                    + int(example["local_game"])
                )
            examples.extend(worker_result["examples"])
        replay.extend(examples)
        if len(replay) > config.replay_size:
            replay = replay[-config.replay_size :]
        total_positions += len(examples)

        train_start = time.perf_counter()
        train_history = train_from_replay(
            model=model,
            optimizer=optimizer,
            replay=replay,
            steps=config.train_steps_per_cycle,
            batch_size=config.batch_size,
            augment_dihedral=config.augment_dihedral,
        )
        train_seconds = time.perf_counter() - train_start
        total_train_steps += len(train_history)

        write_start = time.perf_counter()
        first_worker_config = make_selfplay_config(config, config.seed + cycle * 10_000 + 1)
        cycle_trace = build_trace(first_worker_config, examples)
        write_sgf(out_dir / "latest-cycle.sgf", first_worker_config, examples)
        write_json(out_dir / "latest-cycle-trace.json", cycle_trace)
        if cycle % 10 == 0:
            records_dir = out_dir / "cycle-records"
            write_sgf(records_dir / f"cycle-{cycle:05d}.sgf", first_worker_config, examples)
            write_json(records_dir / f"cycle-{cycle:05d}-trace.json", cycle_trace)
        write_seconds = time.perf_counter() - write_start
        total_seconds = time.perf_counter() - cycle_start

        metrics = summarize_cycle(
            cycle=cycle,
            total_seconds=total_seconds,
            selfplay_seconds=selfplay_seconds,
            train_seconds=train_seconds,
            write_seconds=write_seconds,
            examples=examples,
            train_history=train_history,
            replay_size=len(replay),
            total_positions=total_positions,
            total_train_steps=total_train_steps,
            worker_summaries=worker_results,
        )
        last_metrics = metrics
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
            handle.flush()

        checkpoint_config = to_overnight_config(config)
        save_checkpoint(
            out_dir / "latest.pt",
            checkpoint_config,
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
                checkpoint_config,
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
    config = MultiWorkerConfig(
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        simulations=args.simulations,
        komi=args.komi,
        score_komi=args.score_komi,
        input_komi=args.input_komi,
        terminal_dead_stone_cleanup=args.terminal_dead_stone_cleanup,
        score_margin_reward_scale=args.score_margin_reward_scale,
        workers=args.workers,
        games_per_worker=args.games_per_worker,
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
