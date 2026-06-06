from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diamondgo.batched_demo import make_model, play_batched_games
from diamondgo.demo_cpu import build_trace, write_json, write_sgf
from diamondgo.multiworker_train import (
    MultiWorkerConfig,
    build_parser as build_multiworker_parser,
    cpu_state_dict,
    make_selfplay_config,
    summarize_cycle,
    to_overnight_config,
)
from diamondgo.overnight_train import load_checkpoint, save_checkpoint, train_from_replay


@dataclass(frozen=True)
class DualGpuConfig(MultiWorkerConfig):
    selfplay_devices: str = ""


def base_config(config: DualGpuConfig) -> MultiWorkerConfig:
    base_fields = {item.name for item in fields(MultiWorkerConfig)}
    return MultiWorkerConfig(**{key: value for key, value in asdict(config).items() if key in base_fields})


def parse_selfplay_devices(config: DualGpuConfig) -> list[str]:
    if config.selfplay_devices.strip():
        return [item.strip() for item in config.selfplay_devices.split(",") if item.strip()]
    if config.device.startswith("cuda") and torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 1:
            return [f"cuda:{index}" for index in range(count)]
    return [config.device]


def worker_selfplay_on_device(
    worker_id: int,
    config_dict: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    seed: int,
    device: str,
) -> dict[str, Any]:
    config = replace(MultiWorkerConfig(**config_dict), device=device)
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
        "device": device,
        "positions": len(examples),
        "seconds": round(elapsed, 3),
        "positions_per_second": round(len(examples) / max(elapsed, 1e-9), 3),
        "examples": examples,
        "selfplay": {
            key: value for key, value in stats.items() if key != "batch_sizes"
        },
    }


def run(config: DualGpuConfig, out_dir: Path, resume: str = "") -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but CUDA is unavailable")

    base = base_config(config)
    selfplay_devices = parse_selfplay_devices(config)
    model = make_model(make_selfplay_config(base, config.seed))
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
    base_config_dict = asdict(base)

    for cycle in range(start_cycle + 1, config.cycles + 1):
        if config.time_limit_minutes > 0 and (time.perf_counter() - started) >= config.time_limit_minutes * 60:
            break

        cycle_start = time.perf_counter()
        state_dict = cpu_state_dict(model)
        selfplay_start = time.perf_counter()
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=context,
        ) as executor:
            futures = []
            for worker_id in range(1, config.workers + 1):
                device = selfplay_devices[(worker_id - 1) % len(selfplay_devices)]
                futures.append(
                    executor.submit(
                        worker_selfplay_on_device,
                        worker_id,
                        base_config_dict,
                        state_dict,
                        config.seed + cycle * 10_000 + worker_id,
                        device,
                    )
                )
            worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        worker_results.sort(key=lambda item: item["worker_id"])
        selfplay_seconds = time.perf_counter() - selfplay_start

        examples: list[dict[str, object]] = []
        for worker_result in worker_results:
            worker_id = int(worker_result["worker_id"])
            for summary in worker_result["selfplay"].get("game_summaries", []):
                summary["worker_id"] = worker_id
                summary["device"] = worker_result["device"]
                summary["local_game"] = int(summary["game"])
                summary["game"] = (worker_id - 1) * config.games_per_worker + int(summary["local_game"])
            for example in worker_result["examples"]:
                example["worker_id"] = worker_id
                example["device"] = worker_result["device"]
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
        first_worker_config = make_selfplay_config(base, config.seed + cycle * 10_000 + 1)
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
        device_by_worker = {int(item["worker_id"]): item["device"] for item in worker_results}
        for worker in metrics.get("workers", []):
            worker["device"] = device_by_worker.get(int(worker["worker_id"]), "")
        metrics["selfplay_devices"] = selfplay_devices
        last_metrics = metrics
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
            handle.flush()

        checkpoint_config = to_overnight_config(base)
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
        if cycle % max(1, config.checkpoint_every) == 0:
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
        "selfplay_devices": selfplay_devices,
        "total_positions": total_positions,
        "total_train_steps": total_train_steps,
        "latest_metrics": last_metrics,
        "checkpoint": str(out_dir / "latest.pt"),
        "metrics_path": str(metrics_path),
        "cycle_records_dir": str(out_dir / "cycle-records"),
    }


def build_parser():
    parser = build_multiworker_parser()
    parser.description = "Run multi-worker 9x9 self-play with self-play workers spread across GPUs."
    parser.add_argument(
        "--selfplay-devices",
        default=DualGpuConfig.selfplay_devices,
        help="Comma-separated devices for self-play workers, e.g. cuda:0,cuda:1. Defaults to all CUDA devices.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DualGpuConfig(
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
        selfplay_devices=args.selfplay_devices,
    )
    summary = run(config, Path(args.out_dir), resume=args.resume)
    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
