from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamondgo.batched_demo import BatchedConfig, make_model, play_batched_games
from diamondgo.demo_cpu import build_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate replayable self-play traces from checkpoints.")
    parser.add_argument("--manifest", required=True, help="JSON list with cycle, label, checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--root-noise-fraction", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=606)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest must be a non-empty JSON list")
    items = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("manifest entries must be objects")
        if "cycle" not in item or "checkpoint" not in item:
            raise ValueError("manifest entries need cycle and checkpoint")
        items.append(dict(item))
    return items


def make_config(raw: dict[str, Any], args: argparse.Namespace, cycle: int) -> BatchedConfig:
    values = {
        "board_size": int(raw.get("board_size", 9)),
        "komi": float(raw.get("komi", 0.5)),
        "score_komi": float(raw.get("score_komi", raw.get("komi", 0.5))),
        "input_komi": bool(raw.get("input_komi", False)),
        "channels": int(raw.get("channels", 64)),
        "residual_blocks": int(raw.get("residual_blocks", 4)),
        "simulations": int(args.simulations),
        "max_moves": int(raw.get("max_moves", 120)),
        "games": int(args.games),
        "train_steps": 0,
        "batch_size": 256,
        "c_puct": float(raw.get("c_puct", 1.5)),
        "temperature": float(args.temperature),
        "temperature_moves": 0,
        "late_temperature": float(args.temperature),
        "root_dirichlet_alpha": float(args.root_dirichlet_alpha),
        "root_noise_fraction": float(args.root_noise_fraction),
        "root_policy_temperature": float(raw.get("root_policy_temperature", 1.0)),
        "seed": int(cycle * 1000 + args.seed_base),
        "device": str(args.device),
        "rules_backend": str(raw.get("rules_backend", "sgfmill")),
    }
    allowed = {field.name for field in fields(BatchedConfig)}
    return BatchedConfig(**{key: value for key, value in values.items() if key in allowed})


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for item in load_manifest(Path(args.manifest)):
        cycle = int(item["cycle"])
        label = str(item.get("label") or f"cycle-{cycle:05d}")
        checkpoint = Path(str(item["checkpoint"]))
        trace_path = out_dir / f"cycle-{cycle:05d}-moves.json"
        start = time.perf_counter()
        print(f"[showcase] loading {label}: {checkpoint}", flush=True)
        payload = torch.load(checkpoint, map_location=torch.device(args.device))
        config = make_config(dict(payload["config"]), args, cycle)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        model = make_model(config)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        examples, stats = play_batched_games(config, model)
        trace = build_trace(config, examples)
        trace["checkpoint"] = str(checkpoint)
        trace["checkpoint_label"] = label
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        summary.append(
            {
                "cycle": cycle,
                "label": label,
                "checkpoint": str(checkpoint),
                "trace": trace_path.name,
                "config": asdict(config),
                "positions": len(examples),
                "games": config.games,
                "simulations": config.simulations,
                "game_summaries": stats.get("game_summaries", []),
                "seconds": round(time.perf_counter() - start, 3),
            }
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[showcase] wrote {trace_path.name} positions={len(examples)}", flush=True)

    dataset = out_dir.name
    links = [
        "<!doctype html><meta charset='utf-8'><title>DiamondGo self-play showcase</title>",
        "<style>body{font-family:system-ui;margin:24px;line-height:1.55}a{display:block;margin:8px 0}</style>",
        "<h1>DiamondGo self-play showcase</h1>",
    ]
    for item in summary:
        links.append(
            "<a href='../viewers/selfplay-catalog-viewer.html?"
            f"dataset={dataset}&cycle={item['cycle']}&game=1'>"
            f"{item['label']} ({item['positions']} moves)</a>"
        )
    (out_dir / "index.html").write_text("\n".join(links), encoding="utf-8")
    print(f"[showcase] done {out_dir}", flush=True)


if __name__ == "__main__":
    main()
