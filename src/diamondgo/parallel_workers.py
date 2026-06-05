from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch multiple DiamondGo self-play workers.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games-per-worker", type=int, default=8)
    parser.add_argument("--max-moves", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rules", choices=["simple", "sgfmill"], default="sgfmill")
    parser.add_argument("--out-dir", default="artifacts/parallel-workers")
    parser.add_argument("--json", action="store_true")
    return parser


def worker_command(args: argparse.Namespace, worker_id: int, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "diamondgo.batched_demo",
        "--json",
        "--rules",
        args.rules,
        "--device",
        args.device,
        "--games",
        str(args.games_per_worker),
        "--max-moves",
        str(args.max_moves),
        "--simulations",
        str(args.simulations),
        "--train-steps",
        str(args.train_steps),
        "--batch-size",
        str(args.batch_size),
        "--channels",
        str(args.channels),
        "--residual-blocks",
        str(args.residual_blocks),
        "--sgf",
        str(out_dir / f"worker-{worker_id}.sgf"),
        "--trace",
        str(out_dir / f"worker-{worker_id}.json"),
        "--dashboard",
        str(out_dir / f"worker-{worker_id}.html"),
        "--overview-svg",
        str(out_dir / f"worker-{worker_id}.svg"),
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "src")

    start = time.perf_counter()
    processes = []
    for worker_id in range(1, args.workers + 1):
        log_path = out_dir / f"worker-{worker_id}.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(worker_command(args, worker_id, out_dir), stdout=log_file, stderr=subprocess.STDOUT, env=env)
        processes.append((worker_id, process, log_file, log_path))

    worker_summaries = []
    for worker_id, process, log_file, log_path in processes:
        return_code = process.wait()
        log_file.close()
        text = log_path.read_text(encoding="utf-8")
        summary = json.loads(text)
        summary["worker_id"] = worker_id
        summary["return_code"] = return_code
        worker_summaries.append(summary)

    wall_seconds = time.perf_counter() - start
    total_positions = sum(int(item["positions"]) for item in worker_summaries)
    network_seconds = sum(float(item["timing"].get("network_seconds", 0.0)) for item in worker_summaries)
    network_calls = sum(int(item["timing"].get("network_calls", 0)) for item in worker_summaries)
    aggregate = {
        "workers": args.workers,
        "games_per_worker": args.games_per_worker,
        "total_positions": total_positions,
        "wall_seconds": round(wall_seconds, 3),
        "positions_per_second": round(total_positions / max(wall_seconds, 1e-9), 3),
        "summed_network_seconds": round(network_seconds, 3),
        "summed_network_calls": network_calls,
        "worker_positions_per_second": [
            item["timing"]["positions_per_second"] for item in worker_summaries
        ],
        "workers_detail": worker_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
