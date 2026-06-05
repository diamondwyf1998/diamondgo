from __future__ import annotations

import argparse
import json
from pathlib import Path

from diamondgo import eval_checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a standard DiamondGo evaluation suite.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", default="50,200,500")
    parser.add_argument("--opponents", default="initial,previous")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--max-moves", type=int, default=120)
    parser.add_argument("--sample-games", type=int, default=2)
    parser.add_argument("--include-latest", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true")
    return parser


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one evaluation step")
    return sorted(set(values))


def parse_opponents(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"initial", "previous"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown opponents: {', '.join(unknown)}")
    if not values:
        raise ValueError("expected at least one opponent")
    return values


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = parse_int_list(args.steps)
    opponents = parse_opponents(args.opponents)
    runs = []
    for step in steps:
        for opponent in opponents:
            tier_out = out_dir / f"step-{step:05d}-vs-{opponent}"
            eval_args = argparse.Namespace(
                checkpoint_dir=args.checkpoint_dir,
                out_dir=str(tier_out),
                step=step,
                games=args.games,
                simulations=args.simulations,
                max_moves=args.max_moves,
                sample_games=args.sample_games,
                opponent=opponent,
                include_latest=args.include_latest,
                device=args.device,
                json=True,
            )
            summary = eval_checkpoints.run(eval_args)
            runs.append(
                {
                    "step": step,
                    "opponent": opponent,
                    "out_dir": str(tier_out),
                    "results_path": summary["results_path"],
                    "report_path": summary["report_path"],
                    "dashboard_path": summary["dashboard_path"],
                    "matches": len(summary["results"]),
                }
            )
            (out_dir / "suite_summary.json").write_text(
                json.dumps({"runs": runs}, indent=2),
                encoding="utf-8",
            )
    summary = {"steps": steps, "opponents": opponents, "runs": runs}
    (out_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
