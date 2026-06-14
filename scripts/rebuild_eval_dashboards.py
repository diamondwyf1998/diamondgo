from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamondgo import eval_checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild per-directory evaluation replay dashboards from saved "
            "results.json files. Each evaluation directory stays separate."
        )
    )
    parser.add_argument("paths", nargs="+", help="Evaluation dirs, results.json files, or parent dirs to scan.")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--recursive", action="store_true", help="Scan descendants for results.json.")
    return parser


def result_paths(paths: list[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name == "results.json":
            found.append(path)
        elif path.is_dir():
            direct = path / "results.json"
            if direct.exists():
                found.append(direct)
            if recursive:
                found.extend(sorted(path.rglob("results.json")))
        else:
            raise FileNotFoundError(path)
    unique = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def rebuild(path: Path, board_size: int) -> Path:
    results = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError(f"{path} must contain a JSON list")
    if not all(isinstance(item, dict) and "games_detail" in item for item in results):
        raise ValueError(f"{path} does not contain saved games_detail entries")
    out = path.parent / "games_dashboard.html"
    out.write_text(eval_checkpoints.render_eval_dashboard(results, board_size), encoding="utf-8")
    return out


def main() -> None:
    args = build_parser().parse_args()
    paths = result_paths(args.paths, args.recursive)
    if not paths:
        raise RuntimeError("no results.json files found")
    for path in paths:
        out = rebuild(path, args.board_size)
        print(out)


if __name__ == "__main__":
    main()
