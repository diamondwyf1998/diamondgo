from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_cycles(value: str) -> list[int]:
    cycles: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            cycles.extend(range(start, end + step, step))
        else:
            cycles.append(int(part))
    return sorted(dict.fromkeys(cycles))


def keep_game(value: Any, games: set[int]) -> bool:
    try:
        return int(value) in games
    except (TypeError, ValueError):
        return False


def filter_by_game(value: Any, games: set[int]) -> Any:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if keep_game(key, games)}
    if isinstance(value, list):
        filtered = []
        for item in value:
            if isinstance(item, dict) and keep_game(item.get("game"), games):
                filtered.append(item)
        return filtered
    return value


def build_index(out_dir: Path, summary: list[dict[str, Any]], title: str) -> None:
    dataset = out_dir.name
    links = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;line-height:1.55;background:#f6f4ed;color:#172026}",
        "a{display:block;margin:8px 0;color:#0b7285;font-weight:650}",
        ".note{color:#52606d;max-width:820px}",
        "</style>",
        f"<h1>{html.escape(title)}</h1>",
        "<p class='note'>Recorded training self-play subset. Each listed cycle keeps a bounded set of games so the shared DiamondGo viewer can load them quickly while preserving saved MCTS/root judgement fields.</p>",
    ]
    for item in summary:
        links.append(
            "<a href='../viewers/selfplay-catalog-viewer.html?"
            f"dataset={html.escape(dataset)}&cycle={int(item['cycle'])}&game=1'>"
            f"cycle-{int(item['cycle']):05d}: {int(item['games'])} games, "
            f"{int(item['positions'])} moves</a>"
        )
    links.append('<script src="/viewers/return-nav.js" defer></script>')
    out_dir.joinpath("index.html").write_text("\n".join(links), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small viewer dataset from recorded self-play traces.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing cycle-records/.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cycles", required=True, help="Comma-separated cycles or ranges.")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--title", default="DiamondGo recorded self-play subset")
    args = parser.parse_args()

    source_dir = args.source_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_games = set(range(1, max(1, int(args.games)) + 1))
    summary: list[dict[str, Any]] = []

    for cycle in parse_cycles(args.cycles):
        source_trace = source_dir / "cycle-records" / f"cycle-{cycle:05d}-trace.json"
        if not source_trace.exists():
            raise FileNotFoundError(source_trace)
        trace = json.loads(source_trace.read_text(encoding="utf-8"))
        moves = [
            move
            for move in trace.get("moves", [])
            if isinstance(move, dict) and keep_game(move.get("game"), wanted_games)
        ]
        if not moves:
            raise ValueError(f"{source_trace} has no selected games")
        trace["moves"] = moves
        if "game_summaries" in trace:
            trace["game_summaries"] = filter_by_game(trace.get("game_summaries"), wanted_games)
        if "initial_stones_by_game" in trace:
            trace["initial_stones_by_game"] = filter_by_game(trace.get("initial_stones_by_game"), wanted_games)
        trace["subset"] = {
            "source_trace": source_trace.as_posix(),
            "selected_games": sorted(wanted_games),
            "selected_cycle": cycle,
        }
        dest = out_dir / f"cycle-{cycle:05d}-moves.json"
        dest.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        actual_games = sorted(
            {
                int(move["game"])
                for move in moves
                if isinstance(move, dict) and keep_game(move.get("game"), wanted_games)
            }
        )
        config = trace.get("config") if isinstance(trace.get("config"), dict) else {}
        summary.append(
            {
                "cycle": cycle,
                "label": f"cycle-{cycle:05d} recorded games 1-{max(actual_games)}",
                "trace": dest.name,
                "source_trace": source_trace.as_posix(),
                "games": len(actual_games),
                "selected_games": actual_games,
                "positions": len(moves),
                "board_size": config.get("board_size"),
                "config": config,
            }
        )
        print(f"[subset] cycle-{cycle:05d}: games={len(actual_games)} moves={len(moves)}")

    out_dir.joinpath("summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_index(out_dir, summary, args.title)
    print(f"[subset] wrote {out_dir}")


if __name__ == "__main__":
    main()
