from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamondgo import eval_checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all-pairs checkpoint matches and write a matrix dashboard.")
    parser.add_argument("--manifest", required=True, help="JSON list with cycle, label, checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--max-moves", type=int, default=150)
    parser.add_argument("--filter-early-double-pass", type=int, default=0)
    parser.add_argument("--valid-per-color", type=int, default=5)
    parser.add_argument("--max-attempt-batches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("manifest must contain at least two checkpoints")
    items = []
    for item in raw:
        if not isinstance(item, dict) or "cycle" not in item or "checkpoint" not in item:
            raise ValueError("each manifest item needs cycle and checkpoint")
        copied = dict(item)
        copied["cycle"] = int(copied["cycle"])
        copied["label"] = str(copied.get("label") or f"cycle-{copied['cycle']:05d}")
        copied["checkpoint"] = str(copied["checkpoint"])
        items.append(copied)
    return sorted(items, key=lambda item: int(item["cycle"]))


def is_early_double_pass(game: dict[str, Any], threshold: int) -> bool:
    if threshold <= 0:
        return False
    moves = list(game.get("moves", []))
    if len(moves) < 2:
        return False
    return (
        str(moves[-1].get("move")) == "pass"
        and str(moves[-2].get("move")) == "pass"
        and int(game.get("moves_played", len(moves))) <= threshold
    )


def renumber_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = []
    for index, game in enumerate(games, start=1):
        item = dict(game)
        item["game"] = index
        copied.append(item)
    return copied


def summarize_games(
    *,
    candidate_name: str,
    opponent_name: str,
    games: list[dict[str, Any]],
    seconds: float,
    rejected_count: int,
    attempt_batches: int,
) -> dict[str, Any]:
    games = renumber_games(games)
    wins = sum(1 for game in games if bool(game["candidate_won"]))
    black_wins = sum(
        1
        for game in games
        if bool(game["candidate_won"]) and str(game["candidate_color"]) == "b"
    )
    white_wins = sum(
        1
        for game in games
        if bool(game["candidate_won"]) and str(game["candidate_color"]) == "w"
    )
    return {
        "candidate": candidate_name,
        "opponent": opponent_name,
        "games": len(games),
        "candidate_wins": wins,
        "candidate_losses": len(games) - wins,
        "win_rate": round(wins / max(len(games), 1), 4),
        "candidate_black_wins": black_wins,
        "candidate_white_wins": white_wins,
        "candidate_black_games": sum(1 for game in games if str(game["candidate_color"]) == "b"),
        "candidate_white_games": sum(1 for game in games if str(game["candidate_color"]) == "w"),
        "seconds": round(seconds, 3),
        "games_per_second": round(len(games) / max(seconds, 1e-9), 3),
        "pass_behavior": eval_checkpoints.summarize_pass_behavior(games),
        "rejected_early_double_pass": rejected_count,
        "attempt_batches": attempt_batches,
        "games_detail": games,
    }


def play_pair(
    *,
    config: eval_checkpoints.MatchConfig,
    candidate_model: torch.nn.Module,
    opponent_model: torch.nn.Module,
    candidate_name: str,
    opponent_name: str,
    out_dir: Path,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = time.perf_counter()
    rejected = []
    if args.filter_early_double_pass <= 0:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        result = eval_checkpoints.play_match(
            replace(config, seed=seed),
            candidate_model,
            opponent_model,
            candidate_name,
            opponent_name,
            out_dir,
            sample_games=0,
        )
        return result, rejected

    accepted = {"b": [], "w": []}
    attempts = 0
    while (
        len(accepted["b"]) < args.valid_per_color
        or len(accepted["w"]) < args.valid_per_color
    ) and attempts < args.max_attempt_batches:
        attempts += 1
        batch_seed = seed + attempts
        random.seed(batch_seed)
        np.random.seed(batch_seed % (2**32 - 1))
        torch.manual_seed(batch_seed)
        batch = eval_checkpoints.play_match(
            replace(config, seed=batch_seed),
            candidate_model,
            opponent_model,
            candidate_name,
            opponent_name,
            out_dir,
            sample_games=0,
        )
        for game in batch["games_detail"]:
            color = str(game["candidate_color"])
            copied = dict(game)
            copied["attempt_batch"] = attempts
            copied["early_double_pass"] = is_early_double_pass(game, args.filter_early_double_pass)
            if copied["early_double_pass"]:
                rejected.append(copied)
                continue
            if len(accepted[color]) < args.valid_per_color:
                accepted[color].append(copied)

    if len(accepted["b"]) < args.valid_per_color or len(accepted["w"]) < args.valid_per_color:
        raise RuntimeError(
            f"not enough valid games for {candidate_name} vs {opponent_name}: "
            f"b={len(accepted['b'])}, w={len(accepted['w'])}, rejected={len(rejected)}"
        )
    result = summarize_games(
        candidate_name=candidate_name,
        opponent_name=opponent_name,
        games=accepted["b"] + accepted["w"],
        seconds=time.perf_counter() - start,
        rejected_count=len(rejected),
        attempt_batches=attempts,
    )
    return result, rejected


def matrix_summary(
    *,
    items: list[dict[str, Any]],
    results: list[dict[str, Any]],
    config: eval_checkpoints.MatchConfig,
    args: argparse.Namespace,
    rejected_games: list[dict[str, Any]],
) -> dict[str, Any]:
    cycles = [int(item["cycle"]) for item in items]
    win_counts: dict[str, dict[str, object]] = {
        str(cycle): {str(other): None for other in cycles} for cycle in cycles
    }
    win_rates: dict[str, dict[str, float | None]] = {
        str(cycle): {str(other): None for other in cycles} for cycle in cycles
    }
    for cycle in cycles:
        win_counts[str(cycle)][str(cycle)] = "-"
    for result in results:
        a = int(result["candidate_cycle"])
        b = int(result["opponent_cycle"])
        wins = int(result["candidate_wins"])
        losses = int(result["candidate_losses"])
        games = int(result["games"])
        win_counts[str(a)][str(b)] = wins
        win_counts[str(b)][str(a)] = losses
        win_rates[str(a)][str(b)] = round(wins / games, 4)
        win_rates[str(b)][str(a)] = round(losses / games, 4)

    aggregate = []
    games_per_pair = int(results[0]["games"]) if results else args.games
    for cycle in cycles:
        wins = sum(int(win_counts[str(cycle)][str(other)]) for other in cycles if other != cycle)
        games = games_per_pair * (len(cycles) - 1)
        aggregate.append({"cycle": cycle, "wins": wins, "games": games, "win_rate": round(wins / games, 4)})
    aggregate.sort(key=lambda item: (-item["win_rate"], -item["wins"], item["cycle"]))
    return {
        "config": asdict(config),
        "cycles": cycles,
        "simulations": args.simulations,
        "games_per_pair": games_per_pair,
        "filter_early_double_pass": args.filter_early_double_pass,
        "total_games": games_per_pair * len(results),
        "total_rejected_games": len(rejected_games),
        "results": [{key: value for key, value in result.items() if key != "games_detail"} for result in results],
        "win_counts": win_counts,
        "win_rates": win_rates,
        "aggregate": aggregate,
    }


def render_matrix(summary: dict[str, Any]) -> str:
    cycles = [int(cycle) for cycle in summary["cycles"]]
    games_per_pair = int(summary["games_per_pair"])
    headers = "".join(f"<th>{cycle}</th>" for cycle in cycles)
    rows = []
    for row_cycle in cycles:
        cells = []
        for col_cycle in cycles:
            if row_cycle == col_cycle:
                cells.append("<td class='diag'>-</td>")
                continue
            rate = float(summary["win_rates"][str(row_cycle)][str(col_cycle)])
            count = int(summary["win_counts"][str(row_cycle)][str(col_cycle)])
            css = "win" if rate > 0.5 else "loss" if rate < 0.5 else "even"
            cells.append(f"<td class='{css}'>{count}/{games_per_pair}<br><small>{rate*100:.0f}%</small></td>")
        rows.append(f"<tr><th>{row_cycle}</th>{''.join(cells)}</tr>")
    rank_rows = "".join(
        f"<tr><td>{index+1}</td><td>{item['cycle']}</td><td>{item['wins']}/{item['games']}</td><td>{item['win_rate']*100:.1f}%</td></tr>"
        for index, item in enumerate(summary["aggregate"])
    )
    detail_rows = "".join(
        f"<tr><td>{item['candidate_cycle']}</td><td>{item['opponent_cycle']}</td><td>{item['candidate_wins']}/{item['games']}</td><td>{item['candidate_black_wins']}/{item['candidate_black_games']}</td><td>{item['candidate_white_wins']}/{item['candidate_white_games']}</td><td>{item.get('rejected_early_double_pass', 0)}</td><td>{item['win_rate']*100:.1f}%</td></tr>"
        for item in summary["results"]
    )
    return f"""<!doctype html><meta charset='utf-8'><title>DiamondGo pairwise matrix</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#f7f5ef;color:#172026}}table{{border-collapse:collapse;background:#fff;margin:16px 0;box-shadow:0 0 0 1px #ddd}}th,td{{border:1px solid #ddd;padding:8px 10px;text-align:center;vertical-align:middle}}th{{background:#f0ece2}}.win{{background:#d9f2df}}.loss{{background:#f7d8d8}}.even{{background:#fff5cc}}.diag{{background:#eee;color:#777}}small,.note{{color:#52606d}}.note{{max-width:980px;line-height:1.55}}</style>
<h1>DiamondGo pairwise matrix</h1>
<p class='note'>Games per pair: {summary['games_per_pair']}. Simulations: {summary['simulations']}. Early double-pass filter: {summary['filter_early_double_pass']}. Rejected games: {summary['total_rejected_games']}.</p>
<p><a href='games_dashboard.html'>Open game replay dashboard</a></p>
<h2>Win Matrix</h2><table><thead><tr><th>row vs col</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Aggregate</h2><table><thead><tr><th>rank</th><th>cycle</th><th>wins</th><th>win rate</th></tr></thead><tbody>{rank_rows}</tbody></table>
<h2>Pair Details</h2><table><thead><tr><th>candidate</th><th>opponent</th><th>wins</th><th>black wins</th><th>white wins</th><th>rejected</th><th>win rate</th></tr></thead><tbody>{detail_rows}</tbody></table>
<pre id='raw'>{json.dumps(summary, indent=2)}</pre>"""


def main() -> None:
    args = build_parser().parse_args()
    items = load_manifest(Path(args.manifest))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first_payload = eval_checkpoints.load_checkpoint_payload(Path(items[0]["checkpoint"]), args.device)
    config = eval_checkpoints.config_from_payload(first_payload, args.device, args.simulations, args.games, args.max_moves)
    if args.filter_early_double_pass > 0:
        config = replace(config, games=args.valid_per_color * 2)
    models = {
        int(item["cycle"]): eval_checkpoints.make_checkpoint_model(config, Path(item["checkpoint"]))
        for item in items
    }
    results = []
    rejected_games = []
    started = time.perf_counter()
    for i, candidate in enumerate(items):
        for opponent in items[i + 1 :]:
            a = int(candidate["cycle"])
            b = int(opponent["cycle"])
            name_a = str(candidate["label"])
            name_b = str(opponent["label"])
            seed = args.seed + a * 100000 + b
            print(f"[pairwise] {name_a} vs {name_b}", flush=True)
            result, rejected = play_pair(
                config=config,
                candidate_model=models[a],
                opponent_model=models[b],
                candidate_name=name_a,
                opponent_name=name_b,
                out_dir=out_dir / "sample-games",
                args=args,
                seed=seed,
            )
            result["candidate_cycle"] = a
            result["opponent_cycle"] = b
            result["candidate_checkpoint"] = candidate["checkpoint"]
            result["opponent_checkpoint"] = opponent["checkpoint"]
            results.append(result)
            rejected_games.extend(rejected)
            (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            (out_dir / "rejected_games.json").write_text(json.dumps(rejected_games, indent=2), encoding="utf-8")
            (out_dir / "games_dashboard.html").write_text(eval_checkpoints.render_eval_dashboard(results, config.board_size), encoding="utf-8")
            summary = matrix_summary(items=items, results=results, config=config, args=args, rejected_games=rejected_games)
            summary["elapsed_seconds_so_far"] = round(time.perf_counter() - started, 3)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (out_dir / "dashboard.html").write_text(render_matrix(summary), encoding="utf-8")
            print(f"[pairwise] {name_a} wins {result['candidate_wins']}/{result['games']}", flush=True)


if __name__ == "__main__":
    main()
