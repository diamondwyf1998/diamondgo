from __future__ import annotations

import argparse
import json
import random
import re
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
from diamondgo.batched_demo import BatchedConfig, run_batched_mcts
from diamondgo.config import ModelConfig, input_plane_count
from diamondgo.demo_cpu import action_to_gtp, make_rules
from diamondgo.model import PolicyValueNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run reusable cross-run checkpoint matches. Pairing is explicit: "
            "use --pair, --pairs-json, --match, or opt into --same-cycles."
        )
    )
    parser.add_argument("--candidate-dir", default="", help="Run directory containing candidate checkpoints/.")
    parser.add_argument("--opponent-dir", default="", help="Run directory containing opponent checkpoints/.")
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Cycle mapping CANDIDATE_CYCLE:OPPONENT_CYCLE, for use with --candidate-dir and --opponent-dir.",
    )
    parser.add_argument(
        "--same-cycles",
        action="store_true",
        help="With --candidate-dir/--opponent-dir, match cycles that exist in both dirs.",
    )
    parser.add_argument(
        "--cycles",
        default="",
        help="Optional comma-separated candidate cycle filter for --same-cycles.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Explicit pair CANDIDATE_PT:OPPONENT_PT[:CANDIDATE_LABEL[:OPPONENT_LABEL]].",
    )
    parser.add_argument(
        "--pairs-json",
        default="",
        help="JSON list of {candidate, opponent, candidate_label, opponent_label, id}.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--candidate-sims", type=int, default=100)
    parser.add_argument("--opponent-sims", type=int, default=100)
    parser.add_argument("--max-moves", type=int, default=150)
    parser.add_argument("--sample-games", type=int, default=2)
    parser.add_argument("--opening-temperature-moves", type=int, default=6)
    parser.add_argument(
        "--rules-from",
        choices=["candidate", "opponent"],
        default="candidate",
        help="Which checkpoint config supplies board/rules/scoring environment.",
    )
    parser.add_argument("--score-komi", type=float, default=None, help="Override match scoring komi.")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def cycle_path(run_dir: Path, cycle: int) -> Path:
    return run_dir / "checkpoints" / f"cycle-{cycle:05d}.pt"


def parse_cycle(text: str) -> int:
    return int(text.strip().removeprefix("cycle-"))


def label_for(path: Path, fallback_prefix: str) -> str:
    match = re.search(r"cycle-(\d+)\.pt$", path.name)
    if match:
        return f"{fallback_prefix}-cycle-{int(match.group(1)):05d}"
    return f"{fallback_prefix}-{path.stem}"


def load_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    if args.pairs_json:
        raw = json.loads(Path(args.pairs_json).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("--pairs-json must contain a JSON list")
        for index, item in enumerate(raw, start=1):
            pairs.append(
                {
                    "id": str(item.get("id") or f"pair-{index:03d}"),
                    "candidate": str(item["candidate"]),
                    "opponent": str(item["opponent"]),
                    "candidate_label": str(item.get("candidate_label") or label_for(Path(item["candidate"]), "candidate")),
                    "opponent_label": str(item.get("opponent_label") or label_for(Path(item["opponent"]), "opponent")),
                }
            )

    for index, spec in enumerate(args.pair, start=1):
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(f"bad --pair spec: {spec}")
        candidate = Path(parts[0])
        opponent = Path(parts[1])
        pairs.append(
            {
                "id": f"explicit-{index:03d}",
                "candidate": str(candidate),
                "opponent": str(opponent),
                "candidate_label": parts[2] if len(parts) > 2 else label_for(candidate, "candidate"),
                "opponent_label": parts[3] if len(parts) > 3 else label_for(opponent, "opponent"),
            }
        )

    if args.match or args.same_cycles:
        if not args.candidate_dir or not args.opponent_dir:
            raise ValueError("--match/--same-cycles require --candidate-dir and --opponent-dir")
        candidate_dir = Path(args.candidate_dir)
        opponent_dir = Path(args.opponent_dir)
        cycle_pairs: list[tuple[int, int]] = []
        if args.match:
            for spec in args.match:
                left, right = spec.split(":", 1)
                cycle_pairs.append((parse_cycle(left), parse_cycle(right)))
        else:
            requested = {parse_cycle(item) for item in args.cycles.split(",") if item.strip()}
            candidate_cycles = {
                eval_checkpoints.cycle_from_path(path)
                for path in (candidate_dir / "checkpoints").glob("cycle-*.pt")
            }
            opponent_cycles = {
                eval_checkpoints.cycle_from_path(path)
                for path in (opponent_dir / "checkpoints").glob("cycle-*.pt")
            }
            common = sorted(candidate_cycles & opponent_cycles)
            cycle_pairs = [(cycle, cycle) for cycle in common if not requested or cycle in requested]
        for candidate_cycle, opponent_cycle in cycle_pairs:
            candidate = cycle_path(candidate_dir, candidate_cycle)
            opponent = cycle_path(opponent_dir, opponent_cycle)
            pairs.append(
                {
                    "id": f"cycle-{candidate_cycle:05d}-vs-{opponent_cycle:05d}",
                    "candidate": str(candidate),
                    "opponent": str(opponent),
                    "candidate_label": f"candidate-cycle-{candidate_cycle:05d}",
                    "opponent_label": f"opponent-cycle-{opponent_cycle:05d}",
                    "candidate_cycle": candidate_cycle,
                    "opponent_cycle": opponent_cycle,
                }
            )

    if not pairs:
        raise ValueError("no pairs specified; use --pair, --pairs-json, --match, or --same-cycles")
    for pair in pairs:
        for key in ("candidate", "opponent"):
            path = Path(pair[key])
            if not path.exists():
                raise FileNotFoundError(path)
    return pairs


def model_from_checkpoint(path: Path, config: eval_checkpoints.MatchConfig) -> PolicyValueNet:
    model = PolicyValueNet(
        board_size=config.board_size,
        config=ModelConfig(channels=config.channels, residual_blocks=config.residual_blocks),
        input_planes=input_plane_count(config.input_komi, config.history_moves),
    )
    model.to(torch.device(config.device))
    model.eval()
    payload = eval_checkpoints.load_checkpoint_payload(path, config.device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def batched_config(config: eval_checkpoints.MatchConfig, active_games: int) -> BatchedConfig:
    return eval_checkpoints.batched_config(replace(config, games=active_games), active_games)


def play_cross_match(
    *,
    candidate_model: torch.nn.Module,
    opponent_model: torch.nn.Module,
    candidate_config: eval_checkpoints.MatchConfig,
    opponent_config: eval_checkpoints.MatchConfig,
    state_config: eval_checkpoints.MatchConfig,
    candidate_name: str,
    opponent_name: str,
    out_dir: Path,
    sample_games: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    states = [make_rules(batched_config(state_config, state_config.games)) for _ in range(state_config.games)]
    active = [True for _ in states]
    candidate_colors = ["b" if index % 2 == 0 else "w" for index in range(state_config.games)]
    move_counts = [0 for _ in states]
    game_records: list[dict[str, Any]] = [
        {"game": index + 1, "candidate_color": candidate_colors[index], "moves": []}
        for index in range(state_config.games)
    ]
    stats: dict[str, object] = {"candidate": {}, "opponent": {}}

    while any(active):
        active_indices = [
            index
            for index, state in enumerate(states)
            if active[index] and not state.is_terminal() and move_counts[index] < state_config.max_moves
        ]
        if not active_indices:
            break
        grouped: dict[str, list[int]] = {"candidate": [], "opponent": []}
        for index in active_indices:
            key = "candidate" if states[index].to_play == candidate_colors[index] else "opponent"
            grouped[key].append(index)

        roots_by_index = {}
        for key, indices in grouped.items():
            if not indices:
                continue
            model = candidate_model if key == "candidate" else opponent_model
            config = candidate_config if key == "candidate" else opponent_config
            group_states = [states[index] for index in indices]
            roots = run_batched_mcts(model, group_states, batched_config(config, len(indices)), stats[key])
            roots_by_index.update(zip(indices, roots))

        for index in active_indices:
            state = states[index]
            root = roots_by_index[index]
            model_name = "candidate" if state.to_play == candidate_colors[index] else "opponent"
            action = eval_checkpoints.choose_action(
                root,
                state.action_size,
                move_counts[index] + 1,
                state_config.opening_temperature_moves,
            )
            game_records[index]["moves"].append(
                {
                    "move_number": move_counts[index] + 1,
                    "player": state.to_play,
                    "model": model_name,
                    "action": int(action),
                    "move": action_to_gtp(int(action), state_config.board_size),
                    "root_value": round(root.value, 4),
                    "top_actions": root.top_actions(state_config.board_size, limit=None),
                }
            )
            state.play_action(action)
            move_counts[index] += 1
            if state.is_terminal() or move_counts[index] >= state_config.max_moves:
                active[index] = False

    candidate_wins = 0
    candidate_black_wins = 0
    candidate_white_wins = 0
    for index, state in enumerate(states):
        value_for_to_play = float(state.terminal_value())
        winner = state.to_play if value_for_to_play > 0 else eval_checkpoints.other_player(state.to_play)
        candidate_won = winner == candidate_colors[index]
        cleanup = state.terminal_cleanup_counts() if hasattr(state, "terminal_cleanup_counts") else {"b": 0, "w": 0}
        candidate_wins += int(candidate_won)
        candidate_black_wins += int(candidate_won and candidate_colors[index] == "b")
        candidate_white_wins += int(candidate_won and candidate_colors[index] == "w")
        game_records[index]["winner"] = winner
        game_records[index]["candidate_won"] = candidate_won
        game_records[index]["moves_played"] = move_counts[index]
        game_records[index]["black_score_margin"] = round(float(state.terminal_score_margin()), 3)
        game_records[index]["terminal_cleanup_black_stones"] = int(cleanup.get("b", 0))
        game_records[index]["terminal_cleanup_white_stones"] = int(cleanup.get("w", 0))

    for game in game_records[:sample_games]:
        eval_checkpoints.write_match_sgf(
            out_dir / "sample-sgf" / f"{candidate_name}_vs_{opponent_name}_game-{game['game']:02d}.sgf",
            state_config,
            game,
            candidate_name,
            opponent_name,
        )

    elapsed = time.perf_counter() - start
    return {
        "candidate": candidate_name,
        "opponent": opponent_name,
        "games": state_config.games,
        "candidate_wins": candidate_wins,
        "candidate_losses": state_config.games - candidate_wins,
        "win_rate": round(candidate_wins / state_config.games, 4),
        "candidate_black_wins": candidate_black_wins,
        "candidate_white_wins": candidate_white_wins,
        "candidate_black_games": sum(1 for color in candidate_colors if color == "b"),
        "candidate_white_games": sum(1 for color in candidate_colors if color == "w"),
        "seconds": round(elapsed, 3),
        "games_per_second": round(state_config.games / max(elapsed, 1e-9), 3),
        "pass_behavior": eval_checkpoints.summarize_pass_behavior(game_records),
        "games_detail": game_records,
    }


def render_summary(results: list[dict[str, Any]], config: eval_checkpoints.MatchConfig) -> str:
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{result['id']}</td><td>{result['candidate']}</td><td>{result['opponent']}</td>"
            f"<td>{result['candidate_wins']}/{result['games']}</td>"
            f"<td>{result['candidate_black_wins']}/{result['candidate_black_games']}</td>"
            f"<td>{result['candidate_white_wins']}/{result['candidate_white_games']}</td>"
            f"<td>{result['win_rate'] * 100:.1f}%</td>"
            f"<td>{result['pass_behavior'].get('early_first_pass_rate_40', '')}</td>"
            "</tr>"
        )
    return f"""<!doctype html><meta charset='utf-8'><title>DiamondGo cross-run matches</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#f7f5ef;color:#172026}}table{{border-collapse:collapse;background:#fff;margin:16px 0}}th,td{{border:1px solid #ddd;padding:8px 10px;text-align:center}}th{{background:#f0ece2}}.note{{color:#52606d;max-width:920px;line-height:1.55}}</style>
<h1>DiamondGo cross-run matches</h1>
<p class='note'>Games per pair: {results[0]['games'] if results else 0}. Environment: score komi {config.score_komi}, max moves {config.max_moves}, rules {config.rules_backend}. Candidate/opponent pairings are explicit; this is not necessarily same-cycle matching.</p>
<p><a href='games_dashboard.html'>Open game replay dashboard</a></p>
<table><thead><tr><th>pair</th><th>candidate</th><th>opponent</th><th>wins</th><th>black</th><th>white</th><th>win rate</th><th>early pass <=40</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"""


def main() -> None:
    args = build_parser().parse_args()
    if args.games % 2 != 0:
        raise ValueError("--games must be even so colors can split evenly")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args)
    results: list[dict[str, Any]] = []

    for index, pair in enumerate(pairs, start=1):
        candidate_path = Path(pair["candidate"])
        opponent_path = Path(pair["opponent"])
        candidate_payload = eval_checkpoints.load_checkpoint_payload(candidate_path, args.device)
        opponent_payload = eval_checkpoints.load_checkpoint_payload(opponent_path, args.device)
        candidate_config = eval_checkpoints.config_from_payload(
            candidate_payload, args.device, args.candidate_sims, args.games, args.max_moves
        )
        opponent_config = eval_checkpoints.config_from_payload(
            opponent_payload, args.device, args.opponent_sims, args.games, args.max_moves
        )
        base_config = candidate_config if args.rules_from == "candidate" else opponent_config
        if args.score_komi is not None:
            base_config = replace(base_config, score_komi=float(args.score_komi))
        state_config = replace(
            base_config,
            games=args.games,
            max_moves=args.max_moves,
            simulations=max(args.candidate_sims, args.opponent_sims),
            opening_temperature_moves=args.opening_temperature_moves,
            seed=args.seed + index,
        )
        candidate_config = replace(candidate_config, seed=args.seed + index)
        opponent_config = replace(opponent_config, seed=args.seed + index)
        random.seed(state_config.seed)
        np.random.seed(state_config.seed % (2**32 - 1))
        torch.manual_seed(state_config.seed)
        candidate_model = model_from_checkpoint(candidate_path, candidate_config)
        opponent_model = model_from_checkpoint(opponent_path, opponent_config)
        print(f"[cross] {pair['candidate_label']} vs {pair['opponent_label']}", flush=True)
        result = play_cross_match(
            candidate_model=candidate_model,
            opponent_model=opponent_model,
            candidate_config=candidate_config,
            opponent_config=opponent_config,
            state_config=state_config,
            candidate_name=str(pair["candidate_label"]),
            opponent_name=str(pair["opponent_label"]),
            out_dir=out_dir,
            sample_games=args.sample_games,
        )
        result["id"] = str(pair.get("id") or f"pair-{index:03d}")
        result["candidate_checkpoint"] = str(candidate_path)
        result["opponent_checkpoint"] = str(opponent_path)
        result["candidate_sims"] = args.candidate_sims
        result["opponent_sims"] = args.opponent_sims
        result["candidate_cycle"] = pair.get("candidate_cycle")
        result["opponent_cycle"] = pair.get("opponent_cycle")
        results.append(result)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out_dir / "games_dashboard.html").write_text(
            eval_checkpoints.render_eval_dashboard(results, state_config.board_size),
            encoding="utf-8",
        )
        summary = {
            "pairs": pairs,
            "config": asdict(state_config),
            "candidate_sims": args.candidate_sims,
            "opponent_sims": args.opponent_sims,
            "results": [{key: value for key, value in item.items() if key != "games_detail"} for item in results],
            "dashboard_path": str(out_dir / "dashboard.html"),
            "games_dashboard_path": str(out_dir / "games_dashboard.html"),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "dashboard.html").write_text(render_summary(results, state_config), encoding="utf-8")
        print(f"[cross] wins {result['candidate_wins']}/{result['games']}", flush=True)


if __name__ == "__main__":
    main()
