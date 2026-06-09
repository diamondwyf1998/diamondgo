from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diamondgo import eval_checkpoints
from diamondgo.batched_demo import run_batched_mcts
from diamondgo.demo_cpu import action_to_gtp, make_rules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a head-to-head match with different MCTS simulations per side.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--opponent-name", default="opponent")
    parser.add_argument("--candidate-sims", type=int, default=200)
    parser.add_argument("--opponent-sims", action="append", type=int, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-moves", type=int, default=150)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def play_asymmetric(
    *,
    candidate_model: torch.nn.Module,
    opponent_model: torch.nn.Module,
    base_config: eval_checkpoints.MatchConfig,
    candidate_sims: int,
    opponent_sims: int,
    candidate_name: str,
    opponent_name: str,
    seed: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    candidate_config = replace(base_config, simulations=candidate_sims, seed=seed)
    opponent_config = replace(base_config, simulations=opponent_sims, seed=seed)
    state_config = replace(base_config, simulations=max(candidate_sims, opponent_sims), seed=seed)
    states = [make_rules(eval_checkpoints.batched_config(state_config, base_config.games)) for _ in range(base_config.games)]
    active = [True for _ in states]
    candidate_colors = ["b" if index % 2 == 0 else "w" for index in range(base_config.games)]
    move_counts = [0 for _ in states]
    game_records: list[dict[str, object]] = [
        {"game": index + 1, "candidate_color": candidate_colors[index], "moves": []}
        for index in range(base_config.games)
    ]
    stats: dict[str, object] = {"candidate": {}, "opponent": {}}
    start = time.perf_counter()

    while any(active):
        active_indices = [
            index
            for index, state in enumerate(states)
            if active[index] and not state.is_terminal() and move_counts[index] < base_config.max_moves
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
            roots = run_batched_mcts(model, group_states, eval_checkpoints.batched_config(config, len(indices)), stats[key])
            roots_by_index.update(zip(indices, roots))

        for index in active_indices:
            state = states[index]
            root = roots_by_index[index]
            model_name = "candidate" if state.to_play == candidate_colors[index] else "opponent"
            action = eval_checkpoints.choose_action(
                root,
                state.action_size,
                move_counts[index] + 1,
                base_config.opening_temperature_moves,
            )
            game_records[index]["moves"].append(
                {
                    "move_number": move_counts[index] + 1,
                    "player": state.to_play,
                    "model": model_name,
                    "action": int(action),
                    "move": action_to_gtp(int(action), base_config.board_size),
                    "root_value": round(root.value, 4),
                    "top_actions": root.top_actions(base_config.board_size),
                }
            )
            state.play_action(action)
            move_counts[index] += 1
            if state.is_terminal() or move_counts[index] >= base_config.max_moves:
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

    elapsed = time.perf_counter() - start
    return {
        "candidate": f"{candidate_name}-sim{candidate_sims}",
        "opponent": f"{opponent_name}-sim{opponent_sims}",
        "games": base_config.games,
        "candidate_wins": candidate_wins,
        "candidate_losses": base_config.games - candidate_wins,
        "win_rate": round(candidate_wins / base_config.games, 4),
        "candidate_black_wins": candidate_black_wins,
        "candidate_white_wins": candidate_white_wins,
        "candidate_black_games": sum(1 for color in candidate_colors if color == "b"),
        "candidate_white_games": sum(1 for color in candidate_colors if color == "w"),
        "candidate_simulations": candidate_sims,
        "opponent_simulations": opponent_sims,
        "seconds": round(elapsed, 3),
        "games_per_second": round(base_config.games / max(elapsed, 1e-9), 3),
        "pass_behavior": eval_checkpoints.summarize_pass_behavior(game_records),
        "games_detail": game_records,
    }


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = Path(args.candidate)
    opponent_path = Path(args.opponent)
    payload = eval_checkpoints.load_checkpoint_payload(candidate_path, args.device)
    config = eval_checkpoints.config_from_payload(payload, args.device, args.candidate_sims, args.games, args.max_moves)
    config = replace(config, max_moves=args.max_moves)
    candidate_model = eval_checkpoints.make_checkpoint_model(config, candidate_path)
    opponent_model = eval_checkpoints.make_checkpoint_model(config, opponent_path)
    results = []
    for offset, opponent_sims in enumerate(args.opponent_sims):
        result = play_asymmetric(
            candidate_model=candidate_model,
            opponent_model=opponent_model,
            base_config=config,
            candidate_sims=args.candidate_sims,
            opponent_sims=opponent_sims,
            candidate_name=args.candidate_name,
            opponent_name=args.opponent_name,
            seed=args.seed + offset,
        )
        results.append(result)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        dashboard = eval_checkpoints.render_eval_dashboard(results, config.board_size)
        (out_dir / "dashboard.html").write_text(dashboard, encoding="utf-8")
        (out_dir / "games_dashboard.html").write_text(dashboard, encoding="utf-8")
        print(
            f"[asym] {result['candidate']} vs {result['opponent']}: "
            f"{result['candidate_wins']}/{result['games']}",
            flush=True,
        )
    summary = {
        "candidate_checkpoint": str(candidate_path),
        "opponent_checkpoint": str(opponent_path),
        "rules_environment": {
            "komi": config.komi,
            "score_komi": config.score_komi,
            "max_moves": config.max_moves,
            "rules_backend": config.rules_backend,
        },
        "results": [{key: value for key, value in item.items() if key != "games_detail"} for item in results],
        "dashboard_path": str(out_dir / "dashboard.html"),
        "games_dashboard_path": str(out_dir / "games_dashboard.html"),
        "results_path": str(out_dir / "results.json"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
