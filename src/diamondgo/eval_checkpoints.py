from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from diamondgo.batched_demo import BatchedConfig, run_batched_mcts
from diamondgo.config import ModelConfig
from diamondgo.demo_cpu import action_to_gtp, make_rules
from diamondgo.model import PolicyValueNet


@dataclass(frozen=True)
class MatchConfig:
    board_size: int = 9
    komi: float = 0.5
    channels: int = 32
    residual_blocks: int = 2
    simulations: int = 32
    games: int = 20
    max_moves: int = 80
    c_puct: float = 1.5
    opening_temperature_moves: int = 6
    seed: int = 20260605
    device: str = "cuda"
    rules_backend: str = "sgfmill"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate 9x9 checkpoints by head-to-head games.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-moves", type=int, default=80)
    parser.add_argument("--sample-games", type=int, default=2)
    parser.add_argument("--opponent", choices=["initial", "previous"], default="initial")
    parser.add_argument("--include-latest", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true")
    return parser


def cycle_from_path(path: Path) -> int:
    match = re.search(r"cycle-(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"not a cycle checkpoint: {path}")
    return int(match.group(1))


def load_checkpoint_payload(path: Path, device: str) -> dict[str, object]:
    return torch.load(path, map_location=torch.device(device))


def config_from_payload(payload: dict[str, object], device: str, simulations: int, games: int, max_moves: int) -> MatchConfig:
    raw = dict(payload["config"])
    return MatchConfig(
        board_size=int(raw.get("board_size", 9)),
        komi=float(raw.get("komi", 0.5)),
        channels=int(raw.get("channels", 32)),
        residual_blocks=int(raw.get("residual_blocks", 2)),
        simulations=simulations,
        games=games,
        max_moves=max_moves,
        c_puct=float(raw.get("c_puct", 1.5)),
        seed=int(raw.get("seed", 1)),
        device=device,
        rules_backend=str(raw.get("rules_backend", "sgfmill")),
    )


def make_eval_model(config: MatchConfig) -> PolicyValueNet:
    model = PolicyValueNet(
        board_size=config.board_size,
        config=ModelConfig(channels=config.channels, residual_blocks=config.residual_blocks),
    )
    model.to(torch.device(config.device))
    model.eval()
    return model


def make_initial_model(config: MatchConfig) -> PolicyValueNet:
    torch.manual_seed(config.seed)
    return make_eval_model(config)


def make_checkpoint_model(config: MatchConfig, checkpoint_path: Path) -> PolicyValueNet:
    model = make_eval_model(config)
    payload = load_checkpoint_payload(checkpoint_path, config.device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def other_player(player: str) -> str:
    return "w" if player == "b" else "b"


def choose_action(root, action_size: int, move_number: int, opening_temperature_moves: int) -> int:
    if move_number <= opening_temperature_moves:
        policy = root.policy_target(action_size, temperature=1.0)
        return int(np.random.choice(np.arange(action_size), p=policy))
    return int(max(root.children.items(), key=lambda item: item[1].visit_count)[0])


def sgf_action(action: int, board_size: int) -> str:
    if action == board_size * board_size:
        return ""
    row, col = divmod(action, board_size)
    letters = "abcdefghijklmnopqrstuvwxyz"
    return f"{letters[col]}{letters[row]}"


def escape_sgf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("]", "\\]")


def write_match_sgf(
    path: Path,
    config: MatchConfig,
    game: dict[str, object],
    candidate_name: str,
    opponent_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_color = str(game["candidate_color"])
    black_name = candidate_name if candidate_color == "b" else opponent_name
    white_name = candidate_name if candidate_color == "w" else opponent_name
    winner = str(game["winner"])
    result = "B+R" if winner == "b" else "W+R"
    nodes = [
        (
            f"(;GM[1]FF[4]CA[UTF-8]AP[DiamondGo:evaluate]"
            f"SZ[{config.board_size}]KM[{config.komi}]PB[{escape_sgf(black_name)}]PW[{escape_sgf(white_name)}]"
            f"RE[{result}]C[{escape_sgf('DiamondGo checkpoint evaluation game.')}]"
        )
    ]
    for move in game["moves"]:
        color = "B" if move["player"] == "b" else "W"
        comment_lines = [
            f"model: {move['model']}",
            f"root_value: {move['root_value']}",
            "top actions:",
        ]
        for item in move["top_actions"]:
            comment_lines.append(
                f"- {item['move']}: visits={item['visits']} prior={item['prior']} value={item['value']}"
            )
        nodes.append(
            f";{color}[{sgf_action(int(move['action']), config.board_size)}]"
            f"C[{escape_sgf(chr(10).join(comment_lines))}]"
        )
    nodes.append(")")
    path.write_text("".join(nodes), encoding="utf-8")


def batched_config(config: MatchConfig, active_games: int) -> BatchedConfig:
    return BatchedConfig(
        board_size=config.board_size,
        komi=config.komi,
        channels=config.channels,
        residual_blocks=config.residual_blocks,
        simulations=config.simulations,
        max_moves=config.max_moves,
        games=active_games,
        train_steps=0,
        batch_size=256,
        c_puct=config.c_puct,
        temperature=1.0,
        seed=config.seed,
        device=config.device,
        rules_backend=config.rules_backend,
    )


def play_match(
    config: MatchConfig,
    candidate_model: PolicyValueNet,
    opponent_model: PolicyValueNet,
    candidate_name: str,
    opponent_name: str,
    out_dir: Path,
    sample_games: int,
) -> dict[str, object]:
    start = time.perf_counter()
    states = [make_rules(batched_config(config, config.games)) for _ in range(config.games)]
    active = [True for _ in states]
    candidate_colors = ["b" if index % 2 == 0 else "w" for index in range(config.games)]
    move_counts = [0 for _ in states]
    game_records: list[dict[str, object]] = [
        {"game": index + 1, "candidate_color": candidate_colors[index], "moves": []}
        for index in range(config.games)
    ]
    stats: dict[str, object] = {"candidate": {}, "opponent": {}}

    while any(active):
        active_indices = [
            index
            for index, state in enumerate(states)
            if active[index] and not state.is_terminal() and move_counts[index] < config.max_moves
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
            group_states = [states[index] for index in indices]
            roots = run_batched_mcts(model, group_states, batched_config(config, len(indices)), stats[key])
            roots_by_index.update(zip(indices, roots))

        for index in active_indices:
            state = states[index]
            root = roots_by_index[index]
            model_name = "candidate" if state.to_play == candidate_colors[index] else "opponent"
            action = choose_action(
                root,
                state.action_size,
                move_counts[index] + 1,
                config.opening_temperature_moves,
            )
            game_records[index]["moves"].append(
                {
                    "move_number": move_counts[index] + 1,
                    "player": state.to_play,
                    "model": model_name,
                    "action": action,
                    "move": action_to_gtp(action, config.board_size),
                    "root_value": round(root.value, 4),
                    "top_actions": root.top_actions(config.board_size),
                }
            )
            state.play_action(action)
            move_counts[index] += 1
            if state.is_terminal() or move_counts[index] >= config.max_moves:
                active[index] = False

    candidate_wins = 0
    candidate_black_wins = 0
    candidate_white_wins = 0
    for index, state in enumerate(states):
        value_for_to_play = float(state.terminal_value())
        winner = state.to_play if value_for_to_play > 0 else other_player(state.to_play)
        candidate_won = winner == candidate_colors[index]
        candidate_wins += int(candidate_won)
        candidate_black_wins += int(candidate_won and candidate_colors[index] == "b")
        candidate_white_wins += int(candidate_won and candidate_colors[index] == "w")
        game_records[index]["winner"] = winner
        game_records[index]["candidate_won"] = candidate_won
        game_records[index]["moves_played"] = move_counts[index]

    for record in game_records[:sample_games]:
        write_match_sgf(
            out_dir / "sample-sgf" / f"{candidate_name}_vs_{opponent_name}_game-{record['game']:02d}.sgf",
            config,
            record,
            candidate_name,
            opponent_name,
        )

    elapsed = time.perf_counter() - start
    return {
        "candidate": candidate_name,
        "opponent": opponent_name,
        "games": config.games,
        "candidate_wins": candidate_wins,
        "candidate_losses": config.games - candidate_wins,
        "win_rate": round(candidate_wins / config.games, 4),
        "candidate_black_wins": candidate_black_wins,
        "candidate_white_wins": candidate_white_wins,
        "candidate_black_games": sum(1 for color in candidate_colors if color == "b"),
        "candidate_white_games": sum(1 for color in candidate_colors if color == "w"),
        "seconds": round(elapsed, 3),
        "games_per_second": round(config.games / max(elapsed, 1e-9), 3),
        "sample_sgf_dir": str(out_dir / "sample-sgf"),
        "games_detail": game_records,
    }


def markdown_report(results: list[dict[str, object]]) -> str:
    lines = [
        "# DiamondGo checkpoint evaluation",
        "",
        "| candidate | opponent | games | win rate | wins | black wins | white wins | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['candidate']} | {item['opponent']} | {item['games']} | "
            f"{100 * float(item['win_rate']):.1f}% | {item['candidate_wins']} | "
            f"{item['candidate_black_wins']}/{item['candidate_black_games']} | "
            f"{item['candidate_white_wins']}/{item['candidate_white_games']} | {item['seconds']} |"
        )
    lines.append("")
    lines.append("Each match alternates candidate colors. The first six moves are sampled from MCTS visits; later moves are max-visit.")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = sorted(checkpoint_dir.glob("cycle-*.pt"), key=cycle_from_path)
    selected = [path for path in checkpoint_paths if cycle_from_path(path) % args.step == 0]
    if args.include_latest:
        selected.append(Path(args.include_latest))
    if not selected:
        raise RuntimeError(f"no checkpoints selected from {checkpoint_dir}")

    first_payload = load_checkpoint_payload(selected[0], args.device)
    config = config_from_payload(
        first_payload,
        device=args.device,
        simulations=args.simulations,
        games=args.games,
        max_moves=args.max_moves,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but CUDA is unavailable")

    initial_model = make_initial_model(config)
    previous_model = initial_model
    previous_name = "cycle-00000"
    results: list[dict[str, object]] = []

    for checkpoint_path in selected:
        cycle = int(load_checkpoint_payload(checkpoint_path, args.device).get("cycle", cycle_from_path(checkpoint_path)))
        candidate_name = f"cycle-{cycle:05d}"
        candidate_model = make_checkpoint_model(config, checkpoint_path)
        if args.opponent == "initial":
            opponent_model = initial_model
            opponent_name = "cycle-00000"
        else:
            opponent_model = previous_model
            opponent_name = previous_name

        result = play_match(
            config=config,
            candidate_model=candidate_model,
            opponent_model=opponent_model,
            candidate_name=candidate_name,
            opponent_name=opponent_name,
            out_dir=out_dir,
            sample_games=args.sample_games,
        )
        result["checkpoint_path"] = str(checkpoint_path)
        results.append(result)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out_dir / "report.md").write_text(markdown_report(results), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "games_detail"}), flush=True)
        previous_model = candidate_model
        previous_name = candidate_name

    return {
        "config": config.__dict__,
        "results": results,
        "report_path": str(out_dir / "report.md"),
        "results_path": str(out_dir / "results.json"),
        "sample_sgf_dir": str(out_dir / "sample-sgf"),
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    if args.json:
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
