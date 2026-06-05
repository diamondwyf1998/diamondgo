from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from diamondgo.config import ModelConfig
from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_MAX_MOVES, DEFAULT_9X9_SCORE_KOMI
from diamondgo.demo_cpu import (
    action_to_gtp,
    build_trace,
    make_rules,
    train_steps,
    write_dashboard,
    write_json,
    write_overview_svg,
    write_sgf,
)
from diamondgo.mcts import SearchNode, play_search_action, select_child
from diamondgo.model import PolicyValueNet


@dataclass(frozen=True)
class BatchedConfig:
    board_size: int = 9
    komi: float = DEFAULT_9X9_KOMI
    score_komi: float = DEFAULT_9X9_SCORE_KOMI
    input_komi: bool = True
    channels: int = 32
    residual_blocks: int = 2
    simulations: int = 64
    max_moves: int = DEFAULT_9X9_MAX_MOVES
    games: int = 16
    train_steps: int = 16
    batch_size: int = 256
    learning_rate: float = 1e-3
    c_puct: float = 1.5
    temperature: float = 1.0
    temperature_moves: int = 0
    late_temperature: float = 1.0
    root_dirichlet_alpha: float = 0.0
    root_noise_fraction: float = 0.0
    root_policy_temperature: float = 1.0
    seed: int = 1
    device: str = "cuda"
    rules_backend: str = "sgfmill"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batched 9x9 self-play with GPU leaf evaluation.")
    parser.add_argument("--games", type=int, default=BatchedConfig.games)
    parser.add_argument("--komi", type=float, default=BatchedConfig.komi)
    parser.add_argument("--score-komi", type=float, default=BatchedConfig.score_komi)
    parser.add_argument("--input-komi", action=argparse.BooleanOptionalAction, default=BatchedConfig.input_komi)
    parser.add_argument("--simulations", type=int, default=BatchedConfig.simulations)
    parser.add_argument("--max-moves", type=int, default=BatchedConfig.max_moves)
    parser.add_argument("--train-steps", type=int, default=BatchedConfig.train_steps)
    parser.add_argument("--batch-size", type=int, default=BatchedConfig.batch_size)
    parser.add_argument("--channels", type=int, default=BatchedConfig.channels)
    parser.add_argument("--residual-blocks", type=int, default=BatchedConfig.residual_blocks)
    parser.add_argument("--c-puct", type=float, default=BatchedConfig.c_puct)
    parser.add_argument("--temperature", type=float, default=BatchedConfig.temperature)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=BatchedConfig.root_dirichlet_alpha)
    parser.add_argument("--root-noise-fraction", type=float, default=BatchedConfig.root_noise_fraction)
    parser.add_argument("--root-policy-temperature", type=float, default=BatchedConfig.root_policy_temperature)
    parser.add_argument("--temperature-moves", type=int, default=BatchedConfig.temperature_moves)
    parser.add_argument("--late-temperature", type=float, default=BatchedConfig.late_temperature)
    parser.add_argument("--seed", type=int, default=BatchedConfig.seed)
    parser.add_argument("--device", default=BatchedConfig.device)
    parser.add_argument("--rules", choices=["simple", "sgfmill"], default=BatchedConfig.rules_backend)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--sgf", default="artifacts/batched-9x9.sgf")
    parser.add_argument("--trace", default="artifacts/batched-9x9.json")
    parser.add_argument("--dashboard", default="artifacts/visualizations/batched-9x9-dashboard.html")
    parser.add_argument("--overview-svg", default="artifacts/visualizations/batched-9x9-overview.svg")
    return parser


def make_model(config: BatchedConfig) -> PolicyValueNet:
    model = PolicyValueNet(
        board_size=config.board_size,
        config=ModelConfig(channels=config.channels, residual_blocks=config.residual_blocks),
        input_planes=4 if config.input_komi else 3,
    )
    model.to(torch.device(config.device))
    model.eval()
    return model


def evaluate_batch(
    model: PolicyValueNet,
    states: list[object],
    stats: dict[str, object] | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    device = next(model.parameters()).device
    encode_start = time.perf_counter()
    encoded = np.stack([state.encode() for state in states])
    encode_elapsed = time.perf_counter() - encode_start
    tensor_start = time.perf_counter()
    features = torch.tensor(encoded, dtype=torch.float32).to(device)
    tensor_elapsed = time.perf_counter() - tensor_start
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        logits, values = model(features)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if stats is not None:
        stats["network_calls"] = int(stats.get("network_calls", 0)) + 1
        stats["network_seconds"] = float(stats.get("network_seconds", 0.0)) + elapsed
        stats["encode_seconds"] = float(stats.get("encode_seconds", 0.0)) + encode_elapsed
        stats["tensor_seconds"] = float(stats.get("tensor_seconds", 0.0)) + tensor_elapsed
        stats.setdefault("batch_sizes", []).append(len(states))
    priors = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return [priors[i] for i in range(len(states))], [float(v) for v in values.detach().cpu().numpy()]


def backpropagate(path: list[SearchNode], value: float) -> None:
    current = float(value)
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += current
        current = -current


def collect_leaf(
    state,
    root: SearchNode,
    c_puct: float,
    stats: dict[str, object],
) -> tuple[object, list[SearchNode], SearchNode] | None:
    copy_start = time.perf_counter()
    simulation_state = state.copy()
    stats["state_copy_seconds"] = float(stats.get("state_copy_seconds", 0.0)) + (
        time.perf_counter() - copy_start
    )
    node = root
    path = [node]
    while node.expanded() and not simulation_state.is_terminal():
        select_start = time.perf_counter()
        action, child = select_child(node, c_puct)
        stats["select_seconds"] = float(stats.get("select_seconds", 0.0)) + (
            time.perf_counter() - select_start
        )
        play_start = time.perf_counter()
        play_search_action(simulation_state, action)
        stats["play_search_seconds"] = float(stats.get("play_search_seconds", 0.0)) + (
            time.perf_counter() - play_start
        )
        node = child
        path.append(node)

    if simulation_state.is_terminal():
        backpropagate(path, float(simulation_state.terminal_value()))
        return None
    return simulation_state, path, node


def run_batched_mcts(
    model: PolicyValueNet,
    states: list[object],
    config: BatchedConfig,
    stats: dict[str, object],
) -> list[SearchNode]:
    roots = [SearchNode(prior=1.0) for _ in states]
    priors_batch, values = evaluate_batch(model, states, stats)
    for root, state, priors, value in zip(roots, states, priors_batch, values):
        legal_start = time.perf_counter()
        legal_actions = state.legal_actions()
        stats["legal_actions_seconds"] = float(stats.get("legal_actions_seconds", 0.0)) + (
            time.perf_counter() - legal_start
        )
        root.expand(apply_policy_temperature(priors, config.root_policy_temperature), legal_actions)
        add_root_dirichlet_noise(root, config.root_dirichlet_alpha, config.root_noise_fraction)
        root.visit_count = 1
        root.value_sum = float(value)

    for _ in range(config.simulations):
        pending: list[tuple[object, list[SearchNode], SearchNode]] = []
        for state, root in zip(states, roots):
            leaf = collect_leaf(state, root, config.c_puct, stats)
            if leaf is not None:
                pending.append(leaf)
        if not pending:
            continue
        leaf_states = [item[0] for item in pending]
        priors_batch, values = evaluate_batch(model, leaf_states, stats)
        for (_, path, node), priors, value, leaf_state in zip(pending, priors_batch, values, leaf_states):
            legal_start = time.perf_counter()
            legal_actions = leaf_state.legal_actions()
            stats["legal_actions_seconds"] = float(stats.get("legal_actions_seconds", 0.0)) + (
                time.perf_counter() - legal_start
            )
            node.expand(priors, legal_actions)
            backpropagate(path, float(value))
    return roots


def apply_policy_temperature(priors: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0 or abs(temperature - 1.0) <= 1e-6:
        return priors
    adjusted = np.power(np.clip(priors, 1e-12, 1.0), 1.0 / temperature)
    total = float(adjusted.sum())
    if total <= 0:
        return priors
    return adjusted / total


def temperature_for_move(config: BatchedConfig, played_moves: int) -> float:
    if config.temperature_moves <= 0:
        return config.temperature
    return config.temperature if played_moves < config.temperature_moves else config.late_temperature


def add_root_dirichlet_noise(root: SearchNode, alpha: float, fraction: float) -> None:
    if alpha <= 0.0 or fraction <= 0.0 or not root.children:
        return
    actions = list(root.children)
    noise = np.random.dirichlet([alpha] * len(actions))
    mix = min(max(fraction, 0.0), 1.0)
    for action, noise_value in zip(actions, noise):
        child = root.children[action]
        child.prior = (1.0 - mix) * child.prior + mix * float(noise_value)


def play_batched_games(config: BatchedConfig, model: PolicyValueNet) -> tuple[list[dict[str, object]], dict[str, object]]:
    states = [make_rules(config) for _ in range(config.games)]
    active = [True for _ in states]
    move_counts = [0 for _ in states]
    pass_counts = [0 for _ in states]
    capture_move_counts = [0 for _ in states]
    captured_stone_counts = [0 for _ in states]
    examples: list[dict[str, object]] = []
    stats: dict[str, object] = {"network_calls": 0, "network_seconds": 0.0, "batch_sizes": []}

    while any(active):
        active_indices = [
            index
            for index, state in enumerate(states)
            if active[index] and not state.is_terminal() and move_counts[index] < config.max_moves
        ]
        if not active_indices:
            break

        active_states = [states[index] for index in active_indices]
        roots = run_batched_mcts(model, active_states, config, stats)

        for state_index, root in zip(active_indices, roots):
            state = states[state_index]
            player = state.to_play
            encode_start = time.perf_counter()
            features = state.encode()
            stats["sample_encode_seconds"] = float(stats.get("sample_encode_seconds", 0.0)) + (
                time.perf_counter() - encode_start
            )
            policy = root.policy_target(state.action_size, temperature_for_move(config, move_counts[state_index]))
            action = int(np.random.choice(np.arange(state.action_size), p=policy))
            stone_start = time.perf_counter()
            stones_before = _stone_counts(state)
            stats["stone_count_seconds"] = float(stats.get("stone_count_seconds", 0.0)) + (
                time.perf_counter() - stone_start
            )
            move_counts[state_index] += 1
            is_pass = action == state.action_size - 1
            if is_pass:
                pass_counts[state_index] += 1
            examples.append(
                {
                    "features": features,
                    "policy": policy,
                    "player": player,
                    "top_actions": root.top_actions(config.board_size),
                    "root_value": round(root.value, 4),
                    "chosen_action": action,
                    "game": state_index + 1,
                    "move_in_game": move_counts[state_index],
                    "chosen_move": action_to_gtp(action, config.board_size),
                    "is_pass": is_pass,
                }
            )
            state.play_action(action)
            stone_start = time.perf_counter()
            captures = _captures_for_move(player, stones_before, _stone_counts(state))
            stats["stone_count_seconds"] = float(stats.get("stone_count_seconds", 0.0)) + (
                time.perf_counter() - stone_start
            )
            examples[-1]["captures"] = captures
            if captures > 0:
                capture_move_counts[state_index] += 1
                captured_stone_counts[state_index] += captures
            if state.is_terminal() or move_counts[state_index] >= config.max_moves:
                active[state_index] = False

    terminal_values_by_game = {}
    game_summaries = []
    for game_index, state in enumerate(states, start=1):
        value_for_to_play = float(state.terminal_value())
        terminal_values_by_game[(game_index, state.to_play)] = value_for_to_play
        terminal_values_by_game[(game_index, "w" if state.to_play == "b" else "b")] = -value_for_to_play
        black_margin = _black_score_margin(state)
        winner = "b" if black_margin > 0 else "w"
        ended_by_pass = bool(state.is_terminal())
        game_summaries.append(
            {
                "game": game_index,
                "moves": move_counts[game_index - 1],
                "ended_by": "pass" if ended_by_pass else "max_moves",
                "passes": pass_counts[game_index - 1],
                "capture_moves": capture_move_counts[game_index - 1],
                "captured_stones": captured_stone_counts[game_index - 1],
                "black_score_margin": round(black_margin, 3),
                "winner": winner,
                "to_play_at_end": state.to_play,
            }
        )

    for example in examples:
        example["value_target"] = terminal_values_by_game[(int(example["game"]), example["player"])]
    batch_sizes = list(stats.get("batch_sizes", []))
    stats["average_batch_size"] = round(float(np.mean(batch_sizes)), 3) if batch_sizes else 0.0
    stats["max_batch_size"] = int(max(batch_sizes)) if batch_sizes else 0
    for key in [
        "network_seconds",
        "encode_seconds",
        "tensor_seconds",
        "legal_actions_seconds",
        "state_copy_seconds",
        "select_seconds",
        "play_search_seconds",
        "sample_encode_seconds",
        "stone_count_seconds",
    ]:
        stats[key] = round(float(stats.get(key, 0.0)), 3)
    stats["game_summaries"] = game_summaries
    return examples, stats


def _stone_counts(state: object) -> dict[str, int]:
    board_array = getattr(state, "board_array", None)
    if board_array is not None:
        array = np.asarray(board_array)
        return {"b": int((array == 1).sum()), "w": int((array == -1).sum())}
    board = getattr(state, "board", None)
    size = int(getattr(state, "size"))
    counts = {"b": 0, "w": 0}
    if board is None:
        return counts
    get = getattr(board, "get", None)
    if get is not None:
        for row in range(size):
            for col in range(size):
                colour = get(row, col)
                if colour in counts:
                    counts[colour] += 1
        return counts
    array = np.asarray(board)
    counts["b"] = int((array == 1).sum())
    counts["w"] = int((array == -1).sum())
    return counts


def _captures_for_move(player: str, before: dict[str, int], after: dict[str, int]) -> int:
    opponent = "w" if player == "b" else "b"
    return max(0, int(before.get(opponent, 0)) - int(after.get(opponent, 0)))


def _black_score_margin(state: object) -> float:
    board = getattr(state, "board", None)
    score_komi = float(getattr(state, "score_komi", getattr(state, "komi")))
    area_score = getattr(board, "area_score", None)
    if area_score is not None:
        return float(area_score()) - score_komi
    return float(np.asarray(board).sum()) - score_komi


def run(config: BatchedConfig, sgf_path: str, trace_path: str, dashboard_path: str, overview_svg_path: str) -> dict[str, object]:
    total_start = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but CUDA is unavailable")

    model = make_model(config)
    params = sum(parameter.numel() for parameter in model.parameters())
    selfplay_start = time.perf_counter()
    examples, selfplay_stats = play_batched_games(config, model)
    selfplay_seconds = time.perf_counter() - selfplay_start

    train_start = time.perf_counter()
    loss_history = train_steps(config, model, examples)
    train_seconds = time.perf_counter() - train_start

    write_start = time.perf_counter()
    write_sgf(sgf_path, config, examples)
    write_json(trace_path, build_trace(config, examples))
    write_dashboard(dashboard_path, config, params, examples, loss_history, sgf_path, trace_path)
    write_overview_svg(overview_svg_path, config, params, examples, loss_history)
    write_seconds = time.perf_counter() - write_start
    total_seconds = time.perf_counter() - total_start

    first = examples[0]
    return {
        "config": asdict(config),
        "parameters": params,
        "positions": len(examples),
        "sgf_path": str(sgf_path),
        "trace_path": str(trace_path),
        "dashboard_path": str(dashboard_path),
        "overview_svg_path": str(overview_svg_path),
        "first_position": {
            "root_value": first["root_value"],
            "top_actions": first["top_actions"],
            "chosen_action": first["chosen_action"],
        },
        "train_metrics": loss_history[-1],
        "loss_history": loss_history,
        "timing": {
            "selfplay_seconds": round(selfplay_seconds, 3),
            "train_seconds": round(train_seconds, 3),
            "write_seconds": round(write_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "positions_per_second": round(len(examples) / max(selfplay_seconds, 1e-9), 3),
            "network_seconds": selfplay_stats["network_seconds"],
            "network_calls": selfplay_stats["network_calls"],
            "average_network_batch": selfplay_stats["average_batch_size"],
            "max_network_batch": selfplay_stats["max_batch_size"],
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    config = BatchedConfig(
        games=args.games,
        komi=args.komi,
        score_komi=args.score_komi,
        input_komi=args.input_komi,
        simulations=args.simulations,
        max_moves=args.max_moves,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        c_puct=args.c_puct,
        temperature=args.temperature,
        temperature_moves=args.temperature_moves,
        late_temperature=args.late_temperature,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_noise_fraction=args.root_noise_fraction,
        root_policy_temperature=args.root_policy_temperature,
        seed=args.seed,
        device=args.device,
        rules_backend=args.rules,
    )
    summary = run(config, args.sgf, args.trace, args.dashboard, args.overview_svg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
