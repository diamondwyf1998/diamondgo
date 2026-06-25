from __future__ import annotations

import argparse
import html
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diamondgo.config import MCTSConfig, ModelConfig, input_plane_count
from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_SCORE_KOMI
from diamondgo.mcts import run_mcts
from diamondgo.model import PolicyValueNet, policy_value, policy_value_final_board
from diamondgo.rules import SgfmillRules, SimpleAreaRules


@dataclass(frozen=True)
class CpuDemoConfig:
    board_size: int = 9
    komi: float = DEFAULT_9X9_KOMI
    score_komi: float = DEFAULT_9X9_SCORE_KOMI
    input_komi: bool = True
    history_moves: int = 0
    terminal_dead_stone_cleanup: bool = False
    score_margin_reward_scale: float = 0.0
    final_board_loss_weight: float = 0.25
    channels: int = 16
    residual_blocks: int = 1
    simulations: int = 8
    max_moves: int = 20
    games: int = 1
    train_steps: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    c_puct: float = 1.5
    temperature: float = 1.0
    seed: int = 1
    device: str = "cpu"
    rules_backend: str = "simple"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny CPU-only 9x9 baby-zero demo.")
    parser.add_argument("--games", type=int, default=CpuDemoConfig.games)
    parser.add_argument("--komi", type=float, default=CpuDemoConfig.komi)
    parser.add_argument("--score-komi", type=float, default=CpuDemoConfig.score_komi)
    parser.add_argument("--input-komi", action=argparse.BooleanOptionalAction, default=CpuDemoConfig.input_komi)
    parser.add_argument(
        "--history-moves",
        type=int,
        default=CpuDemoConfig.history_moves,
        help="Append this many previous-move location planes to the neural-network input.",
    )
    parser.add_argument(
        "--terminal-dead-stone-cleanup",
        action=argparse.BooleanOptionalAction,
        default=CpuDemoConfig.terminal_dead_stone_cleanup,
        help="At terminal scoring, remove conservatively detected obvious dead groups.",
    )
    parser.add_argument(
        "--score-margin-reward-scale",
        type=float,
        default=CpuDemoConfig.score_margin_reward_scale,
        help="Scale for the capped +/-0.6 score-margin component; enabled targets use +/-0.4 win/loss base.",
    )
    parser.add_argument(
        "--final-board-loss-weight",
        type=float,
        default=CpuDemoConfig.final_board_loss_weight,
        help="Auxiliary loss weight for predicting the final black/white ownership board.",
    )
    parser.add_argument("--simulations", type=int, default=CpuDemoConfig.simulations)
    parser.add_argument("--max-moves", type=int, default=CpuDemoConfig.max_moves)
    parser.add_argument("--train-steps", type=int, default=CpuDemoConfig.train_steps)
    parser.add_argument("--batch-size", type=int, default=CpuDemoConfig.batch_size)
    parser.add_argument("--channels", type=int, default=CpuDemoConfig.channels)
    parser.add_argument("--residual-blocks", type=int, default=CpuDemoConfig.residual_blocks)
    parser.add_argument("--seed", type=int, default=CpuDemoConfig.seed)
    parser.add_argument("--device", default=CpuDemoConfig.device)
    parser.add_argument("--rules", choices=["simple", "sgfmill"], default=CpuDemoConfig.rules_backend)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    parser.add_argument(
        "--sgf",
        default="artifacts/cpu-demo-9x9.sgf",
        help="Path for the self-play SGF with search comments.",
    )
    parser.add_argument(
        "--trace",
        default="artifacts/cpu-demo-9x9.json",
        help="Path for compact self-play/search trace JSON.",
    )
    parser.add_argument(
        "--dashboard",
        default="artifacts/visualizations/cpu-demo-9x9-dashboard.html",
        help="Path for a standalone HTML visualization dashboard.",
    )
    parser.add_argument(
        "--overview-svg",
        default="artifacts/visualizations/cpu-demo-9x9-overview.svg",
        help="Path for a compact SVG overview.",
    )
    return parser


def make_model(config: CpuDemoConfig) -> PolicyValueNet:
    model_config = ModelConfig(channels=config.channels, residual_blocks=config.residual_blocks)
    model = PolicyValueNet(
        config.board_size,
        model_config,
        input_planes=input_plane_count(config.input_komi, config.history_moves),
    )
    model.to(torch.device(config.device))
    model.eval()
    return model


def evaluate_with_model(model: PolicyValueNet, state: SimpleAreaRules) -> tuple[np.ndarray, float]:
    device = next(model.parameters()).device
    features = torch.from_numpy(state.encode()).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = policy_value(model(features))
    priors = torch.softmax(logits[0], dim=0).detach().cpu().numpy()
    return priors, float(value.item())


def make_rules(config: CpuDemoConfig):
    if config.rules_backend == "sgfmill":
        return SgfmillRules(
            size=config.board_size,
            komi=config.komi,
            score_komi=config.score_komi,
            input_komi=config.input_komi,
            history_moves=config.history_moves,
            terminal_dead_stone_cleanup=config.terminal_dead_stone_cleanup,
            score_margin_reward_scale=config.score_margin_reward_scale,
        )
    return SimpleAreaRules(
        size=config.board_size,
        komi=config.komi,
        score_komi=config.score_komi,
        input_komi=config.input_komi,
        history_moves=config.history_moves,
        terminal_dead_stone_cleanup=config.terminal_dead_stone_cleanup,
        score_margin_reward_scale=config.score_margin_reward_scale,
    )


def play_game(config: CpuDemoConfig, model: PolicyValueNet) -> tuple[list[dict[str, object]], float]:
    state = make_rules(config)
    examples: list[dict[str, object]] = []
    search_config = MCTSConfig(
        simulations=config.simulations,
        c_puct=config.c_puct,
        temperature=config.temperature,
    )

    while not state.is_terminal() and len(examples) < config.max_moves:
        player = state.to_play
        features = state.encode()
        root = run_mcts(
            state,
            evaluator=lambda search_state: evaluate_with_model(model, search_state),
            simulations=search_config.simulations,
            c_puct=search_config.c_puct,
            temperature=search_config.temperature,
        )
        policy = root.policy_target(state.action_size, config.temperature)
        action = int(np.random.choice(np.arange(state.action_size), p=policy))
        examples.append(
            {
                "features": features,
                "policy": policy,
                "player": player,
                "top_actions": root.top_actions(config.board_size, limit=None),
                "root_value": round(root.value, 4),
                "chosen_action": action,
            }
        )
        state.play_action(action)

    outcome_for_to_play = state.terminal_value()
    winner_value_by_player = {state.to_play: outcome_for_to_play}
    winner_value_by_player["w" if state.to_play == "b" else "b"] = -outcome_for_to_play
    final_board_target = np.asarray(state.terminal_ownership(), dtype=np.float32).reshape(-1)
    for example in examples:
        example["value_target"] = winner_value_by_player[example["player"]]
        example["final_board_target"] = final_board_target
    return examples, float(outcome_for_to_play)


def train_steps(
    config: CpuDemoConfig,
    model: PolicyValueNet,
    examples: list[dict[str, object]],
) -> list[dict[str, float]]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = []
    for step in range(1, config.train_steps + 1):
        batch = [random.choice(examples) for _ in range(min(config.batch_size, len(examples)))]
        device = next(model.parameters()).device
        features = torch.tensor(np.stack([item["features"] for item in batch]), dtype=torch.float32).to(device)
        policy_targets = torch.tensor(np.stack([item["policy"] for item in batch]), dtype=torch.float32).to(device)
        value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.float32).to(device)
        final_board_targets = torch.tensor(
            np.stack([item["final_board_target"] for item in batch]),
            dtype=torch.float32,
        ).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, values, final_board = policy_value_final_board(model(features))
        policy_loss = -(policy_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)
        final_board_loss = (
            F.mse_loss(final_board, final_board_targets)
            if final_board is not None and config.final_board_loss_weight > 0.0
            else torch.zeros((), dtype=torch.float32, device=device)
        )
        loss = policy_loss + value_loss + config.final_board_loss_weight * final_board_loss
        loss.backward()
        optimizer.step()

        history.append(
            {
                "step": step,
                "loss": round(float(loss.item()), 6),
                "policy_loss": round(float(policy_loss.item()), 6),
                "value_loss": round(float(value_loss.item()), 6),
                "final_board_loss": round(float(final_board_loss.item()), 6),
                "final_board_loss_weight": round(float(config.final_board_loss_weight), 6),
            }
        )
    return history


def run_demo(
    config: CpuDemoConfig,
    sgf_path: str | Path,
    trace_path: str | Path,
    dashboard_path: str | Path,
    overview_svg_path: str | Path,
) -> dict[str, object]:
    total_start = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but torch.cuda.is_available() is false")

    model = make_model(config)
    params = sum(parameter.numel() for parameter in model.parameters())
    all_examples: list[dict[str, object]] = []
    selfplay_start = time.perf_counter()
    for game_index in range(1, config.games + 1):
        examples, _ = play_game(config, model)
        for move_index, example in enumerate(examples, start=1):
            example["game"] = game_index
            example["move_in_game"] = move_index
        all_examples.extend(examples)
    selfplay_seconds = time.perf_counter() - selfplay_start
    train_start = time.perf_counter()
    loss_history = train_steps(config, model, all_examples)
    train_seconds = time.perf_counter() - train_start
    metrics = loss_history[-1]
    write_start = time.perf_counter()
    write_sgf(sgf_path, config, all_examples)
    trace = build_trace(config, all_examples)
    write_json(trace_path, trace)
    write_dashboard(dashboard_path, config, params, all_examples, loss_history, sgf_path, trace_path)
    write_overview_svg(overview_svg_path, config, params, all_examples, loss_history)
    write_seconds = time.perf_counter() - write_start
    total_seconds = time.perf_counter() - total_start

    first = all_examples[0]
    return {
        "config": asdict(config),
        "parameters": params,
        "positions": len(all_examples),
        "sgf_path": str(sgf_path),
        "trace_path": str(trace_path),
        "dashboard_path": str(dashboard_path),
        "overview_svg_path": str(overview_svg_path),
        "first_position": {
            "root_value": first["root_value"],
            "top_actions": first["top_actions"],
            "chosen_action": first["chosen_action"],
        },
        "train_metrics": metrics,
        "loss_history": loss_history,
        "timing": {
            "selfplay_seconds": round(selfplay_seconds, 3),
            "train_seconds": round(train_seconds, 3),
            "write_seconds": round(write_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "positions_per_second": round(len(all_examples) / max(selfplay_seconds, 1e-9), 3),
        },
    }


def build_trace(config: CpuDemoConfig, examples: list[dict[str, object]]) -> dict[str, object]:
    moves = []
    for index, example in enumerate(examples, start=1):
        action = int(example["chosen_action"])
        policy = np.asarray(example["policy"], dtype=np.float32)
        entropy = float(-(policy * np.log(np.clip(policy, 1e-9, 1.0))).sum())
        moves.append(
            {
                "move_number": index,
                "game": int(example.get("game", 1)),
                "worker_id": int(example.get("worker_id", 1)),
                "local_game": int(example.get("local_game", example.get("game", 1))),
                "move_in_game": int(example.get("move_in_game", index)),
                "player": example["player"],
                "chosen_action": action,
                "chosen_move": action_to_gtp(action, config.board_size),
                "root_value": example["root_value"],
                "value_target": example["value_target"],
                "policy_entropy": round(entropy, 4),
                "top_actions": example["top_actions"],
                "is_pass": bool(example.get("is_pass", False)),
                "captures": int(example.get("captures", 0)),
            }
        )
    return {"config": asdict(config), "moves": moves}


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_sgf(path: str | Path, config: CpuDemoConfig, examples: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[dict[str, object]]] = {}
    for example in examples:
        groups.setdefault(int(example.get("game", 1)), []).append(example)

    trees = []
    for game_id, game_examples in sorted(groups.items()):
        game_examples.sort(key=lambda item: int(item.get("move_in_game", 0)))
        comment = f"DiamondGo baby-zero demo with {config.rules_backend} rules."
        if len(groups) > 1:
            comment = f"{comment} Game {game_id} in this SGF collection."
        nodes = [
            (
                f"(;GM[1]FF[4]CA[UTF-8]AP[DiamondGo:cpu-demo]"
                f"SZ[{config.board_size}]KM[{config.score_komi}]"
                f"PB[DiamondGo random-init]PW[DiamondGo random-init]"
                f"C[{_escape_sgf_text(comment)}]"
            )
        ]
        for example in game_examples:
            color = "B" if example["player"] == "b" else "W"
            move = _action_to_sgf(int(example["chosen_action"]), config.board_size)
            comment = _format_comment(example)
            nodes.append(f";{color}[{move}]C[{_escape_sgf_text(comment)}]")
        nodes.append(")")
        trees.append("".join(nodes))
    path.write_text("".join(trees), encoding="utf-8")


def _format_comment(example: dict[str, object]) -> str:
    lines = [
        f"root_value: {example['root_value']}",
        f"captures: {int(example.get('captures', 0))}",
        "top actions:",
    ]
    for action in example["top_actions"]:
        lines.append(
            f"- {action['move']}: visits={action['visits']} "
            f"prior={action['prior']} value={action['value']}"
        )
    return "\n".join(lines)


def action_to_gtp(action: int, board_size: int) -> str:
    if action == board_size * board_size:
        return "pass"
    row, col = divmod(action, board_size)
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    return f"{letters[col]}{board_size - row}"


def _action_to_sgf(action: int, board_size: int) -> str:
    if action == board_size * board_size:
        return ""
    row, col = divmod(action, board_size)
    letters = "abcdefghijklmnopqrstuvwxyz"
    return f"{letters[col]}{letters[row]}"


def _escape_sgf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("]", "\\]")


def write_dashboard(
    path: str | Path,
    config: CpuDemoConfig,
    params: int,
    examples: list[dict[str, object]],
    loss_history: list[dict[str, float]],
    sgf_path: str | Path,
    trace_path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_dashboard(config, params, examples, loss_history, sgf_path, trace_path)
    path.write_text(html_text, encoding="utf-8")


def _build_dashboard_state(config: CpuDemoConfig, examples: list[dict[str, object]]) -> dict[str, object]:
    moves = []
    for index, example in enumerate(examples, start=1):
        action = int(example["chosen_action"])
        top_actions = []
        for item in list(example["top_actions"])[:5]:
            move = str(item["move"])
            top_actions.append(
                {
                    "move": move,
                    "action": _gtp_to_action(move, config.board_size),
                    "visits": int(item["visits"]),
                    "prior": float(item["prior"]),
                    "value": float(item["value"]),
                }
            )
        moves.append(
            {
                "index": index,
                "game": int(example.get("game", 1)),
                "moveInGame": int(example.get("move_in_game", index)),
                "player": str(example["player"]),
                "action": action,
                "chosenMove": action_to_gtp(action, config.board_size),
                "rootValue": float(example["root_value"]),
                "topActions": top_actions,
            }
        )
    return {
        "boardSize": config.board_size,
        "cell": 42,
        "pad": 34,
        "moves": moves,
    }


def _gtp_to_action(move: str, board_size: int) -> int:
    if move == "pass":
        return board_size * board_size
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    col = letters.index(move[0].upper())
    row = board_size - int(move[1:])
    return row * board_size + col


def write_overview_svg(
    path: str | Path,
    config: CpuDemoConfig,
    params: int,
    examples: list[dict[str, object]],
    loss_history: list[dict[str, float]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = render_overview_svg(config, params, examples, loss_history)
    path.write_text(svg, encoding="utf-8")


def render_overview_svg(
    config: CpuDemoConfig,
    params: int,
    examples: list[dict[str, object]],
    loss_history: list[dict[str, float]],
) -> str:
    board = _render_board_svg_at(config, examples, x0=40, y0=105, cell=34, pad=24)
    values = [float(item["root_value"]) for item in examples]
    first_actions = list(examples[0]["top_actions"])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760">
  <rect width="1200" height="760" fill="#f7f4ef"/>
  <text x="40" y="48" font-family="Segoe UI, Arial" font-size="28" font-weight="700" fill="#1d2a2e">DiamondGo CPU Baby-Zero 9x9</text>
  <text x="40" y="76" font-family="Segoe UI, Arial" font-size="14" fill="#657174">simplified rules smoke run - {len(examples)} positions - {params:,} parameters - {config.simulations} simulations/move</text>
  <text x="40" y="102" font-family="Segoe UI, Arial" font-size="18" font-weight="700" fill="#1d2a2e">Final board</text>
  {board}
  <text x="430" y="102" font-family="Segoe UI, Arial" font-size="18" font-weight="700" fill="#1d2a2e">Root value by move</text>
  {_render_line_svg_at(values, x0=430, y0=122, width=700, height=190)}
  <text x="430" y="350" font-family="Segoe UI, Arial" font-size="18" font-weight="700" fill="#1d2a2e">First search top moves</text>
  {_render_top_actions_svg_at(first_actions, x0=430, y0=374, width=660)}
  <text x="430" y="580" font-family="Segoe UI, Arial" font-size="18" font-weight="700" fill="#1d2a2e">Training loss history</text>
  {_render_training_history_svg_at(loss_history, x0=430, y0=604, width=660, height=110)}
</svg>
"""


def render_dashboard(
    config: CpuDemoConfig,
    params: int,
    examples: list[dict[str, object]],
    loss_history: list[dict[str, float]],
    sgf_path: str | Path,
    trace_path: str | Path,
) -> str:
    metrics = loss_history[-1]
    move_rows = "\n".join(_render_move_row(config, index, example) for index, example in enumerate(examples, 1))
    dashboard_state = _build_dashboard_state(config, examples)
    dashboard_json = json.dumps(dashboard_state, separators=(",", ":"))
    cards = [
        ("Board", f"{config.board_size}x{config.board_size}"),
        ("Rules", config.rules_backend),
        ("Positions", str(len(examples))),
        ("Params", f"{params:,}"),
        ("Simulations", f"{config.simulations}/move"),
        ("Loss", str(metrics["loss"])),
        ("Policy loss", str(metrics["policy_loss"])),
        ("Value loss", str(metrics["value_loss"])),
    ]
    card_html = "\n".join(f"<div class='card'><b>{label}</b><span>{value}</span></div>" for label, value in cards)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DiamondGo CPU Baby-Zero Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: #222;
      background: #f5f7f6;
    }}
    header {{
      padding: 28px 36px 18px;
      background: #1d2a2e;
      color: #fff;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    main {{ padding: 24px 36px 40px; max-width: 1240px; }}
    .subtle {{ color: #657174; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(136px, 1fr));
      gap: 10px;
      margin: 18px 0 8px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #dbe2df;
      border-radius: 8px;
      padding: 12px;
    }}
    .card b {{ display: block; font-size: 12px; color: #687173; margin-bottom: 6px; }}
    .card span {{ font-size: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(360px, 520px) minmax(420px, 1fr);
      gap: 22px;
      align-items: start;
    }}
    section {{
      background: #fff;
      border: 1px solid #dbe2df;
      border-radius: 8px;
      padding: 16px;
    }}
    .replay {{
      display: grid;
      grid-template-columns: minmax(340px, 520px) minmax(320px, 1fr);
      gap: 22px;
      align-items: start;
    }}
    .board-wrap {{
      background: #e6bf78;
      border: 1px solid #b99255;
      border-radius: 8px;
      padding: 12px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto auto 1fr auto;
      gap: 10px;
      align-items: center;
      margin: 14px 0 4px;
    }}
    button {{
      min-width: 42px;
      height: 34px;
      border: 1px solid #b8c4c0;
      border-radius: 6px;
      background: #ffffff;
      color: #203033;
      font-size: 17px;
      cursor: pointer;
    }}
    button:hover {{ background: #edf3f1; }}
    input[type="range"] {{
      width: 100%;
      accent-color: #2f6f73;
    }}
    .move-panel {{
      display: grid;
      gap: 12px;
    }}
    .move-card {{
      border: 1px solid #dbe2df;
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfb;
    }}
    .move-card b {{ display: block; color: #657174; font-size: 12px; margin-bottom: 4px; }}
    .move-card span {{ font-size: 18px; }}
    .candidate-list {{
      display: grid;
      gap: 9px;
      margin-top: 8px;
    }}
    .candidate {{
      display: grid;
      grid-template-columns: 28px 52px 1fr;
      gap: 8px;
      align-items: center;
      font-size: 13px;
    }}
    .rank {{
      width: 24px;
      height: 24px;
      border-radius: 999px;
      display: inline-grid;
      place-items: center;
      color: #fff;
      font-weight: 700;
      background: #2f6f73;
    }}
    tr.active {{ background: #eaf3f1; }}
    svg {{ max-width: 100%; height: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #fff;
    }}
    th, td {{
      border-bottom: 1px solid #e8eeee;
      padding: 8px 7px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: #4f5c5f; font-weight: 700; background: #faf8f4; }}
    code {{
      background: #e9efed;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    .files {{ margin: 10px 0 0; }}
    .files li {{ margin: 6px 0; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>DiamondGo CPU Baby-Zero 9x9</h1>
    <div class="subtle">Local self-play with {config.rules_backend} rules, MCTS diagnostics, and optimizer updates.</div>
  </header>
  <main>
    <div class="cards">{card_html}</div>
    <section>
      <h2>Move Replay And Top 3 Candidates</h2>
      <div class="replay">
        <div>
          <div class="board-wrap">
            <svg id="replay-board" viewBox="0 0 404 404" role="img" aria-label="move replay board"></svg>
          </div>
          <div class="controls">
            <button type="button" id="prev-move" title="Previous move">‹</button>
            <button type="button" id="next-move" title="Next move">›</button>
            <input id="move-slider" type="range" min="0" max="{max(len(examples), 1) - 1}" value="0">
            <span id="move-count"></span>
          </div>
        </div>
        <div class="move-panel">
          <div class="move-card"><b>Current move</b><span id="current-move"></span></div>
          <div class="move-card"><b>Root value</b><span id="current-value"></span></div>
          <div class="move-card">
            <b>Top 3 search candidates before this move</b>
            <div id="candidate-list" class="candidate-list"></div>
          </div>
        </div>
      </div>
    </section>
    <div class="grid">
      <section>
        <h2>Final Board And Move Order</h2>
        {_render_board_svg(config, examples)}
      </section>
      <section>
        <h2>Root Value By Move</h2>
        {_render_line_svg([float(item["root_value"]) for item in examples], "root value")}
        <h2>Training Loss History</h2>
        {_render_training_history_svg(loss_history)}
        <h2>Final Loss Breakdown</h2>
        {_render_loss_svg(metrics)}
      </section>
    </div>
    <section>
      <h2>First Search Top Moves</h2>
      {_render_top_actions_svg(examples[0]["top_actions"])}
    </section>
    <section>
      <h2>Move Trace</h2>
      <table>
        <thead><tr><th>#</th><th>Player</th><th>Chosen</th><th>Root value</th><th>Policy entropy</th><th>Top search actions</th></tr></thead>
        <tbody>{move_rows}</tbody>
      </table>
    </section>
    <section>
      <h2>Files</h2>
      <ul class="files">
        <li>SGF replay: <code>{html.escape(str(sgf_path))}</code></li>
        <li>Trace JSON: <code>{html.escape(str(trace_path))}</code></li>
      </ul>
    </section>
  </main>
  <script>
    const dashboardState = {dashboard_json};
    const letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    const board = document.getElementById("replay-board");
    const slider = document.getElementById("move-slider");
    const prevButton = document.getElementById("prev-move");
    const nextButton = document.getElementById("next-move");
    const moveCount = document.getElementById("move-count");
    const currentMove = document.getElementById("current-move");
    const currentValue = document.getElementById("current-value");
    const candidateList = document.getElementById("candidate-list");
    const rows = Array.from(document.querySelectorAll("tr[data-index]"));

    function makeSvg(name, attributes = {{}}, text = "") {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
      if (text) node.textContent = text;
      return node;
    }}

    function pointFor(action) {{
      const row = Math.floor(action / dashboardState.boardSize);
      const col = action % dashboardState.boardSize;
      return {{
        x: dashboardState.pad + col * dashboardState.cell,
        y: dashboardState.pad + row * dashboardState.cell
      }};
    }}

    function drawBoard(index) {{
      const move = dashboardState.moves[index];
      const size = dashboardState.boardSize;
      const gameMoves = dashboardState.moves.filter((item, itemIndex) =>
        itemIndex <= index && item.game === move.game && item.action < size * size
      );
      board.replaceChildren();
      for (let i = 0; i < size; i += 1) {{
        const pos = dashboardState.pad + i * dashboardState.cell;
        board.appendChild(makeSvg("line", {{ x1: dashboardState.pad, y1: pos, x2: 370, y2: pos, stroke: "#765933" }}));
        board.appendChild(makeSvg("line", {{ x1: pos, y1: dashboardState.pad, x2: pos, y2: 370, stroke: "#765933" }}));
      }}
      move.topActions.slice(0, 3).filter((item) => item.action < size * size).forEach((item, rank) => {{
        const p = pointFor(item.action);
        board.appendChild(makeSvg("circle", {{
          cx: p.x, cy: p.y, r: 13 + rank * 2, fill: "none", stroke: ["#2f6f73", "#8a6fb0", "#b5654d"][rank],
          "stroke-width": 3, "stroke-dasharray": rank === 0 ? "0" : "5 4"
        }}));
        board.appendChild(makeSvg("text", {{
          x: p.x, y: p.y + 5, "text-anchor": "middle", "font-size": 12, "font-weight": 700,
          fill: ["#2f6f73", "#8a6fb0", "#b5654d"][rank]
        }}, String(rank + 1)));
      }});
      gameMoves.forEach((item) => {{
        const p = pointFor(item.action);
        const isBlack = item.player === "b";
        board.appendChild(makeSvg("circle", {{
          cx: p.x, cy: p.y, r: 16, fill: isBlack ? "#161616" : "#f4f1e8",
          stroke: isBlack ? "#161616" : "#7b756c", "stroke-width": 1.5
        }}));
        board.appendChild(makeSvg("text", {{
          x: p.x, y: p.y + 4, "text-anchor": "middle", "font-size": 12,
          fill: isBlack ? "#fff" : "#111"
        }}, String(item.moveInGame)));
      }});
    }}

    function render(index) {{
      const safeIndex = Math.max(0, Math.min(dashboardState.moves.length - 1, index));
      const move = dashboardState.moves[safeIndex];
      slider.value = String(safeIndex);
      moveCount.textContent = `${{safeIndex + 1}} / ${{dashboardState.moves.length}}`;
      currentMove.textContent = `Game ${{move.game}}, ${{move.player === "b" ? "Black" : "White"}} ${{move.chosenMove}}`;
      currentValue.textContent = move.rootValue.toFixed(4);
      candidateList.replaceChildren(...move.topActions.slice(0, 3).map((item, rank) => {{
        const row = document.createElement("div");
        row.className = "candidate";
        row.innerHTML = `<span class="rank">${{rank + 1}}</span><b>${{item.move}}</b><span>visits=${{item.visits}} prior=${{item.prior}} value=${{item.value}}</span>`;
        return row;
      }}));
      rows.forEach((row) => row.classList.toggle("active", Number(row.dataset.index) === safeIndex));
      drawBoard(safeIndex);
    }}

    prevButton.addEventListener("click", () => render(Number(slider.value) - 1));
    nextButton.addEventListener("click", () => render(Number(slider.value) + 1));
    slider.addEventListener("input", () => render(Number(slider.value)));
    render(0);
  </script>
</body>
</html>
"""


def _render_board_svg(config: CpuDemoConfig, examples: list[dict[str, object]]) -> str:
    size = config.board_size
    cell = 42
    pad = 34
    width = pad * 2 + cell * (size - 1)
    stones: dict[int, tuple[int, str]] = {}
    for number, example in enumerate(examples, start=1):
        action = int(example["chosen_action"])
        if action < size * size:
            stones[action] = (number, str(example["player"]))
    lines = []
    for i in range(size):
        pos = pad + i * cell
        lines.append(f"<line x1='{pad}' y1='{pos}' x2='{width - pad}' y2='{pos}' stroke='#7b6140'/>")
        lines.append(f"<line x1='{pos}' y1='{pad}' x2='{pos}' y2='{width - pad}' stroke='#7b6140'/>")
    stone_svg = []
    for action, (number, player) in stones.items():
        row, col = divmod(action, size)
        x = pad + col * cell
        y = pad + row * cell
        fill = "#161616" if player == "b" else "#f4f1e8"
        stroke = "#161616" if player == "b" else "#7b756c"
        text_fill = "#fff" if player == "b" else "#111"
        stone_svg.append(f"<circle cx='{x}' cy='{y}' r='16' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>")
        stone_svg.append(
            f"<text x='{x}' y='{y + 4}' text-anchor='middle' font-size='12' fill='{text_fill}'>{number}</text>"
        )
    return f"<svg viewBox='0 0 {width} {width}' role='img'>{''.join(lines)}{''.join(stone_svg)}</svg>"


def _render_board_svg_at(
    config: CpuDemoConfig,
    examples: list[dict[str, object]],
    x0: int,
    y0: int,
    cell: int,
    pad: int,
) -> str:
    size = config.board_size
    board_width = pad * 2 + cell * (size - 1)
    fill = f"<rect x='{x0}' y='{y0}' width='{board_width}' height='{board_width}' rx='8' fill='#d4a85f'/>"
    lines = []
    for i in range(size):
        pos = pad + i * cell
        lines.append(
            f"<line x1='{x0 + pad}' y1='{y0 + pos}' x2='{x0 + board_width - pad}' y2='{y0 + pos}' stroke='#6f5636'/>"
        )
        lines.append(
            f"<line x1='{x0 + pos}' y1='{y0 + pad}' x2='{x0 + pos}' y2='{y0 + board_width - pad}' stroke='#6f5636'/>"
        )
    stones: dict[int, tuple[int, str]] = {}
    for number, example in enumerate(examples, start=1):
        action = int(example["chosen_action"])
        if action < size * size:
            stones[action] = (number, str(example["player"]))
    stone_svg = []
    for action, (number, player) in stones.items():
        row, col = divmod(action, size)
        x = x0 + pad + col * cell
        y = y0 + pad + row * cell
        fill_color = "#161616" if player == "b" else "#f4f1e8"
        stroke = "#161616" if player == "b" else "#7b756c"
        text_fill = "#fff" if player == "b" else "#111"
        stone_svg.append(f"<circle cx='{x}' cy='{y}' r='13' fill='{fill_color}' stroke='{stroke}' stroke-width='1.3'/>")
        stone_svg.append(
            f"<text x='{x}' y='{y + 4}' text-anchor='middle' font-family='Segoe UI, Arial' font-size='10' fill='{text_fill}'>{number}</text>"
        )
    return fill + "".join(lines) + "".join(stone_svg)


def _render_line_svg(values: list[float], label: str) -> str:
    width, height, pad = 640, 220, 32
    if not values:
        return ""
    low = min(values + [-1.0])
    high = max(values + [1.0])
    span = high - low or 1.0
    points = []
    for index, value in enumerate(values):
        x = pad + index * ((width - 2 * pad) / max(1, len(values) - 1))
        y = height - pad - ((value - low) / span) * (height - 2 * pad)
        points.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    zero_y = height - pad - ((0 - low) / span) * (height - 2 * pad)
    dots = "".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.2' fill='#2f6f73'/>" for x, y in points)
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(label)}'>"
        f"<line x1='{pad}' y1='{zero_y:.1f}' x2='{width - pad}' y2='{zero_y:.1f}' stroke='#d0c8bd'/>"
        f"<polyline points='{polyline}' fill='none' stroke='#2f6f73' stroke-width='2.5'/>"
        f"{dots}<text x='{pad}' y='20' font-size='12' fill='#687173'>{html.escape(label)}</text></svg>"
    )


def _render_line_svg_at(values: list[float], x0: int, y0: int, width: int, height: int) -> str:
    if not values:
        return ""
    pad = 30
    low = min(values + [-1.0])
    high = max(values + [1.0])
    span = high - low or 1.0
    points = []
    for index, value in enumerate(values):
        x = x0 + pad + index * ((width - 2 * pad) / max(1, len(values) - 1))
        y = y0 + height - pad - ((value - low) / span) * (height - 2 * pad)
        points.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    zero_y = y0 + height - pad - ((0 - low) / span) * (height - 2 * pad)
    dots = "".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='#2f6f73'/>" for x, y in points)
    return (
        f"<rect x='{x0}' y='{y0}' width='{width}' height='{height}' rx='8' fill='#ffffff' stroke='#ded8ce'/>"
        f"<line x1='{x0 + pad}' y1='{zero_y:.1f}' x2='{x0 + width - pad}' y2='{zero_y:.1f}' stroke='#d0c8bd'/>"
        f"<polyline points='{polyline}' fill='none' stroke='#2f6f73' stroke-width='2.5'/>"
        f"{dots}"
    )


def _render_loss_svg(metrics: dict[str, float]) -> str:
    items = [("policy", metrics["policy_loss"], "#52796f"), ("value", metrics["value_loss"], "#b5654d")]
    max_value = max(value for _, value, _ in items) or 1.0
    width, height = 640, 130
    rows = []
    for index, (label, value, color) in enumerate(items):
        y = 26 + index * 44
        bar_width = 420 * (value / max_value)
        rows.append(f"<text x='0' y='{y + 15}' font-size='13'>{label}</text>")
        rows.append(f"<rect x='78' y='{y}' width='{bar_width:.1f}' height='22' fill='{color}' rx='4'/>")
        rows.append(f"<text x='{86 + bar_width:.1f}' y='{y + 16}' font-size='13'>{value}</text>")
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(rows)}</svg>"


def _render_training_history_svg(history: list[dict[str, float]]) -> str:
    values = [float(item["loss"]) for item in history]
    return _render_line_svg(values, "total loss")


def _render_training_history_svg_at(
    history: list[dict[str, float]],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> str:
    values = [float(item["loss"]) for item in history]
    if not values:
        return ""
    low = min(values)
    high = max(values)
    span = high - low or 1.0
    pad = 22
    points = []
    for index, value in enumerate(values):
        x = x0 + pad + index * ((width - 2 * pad) / max(1, len(values) - 1))
        y = y0 + height - pad - ((value - low) / span) * (height - 2 * pad)
        points.append((x, y))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dots = "".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='#7d5a9b'/>" for x, y in points)
    return (
        f"<rect x='{x0}' y='{y0}' width='{width}' height='{height}' rx='8' fill='#ffffff' stroke='#ded8ce'/>"
        f"<polyline points='{polyline}' fill='none' stroke='#7d5a9b' stroke-width='2.5'/>"
        f"{dots}<text x='{x0 + 16}' y='{y0 + 24}' font-family='Segoe UI, Arial' font-size='12' fill='#687173'>"
        f"loss {values[0]:.3f} -> {values[-1]:.3f}</text>"
    )


def _render_loss_svg_at(metrics: dict[str, float], x0: int, y0: int, width: int) -> str:
    items = [("policy", metrics["policy_loss"], "#52796f"), ("value", metrics["value_loss"], "#b5654d")]
    max_value = max(value for _, value, _ in items) or 1.0
    rows = [f"<rect x='{x0}' y='{y0}' width='{width}' height='110' rx='8' fill='#ffffff' stroke='#ded8ce'/>"]
    for index, (label, value, color) in enumerate(items):
        y = y0 + 26 + index * 38
        bar_width = (width - 190) * (value / max_value)
        rows.append(f"<text x='{x0 + 18}' y='{y + 15}' font-family='Segoe UI, Arial' font-size='13'>{label}</text>")
        rows.append(f"<rect x='{x0 + 88}' y='{y}' width='{bar_width:.1f}' height='21' fill='{color}' rx='4'/>")
        rows.append(
            f"<text x='{x0 + 96 + bar_width:.1f}' y='{y + 15}' font-family='Segoe UI, Arial' font-size='13'>{value}</text>"
        )
    return "".join(rows)


def _render_top_actions_svg(actions: object) -> str:
    rows = []
    action_rows = list(actions) if isinstance(actions, list) else []
    max_visits = max([int(item["visits"]) for item in action_rows] + [1])
    width, height = 640, 36 + 32 * len(action_rows)
    for index, item in enumerate(action_rows):
        y = 28 + index * 32
        visits = int(item["visits"])
        bar = 360 * visits / max_visits
        rows.append(f"<text x='0' y='{y + 16}' font-size='13'>{html.escape(str(item['move']))}</text>")
        rows.append(f"<rect x='60' y='{y}' width='{bar:.1f}' height='20' fill='#4c956c' rx='4'/>")
        rows.append(
            f"<text x='{74 + bar:.1f}' y='{y + 15}' font-size='12'>"
            f"visits={visits} prior={item['prior']} value={item['value']}</text>"
        )
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{''.join(rows)}</svg>"


def _render_top_actions_svg_at(actions: list[dict[str, object]], x0: int, y0: int, width: int) -> str:
    max_visits = max([int(item["visits"]) for item in actions] + [1])
    rows = [f"<rect x='{x0}' y='{y0}' width='{width}' height='174' rx='8' fill='#ffffff' stroke='#ded8ce'/>"]
    for index, item in enumerate(actions):
        y = y0 + 22 + index * 29
        visits = int(item["visits"])
        bar = (width - 260) * visits / max_visits
        rows.append(
            f"<text x='{x0 + 18}' y='{y + 15}' font-family='Segoe UI, Arial' font-size='13'>{html.escape(str(item['move']))}</text>"
        )
        rows.append(f"<rect x='{x0 + 70}' y='{y}' width='{bar:.1f}' height='19' fill='#4c956c' rx='4'/>")
        rows.append(
            f"<text x='{x0 + 82 + bar:.1f}' y='{y + 14}' font-family='Segoe UI, Arial' font-size='12'>"
            f"visits={visits} prior={item['prior']} value={item['value']}</text>"
        )
    return "".join(rows)


def _render_move_row(config: CpuDemoConfig, index: int, example: dict[str, object]) -> str:
    policy = np.asarray(example["policy"], dtype=np.float32)
    entropy = float(-(policy * np.log(np.clip(policy, 1e-9, 1.0))).sum())
    top_actions = example["top_actions"]
    top_text = "; ".join(
        f"{item['move']} v={item['visits']} p={item['prior']} q={item['value']}"
        for item in top_actions
    )
    return (
        f"<tr data-index='{index - 1}'>"
        f"<td>{index}</td>"
        f"<td>{'Black' if example['player'] == 'b' else 'White'}</td>"
        f"<td>{html.escape(action_to_gtp(int(example['chosen_action']), config.board_size))}</td>"
        f"<td>{example['root_value']}</td>"
        f"<td>{entropy:.4f}</td>"
        f"<td>{html.escape(top_text)}</td>"
        "</tr>"
    )


def main() -> None:
    args = build_parser().parse_args()
    config = replace(
        CpuDemoConfig(),
        games=args.games,
        komi=args.komi,
        score_komi=args.score_komi,
        input_komi=args.input_komi,
        history_moves=args.history_moves,
        terminal_dead_stone_cleanup=args.terminal_dead_stone_cleanup,
        score_margin_reward_scale=args.score_margin_reward_scale,
        final_board_loss_weight=args.final_board_loss_weight,
        simulations=args.simulations,
        max_moves=args.max_moves,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        seed=args.seed,
        device=args.device,
        rules_backend=args.rules,
    )
    summary = run_demo(config, args.sgf, args.trace, args.dashboard, args.overview_svg)
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("DiamondGo CPU baby-zero demo")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
