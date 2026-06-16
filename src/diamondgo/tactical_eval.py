from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from diamondgo.batched_demo import BatchedConfig, make_model, run_batched_mcts
from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_SCORE_KOMI
from diamondgo.demo_cpu import action_to_gtp
from diamondgo.overnight_train import load_checkpoint
from diamondgo.rules import BLACK, WHITE, SgfmillRules


@dataclass(frozen=True)
class TacticalCase:
    name: str
    to_play: str
    black: tuple[tuple[int, int], ...]
    white: tuple[tuple[int, int], ...]
    target: tuple[int, int]
    note: str


TACTICAL_CASES = [
    TacticalCase(
        name="black_capture_one_stone",
        to_play=BLACK,
        black=((0, 1), (1, 0), (2, 1)),
        white=((1, 1),),
        target=(1, 2),
        note="Black can capture the white stone by filling its last liberty.",
    ),
    TacticalCase(
        name="white_capture_one_stone",
        to_play=WHITE,
        black=((4, 4),),
        white=((3, 4), (4, 3), (5, 4)),
        target=(4, 5),
        note="White can capture the black stone by filling its last liberty.",
    ),
    TacticalCase(
        name="black_escape_atari",
        to_play=BLACK,
        black=((2, 2),),
        white=((1, 2), (2, 1), (3, 2)),
        target=(2, 3),
        note="Black is in atari and can extend to the only liberty.",
    ),
    TacticalCase(
        name="white_escape_atari",
        to_play=WHITE,
        black=((5, 4), (6, 3), (7, 4)),
        white=((6, 4),),
        target=(6, 5),
        note="White is in atari and can extend to the only liberty.",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fixed 9x9 tactical probes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true")
    return parser


def action_for(point: tuple[int, int], board_size: int) -> int:
    row, col = point
    return row * board_size + col


def make_case_state(
    case: TacticalCase,
    board_size: int,
    komi: float,
    score_komi: float,
    input_komi: bool,
    history_moves: int,
    terminal_dead_stone_cleanup: bool,
    score_margin_reward_scale: float,
) -> SgfmillRules:
    state = SgfmillRules(
        size=board_size,
        komi=komi,
        score_komi=score_komi,
        input_komi=input_komi,
        history_moves=history_moves,
        terminal_dead_stone_cleanup=terminal_dead_stone_cleanup,
        score_margin_reward_scale=score_margin_reward_scale,
    )
    for row, col in case.black:
        state.board.board[row][col] = BLACK
    for row, col in case.white:
        state.board.board[row][col] = WHITE
    state.board._is_empty = False
    state.to_play = case.to_play
    state._passes = 0
    state._ko_forbidden = None
    state._legal_actions_cache = None
    state._sync_board_array()
    return state


def load_model_config(checkpoint: Path, device: str) -> tuple[BatchedConfig, torch.nn.Module]:
    payload = torch.load(checkpoint, map_location=torch.device(device))
    raw = dict(payload["config"])
    config = BatchedConfig(
        board_size=int(raw.get("board_size", 9)),
        komi=float(raw.get("komi", DEFAULT_9X9_KOMI)),
        score_komi=float(raw.get("score_komi", raw.get("komi", DEFAULT_9X9_SCORE_KOMI))),
        input_komi=bool(raw.get("input_komi", True)),
        history_moves=int(raw.get("history_moves", 0)),
        terminal_dead_stone_cleanup=bool(raw.get("terminal_dead_stone_cleanup", False)),
        score_margin_reward_scale=float(raw.get("score_margin_reward_scale", 0.0)),
        channels=int(raw.get("channels", 32)),
        residual_blocks=int(raw.get("residual_blocks", 2)),
        simulations=1,
        max_moves=int(raw.get("max_moves", 120)),
        games=1,
        train_steps=0,
        batch_size=256,
        c_puct=float(raw.get("c_puct", 1.5)),
        temperature=1.0,
        seed=int(raw.get("seed", 1)),
        device=device,
        rules_backend=str(raw.get("rules_backend", "sgfmill")),
    )
    model = make_model(config)
    load_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()))
    model.eval()
    return config, model


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config, model = load_model_config(checkpoint, args.device)
    search_config = BatchedConfig(
        **{
            **config.__dict__,
            "simulations": args.simulations,
            "games": 1,
            "train_steps": 0,
        }
    )
    rows = []
    for case in TACTICAL_CASES:
        state = make_case_state(
            case,
            config.board_size,
            config.komi,
            config.score_komi,
            config.input_komi,
            config.history_moves,
            config.terminal_dead_stone_cleanup,
            config.score_margin_reward_scale,
        )
        root = run_batched_mcts(model, [state], search_config, stats={})[0]
        target_action = action_for(case.target, config.board_size)
        top_actions = root.top_actions(config.board_size, limit=10)
        top_action_ids = [
            action
            for action, _child in sorted(
                root.children.items(),
                key=lambda item: item[1].visit_count,
                reverse=True,
            )[:10]
        ]
        rows.append(
            {
                "case": case.name,
                "to_play": case.to_play,
                "target": action_to_gtp(target_action, config.board_size),
                "target_action": target_action,
                "top1_hit": bool(top_action_ids and top_action_ids[0] == target_action),
                "top3_hit": target_action in top_action_ids[:3],
                "target_rank": (
                    top_action_ids.index(target_action) + 1
                    if target_action in top_action_ids
                    else None
                ),
                "top_actions": top_actions,
                "note": case.note,
            }
        )
    summary = {
        "checkpoint": str(checkpoint),
        "simulations": args.simulations,
        "komi": config.komi,
        "score_komi": config.score_komi,
        "input_komi": config.input_komi,
        "history_moves": config.history_moves,
        "cases": rows,
        "top1_hits": sum(1 for item in rows if item["top1_hit"]),
        "top3_hits": sum(1 for item in rows if item["top3_hit"]),
        "case_count": len(rows),
    }
    (out_dir / "tactical_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# DiamondGo tactical probes",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- simulations: {args.simulations}",
        f"- komi: {config.komi}",
        f"- score_komi: {config.score_komi}",
        f"- input_komi: {config.input_komi}",
        f"- history_moves: {config.history_moves}",
        "",
        "| case | target | top1 | top3 | rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in rows:
        report.append(
            f"| {item['case']} | {item['target']} | {item['top1_hit']} | "
            f"{item['top3_hit']} | {item['target_rank']} |"
        )
    (out_dir / "tactical_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
