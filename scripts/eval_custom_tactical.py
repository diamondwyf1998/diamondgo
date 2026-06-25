from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from diamondgo.batched_demo import BatchedConfig, make_model, run_batched_mcts
from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_MAX_MOVES, DEFAULT_9X9_SCORE_KOMI
from diamondgo.demo_cpu import action_to_gtp
from diamondgo.model import load_model_state_dict
from diamondgo.rules import BLACK, WHITE, SgfmillRules


LETTERS = "ABCDEFGHJ"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate custom DiamondGo tactical cases.")
    parser.add_argument("--cases", required=True, help="JSON exported from puzzle-author.html")
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint .pt path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", action="store_true")
    return parser


def point_to_action(point: list[int] | tuple[int, int], board_size: int) -> int:
    row, col = int(point[0]), int(point[1])
    return row * board_size + col


def point_to_gtp(point: list[int] | tuple[int, int], board_size: int) -> str:
    return action_to_gtp(point_to_action(point, board_size), board_size)


def load_cases(path: Path) -> tuple[int, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    board_size = int(raw.get("board_size", 9)) if isinstance(raw, dict) else 9
    raw_cases = raw.get("cases", raw) if isinstance(raw, dict) else raw
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases JSON must contain a non-empty cases array")
    cases = []
    for index, item in enumerate(raw_cases, start=1):
        case = {
            "name": str(item.get("name") or f"custom_case_{index:02d}"),
            "category": str(item.get("category") or "custom"),
            "subcategory": str(item.get("subcategory") or "manual"),
            "to_play": WHITE if item.get("to_play") == WHITE else BLACK,
            "black": sorted([tuple(map(int, p)) for p in item.get("black", [])]),
            "white": sorted([tuple(map(int, p)) for p in item.get("white", [])]),
            "good": sorted([tuple(map(int, p)) for p in item.get("good", [])]),
            "bad": sorted([tuple(map(int, p)) for p in item.get("bad", [])]),
            "note": str(item.get("note") or ""),
        }
        validate_case(case, board_size)
        cases.append(case)
    return board_size, cases


def validate_case(case: dict[str, Any], board_size: int) -> None:
    occupied: dict[tuple[int, int], str] = {}
    for color in ("black", "white"):
        for row, col in case[color]:
            if not (0 <= row < board_size and 0 <= col < board_size):
                raise ValueError(f"{case['name']} has out-of-board point {(row, col)}")
            if (row, col) in occupied:
                raise ValueError(f"{case['name']} overlaps stones at {(row, col)}")
            occupied[(row, col)] = color
    for label in ("good", "bad"):
        for row, col in case[label]:
            if not (0 <= row < board_size and 0 <= col < board_size):
                raise ValueError(f"{case['name']} has out-of-board target {(row, col)}")
            if (row, col) in occupied:
                raise ValueError(f"{case['name']} target overlaps a stone at {(row, col)}")
    if not case["good"]:
        raise ValueError(f"{case['name']} has no good target point")


def make_config(raw: dict[str, Any], device: str, simulations: int) -> BatchedConfig:
    values = {
        "board_size": int(raw.get("board_size", 9)),
        "komi": float(raw.get("komi", DEFAULT_9X9_KOMI)),
        "score_komi": float(raw.get("score_komi", raw.get("komi", DEFAULT_9X9_SCORE_KOMI))),
        "input_komi": bool(raw.get("input_komi", True)),
        "history_moves": int(raw.get("history_moves", 0)),
        "terminal_dead_stone_cleanup": bool(raw.get("terminal_dead_stone_cleanup", False)),
        "score_margin_reward_scale": float(raw.get("score_margin_reward_scale", 0.0)),
        "channels": int(raw.get("channels", 32)),
        "residual_blocks": int(raw.get("residual_blocks", 2)),
        "simulations": simulations,
        "max_moves": int(raw.get("max_moves", DEFAULT_9X9_MAX_MOVES)),
        "games": 1,
        "train_steps": 0,
        "batch_size": 256,
        "c_puct": float(raw.get("c_puct", 1.5)),
        "temperature": 1.0,
        "seed": int(raw.get("seed", 1)),
        "device": device,
        "rules_backend": str(raw.get("rules_backend", "sgfmill")),
    }
    allowed = {field.name for field in fields(BatchedConfig)}
    return BatchedConfig(**{key: value for key, value in values.items() if key in allowed})


def load_model(checkpoint: Path, device: str, simulations: int) -> tuple[BatchedConfig, torch.nn.Module]:
    payload = torch.load(checkpoint, map_location=torch.device(device))
    config = make_config(dict(payload["config"]), device, simulations)
    model = make_model(config)
    load_model_state_dict(model, payload["model_state_dict"])
    model.eval()
    return config, model


def make_state(case: dict[str, Any], config: BatchedConfig) -> SgfmillRules:
    state = SgfmillRules(
        size=config.board_size,
        komi=config.komi,
        score_komi=config.score_komi,
        input_komi=config.input_komi,
        history_moves=config.history_moves,
        terminal_dead_stone_cleanup=config.terminal_dead_stone_cleanup,
        score_margin_reward_scale=config.score_margin_reward_scale,
    )
    for row, col in case["black"]:
        state.board.board[row][col] = BLACK
    for row, col in case["white"]:
        state.board.board[row][col] = WHITE
    state.board._is_empty = False
    state.to_play = case["to_play"]
    state._passes = 0
    state._ko_forbidden = None
    state._legal_actions_cache = None
    state._sync_board_array()
    return state


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        bucket = out.setdefault(
            row["category"], {"cases": 0, "top1_good": 0, "top3_good": 0, "top10_good": 0}
        )
        bucket["cases"] += 1
        bucket["top1_good"] += int(row["top1_good"])
        bucket["top3_good"] += int(row["top3_good"])
        bucket["top10_good"] += int(row["top10_good"])
    return out


def eval_checkpoint(
    checkpoint: Path,
    cases: list[dict[str, Any]],
    device: str,
    simulations: int,
) -> dict[str, Any]:
    config, model = load_model(checkpoint, device, simulations)
    rows = []
    for case in cases:
        state = make_state(case, config)
        legal = state.legal_actions()
        root = run_batched_mcts(model, [state], config, stats={})[0]
        ranked = [
            action
            for action, _child in sorted(
                root.children.items(), key=lambda item: item[1].visit_count, reverse=True
            )[:10]
        ]
        good_actions = [point_to_action(point, config.board_size) for point in case["good"]]
        bad_actions = [point_to_action(point, config.board_size) for point in case["bad"]]
        top1 = ranked[0] if ranked else None
        rows.append(
            {
                "case": case["name"],
                "category": case["category"],
                "subcategory": case["subcategory"],
                "to_play": case["to_play"],
                "black": case["black"],
                "white": case["white"],
                "good_points": case["good"],
                "bad_points": case["bad"],
                "good": [point_to_gtp(point, config.board_size) for point in case["good"]],
                "bad": [point_to_gtp(point, config.board_size) for point in case["bad"]],
                "good_legal": [bool(legal[action]) for action in good_actions],
                "bad_legal": [bool(legal[action]) for action in bad_actions],
                "top1": action_to_gtp(top1, config.board_size) if top1 is not None else None,
                "top_actions": root.top_actions(config.board_size, limit=10),
                "note": case["note"],
                "good_ranks": {
                    action_to_gtp(action, config.board_size): ranked.index(action) + 1
                    for action in good_actions
                    if action in ranked
                },
                "bad_ranks": {
                    action_to_gtp(action, config.board_size): ranked.index(action) + 1
                    for action in bad_actions
                    if action in ranked
                },
                "top1_good": bool(good_actions and top1 in good_actions),
                "top3_good": any(action in ranked[:3] for action in good_actions),
                "top10_good": any(action in ranked[:10] for action in good_actions),
            }
        )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "simulations": simulations,
        "komi": config.komi,
        "score_komi": config.score_komi,
        "input_komi": config.input_komi,
        "history_moves": config.history_moves,
        "channels": config.channels,
        "residual_blocks": config.residual_blocks,
        "category_summary": summarize(rows),
        "cases": rows,
    }


def write_report(out_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = ["# Custom DiamondGo tactical probes", ""]
    for result in results:
        lines.extend(
            [
                f"## {result['checkpoint_name']}",
                "",
                f"- simulations: `{result['simulations']}`",
                f"- model: `{result['channels']}x{result['residual_blocks']}`",
                f"- komi: `{result['komi']}`, score komi: `{result['score_komi']}`",
                "",
                "| case | target | top1 | top3 | rank |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in result["cases"]:
            ranks = ", ".join(f"{move}:{rank}" for move, rank in row["good_ranks"].items()) or "-"
            lines.append(
                f"| {row['case']} | {', '.join(row['good'])} | {row['top1_good']} | "
                f"{row['top3_good']} | {ranks} |"
            )
        lines.append("")
    (out_dir / "tactical_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_casebook(out_dir: Path, source_cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    data = json.dumps({"cases": source_cases, "results": results}, ensure_ascii=False)
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>DiamondGo custom tactical casebook</title>
<style>
:root{{color:#172026;background:#f6f4ed;font-family:system-ui,-apple-system,Segoe UI,sans-serif}}
body{{margin:0}} header{{padding:18px 22px;background:#fbfaf6;border-bottom:1px solid #d7d2c5}}
h1{{font-size:22px;margin:0}} .controls{{display:flex;gap:10px;flex-wrap:wrap;padding:14px 22px;background:#fff;border-bottom:1px solid #d7dce2}}
select{{height:34px;border:1px solid #bcc7c1;border-radius:6px;background:#fff;padding:0 10px}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:16px;padding:18px}}
article{{background:#fff;border:1px solid #d7dce2;border-radius:8px;padding:14px}}
h2{{font-size:16px;margin:0 0 4px}} .meta{{font-size:12px;color:#627d98;margin-bottom:10px}}
.layout{{display:grid;grid-template-columns:210px 1fr;gap:12px;align-items:start}}
canvas{{width:210px;height:210px;background:#d8a84f;border:1px solid #7f5539}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border-bottom:1px solid #e4e7eb;padding:5px 6px;text-align:left;vertical-align:top}}
th{{background:#eef2f0;color:#334e68}} .hit{{color:#166534;font-weight:700}} .miss{{color:#9f1239;font-weight:700}}
@media(max-width:520px){{main{{grid-template-columns:1fr;padding:12px}}.layout{{grid-template-columns:1fr}}}}
</style>
<header><h1>DiamondGo Custom Tactical Casebook</h1></header>
<div class="controls"><label>Checkpoint <select id="checkpoint"></select></label></div>
<main id="cases"></main>
<script>
const data = {data};
const checkpointSelect = document.querySelector("#checkpoint");
const root = document.querySelector("#cases");
function draw(canvas, item) {{
  const ctx=canvas.getContext("2d"), size=210, pad=18, gap=(size-pad*2)/8;
  ctx.clearRect(0,0,size,size); ctx.fillStyle="#d8a84f"; ctx.fillRect(0,0,size,size);
  ctx.strokeStyle="#4a3422"; ctx.lineWidth=1.5;
  for(let i=0;i<9;i++){{const x=pad+i*gap,y=pad+i*gap;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(size-pad,y);ctx.stroke();ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,size-pad);ctx.stroke();}}
  for(const [r,c] of [[2,2],[2,6],[4,4],[6,2],[6,6]]){{ctx.fillStyle="#3f2d20";ctx.beginPath();ctx.arc(pad+c*gap,pad+r*gap,3,0,Math.PI*2);ctx.fill();}}
  for(const [color,points] of [["black",item.black],["white",item.white]]) for(const [r,c] of points){{const x=pad+c*gap,y=pad+r*gap;ctx.beginPath();ctx.arc(x,y,gap*.38,0,Math.PI*2);ctx.fillStyle=color==="black"?"#111":"#f8fafc";ctx.fill();ctx.strokeStyle=color==="black"?"#000":"#64748b";ctx.lineWidth=1.5;ctx.stroke();}}
  for(const p of item.good_points){{const x=pad+p[1]*gap,y=pad+p[0]*gap;ctx.strokeStyle="#16a34a";ctx.lineWidth=3;ctx.strokeRect(x-8,y-8,16,16);}}
  for(const p of item.bad_points){{const x=pad+p[1]*gap,y=pad+p[0]*gap;ctx.strokeStyle="#dc2626";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-8,y-8);ctx.lineTo(x+8,y+8);ctx.moveTo(x+8,y-8);ctx.lineTo(x-8,y+8);ctx.stroke();}}
}}
function render() {{
  const result = data.results[Number(checkpointSelect.value)] || data.results[0];
  root.replaceChildren(...result.cases.map(item => {{
    const article=document.createElement("article");
    const title=document.createElement("h2"); title.textContent=item.case;
    const meta=document.createElement("div"); meta.className="meta"; meta.textContent=`${{item.category}} / ${{item.subcategory}} - to play ${{item.to_play.toUpperCase()}} - ${{item.note}}`;
    const layout=document.createElement("div"); layout.className="layout";
    const canvas=document.createElement("canvas"); canvas.width=210; canvas.height=210;
    const table=document.createElement("table");
    const status=item.top1_good?"top1 hit":item.top3_good?"top3 hit":item.top10_good?"top10 hit":"miss";
    table.innerHTML=`<tr><th>Metric</th><th>Value</th></tr><tr><td>Target</td><td>${{item.good.join(", ")}}</td></tr><tr><td>Top1</td><td class="${{item.top1_good?"hit":"miss"}}">${{item.top1}} - ${{status}}</td></tr><tr><td>Top actions</td><td>${{item.top_actions.slice(0,5).map(x=>`${{x.move}} v=${{x.visits}} p=${{x.prior}} val=${{x.value}}`).join("<br>")}}</td></tr>`;
    layout.replaceChildren(canvas, table); article.replaceChildren(title, meta, layout); setTimeout(()=>draw(canvas,item)); return article;
  }}));
}}
checkpointSelect.replaceChildren(...data.results.map((r,i)=>{{const o=document.createElement("option");o.value=String(i);o.textContent=r.checkpoint_name;return o;}}));
checkpointSelect.addEventListener("change", render); render();
</script>"""
    (out_dir / "casebook.html").write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _, cases = load_cases(Path(args.cases))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    results = [
        eval_checkpoint(Path(checkpoint), cases, args.device, args.simulations)
        for checkpoint in args.checkpoint
    ]
    summary = {"source_cases": str(Path(args.cases)), "results": results}
    (out_dir / "tactical_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(out_dir, results)
    write_casebook(out_dir, cases, results)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
