from __future__ import annotations

import argparse
import html
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
from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_MAX_MOVES, DEFAULT_9X9_SCORE_KOMI
from diamondgo.demo_cpu import action_to_gtp, make_rules
from diamondgo.model import PolicyValueNet


@dataclass(frozen=True)
class MatchConfig:
    board_size: int = 9
    komi: float = DEFAULT_9X9_KOMI
    score_komi: float = DEFAULT_9X9_SCORE_KOMI
    channels: int = 32
    residual_blocks: int = 2
    simulations: int = 32
    games: int = 20
    max_moves: int = DEFAULT_9X9_MAX_MOVES
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
    parser.add_argument("--max-moves", type=int, default=DEFAULT_9X9_MAX_MOVES)
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
        komi=float(raw.get("komi", DEFAULT_9X9_KOMI)),
        score_komi=float(raw.get("score_komi", raw.get("komi", DEFAULT_9X9_SCORE_KOMI))),
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
            f"SZ[{config.board_size}]KM[{config.score_komi}]PB[{escape_sgf(black_name)}]PW[{escape_sgf(white_name)}]"
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
        score_komi=config.score_komi,
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


def dashboard_state(results: list[dict[str, object]], board_size: int) -> dict[str, object]:
    matches = []
    for match_index, result in enumerate(results):
        games = []
        for game in result["games_detail"]:
            games.append(
                {
                    "game": int(game["game"]),
                    "candidateColor": str(game["candidate_color"]),
                    "winner": str(game["winner"]),
                    "candidateWon": bool(game["candidate_won"]),
                    "movesPlayed": int(game["moves_played"]),
                    "moves": [
                        {
                            "moveNumber": int(move["move_number"]),
                            "player": str(move["player"]),
                            "model": str(move["model"]),
                            "action": int(move["action"]),
                            "move": str(move["move"]),
                            "rootValue": float(move["root_value"]),
                            "topActions": [
                                {
                                    "move": str(item["move"]),
                                    "action": gtp_to_action(str(item["move"]), board_size),
                                    "visits": int(item["visits"]),
                                    "prior": float(item["prior"]),
                                    "value": float(item["value"]),
                                }
                                for item in move["top_actions"]
                            ],
                        }
                        for move in game["moves"]
                    ],
                }
            )
        matches.append(
            {
                "index": match_index,
                "candidate": str(result["candidate"]),
                "opponent": str(result["opponent"]),
                "games": int(result["games"]),
                "candidateWins": int(result["candidate_wins"]),
                "candidateLosses": int(result["candidate_losses"]),
                "winRate": float(result["win_rate"]),
                "seconds": float(result["seconds"]),
                "gamesDetail": games,
            }
        )
    return {"boardSize": board_size, "cell": 42, "pad": 34, "matches": matches}


def gtp_to_action(move: str, board_size: int) -> int:
    if move == "pass":
        return board_size * board_size
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    col = letters.index(move[0].upper())
    row = board_size - int(move[1:])
    return row * board_size + col


def render_eval_dashboard(results: list[dict[str, object]], board_size: int = 9) -> str:
    state_json = json.dumps(dashboard_state(results, board_size), separators=(",", ":"))
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(item['candidate']))}</td>"
        f"<td>{html.escape(str(item['opponent']))}</td>"
        f"<td>{item['games']}</td>"
        f"<td>{100 * float(item['win_rate']):.1f}%</td>"
        f"<td>{item['candidate_wins']}</td>"
        f"<td>{item['candidate_black_wins']}/{item['candidate_black_games']}</td>"
        f"<td>{item['candidate_white_wins']}/{item['candidate_white_games']}</td>"
        f"<td>{item['seconds']}</td>"
        "</tr>"
        for item in results
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DiamondGo Evaluation Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: #222;
      background: #f5f7f6;
    }}
    header {{
      padding: 26px 34px 18px;
      color: #fff;
      background: #1d2a2e;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    main {{ padding: 24px 34px 40px; max-width: 1280px; }}
    section {{
      margin-bottom: 18px;
      padding: 16px;
      border: 1px solid #dbe2df;
      border-radius: 8px;
      background: #fff;
    }}
    .subtle {{ color: #acb8b8; }}
    .summary {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 7px;
      border-bottom: 1px solid #e8eeee;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: #4f5c5f; background: #faf8f4; }}
    .viewer {{
      display: grid;
      grid-template-columns: minmax(340px, 520px) minmax(320px, 1fr);
      gap: 22px;
      align-items: start;
    }}
    .selectors {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 14px;
    }}
    select {{
      height: 34px;
      border: 1px solid #b8c4c0;
      border-radius: 6px;
      background: #fff;
      color: #203033;
      padding: 0 8px;
    }}
    .board-wrap {{
      padding: 12px;
      border: 1px solid #b99255;
      border-radius: 8px;
      background: #e6bf78;
    }}
    svg {{ width: 100%; height: auto; display: block; }}
    .controls {{
      display: grid;
      grid-template-columns: auto auto 1fr auto;
      gap: 10px;
      align-items: center;
      margin-top: 14px;
    }}
    button {{
      min-width: 42px;
      height: 34px;
      border: 1px solid #b8c4c0;
      border-radius: 6px;
      background: #fff;
      color: #203033;
      font-size: 17px;
      cursor: pointer;
    }}
    button:hover {{ background: #edf3f1; }}
    input[type="range"] {{ width: 100%; accent-color: #2f6f73; }}
    .panel {{
      display: grid;
      gap: 12px;
    }}
    .box {{
      padding: 14px;
      border: 1px solid #dbe2df;
      border-radius: 8px;
      background: #fbfcfb;
    }}
    .box b {{ display: block; color: #657174; font-size: 12px; margin-bottom: 4px; }}
    .box span {{ font-size: 18px; }}
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
    @media (max-width: 900px) {{
      .viewer, .selectors {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DiamondGo Evaluation Dashboard</h1>
    <div class="subtle">Checkpoint matches with inline game replay and search candidates.</div>
  </header>
  <main>
    <section>
      <h2>Match Summary</h2>
      <table class="summary">
        <thead><tr><th>candidate</th><th>opponent</th><th>games</th><th>win rate</th><th>wins</th><th>black wins</th><th>white wins</th><th>seconds</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    <section>
      <h2>Game Replay</h2>
      <div class="selectors">
        <select id="match-select"></select>
        <select id="game-select"></select>
      </div>
      <div class="viewer">
        <div>
          <div class="board-wrap">
            <svg id="eval-board" viewBox="0 0 404 404" role="img" aria-label="evaluation game board"></svg>
          </div>
          <div class="controls">
            <button type="button" id="prev-move" title="Previous move">&lt;</button>
            <button type="button" id="next-move" title="Next move">&gt;</button>
            <input id="move-slider" type="range" min="0" value="0">
            <span id="move-count"></span>
          </div>
        </div>
        <div class="panel">
          <div class="box"><b>Game</b><span id="game-info"></span></div>
          <div class="box"><b>Current move</b><span id="move-info"></span></div>
          <div class="box"><b>Root value</b><span id="root-value"></span></div>
          <div class="box">
            <b>Top 5 search candidates before this move</b>
            <div id="candidate-list" class="candidate-list"></div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const state = {state_json};
    const board = document.getElementById("eval-board");
    const matchSelect = document.getElementById("match-select");
    const gameSelect = document.getElementById("game-select");
    const slider = document.getElementById("move-slider");
    const prevButton = document.getElementById("prev-move");
    const nextButton = document.getElementById("next-move");
    const moveCount = document.getElementById("move-count");
    const gameInfo = document.getElementById("game-info");
    const moveInfo = document.getElementById("move-info");
    const rootValue = document.getElementById("root-value");
    const candidateList = document.getElementById("candidate-list");

    function makeSvg(name, attributes = {{}}, text = "") {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
      if (text) node.textContent = text;
      return node;
    }}

    function pointFor(action) {{
      const row = Math.floor(action / state.boardSize);
      const col = action % state.boardSize;
      return {{
        x: state.pad + col * state.cell,
        y: state.pad + row * state.cell
      }};
    }}

    function neighbors(action) {{
      const size = state.boardSize;
      const row = Math.floor(action / size);
      const col = action % size;
      const result = [];
      if (row > 0) result.push(action - size);
      if (row + 1 < size) result.push(action + size);
      if (col > 0) result.push(action - 1);
      if (col + 1 < size) result.push(action + 1);
      return result;
    }}

    function groupAndLiberties(boardState, start) {{
      const color = boardState[start].color;
      const stack = [start];
      const seen = new Set([start]);
      const liberties = new Set();
      while (stack.length) {{
        const action = stack.pop();
        neighbors(action).forEach((next) => {{
          const stone = boardState[next];
          if (!stone) {{
            liberties.add(next);
          }} else if (stone.color === color && !seen.has(next)) {{
            seen.add(next);
            stack.push(next);
          }}
        }});
      }}
      return {{ stones: [...seen], liberties: liberties.size }};
    }}

    function replayBoard(index) {{
      const boardState = Array(state.boardSize * state.boardSize).fill(null);
      const game = currentGame();
      game.moves.slice(0, index + 1).forEach((move) => {{
        if (move.action >= state.boardSize * state.boardSize) return;
        boardState[move.action] = {{ color: move.player, moveNumber: move.moveNumber }};
        const opponent = move.player === "b" ? "w" : "b";
        const checked = new Set();
        neighbors(move.action).forEach((next) => {{
          const stone = boardState[next];
          if (!stone || stone.color !== opponent || checked.has(next)) return;
          const group = groupAndLiberties(boardState, next);
          group.stones.forEach((action) => checked.add(action));
          if (group.liberties === 0) {{
            group.stones.forEach((action) => {{ boardState[action] = null; }});
          }}
        }});
        const own = groupAndLiberties(boardState, move.action);
        if (own.liberties === 0) {{
          own.stones.forEach((action) => {{ boardState[action] = null; }});
        }}
      }});
      return boardState;
    }}

    function currentMatch() {{
      return state.matches[Number(matchSelect.value)] || state.matches[0];
    }}

    function currentGame() {{
      const match = currentMatch();
      return match.gamesDetail[Number(gameSelect.value)] || match.gamesDetail[0];
    }}

    function populateMatches() {{
      matchSelect.replaceChildren(...state.matches.map((match, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${{match.candidate}} vs ${{match.opponent}} - ${{Math.round(match.winRate * 100)}}%`;
        return option;
      }}));
    }}

    function populateGames() {{
      const match = currentMatch();
      gameSelect.replaceChildren(...match.gamesDetail.map((game, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `Game ${{game.game}} - candidate ${{game.candidateColor.toUpperCase()}} - ${{game.candidateWon ? "win" : "loss"}}`;
        return option;
      }}));
      slider.max = String(Math.max(0, currentGame().moves.length - 1));
      render(0);
    }}

    function drawBoard(index) {{
      const game = currentGame();
      const current = game.moves[index];
      const boardState = replayBoard(index);
      const boardEnd = state.pad + (state.boardSize - 1) * state.cell;
      const viewSize = boardEnd + state.pad;
      board.setAttribute("viewBox", `0 0 ${{viewSize}} ${{viewSize}}`);
      board.replaceChildren();
      for (let i = 0; i < state.boardSize; i += 1) {{
        const pos = state.pad + i * state.cell;
        board.appendChild(makeSvg("line", {{ x1: state.pad, y1: pos, x2: boardEnd, y2: pos, stroke: "#765933" }}));
        board.appendChild(makeSvg("line", {{ x1: pos, y1: state.pad, x2: pos, y2: boardEnd, stroke: "#765933" }}));
      }}
      current.topActions.filter((item) => item.action < state.boardSize * state.boardSize).slice(0, 5).forEach((item, rank) => {{
        const p = pointFor(item.action);
        const colors = ["#2f6f73", "#8a6fb0", "#b5654d", "#5b7aa4", "#7d7d46"];
        board.appendChild(makeSvg("circle", {{
          cx: p.x, cy: p.y, r: 13 + rank, fill: "none", stroke: colors[rank],
          "stroke-width": 3, "stroke-dasharray": rank === 0 ? "0" : "5 4"
        }}));
        board.appendChild(makeSvg("text", {{
          x: p.x, y: p.y + 5, "text-anchor": "middle", "font-size": 12, "font-weight": 700,
          fill: colors[rank]
        }}, String(rank + 1)));
      }});
      boardState.forEach((stone, action) => {{
        if (!stone) return;
        const p = pointFor(action);
        const isBlack = stone.color === "b";
        board.appendChild(makeSvg("circle", {{
          cx: p.x, cy: p.y, r: 16, fill: isBlack ? "#161616" : "#f4f1e8",
          stroke: isBlack ? "#161616" : "#7b756c", "stroke-width": 1.5
        }}));
        board.appendChild(makeSvg("text", {{
          x: p.x, y: p.y + 4, "text-anchor": "middle", "font-size": 12,
          fill: isBlack ? "#fff" : "#111"
        }}, String(stone.moveNumber)));
      }});
    }}

    function render(index) {{
      const game = currentGame();
      if (!game || game.moves.length === 0) return;
      const safeIndex = Math.max(0, Math.min(game.moves.length - 1, index));
      const move = game.moves[safeIndex];
      slider.max = String(game.moves.length - 1);
      slider.value = String(safeIndex);
      moveCount.textContent = `${{safeIndex + 1}} / ${{game.moves.length}}`;
      gameInfo.textContent = `Game ${{game.game}}, candidate ${{game.candidateColor.toUpperCase()}}, winner ${{game.winner.toUpperCase()}}`;
      moveInfo.textContent = `${{move.player.toUpperCase()}} ${{move.move}} by ${{move.model}}`;
      rootValue.textContent = move.rootValue.toFixed(4);
      candidateList.replaceChildren(...move.topActions.slice(0, 5).map((item, rank) => {{
        const row = document.createElement("div");
        row.className = "candidate";
        const rankNode = document.createElement("span");
        rankNode.className = "rank";
        rankNode.textContent = String(rank + 1);
        const moveNode = document.createElement("b");
        moveNode.textContent = item.move;
        const detailNode = document.createElement("span");
        detailNode.textContent = `visits=${{item.visits}} prior=${{item.prior}} value=${{item.value}}`;
        row.replaceChildren(rankNode, moveNode, detailNode);
        return row;
      }}));
      drawBoard(safeIndex);
    }}

    matchSelect.addEventListener("change", populateGames);
    gameSelect.addEventListener("change", () => render(0));
    slider.addEventListener("input", () => render(Number(slider.value)));
    prevButton.addEventListener("click", () => render(Number(slider.value) - 1));
    nextButton.addEventListener("click", () => render(Number(slider.value) + 1));
    populateMatches();
    populateGames();
  </script>
</body>
</html>
"""


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
        checkpoint_payload = load_checkpoint_payload(checkpoint_path, args.device)
        cycle = int(
            checkpoint_payload["cycle"]
            if "cycle" in checkpoint_payload
            else cycle_from_path(checkpoint_path)
        )
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
        (out_dir / "dashboard.html").write_text(render_eval_dashboard(results, config.board_size), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "games_detail"}), flush=True)
        previous_model = candidate_model
        previous_name = candidate_name

    return {
        "config": config.__dict__,
        "results": results,
        "report_path": str(out_dir / "report.md"),
        "results_path": str(out_dir / "results.json"),
        "dashboard_path": str(out_dir / "dashboard.html"),
        "sample_sgf_dir": str(out_dir / "sample-sgf"),
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    if args.json:
        print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
