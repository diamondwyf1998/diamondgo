from __future__ import annotations

import argparse
import json
from dataclasses import fields
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from diamondgo.batched_demo import BatchedConfig, make_model, run_batched_mcts
from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_MAX_MOVES, DEFAULT_9X9_SCORE_KOMI
from diamondgo.demo_cpu import action_to_gtp, make_rules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a small browser UI for playing DiamondGo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--static-root", default="artifacts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-simulations", type=int, default=64)
    parser.add_argument("--checkpoint-catalog", default="")
    return parser


def make_config(raw: dict[str, Any], device: str, simulations: int) -> BatchedConfig:
    values = {
        "board_size": int(raw.get("board_size", 9)),
        "komi": float(raw.get("komi", DEFAULT_9X9_KOMI)),
        "score_komi": float(raw.get("score_komi", raw.get("komi", DEFAULT_9X9_SCORE_KOMI))),
        "input_komi": bool(raw.get("input_komi", True)),
        "channels": int(raw.get("channels", 32)),
        "residual_blocks": int(raw.get("residual_blocks", 2)),
        "simulations": int(simulations),
        "max_moves": int(raw.get("max_moves", DEFAULT_9X9_MAX_MOVES)),
        "games": 1,
        "train_steps": 0,
        "batch_size": 256,
        "c_puct": float(raw.get("c_puct", 1.5)),
        "temperature": 0.0,
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
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return config, model


def load_catalog(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("checkpoint catalog must be a JSON list")
    catalog: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "id" not in item or "checkpoint" not in item:
            raise ValueError("checkpoint catalog entries need id and checkpoint")
        catalog.append(dict(item))
    return catalog


class PlayState:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        default_simulations: int,
        checkpoint_catalog: Path | None = None,
    ) -> None:
        self._lock = RLock()
        self.checkpoint = checkpoint
        self.device = device
        self.default_simulations = default_simulations
        self.catalog = load_catalog(checkpoint_catalog)
        self.current_checkpoint_id = self._find_checkpoint_id(checkpoint)
        self.config, self.model = load_model(checkpoint, device, default_simulations)

    def _find_checkpoint_id(self, checkpoint: Path) -> str:
        resolved = checkpoint.resolve()
        for item in self.catalog:
            candidate = Path(str(item["checkpoint"]))
            if candidate.resolve() == resolved:
                return str(item["id"])
        return ""

    def checkpoints(self) -> dict[str, Any]:
        with self._lock:
            return {"checkpoints": self.catalog, "current": self.current_checkpoint_id}

    def load_checkpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = str(payload.get("id", ""))
        entries = [item for item in self.catalog if str(item["id"]) == checkpoint_id]
        if not entries:
            raise ValueError(f"unknown checkpoint id: {checkpoint_id}")
        entry = entries[0]
        checkpoint = Path(str(entry["checkpoint"]))
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        with self._lock:
            self.checkpoint = checkpoint
            self.current_checkpoint_id = checkpoint_id
            self.config, self.model = load_model(checkpoint, self.device, self.default_simulations)
            return self.info()

    def config_for(self, simulations: int) -> BatchedConfig:
        return make_config(
            {
                "board_size": self.config.board_size,
                "komi": self.config.komi,
                "score_komi": getattr(self.config, "score_komi", self.config.komi),
                "input_komi": self.config.input_komi,
                "channels": self.config.channels,
                "residual_blocks": self.config.residual_blocks,
                "max_moves": self.config.max_moves,
                "c_puct": self.config.c_puct,
                "seed": self.config.seed,
                "rules_backend": self.config.rules_backend,
            },
            self.device,
            simulations,
        )

    def info(self) -> dict[str, Any]:
        entry = next(
            (item for item in self.catalog if str(item["id"]) == self.current_checkpoint_id),
            {},
        )
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_id": self.current_checkpoint_id,
            "checkpoint_label": entry.get("label", self.checkpoint.name),
            "checkpoint_name": self.checkpoint.name,
            "device": self.device,
            "default_simulations": int(self.default_simulations),
            "board_size": int(self.config.board_size),
            "komi": float(self.config.komi),
            "score_komi": float(getattr(self.config, "score_komi", self.config.komi)),
            "input_komi": bool(self.config.input_komi),
            "channels": int(self.config.channels),
            "residual_blocks": int(self.config.residual_blocks),
            "rules_backend": str(self.config.rules_backend),
        }

    def state_from_history(self, history: list[int]) -> Any:
        state = make_rules(self.config_for(self.default_simulations))
        for action in history:
            if action < 0 or action >= state.action_size:
                raise ValueError(f"action out of range: {action}")
            if not state.legal_actions()[action]:
                raise ValueError(f"illegal history move: {action_to_gtp(action, state.size)}")
            state.play_action(action)
        return state

    def validate_move(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            history = [int(item) for item in payload.get("history", [])]
            action = int(payload["action"])
            state = self.state_from_history(history)
            if action < 0 or action >= state.action_size:
                return {"legal": False, "reason": f"action out of range: {action}"}
            legal = bool(state.legal_actions()[action])
            return {
                "legal": legal,
                "move": action_to_gtp(action, state.size),
                "player": state.to_play,
                "reason": "" if legal else f"illegal move for {state.to_play}",
            }

    def ai_move(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            history = [int(item) for item in payload.get("history", [])]
            simulations = max(1, min(400, int(payload.get("simulations", self.default_simulations))))
            state = self.state_from_history(history)
            if state.is_terminal():
                return {"terminal": True, "to_play": state.to_play, "value": float(state.terminal_value())}
            search_config = self.config_for(simulations)
            stats: dict[str, object] = {}
            root = run_batched_mcts(self.model, [state], search_config, stats=stats)[0]
            action, child = max(root.children.items(), key=lambda item: item[1].visit_count)
            return {
                "action": int(action),
                "move": action_to_gtp(int(action), self.config.board_size),
                "player": state.to_play,
                "root_value": round(float(root.value), 4),
                "child_value": round(float(child.value), 4),
                "simulations": simulations,
                "move_temperature": 0.0,
                "move_selection": "max_visit",
                "top_actions": root.top_actions(self.config.board_size, limit=8),
                "stats": {
                    key: round(float(value), 4)
                    for key, value in stats.items()
                    if isinstance(value, (int, float))
                },
            }


def make_handler(play_state: PlayState, static_root: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, directory=str(static_root), **kwargs)

        def do_GET(self) -> None:
            if self.path == "/api/info":
                self._write_json(200, play_state.info())
                return
            if self.path == "/api/checkpoints":
                self._write_json(200, play_state.checkpoints())
                return
            super().do_GET()

        def do_POST(self) -> None:
            if self.path not in {"/api/ai_move", "/api/validate_move", "/api/load_checkpoint"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/api/load_checkpoint":
                    response = play_state.load_checkpoint(payload)
                elif self.path == "/api/validate_move":
                    response = play_state.validate_move(payload)
                else:
                    response = play_state.ai_move(payload)
                self._write_json(200, response)
            except Exception as exc:  # noqa: BLE001 - show UI-friendly errors.
                self._write_json(400, {"error": str(exc)})

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = Path(args.checkpoint)
    static_root = Path(args.static_root).resolve()
    checkpoint_catalog = Path(args.checkpoint_catalog) if args.checkpoint_catalog else None
    play_state = PlayState(checkpoint, args.device, args.default_simulations, checkpoint_catalog)
    handler = make_handler(play_state, static_root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Serving DiamondGo play UI at http://{args.host}:{args.port}/viewers/play-ai.html "
        f"using {checkpoint}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
