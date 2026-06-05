from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SearchNode:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return bool(self.children)

    def expand(self, priors: np.ndarray, legal_actions: np.ndarray) -> None:
        masked = np.where(legal_actions, priors, 0.0)
        total = float(masked.sum())
        if total <= 0.0:
            masked = legal_actions.astype(np.float32)
            total = float(masked.sum())
        masked = masked / total
        for action, prior in enumerate(masked):
            if prior > 0:
                self.children[action] = SearchNode(prior=float(prior))

    def policy_target(self, action_size: int, temperature: float) -> np.ndarray:
        visits = np.zeros(action_size, dtype=np.float32)
        for action, child in self.children.items():
            visits[action] = child.visit_count
        if visits.sum() == 0:
            return np.full(action_size, 1.0 / action_size, dtype=np.float32)
        if temperature <= 1e-6:
            target = np.zeros(action_size, dtype=np.float32)
            target[int(visits.argmax())] = 1.0
            return target
        visits = visits ** (1.0 / temperature)
        return visits / visits.sum()

    def top_actions(self, board_size: int, limit: int = 5) -> list[dict[str, object]]:
        rows = []
        for action, child in self.children.items():
            move = "pass" if action == board_size * board_size else _action_to_gtp(action, board_size)
            rows.append(
                {
                    "move": move,
                    "visits": child.visit_count,
                    "prior": round(child.prior, 4),
                    "value": round(child.value, 4),
                }
            )
        return sorted(rows, key=lambda item: item["visits"], reverse=True)[:limit]


def run_mcts(state, evaluator, simulations: int, c_puct: float, temperature: float) -> SearchNode:
    root = SearchNode(prior=1.0)
    priors, value = evaluator(state)
    root.expand(priors, state.legal_actions())
    root.visit_count = 1
    root.value_sum = float(value)

    for _ in range(simulations):
        simulation_state = state.copy()
        _simulate(simulation_state, root, evaluator, c_puct)
    return root


def play_search_action(state, action: int) -> None:
    fast_play = getattr(state, "play_search_action", None)
    if fast_play is None:
        state.play_action(action)
    else:
        fast_play(action)


def select_child(node: SearchNode, c_puct: float) -> tuple[int, SearchNode]:
    total_visits = max(1, node.visit_count)
    best_score = -float("inf")
    best: tuple[int, SearchNode] | None = None
    for action, child in node.children.items():
        q_value = -child.value
        u_value = c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
        score = q_value + u_value
        if score > best_score:
            best_score = score
            best = (action, child)
    if best is None:
        raise RuntimeError("cannot select a child from an unexpanded node")
    return best


def _simulate(state, node: SearchNode, evaluator, c_puct: float) -> float:
    node.visit_count += 1
    if state.is_terminal():
        value = float(state.terminal_value())
        node.value_sum += value
        return value

    if not node.expanded():
        priors, value = evaluator(state)
        node.expand(priors, state.legal_actions())
        node.value_sum += float(value)
        return float(value)

    action, child = select_child(node, c_puct)
    play_search_action(state, action)
    child_value = _simulate(state, child, evaluator, c_puct)
    value = -child_value
    node.value_sum += value
    return value


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _action_to_gtp(action: int, board_size: int) -> str:
    col = action % board_size
    row = action // board_size
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    return f"{letters[col]}{board_size - row}"
