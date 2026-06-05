from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("sgfmill")

from diamondgo.rules import SgfmillRules


def _probe_legal(state: SgfmillRules, row: int, col: int) -> bool:
    if state.board.get(row, col) is not None:
        return False
    if state._ko_forbidden == (row, col):
        return False
    probe = state.board.copy()
    try:
        probe.play(row, col, state.to_play)
    except (IndexError, ValueError):
        return False
    return probe.get(row, col) == state.to_play


def _probe_mask(state: SgfmillRules) -> np.ndarray:
    legal = np.zeros(state.action_size, dtype=np.bool_)
    for row in range(state.size):
        for col in range(state.size):
            if _probe_legal(state, row, col):
                legal[row * state.size + col] = True
    legal[-1] = True
    return legal


def _board_counts(state: SgfmillRules) -> tuple[int, int]:
    black = 0
    white = 0
    for row in range(state.size):
        for col in range(state.size):
            colour = state.board.get(row, col)
            black += colour == "b"
            white += colour == "w"
    return black, white


def test_sgfmill_fast_legal_actions_match_probe_play() -> None:
    random.seed(7)
    state = SgfmillRules(size=9, komi=0.5)
    checked_positions = 0
    for _ in range(80):
        state._legal_actions_cache = None
        fast = state.legal_actions()
        assert np.array_equal(fast, _probe_mask(state))
        assert (
            int((state.board_array == 1).sum()),
            int((state.board_array == -1).sum()),
        ) == _board_counts(state)
        checked_positions += 1
        non_pass_actions = [
            int(action)
            for action in np.flatnonzero(fast)
            if action != state.action_size - 1
        ]
        if not non_pass_actions:
            break
        state.play_action(random.choice(non_pass_actions))
    assert checked_positions > 20
