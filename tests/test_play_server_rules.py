from __future__ import annotations

import pytest

pytest.importorskip("sgfmill")

from diamondgo.batched_demo import BatchedConfig
from diamondgo.play_server import PlayState
from diamondgo.rules import SgfmillRules


def _rules_only_play_state() -> PlayState:
    state = PlayState.__new__(PlayState)
    state.default_simulations = 16
    state.device = "cpu"
    state.config = BatchedConfig(
        board_size=5,
        komi=0.5,
        score_komi=2.5,
        input_komi=False,
        channels=8,
        residual_blocks=1,
        simulations=16,
        games=1,
        rules_backend="sgfmill",
    )
    return state


def test_play_server_history_uses_sgfmill_ko_rules() -> None:
    play_state = _rules_only_play_state()
    history = [
        6,  # B4
        8,  # D4
        15,  # A2
        23,  # D1
        10,  # A3
        9,  # E4
        4,  # E5
        21,  # B1
        22,  # C1
        1,  # B5
        0,  # A5
        5,  # A4, creates ko at A5
    ]

    state = play_state.state_from_history(history)

    assert isinstance(state, SgfmillRules)
    assert state._ko_forbidden == (0, 0)
    assert not state.legal_actions()[0]
    assert play_state.validate_move({"history": history, "action": 0})["legal"] is False
