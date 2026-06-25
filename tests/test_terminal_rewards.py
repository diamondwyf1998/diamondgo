import numpy as np
import pytest

from diamondgo.rules import WHITE, SgfmillRules, SimpleAreaRules


def test_terminal_cleanup_removes_enclosed_single_eye_group() -> None:
    board = np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 1, 1, 1, -1],
            [-1, 1, 0, 1, -1],
            [-1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=np.int8,
    )
    without_cleanup = SimpleAreaRules(size=5, score_komi=0.0)
    with_cleanup = SimpleAreaRules(size=5, score_komi=0.0, terminal_dead_stone_cleanup=True)
    without_cleanup.board = board.copy()
    with_cleanup.board = board.copy()

    assert with_cleanup.terminal_cleanup_counts() == {"b": 8, "w": 0}
    assert with_cleanup.terminal_score_margin() < without_cleanup.terminal_score_margin()
    assert with_cleanup.terminal_ownership().shape == board.shape
    assert np.all(with_cleanup.terminal_ownership() == -1.0)


def test_terminal_cleanup_keeps_two_eye_group_and_edge_group() -> None:
    board = np.array(
        [
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, 1, 1, 1, 1, 1, -1],
            [-1, 1, 0, 1, 0, 1, -1],
            [-1, 1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int8,
    )
    state = SimpleAreaRules(size=7, score_komi=0.0, terminal_dead_stone_cleanup=True)
    state.board = board

    assert state.terminal_cleanup_counts() == {"b": 0, "w": 0}


def test_terminal_cleanup_keeps_edge_group_to_avoid_false_positive() -> None:
    board = np.array(
        [
            [1, -1, -1, 1, 0],
            [1, -1, -1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    state = SimpleAreaRules(size=5, score_komi=0.0, terminal_dead_stone_cleanup=True)
    state.board = board

    assert state.terminal_cleanup_counts() == {"b": 0, "w": 0}


def test_sgfmill_terminal_cleanup_uses_board_array_for_scoring() -> None:
    pytest.importorskip("sgfmill")
    board = np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 1, 1, 1, -1],
            [-1, 1, 0, 1, -1],
            [-1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=np.int8,
    )
    without_cleanup = SgfmillRules(size=5, score_komi=0.0)
    with_cleanup = SgfmillRules(size=5, score_komi=0.0, terminal_dead_stone_cleanup=True)
    without_cleanup.board_array = board.copy()
    with_cleanup.board_array = board.copy()

    assert with_cleanup.terminal_cleanup_counts() == {"b": 8, "w": 0}
    assert with_cleanup.terminal_score_margin() == -25.0
    assert with_cleanup.terminal_score_margin() < float(board.sum())
    assert np.all(with_cleanup.terminal_ownership() == -1.0)


def test_score_margin_reward_adds_signed_fourth_root_bonus() -> None:
    state = SimpleAreaRules(size=5, score_komi=0.0, score_margin_reward_scale=1.0)
    state.board[:4, :4] = 1

    assert np.isclose(state.terminal_score_margin(), 16.0)
    assert np.isclose(state.terminal_value(), 0.8)

    state.to_play = WHITE

    assert np.isclose(state.terminal_value(), -0.8)


def test_score_margin_reward_caps_targets_to_one() -> None:
    state = SimpleAreaRules(size=9, score_komi=0.0, score_margin_reward_scale=2.0)
    state.board[:, :] = 1

    assert np.isclose(state.terminal_score_margin(), 81.0)
    assert np.isclose(state.terminal_value(), 1.0)
