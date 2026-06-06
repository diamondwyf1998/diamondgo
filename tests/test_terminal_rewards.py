import numpy as np

from diamondgo.rules import WHITE, SimpleAreaRules


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


def test_score_margin_reward_adds_signed_fourth_root_bonus() -> None:
    state = SimpleAreaRules(size=5, score_komi=0.0, score_margin_reward_scale=1.0)
    state.board[:4, :4] = 1

    assert np.isclose(state.terminal_score_margin(), 16.0)
    assert np.isclose(state.terminal_value(), 1.4)

    state.to_play = WHITE

    assert np.isclose(state.terminal_value(), -1.4)
