import numpy as np

from diamondgo.mcts import SearchNode, play_search_action
from diamondgo.rules import BLACK, WHITE, SimpleAreaRules


def test_simple_rules_search_action_applies_prevalidated_move() -> None:
    state = SimpleAreaRules(size=3, komi=0.5)

    state.play_search_action(4)

    assert state.board[1, 1] == 1
    assert state.to_play == WHITE


def test_play_search_action_prefers_fast_state_method() -> None:
    class FastState:
        def __init__(self) -> None:
            self.fast_calls = 0
            self.slow_calls = 0

        def play_search_action(self, action: int) -> None:
            self.fast_calls += action

        def play_action(self, action: int) -> None:
            self.slow_calls += action

    state = FastState()

    play_search_action(state, 3)

    assert state.fast_calls == 3
    assert state.slow_calls == 0


def test_node_expand_can_reuse_cached_legal_actions() -> None:
    state = SimpleAreaRules(size=3, komi=0.5)
    priors = np.full(state.action_size, 1.0 / state.action_size, dtype=np.float32)
    legal = state.legal_actions()
    legal[0] = False
    node = SearchNode(prior=1.0)

    node.expand(priors, state.legal_actions())

    assert 0 in node.children
    assert state.to_play == BLACK
