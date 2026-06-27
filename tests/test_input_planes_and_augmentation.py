import numpy as np

from diamondgo.config import input_plane_count
from diamondgo.demo_cpu import CpuDemoConfig, build_trace
from diamondgo.multiworker_train import trace_examples_for_cycle
from diamondgo.overnight_train import (
    apply_board_target_transform,
    apply_dihedral_transform,
    prepare_training_batch,
)
from diamondgo.rules import SimpleAreaRules


def test_rules_can_encode_without_komi_plane() -> None:
    state = SimpleAreaRules(size=3, komi=0.5, score_komi=2.5, input_komi=False)

    encoded = state.encode()

    assert encoded.shape == (3, 3, 3)
    assert np.all(encoded[2] == 1.0)


def test_rules_keep_legacy_komi_plane_by_default() -> None:
    state = SimpleAreaRules(size=3, komi=0.5)

    encoded = state.encode()

    assert encoded.shape == (4, 3, 3)
    assert np.all(encoded[3] == 0.05)


def test_previous_move_planes_track_recent_actions() -> None:
    state = SimpleAreaRules(size=3, input_komi=False, history_moves=2)

    state.play_action(4)
    encoded_after_one = state.encode()

    assert encoded_after_one.shape == (5, 3, 3)
    assert encoded_after_one[3, 1, 1] == 1.0
    assert encoded_after_one[4].sum() == 0.0

    state.play_action(0)
    encoded_after_two = state.encode()

    assert encoded_after_two[3, 0, 0] == 1.0
    assert encoded_after_two[4, 1, 1] == 1.0


def test_previous_move_planes_treat_pass_as_empty_plane_and_copy_history() -> None:
    state = SimpleAreaRules(size=3, input_komi=False, history_moves=2)
    state.play_action(4)
    state.play_action(state.action_size - 1)

    clone = state.copy()
    encoded = clone.encode()

    assert encoded.shape == (5, 3, 3)
    assert encoded[3].sum() == 0.0
    assert encoded[4, 1, 1] == 1.0


def test_input_plane_count_includes_history_and_optional_komi() -> None:
    assert input_plane_count(input_komi=False, history_moves=2) == 5
    assert input_plane_count(input_komi=True, history_moves=2) == 6


def test_trace_examples_keep_full_top_actions_only_for_selected_games() -> None:
    examples = [
        {"game": 1, "top_actions": list(range(10))},
        {"game": 5, "top_actions": list(range(10))},
        {"game": 6, "top_actions": list(range(10))},
    ]

    full_cycle = trace_examples_for_cycle(
        examples,
        cycle=20,
        full_trace_every=20,
        full_trace_games=5,
        trace_top_actions_limit=3,
    )
    normal_cycle = trace_examples_for_cycle(
        examples,
        cycle=21,
        full_trace_every=20,
        full_trace_games=5,
        trace_top_actions_limit=3,
    )

    assert len(full_cycle[0]["top_actions"]) == 10
    assert len(full_cycle[1]["top_actions"]) == 10
    assert len(full_cycle[2]["top_actions"]) == 3
    assert len(normal_cycle[0]["top_actions"]) == 3
    assert examples[0]["top_actions"] == list(range(10))


def test_full_trace_examples_include_lightweight_top5_tree_summary() -> None:
    actions = [
        {"action": index, "move": f"A{index}", "visits": 10 - index, "prior": 0.1, "value": 0.0}
        for index in range(6)
    ]

    traced = trace_examples_for_cycle(
        [{"game": 1, "top_actions": actions}],
        cycle=20,
        full_trace_every=20,
        full_trace_games=1,
        trace_top_actions_limit=5,
    )

    assert len(traced[0]["top_actions"]) == 6
    assert traced[0]["top5_search_tree"] == {
        "root_search_visits": 45,
        "actions": actions[:5],
    }


def test_full_trace_examples_skip_lightweight_tree_for_single_visit_roots() -> None:
    actions = [
        {"action": 0, "move": "A1", "visits": 1, "prior": 0.5, "value": 0.0},
        {"action": 1, "move": "B1", "visits": 0, "prior": 0.5, "value": 0.0},
    ]

    traced = trace_examples_for_cycle(
        [{"game": 1, "top_actions": actions}],
        cycle=20,
        full_trace_every=20,
        full_trace_games=1,
        trace_top_actions_limit=5,
    )

    assert "top5_search_tree" not in traced[0]


def test_build_trace_persists_lightweight_top5_search_tree() -> None:
    tree = {
        "root_search_visits": 3,
        "actions": [{"action": 0, "move": "A3", "visits": 3, "prior": 1.0, "value": 0.25}],
    }

    trace = build_trace(
        CpuDemoConfig(board_size=3),
        [
            {
                "chosen_action": 0,
                "policy": np.full(10, 0.1, dtype=np.float32),
                "player": "b",
                "root_value": 0.25,
                "value_target": 1.0,
                "top_actions": tree["actions"],
                "top5_search_tree": tree,
            }
        ],
    )

    assert trace["moves"][0]["top5_search_tree"] == tree


def test_dihedral_policy_transform_preserves_pass_probability() -> None:
    features = np.arange(3 * 3 * 3, dtype=np.float32).reshape(3, 3, 3)
    policy = np.arange(10, dtype=np.float32)
    policy = policy / policy.sum()

    transformed_features, transformed_policy = apply_dihedral_transform(features, policy, transform=5)

    assert transformed_features.shape == features.shape
    assert transformed_policy.shape == policy.shape
    assert transformed_policy[-1] == policy[-1]
    assert np.isclose(transformed_policy[:-1].sum(), policy[:-1].sum())


def test_final_board_target_transform_matches_board_transform() -> None:
    final_board_target = np.arange(9, dtype=np.float32)

    transformed = apply_board_target_transform(final_board_target, board_size=3, transform=5)

    expected = np.flip(np.rot90(final_board_target.reshape(3, 3), k=1), axis=-1).reshape(-1)
    assert np.array_equal(transformed, expected)


def test_prepare_training_batch_transforms_final_board_targets() -> None:
    features = np.arange(3 * 3 * 3, dtype=np.float32).reshape(3, 3, 3)
    policy = np.arange(10, dtype=np.float32)
    policy = policy / policy.sum()
    final_board_target = np.arange(9, dtype=np.float32)

    import random

    original_randrange = random.randrange
    try:
        random.randrange = lambda _upper: 5
        features_batch, policy_batch, final_board_batch = prepare_training_batch(
            [
                {
                    "features": features,
                    "policy": policy,
                    "final_board_target": final_board_target,
                }
            ],
            augment_dihedral=True,
        )
    finally:
        random.randrange = original_randrange

    expected_board = np.flip(np.rot90(final_board_target.reshape(3, 3), k=1), axis=-1).reshape(-1)
    assert features_batch.shape == (1, 3, 3, 3)
    assert policy_batch.shape == (1, 10)
    assert final_board_batch is not None
    assert np.array_equal(final_board_batch[0], expected_board)
