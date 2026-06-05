import numpy as np

from diamondgo.overnight_train import apply_dihedral_transform
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


def test_dihedral_policy_transform_preserves_pass_probability() -> None:
    features = np.arange(3 * 3 * 3, dtype=np.float32).reshape(3, 3, 3)
    policy = np.arange(10, dtype=np.float32)
    policy = policy / policy.sum()

    transformed_features, transformed_policy = apply_dihedral_transform(features, policy, transform=5)

    assert transformed_features.shape == features.shape
    assert transformed_policy.shape == policy.shape
    assert transformed_policy[-1] == policy[-1]
    assert np.isclose(transformed_policy[:-1].sum(), policy[:-1].sum())
