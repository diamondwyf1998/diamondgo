import torch

from diamondgo.config import ModelConfig
from diamondgo.model import PolicyValueNet


def test_policy_value_shapes() -> None:
    model = PolicyValueNet(board_size=5, config=ModelConfig(channels=16, residual_blocks=1))
    features = torch.zeros(2, 4, 5, 5)

    policy, value, final_board = model(features)

    assert policy.shape == (2, 26)
    assert value.shape == (2,)
    assert final_board.shape == (2, 25)
    assert torch.all(final_board <= 1.0)
    assert torch.all(final_board >= -1.0)
