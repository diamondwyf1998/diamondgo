from diamondgo.config import ExperimentConfig


def test_default_config_is_9x9_baby_zero() -> None:
    config = ExperimentConfig()

    assert config.board.size == 9
    assert config.board.rules_backend == "sgfmill"
    assert config.mcts.simulations == 64
