import numpy as np

from diamondgo.demo_cpu import CpuDemoConfig, write_sgf


def _example(game: int, move_in_game: int, player: str, action: int) -> dict[str, object]:
    action_size = 9 * 9 + 1
    policy = np.full(action_size, 1.0 / action_size, dtype=np.float32)
    return {
        "game": game,
        "move_in_game": move_in_game,
        "player": player,
        "chosen_action": action,
        "policy": policy,
        "root_value": 0.0,
        "value_target": 1.0,
        "top_actions": [{"move": "A9", "visits": 1, "prior": 1.0, "value": 0.0}],
        "captures": 0,
    }


def test_write_sgf_exports_interleaved_games_as_collection(tmp_path):
    sgf_path = tmp_path / "cycle-00010.sgf"
    examples = [
        _example(game=1, move_in_game=1, player="b", action=0),
        _example(game=2, move_in_game=1, player="b", action=20),
        _example(game=1, move_in_game=2, player="w", action=10),
    ]

    write_sgf(sgf_path, CpuDemoConfig(input_komi=False), examples)

    text = sgf_path.read_text(encoding="utf-8")
    assert text.count("(;GM[1]") == 2
    assert text.index("Game 1") < text.index("Game 2")

    game_1_text = text[: text.index("Game 2")]
    assert ";B[aa]" in game_1_text
    assert ";W[bb]" in game_1_text
    assert game_1_text.index(";B[aa]") < game_1_text.index(";W[bb]")

    game_2_text = text[text.index("Game 2") :]
    assert ";B[cc]" in game_2_text
