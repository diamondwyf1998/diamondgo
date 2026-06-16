from diamondgo.mcts import SearchNode


def test_top_actions_include_full_root_distribution_when_unlimited() -> None:
    root = SearchNode(prior=1.0, visit_count=11, value_sum=2.2)
    root.children = {
        0: SearchNode(prior=0.2, visit_count=2, value_sum=0.5),
        40: SearchNode(prior=0.5, visit_count=7, value_sum=-1.4),
        81: SearchNode(prior=0.3, visit_count=1, value_sum=0.1),
    }

    actions = root.top_actions(board_size=9, limit=None)

    assert [item["action"] for item in actions] == [40, 0, 81]
    assert [item["move"] for item in actions] == ["E5", "A9", "pass"]
    assert actions[0]["visit_pct"] == 0.7


def test_root_analysis_reports_black_view() -> None:
    root = SearchNode(prior=1.0, visit_count=5, value_sum=1.0)

    black_to_play = root.root_analysis(board_size=9, to_play="b")
    white_to_play = root.root_analysis(board_size=9, to_play="w")

    assert black_to_play["root_value"] == 0.2
    assert black_to_play["black_value"] == 0.2
    assert black_to_play["favorite"] == "black"
    assert white_to_play["black_value"] == -0.2
    assert white_to_play["favorite"] == "white"
