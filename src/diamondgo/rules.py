from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from diamondgo.defaults import DEFAULT_9X9_KOMI, DEFAULT_9X9_SCORE_KOMI


BLACK = "b"
WHITE = "w"
PASS = None
BLACK_VALUE = 1
WHITE_VALUE = -1
VALUE_WIN_LOSS_BASE_WITH_MARGIN_REWARD = 2.0 / 5.0
VALUE_MARGIN_REWARD_LIMIT = 3.0 / 5.0


@dataclass(frozen=True)
class Move:
    row: int | None
    col: int | None

    @property
    def is_pass(self) -> bool:
        return self.row is None or self.col is None


class GoRules(Protocol):
    size: int
    to_play: str

    def copy(self) -> "GoRules":
        ...

    def legal_actions(self) -> np.ndarray:
        ...

    def play_action(self, action: int) -> None:
        ...

    def play_search_action(self, action: int) -> None:
        ...

    def encode(self) -> np.ndarray:
        ...

    def is_terminal(self) -> bool:
        ...

    def terminal_value(self) -> float:
        ...

    def terminal_score_margin(self) -> float:
        ...

    def terminal_cleanup_counts(self) -> dict[str, int]:
        ...


class SimpleAreaRules:
    """Tiny CPU-demo rules backend.

    This is deliberately not a complete Go rules implementation: stones are
    never captured. It exists so the local AlphaZero loop can run before the
    real rules dependency is installed.
    """

    def __init__(
        self,
        size: int = 9,
        komi: float = DEFAULT_9X9_KOMI,
        score_komi: float = DEFAULT_9X9_SCORE_KOMI,
        input_komi: bool = True,
        terminal_dead_stone_cleanup: bool = False,
        score_margin_reward_scale: float = 0.0,
    ) -> None:
        self.size = size
        self.komi = komi
        self.score_komi = score_komi
        self.input_komi = input_komi
        self.terminal_dead_stone_cleanup = terminal_dead_stone_cleanup
        self.score_margin_reward_scale = score_margin_reward_scale
        self.board = np.zeros((size, size), dtype=np.int8)
        self.to_play = BLACK
        self._passes = 0
        self._moves = 0
        self._legal_actions_cache: np.ndarray | None = None

    @property
    def action_size(self) -> int:
        return self.size * self.size + 1

    def copy(self) -> "SimpleAreaRules":
        clone = SimpleAreaRules(
            self.size,
            self.komi,
            self.score_komi,
            self.input_komi,
            self.terminal_dead_stone_cleanup,
            self.score_margin_reward_scale,
        )
        clone.board = self.board.copy()
        clone.to_play = self.to_play
        clone._passes = self._passes
        clone._moves = self._moves
        clone._legal_actions_cache = (
            None if self._legal_actions_cache is None else self._legal_actions_cache.copy()
        )
        return clone

    def legal_actions(self) -> np.ndarray:
        if self._legal_actions_cache is not None:
            return self._legal_actions_cache.copy()
        legal = np.zeros(self.action_size, dtype=np.bool_)
        legal[:-1] = self.board.reshape(-1) == 0
        legal[-1] = True
        self._legal_actions_cache = legal
        return legal.copy()

    def play_action(self, action: int) -> None:
        if action == self.action_size - 1:
            self._passes += 1
        else:
            row, col = divmod(action, self.size)
            if self.board[row, col] != 0:
                raise ValueError(f"illegal move for {self.to_play}: {(row, col)}")
            self.board[row, col] = 1 if self.to_play == BLACK else -1
            self._passes = 0
        self._moves += 1
        self.to_play = WHITE if self.to_play == BLACK else BLACK
        self._legal_actions_cache = None

    def play_search_action(self, action: int) -> None:
        """Apply an action that came from a precomputed legal child in MCTS."""
        if action == self.action_size - 1:
            self._passes += 1
        else:
            row, col = divmod(action, self.size)
            self.board[row, col] = 1 if self.to_play == BLACK else -1
            self._passes = 0
        self._moves += 1
        self.to_play = WHITE if self.to_play == BLACK else BLACK
        self._legal_actions_cache = None

    def encode(self) -> np.ndarray:
        own_value = 1 if self.to_play == BLACK else -1
        opp_value = -own_value
        own = (self.board == own_value).astype(np.float32)
        opp = (self.board == opp_value).astype(np.float32)
        to_play_plane = np.full((self.size, self.size), 1.0 if self.to_play == BLACK else 0.0)
        planes = [own, opp, to_play_plane]
        if self.input_komi:
            planes.append(np.full((self.size, self.size), self.komi / 10.0, dtype=np.float32))
        return np.stack(planes).astype(np.float32)

    def is_terminal(self) -> bool:
        return self._passes >= 2 or self._moves >= self.action_size + 1

    def terminal_value(self) -> float:
        return _terminal_value_from_margin(
            self.terminal_score_margin(),
            self.to_play,
            self.score_margin_reward_scale,
        )

    def terminal_score_margin(self) -> float:
        if not self.terminal_dead_stone_cleanup:
            return float(self.board.sum()) - self.score_komi
        return _area_score_from_array(_cleaned_terminal_board(self.board)) - self.score_komi

    def terminal_cleanup_counts(self) -> dict[str, int]:
        if not self.terminal_dead_stone_cleanup:
            return {"b": 0, "w": 0}
        return _dead_stone_counts(self.board)


class SgfmillRules:
    """Thin adapter around sgfmill so rules can be replaced without touching MCTS."""

    def __init__(
        self,
        size: int = 5,
        komi: float = DEFAULT_9X9_KOMI,
        score_komi: float = DEFAULT_9X9_SCORE_KOMI,
        input_komi: bool = True,
        terminal_dead_stone_cleanup: bool = False,
        score_margin_reward_scale: float = 0.0,
    ) -> None:
        try:
            from sgfmill import boards
        except ImportError as exc:
            raise RuntimeError(
                "sgfmill is required for the initial rules backend. "
                "Install project dependencies with `pip install -e .`."
            ) from exc

        self.size = size
        self.komi = komi
        self.score_komi = score_komi
        self.input_komi = input_komi
        self.terminal_dead_stone_cleanup = terminal_dead_stone_cleanup
        self.score_margin_reward_scale = score_margin_reward_scale
        self._boards = boards
        self.board = boards.Board(size)
        self.board_array = np.zeros((size, size), dtype=np.int8)
        self.to_play = BLACK
        self._passes = 0
        self._ko_forbidden: tuple[int, int] | None = None
        self._legal_actions_cache: np.ndarray | None = None

    @property
    def action_size(self) -> int:
        return self.size * self.size + 1

    def copy(self) -> "SgfmillRules":
        clone = SgfmillRules(
            self.size,
            self.komi,
            self.score_komi,
            self.input_komi,
            self.terminal_dead_stone_cleanup,
            self.score_margin_reward_scale,
        )
        clone.board = self.board.copy()
        clone.board_array = self.board_array.copy()
        clone.to_play = self.to_play
        clone._passes = self._passes
        clone._ko_forbidden = self._ko_forbidden
        clone._legal_actions_cache = (
            None if self._legal_actions_cache is None else self._legal_actions_cache.copy()
        )
        return clone

    def legal_actions(self) -> np.ndarray:
        if self._legal_actions_cache is not None:
            return self._legal_actions_cache.copy()
        legal = np.zeros(self.action_size, dtype=np.bool_)
        group_ids, group_liberties = self._group_info()
        player = self._player_value()
        opponent = -player
        for row in range(self.size):
            for col in range(self.size):
                if self._is_legal_point(row, col, group_ids, group_liberties, player, opponent):
                    legal[row * self.size + col] = True
        legal[-1] = True
        self._legal_actions_cache = legal
        return legal.copy()

    def play_action(self, action: int) -> None:
        if action == self.action_size - 1:
            self._passes += 1
        else:
            row, col = divmod(action, self.size)
            if not self.legal_actions()[action]:
                raise ValueError(f"illegal move for {self.to_play}: {(row, col)}")
            self._ko_forbidden = self.board.play(row, col, self.to_play)
            self._sync_board_array()
            self._passes = 0
        self.to_play = WHITE if self.to_play == BLACK else BLACK
        self._legal_actions_cache = None

    def play_search_action(self, action: int) -> None:
        """Apply an MCTS-selected legal action without re-probing legality."""
        if action == self.action_size - 1:
            self._passes += 1
        else:
            row, col = divmod(action, self.size)
            self._ko_forbidden = self.board.play(row, col, self.to_play)
            self._sync_board_array()
            self._passes = 0
        self.to_play = WHITE if self.to_play == BLACK else BLACK
        self._legal_actions_cache = None

    def encode(self) -> np.ndarray:
        own_value = self._player_value()
        opp_value = -own_value
        own = (self.board_array == own_value).astype(np.float32)
        opp = (self.board_array == opp_value).astype(np.float32)

        to_play_plane = np.full((self.size, self.size), 1.0 if self.to_play == BLACK else 0.0)
        planes = [own, opp, to_play_plane]
        if self.input_komi:
            planes.append(np.full((self.size, self.size), self.komi / 10.0, dtype=np.float32))
        return np.stack(planes).astype(np.float32)

    def area_winner_value(self) -> float:
        return _terminal_value_from_margin(
            self.terminal_score_margin(),
            self.to_play,
            self.score_margin_reward_scale,
        )

    def is_terminal(self) -> bool:
        return self._passes >= 2

    def terminal_value(self) -> float:
        return self.area_winner_value()

    def terminal_score_margin(self) -> float:
        if not self.terminal_dead_stone_cleanup:
            return self.board.area_score() - self.score_komi
        return _area_score_from_array(_cleaned_terminal_board(self.board_array)) - self.score_komi

    def terminal_cleanup_counts(self) -> dict[str, int]:
        if not self.terminal_dead_stone_cleanup:
            return {"b": 0, "w": 0}
        return _dead_stone_counts(self.board_array)

    def _is_legal_point(
        self,
        row: int,
        col: int,
        group_ids: np.ndarray,
        group_liberties: list[int],
        player: int,
        opponent: int,
    ) -> bool:
        if self.board_array[row, col] != 0:
            return False
        if self._ko_forbidden == (row, col):
            return False
        point_bit = 1 << (row * self.size + col)
        has_empty_neighbor = False
        has_capture = False
        has_own_group_liberty = False
        for neighbour_row, neighbour_col in self._neighbors(row, col):
            colour = int(self.board_array[neighbour_row, neighbour_col])
            if colour == 0:
                has_empty_neighbor = True
                continue
            group_id = int(group_ids[neighbour_row, neighbour_col])
            liberties_without_point = group_liberties[group_id] & ~point_bit
            if colour == opponent and liberties_without_point == 0:
                has_capture = True
            elif colour == player and liberties_without_point != 0:
                has_own_group_liberty = True
        return has_empty_neighbor or has_capture or has_own_group_liberty

    def _neighbors(self, row: int, col: int):
        if row > 0:
            yield row - 1, col
        if row + 1 < self.size:
            yield row + 1, col
        if col > 0:
            yield row, col - 1
        if col + 1 < self.size:
            yield row, col + 1

    def _group_info(self) -> tuple[np.ndarray, list[int]]:
        group_ids = np.full((self.size, self.size), -1, dtype=np.int16)
        group_liberties: list[int] = []
        group_id = 0
        for start_row in range(self.size):
            for start_col in range(self.size):
                colour = int(self.board_array[start_row, start_col])
                if colour == 0 or group_ids[start_row, start_col] >= 0:
                    continue
                stack = [(start_row, start_col)]
                group_ids[start_row, start_col] = group_id
                liberties = 0
                while stack:
                    row, col = stack.pop()
                    for neighbour_row, neighbour_col in self._neighbors(row, col):
                        neighbour_colour = int(self.board_array[neighbour_row, neighbour_col])
                        if neighbour_colour == 0:
                            liberties |= 1 << (neighbour_row * self.size + neighbour_col)
                        elif (
                            neighbour_colour == colour
                            and group_ids[neighbour_row, neighbour_col] < 0
                        ):
                            group_ids[neighbour_row, neighbour_col] = group_id
                            stack.append((neighbour_row, neighbour_col))
                group_liberties.append(liberties)
                group_id += 1
        return group_ids, group_liberties

    def _player_value(self) -> int:
        return BLACK_VALUE if self.to_play == BLACK else WHITE_VALUE

    def _sync_board_array(self) -> None:
        board_rows = getattr(self.board, "board", None)
        if board_rows is None:
            for row in range(self.size):
                for col in range(self.size):
                    self.board_array[row, col] = self._colour_value(self.board.get(row, col))
            return
        for row, row_values in enumerate(board_rows):
            for col, colour in enumerate(row_values):
                self.board_array[row, col] = self._colour_value(colour)

    @staticmethod
    def _colour_value(colour: str | None) -> int:
        if colour == BLACK:
            return BLACK_VALUE
        if colour == WHITE:
            return WHITE_VALUE
        return 0


GomillRules = SgfmillRules


def _terminal_value_from_margin(
    black_score_margin: float,
    to_play: str,
    score_margin_reward_scale: float,
) -> float:
    margin_sign = 1.0 if black_score_margin > 0 else -1.0
    if score_margin_reward_scale > 0.0:
        raw_bonus = (abs(float(black_score_margin)) ** 0.25) / 5.0
        bonus = min(
            raw_bonus * float(score_margin_reward_scale),
            VALUE_MARGIN_REWARD_LIMIT,
        )
        black_value = margin_sign * (VALUE_WIN_LOSS_BASE_WITH_MARGIN_REWARD + bonus)
    else:
        black_value = margin_sign
    return float(black_value if to_play == BLACK else -black_value)


def _cleaned_terminal_board(board: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(board, dtype=np.int8).copy()
    cleaned[_obvious_dead_stone_mask(cleaned)] = 0
    return cleaned


def _dead_stone_counts(board: np.ndarray) -> dict[str, int]:
    board_array = np.asarray(board, dtype=np.int8)
    dead_mask = _obvious_dead_stone_mask(board_array)
    return {
        "b": int(((board_array == BLACK_VALUE) & dead_mask).sum()),
        "w": int(((board_array == WHITE_VALUE) & dead_mask).sum()),
    }


def _area_score_from_array(board: np.ndarray) -> float:
    board_array = np.asarray(board, dtype=np.int8)
    score = float((board_array == BLACK_VALUE).sum() - (board_array == WHITE_VALUE).sum())
    visited = np.zeros_like(board_array, dtype=np.bool_)
    for start_row in range(board_array.shape[0]):
        for start_col in range(board_array.shape[1]):
            if board_array[start_row, start_col] != 0 or visited[start_row, start_col]:
                continue
            region, border_colours = _empty_region(board_array, [(start_row, start_col)])
            for row, col in region:
                visited[row, col] = True
            if border_colours == {BLACK_VALUE}:
                score += len(region)
            elif border_colours == {WHITE_VALUE}:
                score -= len(region)
    return score


def _obvious_dead_stone_mask(board: np.ndarray) -> np.ndarray:
    board_array = np.asarray(board, dtype=np.int8)
    dead = np.zeros_like(board_array, dtype=np.bool_)
    for colour, stones in _stone_groups(board_array):
        if _solid_eye_count(board_array, stones, colour) >= 2:
            continue
        removed = board_array.copy()
        for row, col in stones:
            removed[row, col] = 0
        region, border_colours = _empty_region(removed, stones)
        if _region_touches_edge(board_array, region):
            continue
        if region and border_colours == {-colour}:
            for row, col in stones:
                dead[row, col] = True
    return dead


def _solid_eye_count(board: np.ndarray, stones: list[tuple[int, int]], colour: int) -> int:
    seen_regions: set[frozenset[tuple[int, int]]] = set()
    eyes = 0
    for row, col in stones:
        for neighbour in _neighbors_array(board, row, col):
            neighbour_row, neighbour_col = neighbour
            if board[neighbour_row, neighbour_col] != 0:
                continue
            region, border_colours = _empty_region(board, [neighbour])
            region_key = frozenset(region)
            if region_key in seen_regions:
                continue
            seen_regions.add(region_key)
            if border_colours == {colour}:
                eyes += 1
    return eyes


def _stone_groups(board: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    visited = np.zeros_like(board, dtype=np.bool_)
    groups: list[tuple[int, list[tuple[int, int]]]] = []
    for start_row in range(board.shape[0]):
        for start_col in range(board.shape[1]):
            colour = int(board[start_row, start_col])
            if colour == 0 or visited[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            stones: list[tuple[int, int]] = []
            while stack:
                row, col = stack.pop()
                stones.append((row, col))
                for neighbour_row, neighbour_col in _neighbors_array(board, row, col):
                    if (
                        not visited[neighbour_row, neighbour_col]
                        and int(board[neighbour_row, neighbour_col]) == colour
                    ):
                        visited[neighbour_row, neighbour_col] = True
                        stack.append((neighbour_row, neighbour_col))
            groups.append((colour, stones))
    return groups


def _empty_region(
    board: np.ndarray,
    starts: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[int]]:
    stack = list(starts)
    seen: set[tuple[int, int]] = set()
    region: list[tuple[int, int]] = []
    border_colours: set[int] = set()
    while stack:
        row, col = stack.pop()
        if (row, col) in seen:
            continue
        seen.add((row, col))
        colour = int(board[row, col])
        if colour != 0:
            border_colours.add(colour)
            continue
        region.append((row, col))
        for neighbour in _neighbors_array(board, row, col):
            neighbour_row, neighbour_col = neighbour
            if int(board[neighbour_row, neighbour_col]) == 0:
                if neighbour not in seen:
                    stack.append(neighbour)
            else:
                border_colours.add(int(board[neighbour_row, neighbour_col]))
    return region, border_colours


def _neighbors_array(board: np.ndarray, row: int, col: int):
    if row > 0:
        yield row - 1, col
    if row + 1 < board.shape[0]:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < board.shape[1]:
        yield row, col + 1


def _region_touches_edge(board: np.ndarray, region: list[tuple[int, int]]) -> bool:
    last_row = board.shape[0] - 1
    last_col = board.shape[1] - 1
    return any(row == 0 or row == last_row or col == 0 or col == last_col for row, col in region)
