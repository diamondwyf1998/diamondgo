from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


BLACK = "b"
WHITE = "w"
PASS = None


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


class SimpleAreaRules:
    """Tiny CPU-demo rules backend.

    This is deliberately not a complete Go rules implementation: stones are
    never captured. It exists so the local AlphaZero loop can run before the
    real rules dependency is installed.
    """

    def __init__(self, size: int = 9, komi: float = 0.5) -> None:
        self.size = size
        self.komi = komi
        self.board = np.zeros((size, size), dtype=np.int8)
        self.to_play = BLACK
        self._passes = 0
        self._moves = 0
        self._legal_actions_cache: np.ndarray | None = None

    @property
    def action_size(self) -> int:
        return self.size * self.size + 1

    def copy(self) -> "SimpleAreaRules":
        clone = SimpleAreaRules(self.size, self.komi)
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
        komi_plane = np.full((self.size, self.size), self.komi / 10.0, dtype=np.float32)
        return np.stack([own, opp, to_play_plane, komi_plane]).astype(np.float32)

    def is_terminal(self) -> bool:
        return self._passes >= 2 or self._moves >= self.action_size + 1

    def terminal_value(self) -> float:
        black_minus_white = float(self.board.sum()) - self.komi
        winner = BLACK if black_minus_white > 0 else WHITE
        return 1.0 if winner == self.to_play else -1.0


class SgfmillRules:
    """Thin adapter around sgfmill so rules can be replaced without touching MCTS."""

    def __init__(self, size: int = 5, komi: float = 0.5) -> None:
        try:
            from sgfmill import boards
        except ImportError as exc:
            raise RuntimeError(
                "sgfmill is required for the initial rules backend. "
                "Install project dependencies with `pip install -e .`."
            ) from exc

        self.size = size
        self.komi = komi
        self._boards = boards
        self.board = boards.Board(size)
        self.to_play = BLACK
        self._passes = 0
        self._ko_forbidden: tuple[int, int] | None = None
        self._legal_actions_cache: np.ndarray | None = None

    @property
    def action_size(self) -> int:
        return self.size * self.size + 1

    def copy(self) -> "SgfmillRules":
        clone = SgfmillRules(self.size, self.komi)
        clone.board = self.board.copy()
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
        opponent = WHITE if self.to_play == BLACK else BLACK
        for row in range(self.size):
            for col in range(self.size):
                if self._is_legal_point(row, col, group_ids, group_liberties, opponent):
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
            self._passes = 0
        self.to_play = WHITE if self.to_play == BLACK else BLACK
        self._legal_actions_cache = None

    def encode(self) -> np.ndarray:
        own = np.zeros((self.size, self.size), dtype=np.float32)
        opp = np.zeros((self.size, self.size), dtype=np.float32)
        for row in range(self.size):
            for col in range(self.size):
                colour = self.board.get(row, col)
                if colour == self.to_play:
                    own[row, col] = 1.0
                elif colour is not None:
                    opp[row, col] = 1.0

        to_play_plane = np.full((self.size, self.size), 1.0 if self.to_play == BLACK else 0.0)
        komi_plane = np.full((self.size, self.size), self.komi / 10.0, dtype=np.float32)
        return np.stack([own, opp, to_play_plane, komi_plane]).astype(np.float32)

    def area_winner_value(self) -> float:
        black_minus_white = self.board.area_score() - self.komi
        winner = BLACK if black_minus_white > 0 else WHITE
        return 1.0 if winner == self.to_play else -1.0

    def is_terminal(self) -> bool:
        return self._passes >= 2

    def terminal_value(self) -> float:
        return self.area_winner_value()

    def _is_legal_point(
        self,
        row: int,
        col: int,
        group_ids: np.ndarray,
        group_liberties: list[set[tuple[int, int]]],
        opponent: str,
    ) -> bool:
        if self.board.get(row, col) is not None:
            return False
        if self._ko_forbidden == (row, col):
            return False
        point = (row, col)
        has_empty_neighbor = False
        has_capture = False
        has_own_group_liberty = False
        for neighbour_row, neighbour_col in self._neighbors(row, col):
            colour = self.board.get(neighbour_row, neighbour_col)
            if colour is None:
                has_empty_neighbor = True
                continue
            group_id = int(group_ids[neighbour_row, neighbour_col])
            liberties_without_point = group_liberties[group_id] - {point}
            if colour == opponent and not liberties_without_point:
                has_capture = True
            elif colour == self.to_play and liberties_without_point:
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

    def _group_info(self) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
        group_ids = np.full((self.size, self.size), -1, dtype=np.int16)
        group_liberties: list[set[tuple[int, int]]] = []
        group_id = 0
        for start_row in range(self.size):
            for start_col in range(self.size):
                colour = self.board.get(start_row, start_col)
                if colour is None or group_ids[start_row, start_col] >= 0:
                    continue
                stack = [(start_row, start_col)]
                group_ids[start_row, start_col] = group_id
                liberties: set[tuple[int, int]] = set()
                while stack:
                    row, col = stack.pop()
                    for neighbour_row, neighbour_col in self._neighbors(row, col):
                        neighbour_colour = self.board.get(neighbour_row, neighbour_col)
                        if neighbour_colour is None:
                            liberties.add((neighbour_row, neighbour_col))
                        elif (
                            neighbour_colour == colour
                            and group_ids[neighbour_row, neighbour_col] < 0
                        ):
                            group_ids[neighbour_row, neighbour_col] = group_id
                            stack.append((neighbour_row, neighbour_col))
                group_liberties.append(liberties)
                group_id += 1
        return group_ids, group_liberties


GomillRules = SgfmillRules
