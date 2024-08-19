from dataclasses import dataclass
from bitboard import Board, Stone
from tree import Tree


@dataclass
class Step:
    state: Board
    action: int


class ModelAgent:
    def __init__(self, color: Stone, tree: Tree) -> None:
        self.color: Stone = color
        self.tree: Tree = tree
        self.history: list[Step] = []

    def act(self, state: Board) -> int:
        scores = self.tree.search(state, self.color)
        action = scores.index(max(scores))
        self.history.append(Step(state=state, action=action))
        return action

    def get_history(self) -> list[Step]:
        return self.history
