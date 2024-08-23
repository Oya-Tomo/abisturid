from copy import deepcopy
from dataclasses import dataclass
import random

import torch
from bitboard import Board, Stone, flip
from config import TreeConfig


# black = 1, white = -1
def count_to_score(b: int, w: int) -> float:
    return (b - w) / (b + w)


NEG_INF = -float("inf")


@dataclass
class Node:
    state: Board
    value: float


class Tree:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TreeConfig,
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.config = config

        self.nodes: dict[int, dict[int, Node]] = {}  # tree nodes

    def search(self, state: Board, turn: Stone) -> list[float]:
        self.expand(state, turn)

        actions = state.get_actions(turn)
        scores = [NEG_INF] * 65
        for action in actions:
            next_node = self.nodes[state.key(turn)][action]
            scores[action] = -self.evaluate(
                next_node,
                flip(turn),
                self.config.depth - 1,
                NEG_INF,
                NEG_INF,
            )

        return scores

    def expand(self, state: Board, turn: Stone) -> None:
        if state.key(turn) in self.nodes:
            return

        actions = state.get_actions(turn)
        next_states: list[Board] = []

        for action in actions:
            next_state = deepcopy(state)
            next_state.act(turn, action)
            next_states.append(next_state)

        inputs = torch.stack(
            [ns.to_tensor(flip(turn)) for ns in next_states],
        ).to(self.device)
        with torch.no_grad():
            values = self.model(inputs).cpu().flatten().tolist()

        next_nodes = [(a, s, v) for a, s, v in zip(actions, next_states, values)]
        next_nodes.sort(key=lambda x: x[2], reverse=True)
        self.nodes[state.key(turn)] = {
            a: Node(state=s, value=v) for a, s, v in next_nodes
        }

    def evaluate(
        self,
        current: Node,
        turn: Stone,
        depth: int,
        black_thd: float,
        whihe_thd: float,
    ) -> float:
        state = current.state
        value = current.value
        key = state.key(turn)

        if state.is_over():
            b, w, _ = state.get_count()
            score = count_to_score(b, w)
            if turn == Stone.BLACK:
                return score
            else:
                return -score
        elif depth == 0:
            return value
        else:
            if turn == Stone.BLACK and value >= black_thd:
                black_thd = value
            elif turn == Stone.WHITE and value >= whihe_thd:
                whihe_thd = value
            else:
                return value

            self.expand(state, turn)

            next_nodes = self.nodes[key]

            scores = [-float("inf")] * len(next_nodes)
            for i, (action, node) in enumerate(next_nodes.items()):
                if i < self.config.k:
                    scores[i] = -self.evaluate(
                        node,
                        flip(turn),
                        depth - 1,
                        black_thd,
                        whihe_thd,
                    )
                else:
                    scores[i] = -node.value

            return max(scores)


if __name__ == "__main__":
    from model import SVFModel
    from bitboard import gen_random_board
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVFModel().to(device)
    config = TreeConfig(
        depth=4,
        k=10,
    )
    tree = Tree(model, config)
    board, turn = gen_random_board(0)
    while not board.is_over():
        print(turn)
        print(board)
        scores = tree.search(board, turn)
        print(scores)
        action = scores.index(max(scores))
        print(action)
        board.act(turn, action)
        turn = flip(turn)
        print("\n")

    print(board)
    print(board.get_count())
