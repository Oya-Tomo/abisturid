from copy import deepcopy
import random

import torch
from bitboard import Board, Stone, flip
from config import TreeConfig


# black = 1, white = -1
def count_to_score(b: int, w: int) -> float:
    return (b - w) / (b + w)


class Tree:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TreeConfig,
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.config = config

        self.values: dict[int, float] = {}  # state value function outputs
        self.transitions: dict[int, dict[int, Board]] = {}  # state transition cache

        self.black_max = -float("inf")
        self.white_max = -float("inf")

    def search(self, state: Board, turn: Stone) -> list[float]:
        self.expand(state, turn)

        actions = state.get_actions(turn)
        scores = [-float("inf")] * 65
        for action in actions:
            next_state = self.transitions[state.key(turn)][action]
            scores[action] = -self.evaluate(
                next_state, flip(turn), self.config.depth - 1
            )

        return scores

    def expand(self, state: Board, turn: Stone) -> None:
        if state.key(turn) in self.transitions:
            return

        actions = state.get_actions(turn)
        next_states: list[Board] = []

        self.transitions[state.key(turn)] = {}
        for action in actions:
            next_board = deepcopy(state)
            next_board.act(turn, action)
            self.transitions[state.key(turn)][action] = next_board
            next_states.append(next_board)

        inputs = torch.stack(
            [ns.to_tensor(flip(turn)) for ns in next_states],
        ).to(self.device)
        with torch.no_grad():
            values = self.model(inputs).cpu().flatten().tolist()

        for ns, v in zip(next_states, values):
            self.values[ns.key(flip(turn))] = v

    def evaluate(self, state: Board, turn: Stone, depth: int) -> float:
        self.expand(state, turn)

        key = state.key(turn)
        value = self.values[key]

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
            if turn == Stone.BLACK and value > self.black_max:
                self.black_max = value
            elif turn == Stone.WHITE and value > self.white_max:
                self.white_max = value
            else:
                return value

            transitions = self.transitions[key]
            ns_values = [
                (-self.values[ns.key(flip(turn))], ns) for _, ns in transitions.items()
            ]
            ns_values.sort(key=lambda x: x[0], reverse=True)

            scores = [-float("inf")] * len(transitions)
            for i, (v, ns) in enumerate(ns_values):
                if i < self.config.k:
                    scores[i] = -self.evaluate(ns, flip(turn), depth - 1)
                else:
                    scores[i] = v

            return max(scores)


if __name__ == "__main__":
    from model import SVFModel
    from bitboard import gen_random_board
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVFModel().to(device)
    config = TreeConfig(
        depth=15,
        k=10,
    )
    tree = Tree(model, config)
    board, turn = gen_random_board(0)
    print(board)
    actions = board.get_actions(turn)
    t = time.time()
    scores = tree.search(board, turn)
    print(time.time() - t)
    print(actions)
    print(scores)
    board.act(turn, scores.index(max(scores)))
    print(board)
