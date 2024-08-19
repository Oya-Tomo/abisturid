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

        if state.is_over():
            b, w, _ = state.get_count()
            score = count_to_score(b, w)
            if turn == Stone.BLACK:
                return score
            else:
                return -score
        elif depth == 0:
            return self.values[state.key(turn)]
        else:
            actions = state.get_actions(turn)
            remain_actions = deepcopy(actions)
            search_actions = []
            for action, next_state in self.transitions[state.key(turn)].items():
                if (
                    self.values[next_state.key(flip(turn))]
                    > self.values[state.key(turn)]
                ):
                    search_actions.append(action)
                    remain_actions.remove(action)

            # fill k actions with random actions
            if len(search_actions) < min(
                self.config.k,
                len(actions),
            ):
                search_actions.extend(
                    random.sample(
                        remain_actions,
                        k=min(self.config.k, len(actions)) - len(search_actions),
                    )
                )

            scores = [0.0] * len(search_actions)
            for i, action in enumerate(search_actions):
                next_state = self.transitions[state.key(turn)][action]
                scores[i] = -self.evaluate(next_state, flip(turn), depth - 1)

            return max(scores)


if __name__ == "__main__":
    from model import SVFModel
    from bitboard import gen_random_board

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVFModel().to(device)
    config = TreeConfig(
        depth=7,
        k=2,
    )
    tree = Tree(model, config)
    board, turn = gen_random_board(50)
    print(board)
    actions = board.get_actions(turn)
    scores = tree.search(board, turn)
    print(actions)
    print(scores)
    board.act(turn, scores.index(max(scores)))
    print(board)
