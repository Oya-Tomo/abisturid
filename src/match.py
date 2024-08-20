import random
from typing import Generator
import torch
from torch.multiprocessing import Queue, Process

from agent import ModelAgent, Step
from bitboard import Stone, flip, gen_random_board
from config import SelfPlayConfig, TreeConfig
from model import SVFModel
from tree import Tree, count_to_score


def self_play(
    queue: Queue,
    black_weight,
    white_weight,
    config: TreeConfig,
    random_start: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    black_model = SVFModel().to(device)
    black_model.load_state_dict(black_weight)
    black_tree = Tree(black_model, config)
    black_agent = ModelAgent(Stone.BLACK, black_tree)

    white_model = SVFModel().to(device)
    white_model.load_state_dict(white_weight)
    white_tree = Tree(white_model, config)
    white_agent = ModelAgent(Stone.WHITE, white_tree)

    board, turn = gen_random_board(random_start)

    while not board.is_over():
        if turn == Stone.BLACK:
            action = black_agent.act(board)
        else:
            action = white_agent.act(board)

        board.act(turn, action)
        turn = flip(turn)

    b, w, _ = board.get_count()
    score = count_to_score(b, w)

    queue.put(
        (
            black_agent.get_history(),
            score,
            white_agent.get_history(),
            -score,
        )
    )


def self_play_loop(model: SVFModel, config: SelfPlayConfig) -> Generator[
    tuple[list[Step], float, list[Step], float],
    None,
    None,
]:
    queue = Queue()
    tasks: list[Process] = []
    workers: list[Process] = []

    model_weight = model.cpu().state_dict()

    for _ in range(config.num_games):
        task = Process(
            target=self_play,
            args=(
                queue,
                model_weight,
                model_weight,
                config.tree_config,
                random.randint(0, 58),
            ),
        )
        tasks.append(task)

    for _ in range(config.num_processes):
        task = tasks.pop(0)
        task.start()
        workers.append(task)

    while len(workers) > 0:
        joined = False
        while not joined:
            for i in range(len(workers)):
                if workers[i].exitcode is not None:
                    workers[i].join()
                    workers.pop(i)
                    joined = True
                    break

        if len(tasks) > 0:
            task = tasks.pop(0)
            task.start()
            workers.append(task)

        black_history, black_score, white_history, white_score = queue.get()
        yield black_history, black_score, white_history, white_score


if __name__ == "__main__":
    from multiprocessing import Process

    queue = Queue()

    config = TreeConfig(
        depth=20,
        k=15,
    )

    black_weight = SVFModel().state_dict()
    white_weight = SVFModel().state_dict()

    processes = []
    for _ in range(4):
        process = Process(
            target=self_play,
            args=(queue, black_weight, white_weight, config, 40),
        )
        process.start()
        processes.append(process)

    for i in range(4):
        found = False
        while not found:
            for pidx in range(len(processes)):
                if processes[pidx].exitcode is not None:
                    processes[pidx].join()
                    processes.pop(pidx)
                    found = True
                    break

        black_history, black_score, white_history, white_score = queue.get()
        print(f"Black: {black_score}")
        print(f"White: {white_score}")
