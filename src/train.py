import os
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

from bitboard import Stone
from data import SVFDataset
from config import config
from match import self_play_loop
from model import SVFModel


def train():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SVFDataset(config.dataset_config)

    print("Create objects")

    if config.train_config.restart_epoch == 0:
        model = SVFModel().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.train_config.lr,
            weight_decay=config.train_config.weight_decay,
            momentum=config.train_config.momentum,
            nesterov=config.train_config.nesterov,
        )
        loss_history = []
    else:
        checkpoint = torch.load(config.train_config.load_checkpoint)
        model = SVFModel().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.train_config.lr,
            weight_decay=config.train_config.weight_decay,
            momentum=config.train_config.momentum,
            nesterov=config.train_config.nesterov,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_history = checkpoint["loss_history"]

    print("Warmup start")

    if config.train_config.load_dataset is not None:
        print("Load dataset")
        dataset.load_state_dict(torch.load(config.train_config.load_dataset))
        print(f"Load dataset done:  size -> {len(dataset)} boards")
    else:
        for i, res in enumerate(self_play_loop(model, config.warmup_config)):
            black_steps, black_value, white_steps, white_value = res
            dataset.add(Stone.BLACK, black_steps, black_value)
            dataset.add(Stone.WHITE, white_steps, white_value)
            print(f"    Game: {i} score: {black_value}")

    print("Warmup done")

    if config.train_config.save_dataset is not None:
        print("Save dataset")
        if not os.path.exists("checkpoint"):
            os.makedirs("checkpoint")
        torch.save(dataset.state_dict(), config.train_config.save_dataset)
        print(f"Save dataset done: size -> {len(dataset)} boards")

    print("Train start")

    for loop in range(config.train_config.loops):
        print(f"Loop: {loop}")

        dataset.periodic_delete()
        for i, res in enumerate(self_play_loop(model, config.selfplay_config)):
            black_steps, black_value, white_steps, white_value = res
            dataset.add(Stone.BLACK, black_steps, black_value)
            dataset.add(Stone.WHITE, white_steps, white_value)
            print(f"    Game: {i} score: {black_value}")

        dataloader = DataLoader(
            dataset,
            batch_size=config.train_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        model = model.to(device)
        model.train()

        with tqdm(
            range(config.train_config.epochs),
            ncols=80,
            bar_format="{l_bar}{bar:10}{r_bar}",
        ) as pbar:
            for epoch in pbar:
                total_loss = 0
                for state, value in tqdm(
                    dataloader,
                    ncols=80,
                    bar_format="{l_bar}{bar:10}{r_bar}",
                ):
                    state = state.to(device)
                    value = value.to(device)

                    optimizer.zero_grad()
                    pred_value = model(state)
                    loss = torch.nn.functional.mse_loss(pred_value, value)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                loss_history.append(total_loss / len(dataloader))
                pbar.set_postfix({"loss": f"{total_loss / len(dataloader):.8f}"})

        if (
            loop % config.train_config.save_epochs
            == config.train_config.save_epochs - 1
        ):
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_history": loss_history,
            }

            if not os.path.exists("checkpoint"):
                os.makedirs("checkpoint")
            torch.save(checkpoint, f"checkpoint/model_{loop}.pth")

        if config.train_config.save_dataset is not None:
            print("Save dataset")
            if not os.path.exists("checkpoint"):
                os.makedirs("checkpoint")
            torch.save(dataset.state_dict(), config.train_config.save_dataset)
            print(f"Save dataset done: size -> {len(dataset)} boards")


if __name__ == "__main__":
    train()
