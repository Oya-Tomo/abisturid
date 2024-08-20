from dataclasses import dataclass


@dataclass
class TreeConfig:
    depth: int = 10  # The maximum depth of the tree
    k: int = 5  # The max number of actions that must be selected


@dataclass
class SelfPlayConfig:
    num_processes: int = 10
    num_games: int = 100
    tree_config: TreeConfig = TreeConfig(
        depth=20,
        k=15,
    )


@dataclass
class DatasetConfig:
    periodic_delete: int
    limit_length: int


@dataclass
class TrainConfig:
    loops: int = 1000
    epochs: int = 50

    save_epochs: int = 2

    batch_size: int = 512
    lr: float = 0.002
    weight_decay: float = 1e-6
    momentum: float = 0.9
    nesterov: bool = True

    restart_epoch: int = 0
    load_checkpoint: str = ""

    save_dataset: str | None = "checkpoint/dataset.pth"
    load_dataset: str | None = "checkpoint/dataset.pth"


@dataclass
class Config:
    warmup_config: SelfPlayConfig
    selfplay_config: SelfPlayConfig
    dataset_config: DatasetConfig
    train_config: TrainConfig


train_config = Config(
    warmup_config=SelfPlayConfig(
        num_processes=13,
        num_games=5000,
        tree_config=TreeConfig(
            depth=20,
            k=15,
        ),
    ),
    selfplay_config=SelfPlayConfig(
        num_processes=13,
        num_games=1000,
        tree_config=TreeConfig(
            depth=20,
            k=15,
        ),
    ),
    dataset_config=DatasetConfig(
        periodic_delete=2000,
        limit_length=50000,
    ),
    train_config=TrainConfig(
        loops=1000,
        epochs=50,
        save_epochs=2,
        batch_size=512,
        lr=0.002,
        weight_decay=1e-6,
        restart_epoch=0,
        load_checkpoint="",
        save_dataset="checkpoint/dataset.pth",
        load_dataset=None,
    ),
)

debug_config = Config(
    warmup_config=SelfPlayConfig(
        num_processes=10,
        num_games=20,
        tree_config=TreeConfig(
            depth=20,
            k=15,
        ),
    ),
    selfplay_config=SelfPlayConfig(
        num_processes=10,
        num_games=10,
        tree_config=TreeConfig(
            depth=20,
            k=15,
        ),
    ),
    dataset_config=DatasetConfig(
        periodic_delete=5,
        limit_length=100,
    ),
    train_config=TrainConfig(
        loops=1000,
        epochs=50,
        save_epochs=2,
        batch_size=512,
        lr=0.002,
        weight_decay=1e-6,
        restart_epoch=0,
        load_checkpoint="",
        save_dataset="checkpoint/dataset.pth",
        load_dataset=None,
    ),
)

config = train_config
