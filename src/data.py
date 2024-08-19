import torch
from torch.utils.data import Dataset

from agent import Step
from bitboard import Stone
from config import DatasetConfig


class SVFDataset(Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()

        self.config: DatasetConfig = config
        self.buffer: list[
            tuple[
                torch.Tensor,  # state
                torch.Tensor,  # value (last board score)
            ]
        ] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.buffer[idx]

    def add(self, player: Stone, steps: list[Step], value: float) -> None:
        for step in steps:
            self.buffer.append(
                (
                    step.state.to_tensor(player),
                    torch.tensor([value], dtype=torch.float32),
                )
            )

        if len(self.buffer) > self.config.limit_length:
            p = len(self.buffer) - self.config.limit_length
            self.buffer = self.buffer[p:]

    def periodic_delete(self) -> None:
        if len(self.buffer) > self.config.periodic_delete:
            self.buffer = self.buffer[self.config.periodic_delete :]
        else:
            assert False, "Buffer size is less than periodic delete size"

    def state_dict(self) -> dict:
        return {
            "buffer": self.buffer,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.buffer = state_dict["buffer"]
