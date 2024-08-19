import torch
from torch import nn


class SVFModel(nn.Module):  # SVF: State Value Function
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            ResBlock(3, 64, 128),
            ResBlock(128, 128, 256),
            ResBlock(256, 256, 512),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.output_layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.ds = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.se = SEBlock(out_channels, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_x = self.ds(x.clone())
        x = self.layers(x)
        x = self.se(x)
        return x + skip_x


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        se = self.layers(x)
        se = se.view(b, c, 1, 1)
        return x * self.layers(x)


if __name__ == "__main__":
    import torchinfo

    torchinfo.summary(SVFModel(), (32, 2, 8, 8))
