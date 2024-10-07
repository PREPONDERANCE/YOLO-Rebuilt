import torch

from torch import nn
from yolov1.config import settings


class Reshape(nn.Module):
    def __init__(self, *shapes):
        super().__init__()
        self.shapes = shapes

    def forward(self, x: torch.Tensor):
        return x.view(*self.shapes)


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = []

        # First Block
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.LeakyReLU(negative_slope=0.1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )

        # Second block
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(192, 128, kernel_size=1, stride=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )

        # Third block
        block = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.layers.append(nn.Sequential(*[block] * 4))

        # Forth block
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        )

        # Fifth block
        block = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.layers.append(nn.Sequential(*[block] * 2))

        # Sixth block
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                Reshape(-1, 7 * 7 * 1024),
                nn.Linear(7 * 7 * 1024, 4096),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(),
                nn.Linear(4096, settings.S * settings.S * settings.BOX),
                Reshape(-1, settings.S, settings.S, settings.BOX),
            )
        )

        # Integrate all the layers
        self.final_layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        return self.final_layers(x)
