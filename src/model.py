"""
Author: rainyl
Description: models
License: Apache License 2.0
"""
import math
from torch import nn
import torch
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, in_channel: int, out_class: int, dropout=0.2) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            # 480*480*1
            nn.Conv2d(
                in_channel, 64, kernel_size=11, stride=4, padding=2
            ),  # 119*119*64
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 59*59*64
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 59*59*192
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 29*29*192
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 29*29*384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 29*29*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 29*29*256
            nn.MaxPool2d(kernel_size=3, stride=2),  # 14*14*256
        )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)  # 12*12*256
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(12 * 12 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_class),
        )

        self._init_weights()

    def _init_weights(self):
        for mod in self.convs:
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(mod.weight, a=math.sqrt(5), nonlinearity="leaky_relu")
                nn.init.normal_(mod.bias, 0, 0.05)

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
