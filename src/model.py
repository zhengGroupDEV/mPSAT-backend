"""
Description: models
Author: rainyl
License: Apache License 2.0
"""
import math
from typing import List

import einops
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    r"""from ConvNext
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.ones(normalized_shape))
        self.bias = nn.parameter.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class CNN(nn.Module):
    def __init__(self, in_channel: int, out_class: int, dropout=0.2) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            # (1, 3600)
            # Conv 1
            nn.Conv1d(in_channel, 4, kernel_size=11, stride=7),
            nn.BatchNorm1d(4),
            nn.GELU(),
            # Conv 2
            nn.Conv1d(4, 16, kernel_size=7, stride=5),
            nn.BatchNorm1d(16),
            # Conv 3
            nn.Conv1d(16, 64, kernel_size=5, stride=3),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
            # Conv 4
            nn.Conv1d(64, 256, kernel_size=3, stride=3),
            nn.BatchNorm1d(256),
            # Conv 5
            nn.Conv1d(256, 1024, kernel_size=3, stride=3),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 3, 2048),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(2048, out_class),
        )

        # use the default parameter initialization
        # kaiming_uniform
        # self._init_weights()

    def _init_weights(self):
        for mod in self.convs:
            if isinstance(mod, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    mod.weight,
                    a=math.sqrt(5),
                    nonlinearity="leaky_relu",
                )
                # nn.init.normal_(mod.bias, 0, 0.05)

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        x = einops.rearrange(x, "b d l -> b (d l)")
        x = self.classifier(x)
        return x


class CNN2D(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_class: int,
        dropout=0.2,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        act_layer = lambda name="relu": nn.GELU() if name=="gelu" else nn.ReLU(inplace=True)
        norm_layer = LayerNorm
        # norm_layer = nn.BatchNorm2d
        self.convs = nn.Sequential(
            # 480*480*1
            nn.Conv2d(
                in_channel,
                64,
                kernel_size=11,
                stride=3,
                padding=0,
            ),
            norm_layer(64, eps=norm_eps),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2
            nn.Conv2d(64, 192, kernel_size=7, stride=3, padding=0),
            norm_layer(192, eps=norm_eps),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3
            nn.Conv2d(192, 384, kernel_size=5, stride=3, padding=1),
            norm_layer(384, eps=norm_eps),
            act_layer(),
            # conv4
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1),
            norm_layer(768, eps=norm_eps),
            act_layer(),
            nn.Dropout(p=dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(3 * 3 * 768, 2048),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(2048, out_class),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        x = self.convs(x)
        x = einops.rearrange(x, "b c h w -> b (c h w)")
        x = self.classifier(x)
        return x
