"""
Description: config class
Author: Rainyl
License: Apache License 2.0
"""
import json
from typing import Dict, List, Union

__supported_models__ = ("cnn", "cnn2d")
__supported_device__ = ("cpu", "cuda")


class ConfigBase(object):
    def __init__(self, path: str, model: str = "cnn") -> None:
        assert model in __supported_models__, f"model {model} not implemented"
        self.path: str = path
        self.model: str = model
        self.batch_size: int = 1
        self.shuffle: bool = True
        self.num_class: int = 120
        self.seq_len: int = 3600
        self.num_tokens: int = 1000

    def load(self, p: Union[str, None] = None):
        p = p or self.path
        with open(p, "r", encoding="utf-8") as f:
            conf: Dict[str, Union[str, int, float]] = json.load(f)
        for k, v in conf.items():
            if getattr(self, k) is not None:
                setattr(self, k, v)

    @property
    def json(self):
        raise NotImplementedError()

    def save_init(self, p: Union[str, None] = None):
        p = p or self.path
        d = self.json
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=4)


class ConfigCnn(ConfigBase):
    def __init__(self, path: str, model: str = "cnn") -> None:
        super().__init__(path, model)
        # train
        self.batch_size: int = 512
        self.shuffle: bool = True
        # model
        self.in_channel: int = 1
        self.dropout: float = 0.1
        self.norm_eps: float = 1e-6
        self.num_tokens: int = 1000
        self.num_class: int = 120
        self.seq_len: int = 3600
        self.reserve: int = 3

    @property
    def json(self):
        assert self.path, f"config path is not set"
        d = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "dropout": self.dropout,
            "norm_eps": self.norm_eps,
            "num_tokens": self.num_tokens,
            "num_class": self.num_class,
            "seq_len": self.seq_len,
            "reserve": self.reserve,
        }
        return d


class ConfigCnn2D(ConfigBase):
    def __init__(self, path: str, model: str = "cnn") -> None:
        super().__init__(path, model)
        # train
        self.batch_size: int = 128
        self.shuffle: bool = True
        # model
        self.in_channel: int = 1
        self.dropout: float = 0.1
        self.norm_eps: float = 1e-6
        self.num_tokens: int = 1000
        self.num_class: int = 120
        self.seq_len: int = 3600
        self.reserve: int = 3

    @property
    def json(self):
        assert self.path, f"config path is not set"
        d = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "dropout": self.dropout,
            "norm_eps": self.norm_eps,
            "num_tokens": self.num_tokens,
            "num_class": self.num_class,
            "seq_len": self.seq_len,
            "reserve": self.reserve,
        }
        return d
