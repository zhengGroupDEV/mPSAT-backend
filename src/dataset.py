"""
Description: dataset class
Author: Rainyl
License: Apache License 2.0
"""
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset
from pathlib import Path
import torch
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageChops


class MpDataset(Dataset):
    """
    Dataset for CNN(and other Images models)
    :param path: the dataset path, should be the following format:
        - path
            - 0
                0.ftr
                1.ftr
                ...
            - 1
                0.ftr
                1.ftr
                ...
    :param fmt: str, the format of the images
    """

    def __init__(self, path: str, fmt="ftr", transforms=None) -> None:
        super(MpDataset, self).__init__()
        self.path = path
        if os.path.islink(path):
            self.path = os.path.realpath(path)
        assert os.path.exists(
            self.path
        ), f"dataset path {path} not exist or link broken"
        self.fmt = fmt
        self.transforms = transforms
        if self.path.endswith("/"):
            self.path = self.path[:-1]
        self.data_path = glob.glob(self.path + f"/**/*.{self.fmt}", recursive=True)

        self._len: int = len(self.data_path)

    def normalize(self, x, mean=0.05, std=0.5):
        return (x - mean) / std

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class MpDatasetCNN(MpDataset):
    def __init__(
        self,
        path: Union[str, Path],
        transforms=None,
        fmt: str = "ftr",
        seq_len: int = 3600,
    ):
        if isinstance(path, Path):
            path = str(path.absolute())
        super(MpDatasetCNN, self).__init__(path, fmt, transforms)
        self.seq_len = seq_len
        self.labels = [int(Path(p).parent.name) for p in self.data_path]
        self.specs = [str(Path(p).absolute()) for p in self.data_path]

    def __getitem__(self, idx):
        spec = pd.read_feather(self.specs[idx])
        _, y = spec.values.T
        assert (
            y.shape[0] <= self.seq_len
        ), f"data length {y.shape[0]} overflow!, file {self.specs[idx]}"
        y = torch.from_numpy(y.astype(np.float32))
        label = self.labels[idx]
        if self.transforms:
            y = self.transforms(y)
        return y, label


class MpDatasetCnn2D(MpDataset):
    def __init__(
        self,
        path: Union[str, Path],
        transforms=None,
        fmt: str = "png",
        im_size: Tuple[int, int] = (480, 480),
        resample = Image.Resampling.LANCZOS,
    ):
        if isinstance(path, Path):
            path = str(path.absolute())
        super(MpDatasetCnn2D, self).__init__(path, fmt, transforms)
        self.labels = [int(Path(p).parent.name) for p in self.data_path]
        self.specs = [str(Path(p).absolute()) for p in self.data_path]
        self.im_size = (im_size[1], im_size[0])  # input (h w), PIL need (w h)
        self.resample = resample

    def crop_bbox(self, img: Image.Image):
        bg = Image.new(img.mode, img.size, "white")
        diff = ImageChops.difference(img, bg)
        # diff = ImageChops.add(diff, diff)
        img = img.crop(diff.getbbox())
        return img

    def __getitem__(self, idx):
        label = self.labels[idx]
        im = Image.open(self.specs[idx]).convert("L")
        im = self.crop_bbox(im)
        im = im.resize(self.im_size, resample=self.resample)
        # return im
        arr = np.asarray(im, dtype=np.float32)
        y = torch.from_numpy(arr).unsqueeze(0)
        if self.transforms:
            y = self.transforms(y)
        return y, label


class MpDatasetSK(MpDataset):
    def __init__(
        self,
        path: Union[str, Path],
        transforms=None,
        fmt: str = "ftr",
        shuffle=True,
        seq_len: int = 3600,
    ):
        if isinstance(path, Path):
            path = str(path.absolute())
        super(MpDatasetSK, self).__init__(path, fmt, transforms)
        self.seq_len = seq_len
        if shuffle:
            np.random.shuffle(self.data_path)  # type: ignore
        self.labels = [int(Path(p).parent.name) for p in self.data_path]
        self.specs = [str(Path(p).absolute()) for p in self.data_path]

    def __getitem__(self, idx):
        _, y = pd.read_feather(self.specs[idx]).values.T
        assert (
            y.shape[0] <= self.seq_len
        ), f"data length {y.shape[0]} overflow!, file {self.specs[idx]}"
        y = y.astype(np.float32)
        label = self.labels[idx]
        if self.transforms:
            y = self.transforms(y)
        return y, label


if __name__ == "__main__":
    ...
