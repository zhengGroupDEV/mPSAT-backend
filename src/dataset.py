"""
Description: dataset class
Author: Rainyl
License: Apache License 2.0
"""
from typing import Dict, List, Union
from torch.utils.data import Dataset
from pathlib import Path
import torch
import glob
import torchvision
import os
import numpy as np

class MpDatasetCNN(Dataset):
    """
    Dataset for CNN(and other Images models)
    :param dpath: the dataset path, should be the following format:
        - dpath
            - 0
                1.jpg
                2.jpg
                ...
            - 1
                1.jpg
    :param imgfmt: str, the format of the images
    """

    def __init__(
        self,
        dpath: Union[str, Path],
        transforms: Union[torchvision.transforms.Compose, None] = None,
        imgfmt: str = "jpg",
    ):
        self.transforms = transforms
        self.dpath = dpath
        self.imgfmt = imgfmt
        if os.path.isdir(self.dpath):
            if not os.path.exists(self.dpath):
                raise ValueError(f"Dataset {self.dpath} not exists !")
        elif os.path.islink(self.dpath):
            self.dpath = os.readlink(self.dpath)
        else:
            raise NotImplementedError(f"dataset {self.dpath} not supportd!")
        # load labels and image paths from given dataset path
        self.imgsPaths = list(glob.glob(
            str(Path(dpath).absolute()) + f"/**/*.{imgfmt}",
            recursive=True,
        ))
        self.labels = [int(Path(p).parent.name) for p in self.imgsPaths]
        self.imgs = [str(Path(p).absolute()) for p in self.imgsPaths]

    def __getitem__(self, idx):
        img = torchvision.io.read_image(
            self.imgs[idx], mode=torchvision.io.ImageReadMode.GRAY
        )
        img = img.to(torch.float32)
        label = self.labels[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.labels)


class MpDatasetCNNMCCV(Dataset):
    def __init__(
        self,
        dpath: Union[str, Path],
        transforms: Union[torchvision.transforms.Compose, None] = None,
        imgfmt: str = "jpg",
        ptrain: float = 0.7,
        pval: float = 0.3,
        dataset: str = "train",
    ):
        self.transforms = transforms
        self.dpath = dpath if isinstance(dpath, Path) else Path(dpath)
        self.imgfmt = imgfmt
        self.__dataset = dataset
        self.ptrain = ptrain
        self.pval = pval
        if not os.path.exists(self.dpath):
            raise ValueError(f"Data file {self.dpath} not exists !")
        # load labels and image paths from given dataset path
        # use glob to follow symbal links, soft links
        imgs_path = glob.glob(
            str(Path(dpath).absolute()) + f"/**/*.{imgfmt}",
            recursive=True,
        )
        self.imgsPaths = [Path(p) for p in imgs_path]
        # Path.glob won't follow soft links
        print(f"number of images: {len(self.imgsPaths)}, ptrain: {ptrain}")

        self.shuffle_dataset()

    def shuffle_dataset(self):
        labels = [int(p.parent.name) for p in self.imgsPaths]
        imgs = [str(p.absolute()) for p in self.imgsPaths]
        # split train and validation dataset by shuffle index
        sf_idx = np.arange(len(labels))
        np.random.shuffle(sf_idx)
        n_train = int(self.ptrain * len(labels))
        train_idx = sf_idx[:n_train]
        val_idx = sf_idx[n_train:]
        self.img_train = [imgs[i] for i in train_idx]
        self.label_train = [labels[i] for i in train_idx]
        self.img_val = [imgs[i] for i in val_idx]
        self.label_val = [labels[i] for i in val_idx]

    def set_dataset(self, dataset="train"):
        assert dataset in ["train", "val"], "dataset must be train or val"
        self.__dataset = dataset

    def __getitem__(self, idx):
        if self.__dataset == "train":
            imgs = self.img_train
            labels = self.label_train
        else:
            imgs = self.img_val
            labels = self.label_val

        img = torchvision.io.read_image(
            imgs[idx], mode=torchvision.io.ImageReadMode.GRAY
        )
        img = img.to(torch.float32)
        label = labels[idx]
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        if self.__dataset == "train":
            return len(self.label_train)
        else:
            return len(self.label_val)


class MpDatasetSVM(Dataset):
    def __init__(
        self,
        dpath: Union[str, Path],
        transforms: Union[torchvision.transforms.Compose, None] = None,
        imgfmt: str = "jpg",
        shuffle=False,
    ):
        self.transforms = transforms
        self.dpath = dpath if isinstance(dpath, Path) else Path(dpath)
        self.imgfmt = imgfmt
        if not os.path.exists(self.dpath):
            raise ValueError(f"Data path {self.dpath} not exists !")
        # load labels and image paths from given dataset path
        self.imgsPaths = list(glob.glob(
            str(Path(dpath).absolute()) + f"/**/*.{imgfmt}",
            recursive=True,
        ))
        if shuffle:
            np.random.shuffle(self.imgsPaths)  # type: ignore
        self.labels = [int(Path(p).parent.name) for p in self.imgsPaths]
        self.imgs = [str(Path(p).absolute()) for p in self.imgsPaths]

    def __getitem__(self, idx):
        img = torchvision.io.read_image(
            self.imgs[idx], mode=torchvision.io.ImageReadMode.GRAY
        )
        label = self.labels[idx]
        if self.transforms:
            img = self.transforms(img)
        return img.flatten().numpy() - 255, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    ...
