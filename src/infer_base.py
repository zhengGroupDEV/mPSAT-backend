'''
Author: rainyl
Description: Inference Base Class
License: Apache License 2.0
'''
from typing import List, Union, Dict
import warnings
from matplotlib import pyplot as plt
from PIL import Image
import json
import os

import numpy as np


class MpInferenceBase(object):
    xx = np.arange(400, 4000, 0.1)

    def __init__(self, model_path: str, label_name: str):
        self.model = self.load_model(model_path)
        self.label_name_hash = self.load_label_name(label_name)

    def minmax(self, x: np.ndarray) -> np.ndarray:
        return (x - min(x)) / (max(x) - min(x))

    def plot_spec(self, x, y, PX_H: int = 480, dpi: int = 120) -> Image.Image:
        fig = plt.figure(num=1, figsize=(PX_H / dpi, PX_H / dpi), dpi=dpi)
        xx = np.round(self.xx, 2)
        yy = np.interp(xx, x, y, left=0, right=0)

        ax = fig.add_subplot(111)
        ax.plot(xx, yy, ls="-", lw=1, c="k")
        ax.set_xlim(4000, 400)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        im = Image.frombytes(
            "RGB", canvas.get_width_height(), canvas.tostring_rgb()
        ).convert("L")
        plt.close(1)
        return im

    def load_label_name(self, p: str):
        label_name_hash: Union[None, Dict[str, List[str]]] = None
        if os.path.exists(p):
            warnings.warn(f"label to name file {p} not exists!", ResourceWarning)
        with open(p, "r") as f:
            label_name_hash = json.load(f)
        return label_name_hash

    def label2name(self, labels: List[List[int]]) -> List[List[Union[int, str]]]:
        if not all([all([c in self.label_name_hash for c in row]) for row in labels]):
            warnings.warn(f"some label not in label_name_hash table!", ResourceWarning)
        names = [[self.label_name_hash[str(c)] for c in row] for row in labels]
        return names

    def read_plot_csv(
        self,
        img: str,
        rmco2: bool = False,
        fac: float = 0.2,
    ) -> Image.Image:
        _d = np.loadtxt(img, comments=None, delimiter=",", skiprows=1, dtype=np.float32)
        if rmco2:
            _d = self.rm_co2(_d, fac=fac)
        idx = np.argsort(_d.T[0])
        x = _d.T[0][idx]
        y = self.minmax(_d.T[1][idx])
        im = self.plot_spec(x, y)
        return im

    def rm_co2(self, x: np.ndarray, fac: float = 0.2):
        c1 = (self.xx >= 2280) & (self.xx <= 2400)
        c2 = (self.xx >= 3570) & (self.xx <= 3750)
        idxs = (c1 | c2)
        idx = np.argsort(x.T[0])
        x_ = x.T[0][idx]
        y_ = x.T[1][idx]
        _bins, _height = np.histogram(y_, bins=50)
        most = _height[np.argmax(_bins)]
        yy = np.interp(self.xx, x_, y_, left=0, right=0)
        yy[idxs] = yy[idxs] * fac
        yy = (yy - most) / (yy.max() - yy.min())
        yy[yy < 0] = 0
        # yy = self.minmax(yy)
        res = np.vstack((self.xx, yy)).T
        return res

    def load_model(self, p: str):
        ...

    def __call__(self, x: List[Image.Image]):
        ...

    def preprocess(self, x: Union[Image.Image, List[Image.Image]]):
        ...
