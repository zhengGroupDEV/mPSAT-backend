'''
Author: rainyl
Description: Inference of CNN transformed to ONNX format
License: Apache License 2.0
'''
from typing import List, Union
import onnxruntime as ort
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser

from src.infer_base import MpInferenceBase


class MpInferenceNNONNX(MpInferenceBase):
    model: ort.InferenceSession
    mean = 0.5
    std = 0.5

    def __init__(self, model_path: str, label_name: str = "data/label_name.json"):
        super(MpInferenceNNONNX, self).__init__(model_path, label_name)

    def load_model(self, p: str):
        sess = ort.InferenceSession(p)
        return sess

    def normalize(self, x: np.ndarray):
        return (x - self.mean) / self.std

    def preprocess(self, imgs: Union[Image.Image, List[Image.Image]]):
        if not isinstance(imgs, list):
            imgs = [imgs]
        for i, im in enumerate(imgs):
            if im.mode != "L":
                imgs[i] = im.convert("L")
        x: List[np.ndarray] = [
            np.asarray(im, dtype=np.float32)[np.newaxis, ...] for im in imgs
        ]
        x = [self.normalize(i) for i in x]
        x1 = np.array(x)
        return x1

    def __call__(
        self,
        img: List[Image.Image],
        topk: int = 3,
    ) -> List[List[Union[int, str]]]:
        x = self.preprocess(imgs=img)
        input_name = self.model.get_inputs()[0].name
        inputs = {input_name: x}
        out: np.ndarray = self.model.run(None, inputs)[0]
        out_topk = out.argsort(axis=1)[:, ::-1][:, :topk]
        out1 = self.label2name(out_topk.tolist())
        return out1


def main(img: str, model_path: str, topk: int):
    inferer = MpInferenceNNONNX(model_path)
    if img.endswith(".jpg"):
        im = Image.open(img).convert("L")
    elif img.endswith(".csv"):
        im = inferer.read_plot_csv(img, rmco2=True, fac=0.1)
    else:
        raise NotImplementedError(f"image format {img.split('.')[-1]} not implemented!")
    im.save(img.replace(".csv", ".jpg"))
    out = inferer([im], topk=topk)
    print([["+".join(c) for c in row] for row in out])  # type:ignore


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", dest="image", required=True, type=str)
    parser.add_argument("-m", dest="model_path", required=True, type=str)
    parser.add_argument("-k", dest="topk", default=3, type=int)
    args = parser.parse_args()
    # args = parser.parse_args(
    #     ["-i", "", "-k", "5"]
    # )
    main(args.image, args.model_path, args.topk)
