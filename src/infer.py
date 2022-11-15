"""
Description: Universal inference of ONNX models
Author: rainyl
License: Apache License 2.0
"""
import os
import random
from argparse import ArgumentParser, Namespace
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import onnxruntime as ort
import pandas as pd
from numpy.typing import NDArray
from PIL import Image, ImageChops
from sklearn import metrics

from .infer_base import MpInferenceBase, MpDatasetMPe


def seed_everything(seed: int):
    # from kaggle, https://www.kaggle.com/code/rhythmcam/random-seed-everything
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def eval_ds(inferer: MpInferenceBase, args: Namespace):
    assert isinstance(inferer, (MpInferenceBase))
    # 1. load dataset
    ds_mpe = MpDatasetMPe(
        path=args.ds_path,
        shuffle=False,
        mpid_label=args.mpid_label,
    )
    y_trues = [d[1] for d in ds_mpe]
    specs = [d[0] for d in ds_mpe]
    scores = []
    batch_size = args.batch_size
    for b in range(0, len(specs), batch_size):
        out = inferer(specs[b : b + batch_size], topk=args.topk, return_score=True)
        scores.append(out)
    scores = np.concatenate(scores, axis=0)
    acc = metrics.top_k_accuracy_score(
        y_trues,
        scores,
        k=args.topk,
        labels=np.arange(args.num_class),
    )
    print(
        f"############# result of eval mpe ############\n"
        f"top-{args.topk} Accuracy: {acc}\n"
    )
    if args.saveto:
        names = np.expand_dims([d[2] for d in ds_mpe], 1)
        save_to = Path(args.saveto)
        label_name_hash = inferer.load_label_name(args.label_name)
        y_trues_name = np.expand_dims(
            ["+".join(label_name_hash[str(y)]) for y in y_trues],
            1,
        )
        out_topk = scores.argsort(axis=1)[:, ::-1][:, : args.topk]

        out_scores = scores.copy()
        out_scores.sort(axis=1)
        out_scores = np.hstack((names, out_scores[:, ::-1][:, :args.topk]))
        df_scores = pd.DataFrame(
            data=out_scores,
            columns=["name", *[f"score{i}" for i in range(args.topk)]],
        )
        df_scores.sort_values(by=["name"], ascending=True, inplace=True)
        df_scores.to_csv(save_to / f"preds_eval_scores.csv", index=False, float_format="%.4f")

        out1 = inferer.label2name(out_topk.tolist())
        out1_join = np.asarray([["+".join(c) for c in r] for r in out1], dtype=str)
        out1 = np.hstack((names, y_trues_name, out1_join))
        out_df = pd.DataFrame(
            data=out1,
            columns=["name", "true", *[f"pred{i}" for i in range(args.topk)]],
        )
        out_df["name"] = out_df["name"].astype(int)
        out_df.sort_values(by=["name"], ascending=True, inplace=True)
        save_df_to = save_to / f"preds_eval_ds.csv"
        out_df.to_csv(save_df_to, index=False)
        print(f"Saved eval results to {save_df_to}")
    return acc


class MpInferenceOnnx(MpInferenceBase):
    model: ort.InferenceSession
    __supported_models__ = ("dt", "rf", "cnn", "cnn2d", "vit")

    def __init__(
        self,
        model_path: str,
        model_name: str = "cnn",
        label_name: str = "data/label_name.json",
        device: str = "cpu",
    ):
        _providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            ("CPUExecutionProvider",),
        ]
        if device == "cpu":
            self.providers = _providers[1]
        elif device == "cuda":
            self.providers = _providers
        else:
            raise ValueError(f"device {device} not supported")
        super(MpInferenceOnnx, self).__init__(model_path, model_name, label_name)

    def load_model(self, p: str):
        sess = ort.InferenceSession(p, providers=self.providers)
        return sess

    def crop_bbox(self, img: Image.Image):
        bg = Image.new(img.mode, img.size, "white")
        diff = ImageChops.difference(img, bg)
        # diff = ImageChops.add(diff, diff)
        img = img.crop(diff.getbbox())
        return img

    def plot_fig(self, y: NDArray[np.float32]):
        # y: (B, 3600)
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        # rng = np.random.default_rng()
        # lw = rng.uniform(0.1, 3)
        # figsize = (rng.integers(4, 10), rng.integers(4, 10))
        # plt.figure(figsize=(rng.integers(1, 10), rng.integers(1, 10)), dpi=rng.integers)
        figsize = (5, 5)
        lw = 1
        all_images = []
        for yy in y:
            plt.figure(figsize=figsize, dpi=120)
            plt.plot(self.xx, self.minmax(yy), lw=lw, color="k")
            plt.tight_layout()
            plt.axis("off")
            plt.xlim(4000, 400)
            plt.ylim(-0.01, 1)

            im_stream = BytesIO()
            plt.savefig(im_stream, bbox_inches="tight")
            im = Image.open(im_stream).convert("L")
            all_images.append(im)
            plt.clf()
            plt.close("all")

        return all_images

    def preprocess(self, spec: List[NDArray[np.float32]]):
        x = super().preprocess(spec)
        if self.model_name == "cnn":
            x = np.expand_dims(x, 1)
        elif self.model_name == "cnn2d":
            imgs = self.plot_fig(x)
            # (B, 1, H, W)
            imgs = [self.crop_bbox(im) for im in imgs]
            imgs = [im.resize((480, 480), Image.Resampling.LANCZOS) for im in imgs]
            x = np.asarray(
                [
                    np.expand_dims(
                        np.asarray(im, dtype=np.float32),  # (H, W)
                        0,
                    )
                    for im in imgs
                ],
                dtype=np.float32,
            )
        elif self.model_name in ["dt", "rf"]:
            ...
        else:
            raise ValueError()
        return x

    def __call__(
        self,
        spec: List[NDArray[np.float32]],
        topk: int = 3,
        return_score: bool = False,
    ):
        x = self.preprocess(spec)
        input_name = self.model.get_inputs()[0].name
        inputs = {input_name: x}
        if self.model_name in ("cnn", "cnn2d"):
            out: NDArray[np.float32] = self.model.run(None, inputs)[0]
            out = self.softmax(out, axis=1)
        else:
            out: NDArray[np.float32] = self.model.run(None, inputs)[1]
        if return_score:
            return out
        out_topk = out.argsort(axis=1)[:, ::-1][:, :topk]
        out1 = self.label2name(out_topk.tolist())
        return out1


def main(args: Namespace):
    inferer = MpInferenceOnnx(
        model_path=args.model_path,
        model_name=args.model_name,
        label_name=args.label_name,
        device=args.device,
    )
    if args.eval_ds:
        eval_ds(inferer, args)
    else:
        if args.input.endswith(".ftr"):
            df = pd.read_feather(args.input)
        elif args.input.endswith(".csv"):
            df = pd.read_csv(args.input)
        else:
            raise NotImplementedError(
                f"spectrum format {args.input.split('.')[-1]} not implemented!"
            )
        x = df.values
        out = inferer([x], topk=args.topk)
        print([["+".join(c) for c in row] for row in out])  # type:ignore


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", dest="input", type=str)
    parser.add_argument(
        "-m",
        "--model-path",
        dest="model_path",
        required=True,
        type=str,
    )
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--model-name", dest="model_name", type=str, default="cnn")
    parser.add_argument("--eval-ds", dest="eval_ds", action="store_true")
    parser.add_argument("--dataset-path", dest="ds_path", type=str)
    parser.add_argument("--saveto", dest="saveto", type=str)
    parser.add_argument("--mpid-label", dest="mpid_label", type=str, default="mpid")
    parser.add_argument(
        "--label-name",
        dest="label_name",
        type=str,
        default="data/label_name.json",
    )
    parser.add_argument("-k", dest="topk", default=3, type=int)
    parser.add_argument("--num_class", dest="num_class", default=120, type=int)
    parser.add_argument("--batch-size", dest="batch_size", default=20, type=int)
    args = parser.parse_args()
    assert any((args.input, args.ds_path)), f"must provide one of [input, ds_path]"
    seed_everything(42)
    main(args)
