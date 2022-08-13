"""
Description: evaluation on MPe
Author: Rainyl
License: Apache License 2.0
"""
from pathlib import Path
from typing import Dict, List, Union, Callable
from collections import OrderedDict
import glob
import warnings
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from src.infer_skl import MpInferenceSKL
from src.infer_nn import MpInferenceNNONNX


def wcv_score_ovn(
    pred: List[List[Union[str, int]]],
    real: Union[str, int],
    kk: float = 0.2,
    kn: float = 0.2,
    ks: float = 0.9,
):
    """
    get the weighted count value score of the result using linear function.
    0 - real in pred's 1st prediction, score: 0.8
    1 - real in pred's 2nd prediction, score: 0.6
    2 - real in pred's 3rd prediction, score: 0.4
    3 - real in pred's 4th prediction, score: 0.2
    ni - real in pred's n_th prediction, score: 0.1
    ki - real == i_th prediction, score: 1.0, 0.8, 0.6 ... 0.1
    si - prediction is `similar` to real: score: 0.9 * i_th_score
    `similar` means HDPE, LDPE, LLDPE each other.
    if all above not match, score is 0.0.
    if more than one matched, score is the max.

    :params:
        pred: [[0,], [0, 1], [1, 2]]
        real: 0
        or
        pred: [["HDPE", ], ["HDPE", "LDPE"]]
        real: "HDPE"
        kk: float, (0, 1), default: 0.2
        kn: float, (0, 1), default: 0.2
        ks: float, (0, 1), default: 0.9
    """
    # 0.1 <= ki <= 1
    score_fn_ki: Callable[[int], float] = lambda x: max(-kk * x + 1, 0.1)
    # 0.1 <= ki <= 0.9
    score_fn_ni: Callable[[int], float] = lambda x: max(-kn * x + 0.9, 0.1)
    # 0.81 <= si <= 0.09
    score_fn_si: Callable[[int], float] = lambda x: ks * score_fn_ni(x)
    score_fn_ski: Callable[[int], float] = lambda x: ks * score_fn_ki(x)
    similar = {
        0: 1,
        1: 0,
        "HDPE": "LDPE",
        "LDPE": "HDPE",
    }
    if real not in similar:
        similar[real] = real
    flags: List[str] = []
    scores: List[float] = [0.0]
    for i, p in enumerate(pred):
        if [real] == p:  # k_i
            flags.append(f"k_{i}")
        if similar[real] in p:  # S
            if [similar[real]] == p:
                flags.append(f"sk_{i}")  # sk_i
            else:
                flags.append(f"s_{i}")  # s_i
        if real in p:  # n_i
            flags.append(f"n_{i}")
    # print(f"flags: ", flags)
    if len(flags) == 0:
        return 0
    for flag in flags:
        i = int(flag.split("_")[1])
        if flag.startswith("k_"):
            scores.append(score_fn_ki(i))
        elif flag.startswith("n_"):
            scores.append(score_fn_ni(i))
        elif flag.startswith("s_"):
            scores.append(score_fn_si(i))
        elif flag.startswith("sk_"):
            scores.append(score_fn_ski(i))
        else:
            warnings.warn(f"flag {flag} not supported")
    # print(f"scores", scores)
    return max(scores)


# TODO: nvn
# def wcv_score_nvn():
#     ...


def WCA3(
    preds: Dict[str, List[List[Union[str, int]]]],
    reals: Dict[str, Union[str, int]],
):
    """
    reals: {
        "1": "HDPE",
        ...
    }
    preds: {
        "1": [["HDPE"], ["HDPE", "LDPE"], ["HDPE", "PVC"]],
        ...
    }
    """
    assert len(preds) == len(reals)
    scores = OrderedDict()
    keys = sorted(list(preds.keys()))
    kk, kn, ks = 0.2, 0.2, 0.9
    for k in keys:
        score = wcv_score_ovn(pred=preds[k], real=reals[k], kk=kk, kn=kn, ks=ks)
        scores[k] = score
    wca = sum(list(scores.values())) / len(scores.values())
    msg = f"#### environment samples evaluation report ####"
    msg += f"\n\treal: {len(reals)} samples"
    msg += f"\n\tprediction: {len(preds)} samples"
    msg += f"\n\tWCA-3: {wca:.4f}"
    msg += "\n#############################################"
    print(msg)
    return scores


def eval_samples(
    samples: List[str],
    inferer: Union[MpInferenceSKL, MpInferenceNNONNX],
    topk: int = 3,
    out_eval: str = "eval_sample_result.csv",
):
    sample_names = np.array([Path(s).name.split(".")[0] for s in samples])
    all_samples: List[Image.Image] = []
    for sample in samples:
        if sample.endswith(".csv"):
            im = inferer.read_plot_csv(sample, rmco2=True, fac=0.2)
        elif sample.endswith(".jpg"):
            im = Image.open(sample).convert("L")
        else:
            warnings.warn(f"file format not supported, {sample}")
            continue
        all_samples.append(im)
    out = inferer(all_samples, topk=topk)
    # save result
    out1 = np.array([["+".join(c) for c in r] for r in out], dtype=str)
    header = ",".join(["sample", *[f"pred{i}" for i in range(topk)]])
    idx = np.argsort([int(i) for i in sample_names])
    out1 = np.concatenate([sample_names[idx][np.newaxis, ...], out1[idx].T], axis=0).T
    np.savetxt(
        out_eval,
        out1,
        fmt="%s",
        comments="",
        delimiter=",",
        header=header,
    )
    # get WCA score
    real_csv = "data/identity.csv"
    real_table = np.loadtxt(
        real_csv,
        dtype=str,
        comments=None,
        delimiter=",",
        skiprows=1,
        encoding="utf-8",
    )
    reals: Dict[str, Union[str, int]] = {}
    for row in real_table:
        reals[row[0]] = row[1]
    preds: Dict[str, List[List[Union[str, int]]]] = {}
    for s, row in zip(sample_names, out):
        preds[s] = row
    scores = WCA3(preds=preds, reals=reals)
    return scores


def scores2csv(scores: Dict, saveto: str = "scores.csv"):
    arr = np.array([[k, v] for k, v in scores.items()], dtype=np.float32)
    idx = np.argsort(arr.T[0])
    arr = arr[idx]
    np.savetxt(
        saveto,
        arr,
        fmt="%s",
        delimiter=",",
        comments="",
        header="sample,WCVi",
    )


def main(
    img_dir: str,
    model_path: str,
    model_name: str,
    format="jpg",
    topk: int = 3,
    save_name: str = "eval_sample",
):
    if model_name == "onnx":
        inferer = MpInferenceNNONNX(
            model_path=model_path,
            label_name="data/label_name.json",
        )
    elif model_name == "skl":
        inferer = MpInferenceSKL(
            model_path=model_path,
            label_name="data/label_name.json",
        )
    else:
        raise NotImplementedError(f"model {model_name} not implemented")
    if img_dir.endswith("/"):
        img_dir = img_dir[:-1]
    samples = list(glob.glob(f"{img_dir}/*.{format}"))
    scores = eval_samples(
        samples=samples,
        inferer=inferer,
        topk=topk,
        out_eval=save_name + "_preds.csv",
    )
    scores2csv(scores, saveto=save_name + "_scores.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", dest="source", required=True, type=str)
    parser.add_argument("-p", dest="model_path", required=True, type=str)
    parser.add_argument("-m", dest="model", default="onnx", type=str)
    parser.add_argument("-f", dest="format", default="jpg", type=str)
    parser.add_argument("-k", dest="topk", default=3, type=int)
    parser.add_argument("-n", dest="name", default="eval_sample", type=str)
    args = parser.parse_args()
    # args = parser.parse_args(["-s", "", "-p", "", "-f", "jpg", "-k", "3", "-n", "eval_environ_cnn"])
    # args = parser.parse_args(["-s", "", "-p", "", "-f", "csv", "-k", "3", "-n", "eval_environ_cnn"])
    main(args.source, args.model_path, args.model, args.format, args.topk, args.name)
