"""
Description: generate dataset, with several data augmentation
Author: Rainyl
License: Apache License 2.0
"""
import os
import re
import json
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")


MPID_HASH_INV = {
    'HDPE': 0,
    'LDPE': 1,
    "LLDPE": 2,
    "PET": 3,
    "ABS": 4,
    "PP": 5,
    "PS": 6,
    "PVC": 7,
    "CA": 8,
    "PMMA": 9,
    "PA": 10,
    "PA66": 10,
    "PC": 11,
    "PLA": 12,
    "PBT": 13,
    "CPE": 14,
    "EVA": 15,
    "PU": 16,
    "TPU": 16,
    "PTFE": 17,
    "POM": 18,
    "PVDF": 19,
    "PCL": 20,
}


def interpolate(spec: np.ndarray) -> np.ndarray:
    """
    spec: (N, 2), col-1: wavenum, col-2: intensity
    """
    xx = np.arange(400, 4000, 0.1)
    idx = np.argsort(spec.T[0])
    spec = spec[idx]
    yy = np.interp(xx, spec.T[0], spec.T[1], left=0, right=0)
    return np.vstack((xx, yy)).T


def minmax(x: np.ndarray):
    return (x - x.min()) / (x.max() - x.min())


# 1. uniform noise and gaussian noise
def add_noise(spec: np.ndarray, level: float = 0.1):
    rng = np.random.default_rng()
    m = rng.random()
    len_ = spec.shape[0]
    if m < 0.5:
        noise = rng.uniform(0, 1, size=len_) / 20
    else:
        noise = rng.normal(0.2, 0.02, size=len_)
    spec[:, 1] = spec[:, 1] * (1 - level) + level * noise
    return spec


# 2. bend baseline
def bend_baseline(spec: np.ndarray, level: float = 0.1, step=3140):
    rng = np.random.default_rng()
    num = rng.integers(1, 5)
    for _ in range(num):
        start_x = rng.integers(0, spec.shape[0] - step, 1)[0]
        end_x = start_x + step
        x = np.arange(start_x, end_x)
        bend = minmax(np.sin(x / 100) + 1) / 20  # [0, 1]
        spec[start_x:end_x, 1] = spec[start_x:end_x, 1] * (1 - level) + level * bend
    return spec


# 3. random peaks
def add_rand_peaks(spec: np.ndarray, step=314):
    rng = np.random.default_rng()
    level = rng.uniform(0.0, 0.6)
    num = rng.integers(1, 5)
    for _ in range(num):
        start_x = rng.integers(0, spec.shape[0] - step)
        end_x = start_x + step
        x = np.arange(start_x, end_x)
        bend = minmax(np.sin(x / 200) + 1)  # [0, 1]
        spec[start_x:end_x, 1] = spec[start_x:end_x, 1] * (1 - level) + level * bend
    return spec


# 4. reduce peak intensity
def reduce_peak(spec: np.ndarray):
    rng = np.random.default_rng()
    level = rng.uniform(0.5, 0.99)
    height = rng.uniform(0.1, 0.5)
    # peaks = find_peaks_cwt(spec.T[1], 100)
    # peaks, _ = find_peaks(spec.T[1], height=0.1, distance=1)
    peaks = spec.T[1] > height
    # plt.figure(dpi=400)
    # plt.plot(spec.T[0], spec.T[1], lw=1)
    # plt.scatter(spec.T[0][peaks], spec.T[1][peaks], s=2, marker=".", c="g")
    spec[peaks, 1] = spec[peaks, 1] * level
    # plt.plot(spec.T[0], spec.T[1], lw=1, c="r")
    # plt.plot(spec.T[0], np.ones_like(spec.T[0]) * height, "--", c="k")
    # plt.savefig("reduce_peaks.jpg")
    return spec


def augmentation(
    spec: np.ndarray,
    name: str = "original",
    format="jpg",
    save_dir: str = ".",
    more: int = 1,
):
    funcs_comb = (
        (add_noise,),
        (bend_baseline,),
        (add_rand_peaks,),
        (reduce_peak,),
        (add_noise, bend_baseline),
        (add_noise, add_rand_peaks),
        (add_noise, reduce_peak),
        (bend_baseline, add_rand_peaks),
        (bend_baseline, reduce_peak),
        (add_rand_peaks, reduce_peak),
        (add_noise, bend_baseline, add_rand_peaks),
        (add_noise, bend_baseline, reduce_peak),
        (add_noise, add_rand_peaks, reduce_peak),
        (bend_baseline, add_rand_peaks, reduce_peak),
        (add_noise, bend_baseline, add_rand_peaks, reduce_peak),
    )
    spec = interpolate(spec)
    spec[:, 1] = minmax(spec[:, 1])
    plot_spec(
        spec.T[0], spec.T[1], f"{save_dir}/{name}.0.{format}"
    )  # .0 means no transform
    n_funcs = len(funcs_comb)
    for i in range(more):
        for j, funcs in enumerate(funcs_comb):
            no_ = i * n_funcs + j + 1
            img_name = f"{save_dir}/{name}.{no_}.{format}"
            if os.path.exists(img_name):
                print(f"skip {img_name}")
                continue
            spec_ = spec.copy()
            for func in funcs:
                spec_ = func(spec_)  # type:ignore
            spec_[:, 1] = minmax(spec_[:, 1])
            plot_spec(spec_.T[0], spec_.T[1], img_name)
    return spec


def load_spectra(path: str):
    if not os.path.exists(path):
        raise ValueError(f"path {path} not exists!")
    specs = np.loadtxt(
        path, dtype=float, comments=None, delimiter=",", skiprows=1, encoding="utf-8"
    )
    return specs


def plot_spec(x, y, path: str, PX_H: int = 480, dpi: int = 120):
    fig = plt.figure(figsize=(PX_H / dpi, PX_H / dpi), dpi=dpi)
    xx = np.arange(400, 4000, 0.1)
    xx = np.round(xx, 2)
    yy = np.interp(xx, x, y, left=0, right=0)

    ax = fig.add_subplot(111)
    ax.plot(xx, yy, ls="-", lw=1, c="k")
    ax.set_xlim(4000, 400)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    fig.savefig(path)
    plt.clf()
    plt.close(fig)


def combine2label(plabel: Tuple[int, Path]):
    label, p = plabel
    d = pd.read_csv(str(p.absolute()))
    mpid = d["mpid"].unique().astype(int)
    da = d[d["mpid"] == mpid[0]]
    db = d[d["mpid"] == mpid[1]]
    label_path = img_path / f"{label}"
    if not label_path.exists():
        label_path.mkdir()
    count = 0
    nas = da["name"].unique().astype(int)
    np.random.shuffle(nas)
    for na in nas:
        dai = da[da["name"] == na]
        xa = dai["wavenum"]
        ya = dai["intensity"]
        yya = np.interp(xx, xa, ya, left=0, right=0)
        nbs = db["name"].unique().astype(int)
        np.random.shuffle(nbs)
        for nb in nbs:
            dbi = db[db["name"] == nb]
            xb = dbi["wavenum"]
            yb = dbi["intensity"]
            yyb = np.interp(xx, xb, yb, left=0, right=0)
            yyab = (yya + yyb) * 0.5
            spec_ = np.vstack((xx, yyab)).T
            augmentation(
                spec_,
                f"{na}.{nb}",
                format="jpg",
                save_dir=str(label_path.absolute()),
                more=1,
            )
            count += 16
            if count >= 7000:
                return label, mpid.tolist()
    return label, mpid.tolist()


def single_class(path: str, save_to: str):
    mpid2label = {
        "0": "0",
        "1": "1",
        "2": "-1",
        "3": "2",
        "4": "3",
        "5": "4",
        "6": "5",
        "7": "6",
        "8": "7",
        "9": "8",
        "10": "9",
        "11": "10",
        "12": "11",
        "13": "-1",
        "14": "12",
        "15": "13",
        "16": "14",
        "17": "-1",
        "18": "-1",
        "19": "-1",
        "20": "-1",
    }
    more_label = {
        "0": 1,
        "1": 2,
        "2": 1,
        "3": 2,
        "4": 1,
        "5": 1,
        "6": 2,
        "7": 1,
        "8": 2,
        "9": 2,
        "10": 2,
        "11": 2,
        "12": 3,
        "13": 2,
        "14": 3,
    }
    specs = load_spectra(path)
    sample_names = specs.T[0]
    names = np.unique(sample_names)
    for name in tqdm(names):
        spec = specs[sample_names == name, :]
        mpid = str(int(spec[0, -1]))
        label = mpid2label[mpid]
        if label == "-1":
            continue
        spec = spec[:, 1:3]
        save_dir = f"{save_to}/{label}"
        if not os.path.exists(save_dir):
            Path(save_dir).mkdir(parents=True)
        spec = augmentation(
            spec,
            str(int(name)),
            format="jpg",
            save_dir=save_dir,
            more=more_label[label],
        )


def increment21(path: str, save_path: str):
    import glob

    drop_name = ["PBT", "POM", "PPO", "LLDPE", "PVDF", "PTFE", "PCL"]
    # hash_name = {
    #     "HDPE": 0,
    #     "LDPE": 1,
    #     "PET": 2,
    #     "ABS": 3,
    #     "PP": 4,
    #     "PS": 5,
    #     "PVC": 6,
    #     "CA": 7,
    #     "PMMA": 8,
    #     "PA": 9,
    #     "PC": 10,
    #     "PLA": 11,
    #     "CPE": 12,
    #     "EVA": 13,
    #     "PU": 14,
    # }
    ptn = path + "/*.csv"
    csvs = glob.glob(ptn)
    names = [Path(c).name.split(".")[0] for c in csvs]
    for i, n in enumerate(names):
        if n in drop_name:
            del(csvs[i])
    csv_data = []
    for i, csv in enumerate(csvs):
        sample_name = 3172 + i
        name = Path(csv).name.split(".")[0]
        mpid = MPID_HASH_INV[name]
        d = np.loadtxt(csv, dtype=float, comments=None, delimiter=",", skiprows=1)
        ones = np.ones((d.shape[0], 1))
        d = np.hstack((ones*sample_name, d, ones*mpid))
        csv_data.append(d)
    csv_data_1 = np.vstack(csv_data)
    save_path = save_path + "/increment.csv"
    np.savetxt(save_path, csv_data_1, fmt="%s", delimiter=",", comments="", header="name,wavelength,intensity,mpid")
    return save_path


def increment_generate(csvs: str, save_to: str, mode="single"):
    """
    csvs: csv path
        format: sample_name,wavenum,intensity,mpid
    save_to: save directory path
    """
    # mode: single, multi
    if mode == "single":
        single_class(csvs, save_to=save_to)
    elif mode == "multi":
        ...
    else:
        raise NotImplementedError(f"mode {mode} not implemented")


if __name__ == "__main__":
    xx = np.arange(400, 4000, 0.1)
    ################################################
    # increment
    ################################################
    p = increment21("data/incre", "data")
    increment_generate(p, "data/incre_test", "single")

    ################################################
    # single class
    ################################################
    csv_path = "data/datasetFTIRAll.csv"
    single_class(csv_path, save_to="data/dataset/single")

    ################################################
    # multi class
    ################################################
    img_path = Path("data/dataset/multi")
    csv_path: Path = Path("data/csv2")  # for multi
    two_csv: list = [p for p in csv_path.glob("*.csv") if re.search(r"^\d+\.\d+\.csv$", p.name)]  # type: ignore
    MPIDS_ENABLED = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
    args = [(i + len(MPIDS_ENABLED), two_csv[i]) for i in range(len(two_csv))]
    # combine2label(args[0])
    pool = Pool(12)  # type: ignore

    res = pool.map(combine2label, args)

    pool.close()
    pool.join()

    label_mpid = {i: [v] for i, v in enumerate(MPIDS_ENABLED)}
    label_mpid1 = {r[0]: r[1] for r in res}
    label_mpid.update(label_mpid1)

    with open("data/label_mpid.json", "w", encoding="utf-8") as fout:
        json.dump(label_mpid, fout, indent=4)
        print("Saved label_mpid to label_mpid.json...")
