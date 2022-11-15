"""
Description: train models from sklearn
Author: Rainyl
License: Apache License 2.0
"""
import inspect
import json
import logging
import os
import pickle
import random
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from joblib import parallel_backend
from numpy.typing import NDArray
from sklearn import linear_model, metrics
from sklearn import model_selection as ms
from sklearn import preprocessing as pp
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from .dataset import MpDatasetSK

__supported_models__ = ("lsvm", "rf", "dt", "svm", "logistic")

SEEDS = [0, 21, 42, 63, 84, 100, 221, 442, 663, 884]

logger = logging.getLogger("train_sk")
shandler = logging.StreamHandler()
shandler.setFormatter(
    logging.Formatter(
        "%(asctime)s:%(name)s:%(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)
logger.addHandler(shandler)
logger.setLevel(logging.INFO)


def seed_everything(seed: int):
    # from kaggle, https://www.kaggle.com/code/rhythmcam/random-seed-everything
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(ds: str, shuffle=True):
    dataset = MpDatasetSK(ds, shuffle=shuffle)
    X = []
    Y = []
    logger.info(f"Loading dataset [{ds}]...")
    for spec, label in tqdm(dataset):  # type: ignore
        X.append(spec)
        Y.append(label)
    # return sparse.csr_matrix(X), Y
    return X, Y


def load_clf_pkl(pkl: str):
    logger.info(f"Loading pkl [{pkl}] ...")
    if not os.path.exists(pkl):
        raise ValueError(f"pkl {pkl} not exists")
    with open(pkl, "rb") as f:
        clf = pickle.load(f)
    return clf


def lsvm(path: Union[str, None] = None):
    if path is None:
        clf = svm.LinearSVC(
            C=1,
            dual=False,
            loss="squared_hinge",
            multi_class="ovr",
            class_weight="balanced",
            max_iter=3000,
        )
        # clf = linear_model.SGDClassifier(
        #     loss="hinge",
        #     max_iter=1000,
        #     tol=1e-6,
        #     # class_weight="balanced",
        # )
    else:
        clf = load_clf_pkl(path)
    return clf


def logistic(path: Union[str, None] = None):
    if path is None:
        clf = linear_model.LogisticRegression(
            penalty="l2",
            # dual=True,
            C=1,
            # solver="lbfgs",
            solver="liblinear",
            n_jobs=-1,
            max_iter=1000,
            class_weight="balanced",
            multi_class="ovr",
        )
    else:
        clf = load_clf_pkl(path)
    return clf


def svm_(path: Union[str, None] = None):
    if path is None:
        clf = svm.SVC(
            C=1,
            kernel="rbf",
            gamma="auto",
            probability=True,
            cache_size=500,
            decision_function_shape="ovr",
            class_weight="balanced",
        )
    else:
        clf = load_clf_pkl(path)
    return clf


def rf(path: Union[str, None] = None):
    if path is None:
        clf = RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            bootstrap=True,
            class_weight="balanced",
            min_samples_leaf=10,
            n_jobs=-1,
        )
    else:
        clf = load_clf_pkl(path)
    return clf


def dt(path: Union[str, None] = None):
    if path is None:
        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            class_weight="balanced",
            min_samples_leaf=10,
        )
    else:
        clf = load_clf_pkl(path)
    return clf


def get_clf(model_name: str, path: Union[str, None] = None):
    if model_name == "lsvm":
        clf = lsvm(path)
    elif model_name == "rf":
        clf = rf(path)
    elif model_name == "dt":
        clf = dt(path)
    elif model_name == "svm":
        clf = svm_(path)
    elif model_name == "logistic":
        clf = logistic(path)
    else:
        raise ValueError()
    return clf


def get_clf_name(
    clf: Union[
        svm.LinearSVC,
        RandomForestClassifier,
        tree.DecisionTreeClassifier,
        linear_model.LogisticRegression,
        linear_model.SGDClassifier,
    ],
):
    if isinstance(clf, (svm.LinearSVC, linear_model.SGDClassifier)):
        model_name = "lsvm"
    elif isinstance(clf, RandomForestClassifier):
        model_name = "rf"
    elif isinstance(clf, tree.DecisionTreeClassifier):
        model_name = "dt"
    elif isinstance(clf, svm.SVC):
        model_name = "svm"
    elif isinstance(clf, linear_model.LogisticRegression):
        model_name = "logistic"
    else:
        raise ValueError()
    return model_name


def val_one_sk(
    clf: Union[
        svm.LinearSVC,
        RandomForestClassifier,
        tree.DecisionTreeClassifier,
        linear_model.LogisticRegression,
        linear_model.SGDClassifier,
    ],
    save_to: Path,
    pid: int,
    args: Namespace,
    xy: Union[None, Tuple[NDArray[np.float32], NDArray[np.float32]]] = None,
):
    save_to_file = save_to / f"{pid}" / "report.json"
    if xy is None:
        if args.predict_only:
            X_true, Y_true = load_dataset(args.predict_set, shuffle=False)
            save_to_file = save_to / f"{pid}" / "report_predict.json"
        else:
            X_true, Y_true = load_dataset(args.val_set, shuffle=False)
    else:
        X_true, Y_true = xy
    # predict
    y_pred = clf.predict(X_true)
    y_score = None
    if not args.mccv:
        func_names = [f[0] for f in inspect.getmembers(clf, inspect.isroutine)]
        if "predict_proba" in func_names:
            y_score = clf.predict_proba(X_true)  # type: ignore
        else:
            y_score = clf.decision_function(X_true)  # type: ignore
    # append
    y_true_list = [Y_true]
    y_pred_list = [y_pred]
    y_score_list = None if args.mccv else [y_score]
    # save
    saveMetrics(
        save_to_file=save_to_file,
        y_true=y_true_list,  # type: ignore
        y_pred=y_pred_list,  # type: ignore
        y_score=y_score_list,  # type: ignore
    )
    logger.info(f"validation of [{pid}] saved! ")
    return [Y_true, y_pred, y_score]


def train_one_sk(
    clf: Union[
        svm.LinearSVC,
        RandomForestClassifier,
        tree.DecisionTreeClassifier,
        linear_model.LogisticRegression,
        linear_model.SGDClassifier,
    ],
    pid: int,
    save_to: Path,
    args: Namespace,
):
    if not args.mccv:
        assert pid < len(SEEDS)
        seed_everything(SEEDS[pid])
        logger.info(f"set seed to {SEEDS[pid]} for pid [{pid}]")
    model_name = get_clf_name(clf)
    logger.info(f"model: [{model_name}], proccess id: [{pid}]")
    if args.predict_only:
        val_one_sk(clf, save_to, pid, args, None)
        return clf
    elif args.mccv:
        # print(f"lens: {len(DS_X)}, {len(DS_Y)}")
        x_train, x_test, y_train, y_test = ms.train_test_split(
            DS_X,
            DS_Y,
            test_size=args.test_size,
            shuffle=args.shuffle,
        )
    else:
        x_train, y_train = load_dataset(args.train_set, args.shuffle)
    if args.scaler:
        if args.scaler == "minmax":
            scaler = pp.MinMaxScaler(feature_range=(0, 1))
        elif args.scaler == "standard":
            scaler = pp.StandardScaler()
        else:
            raise ValueError(f"scaler {args.scaler} not supported")
        x_train = scaler.fit_transform(x_train)
    # train
    clf.fit(x_train, y_train)
    if args.mccv:
        val_one_sk(clf, save_to, pid, args, xy=(x_test, y_test))  # type: ignore
    else:
        val_one_sk(clf, save_to, pid, args, None)
    if args.save_pkl:
        pklPath = save_to / f"{pid}" / f"clf_{model_name}.pkl"
        if not pklPath.parent.exists():
            pklPath.parent.mkdir(parents=True)
        with open(pklPath, "wb") as f:
            pickle.dump(clf, f)
            logger.info("Saved model to pickle...")
        return pklPath
    return clf


def saveMetrics(
    save_to_file: Path,
    y_true: List[List],
    y_pred: List[NDArray[np.float32]],
    y_score: Union[None, List[NDArray[np.float32]]],
):
    # print(metrics.classification_report(y_true, y_pred))
    # metric to save: macro-accuracy, precision, recall, f1
    # merics.csv
    # accuracy, precision, recall, f1
    # 0.8,      0.9,       0.8,    0.8
    # metric_data = []
    # columns = ["accuracy", "precision", "recall", "f1"]
    for i in range(len(y_true)):
        report: Dict[str, Any] = metrics.classification_report(
            y_true=y_true[i],
            y_pred=y_pred[i],
            output_dict=True,
        )  # type: ignore
        report["trues"] = y_true[i]
        report["preds"] = y_pred[i].tolist()
        if y_score is not None:  # close it for mccv to save space
            report["scores"] = y_score[i].tolist()
        if not save_to_file.parent.exists():
            save_to_file.parent.mkdir(parents=True)
        with open(save_to_file, "w") as f:
            json.dump(report, f, indent=4)


def getOneROC(y_true: Iterable, y_score: NDArray[np.float32], classes=np.arange(120)):
    y_truebin = pp.label_binarize(y_true, classes=classes)
    y_score = np.array(y_score)
    macro_fpr = np.linspace(0, 1, 3000)
    macro_tpr = np.zeros_like(macro_fpr)
    for label in classes:
        fpr, tpr, _ = metrics.roc_curve(y_truebin[:, label], y_score[:, label])
        # roc_auc = metrics.auc(fpr, tpr)
        macro_tpr += np.interp(macro_fpr, fpr, tpr)
    macro_tpr /= len(classes)
    auc = metrics.auc(macro_fpr, macro_tpr)
    return {"fpr": macro_fpr, "tpr": macro_tpr, "auc": auc}


def main(args: Namespace):
    assert args.stop > args.start, "stop must > start"
    save_to = Path(args.save_dir) / args.model
    res = []
    if args.multiprocess:
        pool_ = Pool(args.cores)
        logger.info("Running with multiprocess")
        for i in range(args.start, args.stop):
            save_to_i = save_to / str(i)
            if args.predict_only:
                clf = get_clf(args.model, save_to_i / f"clf_{args.model}.pkl")
                spath = save_to_i / "report_predict.json"
            else:
                clf = get_clf(args.model)
                spath = save_to_i / "report.json"
            # skip if exist, spath only for detect wether exists
            if spath.exists():
                logger.info(f"Skip [{i}]")
                continue
            r = pool_.apply_async(train_one_sk, (clf, i, save_to, args))
            res.append(r)
        pool_.close()
        pool_.join()
        res = [r.get() for r in res]
    else:
        # only for test
        for i in range(len(SEEDS)):
            if args.predict_only:
                clf = get_clf(args.model, save_to / f"{i}" / f"clf_{args.model}.pkl")
            else:
                clf = get_clf(args.model)
            r = train_one_sk(
                clf,  # type: ignore
                i,
                save_to,
                args,
            )
            # res.append(r)

    logger.info("Finished!")


if __name__ == "__main__":
    cores_num = os.cpu_count()
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        default="dt",
        type=str,
        help="model",
    )
    parser.add_argument(
        "--predict-only",
        dest="predict_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--start",
        dest="start",
        default=0,
        type=int,
        help="start sample",
    )
    parser.add_argument(
        "--stop",
        dest="stop",
        default=10,
        type=int,
        help="stop sample",
    )
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default="repeatSave/",
    )
    parser.add_argument(
        "-c",
        "--cores",
        default=cores_num,
        type=int,
        help="cores of multi-processing",
    )
    parser.add_argument(
        "--njobs",
        dest="njobs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--test-size",
        dest="test_size",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        default=False,
        help="multi process enabled",
    )
    parser.add_argument(
        "--mccv",
        action="store_true",
        default=False,
        help="mccv",
    )
    parser.add_argument("--train-set", dest="train_set", type=str, required=True)
    parser.add_argument("--val-set", dest="val_set", type=str, required=True)
    parser.add_argument("--predict-set", dest="predict_set", type=str, required=False)
    parser.add_argument("--mccv-set", dest="mccv_set", type=str)
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--no-save-pkl",
        dest="save_pkl",
        action="store_false",
        default=True,
    )

    parser.add_argument("--scaler", dest="scaler", type=str, default="")

    args = parser.parse_args(
        [
            "--cores",
            "5",
            "--njobs",
            "1",
            "--scaler",
            "",
            # "minmax",
            # "standard",
            "--model",
            "dt",
            # "rf",
            # "lsvm",
            # "svm",
            # "logistic",
            "--train-set",
            "/shm/dataset/opt/dataset/ds_1k_3600_cb_nrpb/train",
            "--val-set",
            "/shm/dataset/opt/dataset/ds_1k_3600_cb_nrpb/val_test",
            "--start",
            "0",
            "--stop",
            "10",
            "--save-dir",
            "repeatSave/1000",
            "--multiprocess",
            ######################### PREDICT ONLY #############
            # "--predict-only",
            # "--predict-set",
            # "/mnt/f/opt/dataset/mpe/full",
            # "--stop",
            # "10",
            # "--save-dir",
            # "repeatSave/",
            # "--cores",
            # "1",
            # "--njobs",
            # "1",
            ################# PREDICT ONLY END #################
            ######################### MCCV ###################
            # "--mccv-set",
            # "/mnt/f/opt/dataset/dataset_300_3600/full",
            # "--save-dir",
            # "mccvSave/",
            # "--mccv",
            # "--stop",
            # "100",
            ######################### MCCV END ###################
        ]
    )
    assert args.model in __supported_models__, f"model {args.model} not supported"

    # load data globally, need enough memory
    logger.warning(
        (
            f"Running model: [{args.model}], cores: [{args.cores}], MCCV: [{args.mccv}] "
            f"range [{args.start} -> {args.stop}] "
        )
    )
    if args.mccv:
        DS_X, DS_Y = load_dataset(args.mccv_set, args.shuffle)

    main(args)
