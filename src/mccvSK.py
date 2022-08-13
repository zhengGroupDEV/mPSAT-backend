'''
Author: rainyl
Description: MCCV for models from Sklearn
License: Apache License 2.0
'''
from typing import Iterable, List, Tuple, Union, Callable

import logging
import os
import time
from multiprocessing import Pool
# import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy import sparse
from sklearn import preprocessing as pp
from sklearn import model_selection as ms
from sklearn import svm, tree, metrics
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


from dataset import MpDatasetSVM

__supported_models__ = ["lsvm", "rf", "dt"]

# MAX_MCCV = 2000
MCCV_SAVE_PATH = Path("mccvSave")
if not MCCV_SAVE_PATH.exists():
    MCCV_SAVE_PATH.mkdir()

logger = logging.getLogger("MCCV")
shandler = logging.StreamHandler()
shandler.setFormatter(logging.Formatter(
    "%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(shandler)
logger.setLevel(logging.INFO)


def split_dataset() -> Tuple[sparse.csr_matrix, sparse.csr_matrix, List, List]:
    if not PRELOAD:
        DS_PATH = "data/dataset/datasetSplit430/mccv"
        dataset = MpDatasetSVM(DS_PATH, shuffle=True)
        XX = []
        YY = []
        for img, label in tqdm(dataset):
            XX.append(img)
            YY.append(label)
        x_train, x_test, y_train, y_test = ms.train_test_split(sparse.csr_matrix(XX), YY, test_size=TEST_SIZE)
    else:
        x_train, x_test, y_train, y_test = ms.train_test_split(DS_X, DS_Y, test_size=TEST_SIZE)
    logger.debug(f"RETURN")
    return x_train, x_test, y_train, y_test


def lsvm():
    clf = svm.LinearSVC(
        C=1,
        dual=True,
        loss="squared_hinge",
        multi_class="ovr",
        class_weight="balanced",
    )
    return clf


def rf():
    clf = RandomForestClassifier(
        n_estimators=120,
        criterion="gini",
        bootstrap=True,
        class_weight="balanced",
        min_samples_leaf=10,
        n_jobs=-1,
    )
    return clf


def dt():
    clf = tree.DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        class_weight="balanced",
        min_samples_leaf=10,
    )
    return clf


def runOneSK(clf: Union[svm.LinearSVC, RandomForestClassifier, tree.DecisionTreeClassifier], pid: int):
    model_name = get_clf_name(clf)
    logger.info(f"model: [{model_name}], proccess id: [{pid}]")

    save_dir = MCCV_SAVE_PATH / model_name / str(pid)
    y_true_list: List[List] = []
    y_pred_list: List[np.ndarray] = []
    y_score_list: List[np.ndarray] = []
    try:
        time_init = time.time()
        X_train, X_test, Y_train, Y_test = split_dataset()
        time_split = time.time()
        logger.critical(f"split dataset takes [{time_split-time_init}]s")
        # train
        clf.fit(X_train, Y_train)
        time_fit = time.time()
        logger.critical(f"fit takes [{time_fit-time_split}]s")
        # predict
        y_pred = clf.predict(X_test)
        time_predict = time.time()
        logger.critical(f"predict takes [{time_predict-time_fit}]s")
        if model_name == "lsvm":
            y_score = clf.decision_function(X_test)
        else:
            y_score = clf.predict_proba(X_test)
        # append
        y_true_list.append(Y_test)
        y_pred_list.append(y_pred)
        y_score_list.append(y_score)
        # save
        saveMetrics(saveDir=save_dir, y_true=y_true_list, y_pred=y_pred_list, y_score=y_score_list)
        logger.info(f"[{pid}/{MAX_MCCV}] Saved! ")
        return [Y_test, y_pred, y_score]
    except Exception as e:
        logger.error(f"run one sk failed, [{e}]")


def runSK(clf: Union[svm.LinearSVC, RandomForestClassifier, tree.DecisionTreeClassifier]):
    model_name = get_clf_name(clf)

    save_dir = MCCV_SAVE_PATH / model_name
    save_step = 2
    y_true_list: List[List] = [[]] * save_step
    y_pred_list: List[np.ndarray] = [np.array([])] * save_step
    y_score_list: List[np.ndarray] = [np.array([])] * save_step
    # !!!!!!!!!!!!!!!!!!!!NOTE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    # In order to decrese memory usage, y_true_list ... will be overwrite
    # in the next iteration. so, 
    # (MAX_MCCV % save_step) should be 0, 
    # otherwise, y_true_list, y_pred_list and y_score_list
    # will have the former result in the last iteration
    for i in range(MAX_MCCV):
        # load and split dataset
        X_train, X_test, Y_train, Y_test = split_dataset()
        # train
        clf.fit(X_train, Y_train)
        # predict
        y_pred = clf.predict(X_test)
        if model_name == "lsvm":
            y_score = clf.decision_function(X_test)
        else:
            y_score = clf.predict_proba(X_test)
        # append
        y_true_list[i % save_step] = Y_test
        y_pred_list[i % save_step] = y_pred
        y_score_list[i % save_step] = y_score
        # save
        if i != 0 and i % (save_step - 1) == 0:
            saveMetrics(saveDir=save_dir, y_true=y_true_list, y_pred=y_pred_list, y_score=y_score_list)
            logger.info(f"[{i}/{MAX_MCCV}] Saved! ")


def saveMetrics(
    saveDir: Path, y_true: List[List], y_pred: List[np.ndarray],
    y_score: List[np.ndarray], ):
    if not saveDir.exists():
        saveDir.mkdir(parents=True)
    columns = ["accuracy", "precision" ,"recall", "f1"]
    # print(metrics.classification_report(y_true, y_pred))
    # metric to save: macro-accuracy, precision, recall, f1
    # merics.csv
    # accuracy, precision, recall, f1
    # 0.8,      0.9,       0.8,    0.8
    metric_data = []
    for i in range(len(y_true)):
        report = metrics.classification_report(y_true=y_true[i], y_pred=y_pred[i], output_dict=True)
        metric_data.append(
            [
                report["accuracy"], report["macro avg"]["precision"],
                report["macro avg"]["recall"], report["macro avg"]["f1-score"],
            ]
        )

    metric_path = saveDir / f"metrics.csv"
    if metric_path.exists():
        metric_data_old = pd.read_csv(metric_path).values
        metric_data = np.concatenate((metric_data_old, metric_data))
    df = pd.DataFrame(metric_data, columns=columns, )
    df.to_csv(metric_path, index=False)
    logger.info(f"Saved metrics to {metric_path}")

    # save data for ROC
    # roc_auc.csv
    # fpr1-3000, tpr1-3000, auc
    columns = [f"fpr{i}" for i in range(3000)]
    columns.extend([f"tpr{i}" for i in range(3000)])
    columns.append("auc")
    roc_data = []
    for i in range(len(y_pred)):
        roc_one = getOneROC(y_true[i], y_score[i])
        row = np.hstack((roc_one["fpr"], roc_one["tpr"], roc_one["auc"]))
        roc_data.append(row)

    roc_path = saveDir / "roc_auc.csv"
    if roc_path.exists():
        roc_data_old = pd.read_csv(roc_path)
        roc_data = np.concatenate((roc_data_old, roc_data))
    df = pd.DataFrame(data=roc_data, columns=columns)
    df.to_csv(roc_path, index=False)
    logger.info(f"Saved roc data to {roc_path}")


def getOneROC(y_true: Iterable, y_score: np.ndarray, classes=np.arange(120)):
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


def get_clf(model_name: str):
    if model_name == "lsvm":
        clf = lsvm()
    elif model_name == "rf":
        clf = rf()
    elif model_name == "dt":
        clf = dt()
    return clf


def get_clf_name(clf: Union[svm.LinearSVC, RandomForestClassifier, tree.DecisionTreeClassifier]):
    if(isinstance(clf, svm.LinearSVC)):
        model_name = "lsvm"
    elif isinstance(clf, RandomForestClassifier):
        model_name = "rf"
    elif isinstance(clf, tree.DecisionTreeClassifier):
        model_name = "dt"
    return model_name


def main(model: str, pool_, start: int = 0, stop: int = 2000):
    assert stop > start, "stop must > start"
    clf = get_clf(model)
    if pool_ is not None:
        logger.info("Running with multiprocess")
        # res = []
        for i in range(start, stop):
            # skip if exist
            spath = MCCV_SAVE_PATH / model / str(i) / "metrics.csv"
            if spath.exists():
                logger.info(f"Skip [{i}]")
                continue
            r = pool_.apply_async(runOneSK, (clf, i))
    else:
        # run model train and val
        runSK(clf)

    logger.info("Finished!")


if __name__ == "__main__":
    cores_num = os.cpu_count()
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default="lsvm", type=str, help="model")
    parser.add_argument("--preload", action="store_true", default=False, help="preload dataset")
    parser.add_argument("-s", "--maxsample", dest="maxsample", default=2000, type=int, help="max sampling times for mccv")
    parser.add_argument("--start", dest="start", default=0, type=int, help="start sample")
    parser.add_argument("--stop", dest="stop", default=2000, type=int, help="stop sample")
    
    parser.add_argument("-c", "--cores", default=cores_num, type=int, help="cores of multi-processing")
    parser.add_argument("--multiprocess", action="store_true", default=False, help="multi process enabled")

    args = parser.parse_args()
    assert args.model in __supported_models__, f"model {args.model} not supported"

    MAX_MCCV: int = args.maxsample
    MULTI_PROCESS: bool = args.multiprocess
    TEST_SIZE: float = 0.22
    CORES_NUM: int = args.cores

    # load data globally, need enough memory
    PRELOAD = args.preload
    logger.warning(
        (f"Running model: [{args.model}], preload: [{PRELOAD}], cores: [{CORES_NUM}], "
         f"MCCV: [{args.start}-{args.stop}/{MAX_MCCV}], MULTI_PROCESS: [{MULTI_PROCESS}]")
    )
    if PRELOAD:
        # ratio=10320/(36120+10320)=0.22
        # to make mccv has similar result as previous train
        DS_PATH = "data/dataset/datasetSplit430/mccv"

        dataset = MpDatasetSVM(DS_PATH, shuffle=True)
        DS_X: List = []
        DS_Y: List = []
        for img, label in tqdm(dataset):
            DS_X.append(img)
            DS_Y.append(label)
        DS_X = sparse.csr_matrix(DS_X)
    if MULTI_PROCESS:
        # logger.info(f"Multi_processing: Cores: [{CORES_NUM}]")
        POOL_ = Pool(CORES_NUM) # type: ignore
        # run here
        with parallel_backend("loky", n_jobs=10):
            main(args.model, pool_=POOL_, start=args.start, stop=args.stop)
        POOL_.close()
        POOL_.join()
    else:
        main(args.model, pool_=None)
