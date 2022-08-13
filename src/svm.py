'''
Author: rainyl
Description: LSVM
License: Apache License 2.0
'''
from pathlib import Path
from typing import Dict, List, Tuple, Union
import pickle
import json

import numpy as np
import sklearn.svm as svm
import sklearn.metrics as metrics
from scipy import sparse
from tqdm import tqdm

from .dataset import MpDatasetSVM


def load_dataset(ds: str = "train", shuffle=False):
    DS = {
        "train": "data/dataset/datasetSplit430/train",
        "test": "data/dataset/datasetSplit430/test",
    }
    if ds not in DS:
        raise ValueError("ds error")
    dataset = MpDatasetSVM(DS[ds], shuffle=shuffle)
    X = []
    Y = []
    print(f"Loading dataset [{ds}]...")
    for img, label in tqdm(dataset):
        X.append(img)
        Y.append(label)
    return sparse.csr_matrix(X), Y
    # return X, Y


def train(pklPath: Union[str, Path] = "clfLinearSVC.pkl"):
    X_train, Y_train = load_dataset(ds="train", shuffle=True)
    clf = svm.LinearSVC(
        C=1,
        dual=True,
        loss="squared_hinge",
        multi_class="ovr",
        class_weight="balanced",
    )
    print("Training....")
    clf.fit(X_train, Y_train)
    with open(pklPath, "wb") as f:
        pickle.dump(clf, f)
        print("Saved...")
    return clf


def val(clf: svm.LinearSVC, valds="val"):
    X_val, Y_val = load_dataset(ds=valds, shuffle=True)
    print("Validating...")
    Y_pred = clf.predict(X_val)
    Y_score: np.ndarray = clf.decision_function(X_val)
    print(metrics.classification_report(Y_val, Y_pred))
    with open(saveDir / f"valLinearSVC_{valds}.json", "w", encoding="utf-8") as f:
        json.dump(
            metrics.classification_report(Y_val, Y_pred, output_dict=True),
            f,
            indent=4
        )
        print("Saved classification report...")
    with open(saveDir / f"valpredLinearSVC_{valds}.json", "w", encoding="utf-8") as f:
        valpred = {
            "truelabels": Y_val,
            "preds": Y_pred.tolist(),
            "proba": Y_score.tolist(),
        }
        json.dump(valpred, f, indent=4)
        print("Saved validation result...")


def loadpkl(pkl: Union[str, Path]):
    print(f"Loading pkl [{pkl}]...")
    with open(pkl, "rb") as f:
        clf = pickle.load(f)
    return clf


def main(pretrained=False, valds: Union[List[str], str]="val"):
    pklPath = saveDir / "clfLinearSVC.pkl"
    if pretrained:
        clf = loadpkl(pklPath)
    else:
        clf = train(pklPath=pklPath)
    if valds:
        val(clf, valds=valds)


if __name__ == "__main__":
    BEGIN = 1
    REPEAT = 10
    END = BEGIN + REPEAT
    for i in range(BEGIN, END):
        saveDir = Path(f"repeatSave/svmSave/{i}")
        # if saveDir.exists():
        #     print(f"Skip {i}")
        #     continue
        if not saveDir.exists():
            saveDir.mkdir(parents=True)
        main(
            pretrained=True,
            # pretrained=False,
            valds="test",
            # valds=""  # off valds
        )
