'''
Author: rainyl
Description: RF
License: Apache License 2.0
'''
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import json

import numpy as np
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
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


def trainRF(pklPath = "clfRF.pkl"):
    X_train, Y_train = load_dataset(ds="train", shuffle=True)
    clf = RandomForestClassifier(
        n_estimators=120,
        criterion="gini",
        bootstrap=True,
        class_weight="balanced",
        min_samples_leaf=10,
    )
    print("Training....")
    clf.fit(X_train, Y_train)
    with open(pklPath, "wb") as f:
        pickle.dump(clf, f)
        print("Saved model to pickle...")
    return clf


def valRF(clf: RandomForestClassifier, valds="val"):
    X_val, Y_val = load_dataset(ds=valds, shuffle=True)
    print("Validating...")
    Y_pred = clf.predict(X_val)
    Y_score: np.ndarray = clf.predict_proba(X_val)
    print(metrics.classification_report(Y_val, Y_pred))
    with open(saveDir / f"valRF_{valds}.json", "w", encoding="utf-8") as f:
        json.dump(
            metrics.classification_report(Y_val, Y_pred, output_dict=True),
            f,
            indent=4
        )
        print("Saved classification report...")
    with open(saveDir / f"valpredRF_{valds}.json", "w", encoding="utf-8") as f:
        valpred = {
            "truelabels": Y_val,
            "preds": Y_pred.tolist(),
            "proba": Y_score.tolist(),
        }
        json.dump(valpred, f, indent=4)
        print("Saved validation result...")


def loadRFpkl(pkl: str):
    print(f"Loading pkl [{pkl}]...")
    with open(pkl, "rb") as f:
        clf = pickle.load(f)
    return clf


def main(pretrained=False, valds="val"):
    pklPath = saveDir / "clfRF.pkl"
    if pretrained:
        clf = loadRFpkl(pkl=pklPath)
    else:
        clf = trainRF(pklPath=pklPath)
    if valds:
        valRF(clf, valds=valds)


if __name__ == "__main__":
    BEGIN = 1
    REPEAT = 10
    END = BEGIN + REPEAT
    for i in range(BEGIN, END):
        saveDir = Path(f"repeatSave/rfSave/{i}")
        if not saveDir.exists():
            saveDir.mkdir()
        main(
            pretrained=True,
            # pretrained=False,
            valds="test",
            # valds=""
        )
