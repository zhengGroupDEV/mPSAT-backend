'''
Author: rainyl
Description: MCCV for CNN
License: Apache License 2.0
'''
from typing import Iterable, List, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torchvision
import torch
from sklearn import preprocessing as pp
from sklearn import metrics
from torch import optim
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from torch.utils.data import DataLoader

from .model import CNN
from .dataset import MpDatasetCNNMCCV


MAX_MCCV = 300
MCCV_SAVE_PATH = Path("mccvSave")
if not MCCV_SAVE_PATH.exists():
    MCCV_SAVE_PATH.mkdir()

logger = logging.getLogger("MCCV")
shandler = logging.StreamHandler()
shandler.setFormatter(
    logging.Formatter(
        "%(asctime)s:%(name)s:%(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)
logger.addHandler(shandler)
logger.setLevel(logging.DEBUG)


def runOneCnn(alexset: MpDatasetCNNMCCV):
    EPOCHS = 20
    LR = 0.0001
    NUM_WORKERS = 5
    BATCH_SIZE = 128

    model = CNN(1, 120, dropout=0.2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    criteria = CrossEntropyLoss()

    alexset.set_dataset(dataset="train")
    alexset.shuffle_dataset()
    alexloader = DataLoader(
        alexset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    for epoch in range(EPOCHS):
        # train
        model.train()
        meanLoss: Union[int, float] = 0
        trainLoss = 0
        correct = 0
        # for idx, (data, target) in enumerate(alexloader):
        for data, target in alexloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criteria(output, target.long())
            correct += output.argmax(dim=1).eq(target).sum().item()
            trainLoss += loss.item()
            meanLoss = torch.mean(torch.Tensor([meanLoss, loss.item()])).item()
            loss.backward()
            optimizer.step()
        logger.info(
            (
                f"Train Epoch: [{epoch}], trainLoss: [{trainLoss:.5f}], "
                f"correct: [{correct}/{len(alexset.label_train)}], "
                f"accuracy: {correct / len(alexset.label_train):.5f}"
            )
        )

    model.eval()
    Y_pred, Y_val, Y_score = [], [], []
    alexset.set_dataset("val")
    valloader = DataLoader(
        alexset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    with torch.no_grad():
        for data, label in valloader:
            data, label = data.to(device), label.to(device)
            out: torch.Tensor = F.softmax(model(data), dim=1)
            _, best = out.max(dim=1)
            Y_pred.extend(best.tolist())
            Y_val.extend(label.tolist())
            Y_score.extend(out.tolist())
    return Y_val, Y_pred, Y_score


def runCnn():
    DS = "data/dataset/datasetSplit430/mccv"
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=(0.5,),
                std=(0.5,),
            )
        ]
    )
    alexset = MpDatasetCNNMCCV(DS, transforms=transforms, ptrain=0.78)

    save_dir = MCCV_SAVE_PATH / "cnn"
    save_step = 5

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
        y_true, y_pred, y_score = runOneCnn(alexset=alexset)
        # append
        y_true_list[i % save_step] = y_true
        y_pred_list[i % save_step] = y_pred
        y_score_list[i % save_step] = y_score
        # save
        if (i + 1) % save_step == 0:
            saveMetrics(
                saveDir=save_dir,
                y_true=y_true_list,
                y_pred=y_pred_list,
                y_score=y_score_list,
            )
            logger.info(f"[{i+1}/{MAX_MCCV}] Saved! ")


def saveMetrics(
    saveDir: Path,
    y_true: List[List],
    y_pred: List[np.ndarray],
    y_score: List[np.ndarray],
):
    if not saveDir.exists():
        saveDir.mkdir(parents=True)
    columns = ["accuracy", "precision", "recall", "f1"]
    # print(metrics.classification_report(y_true, y_pred))
    # metric to save: macro-accuracy, precision, recall, f1
    # merics.csv
    # accuracy, precision, recall, f1
    # 0.8,      0.9,       0.8,    0.8
    metric_data = []
    for i in range(len(y_true)):
        report = metrics.classification_report(
            y_true=y_true[i], y_pred=y_pred[i], output_dict=True
        )
        metric_data.append(
            [
                report["accuracy"],
                report["macro avg"]["precision"],
                report["macro avg"]["recall"],
                report["macro avg"]["f1-score"],
            ]
        )

    metric_path = saveDir / f"metrics.csv"
    if metric_path.exists():
        metric_data_old = pd.read_csv(metric_path).values
        metric_data = np.concatenate((metric_data_old, metric_data))
    df = pd.DataFrame(
        metric_data,
        columns=columns,
    )
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


def main(model: str = "cnn"):
    # run model train and val
    logger.warning(f"Running MCCV for model [{model}]")
    runCnn()


if __name__ == "__main__":
    main()
