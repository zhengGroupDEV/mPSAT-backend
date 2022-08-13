'''
Author: rainyl
Description: CNN training
License: Apache License 2.0
'''
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, Union
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch import optim

from src.dataset import MpDatasetCNN
from src.model import CNN


CHECK_POINTS = "checkpoints"

def main():
    CLASS_IN = 1
    CLASS_OUT = 120
    BATCH_SIZE = 128
    DROP_OUT = 0.2
    LR = 0.0001
    EPOCHS = 20
    CRITERIA = CrossEntropyLoss()
    DS_TRAIN = ""
    DS_VAL = ""
    DATASET = "ds1000"
    DATASET = "ds430"
    CMT = ""

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(
            mean=(0.5, ),
            std=(0.5, ),
        )
    ])
    trainloader = DataLoader(
        dataset=MpDatasetCNN(DS_TRAIN, transforms=transforms),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    valloader = DataLoader(
        dataset=MpDatasetCNN(DS_VAL, transforms=transforms),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    model = CNN(CLASS_IN, CLASS_OUT, dropout=DROP_OUT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device {}".format(device))
    # device = "cpu"
    model.to(device)
    myoptimizer = optim.Adam(model.parameters(), lr=LR)  # type: ignore
    COMMENT = "-NET_{}-{}-{}-{}-{}-LR_{}-DS_{}_CMT_{}".format(
        model._get_name(),
        device,
        CRITERIA._get_name(),
        myoptimizer.__class__.__name__,
        BATCH_SIZE,
        LR,
        DATASET,
        CMT,
    )

    tb = SummaryWriter(comment=COMMENT)
    _diter = iter(valloader)
    images, labels = next(_diter)
    tb.add_graph(model, images.to(device))
    tb.add_image("images", torchvision.utils.make_grid(images), 0)
    len_train = len(trainloader.dataset)  # type:ignore
    len_train_loader = len(trainloader)

    for epoch in range(EPOCHS):
        # train
        model.train()
        mean_loss: Union[int, float] = 0
        train_loss = 0.0
        correct = 0.0
        for idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            myoptimizer.zero_grad()
            output: torch.Tensor = model(data)
            loss: torch.Tensor = CRITERIA(output, target.long())
            train_loss += loss.item()
            tb.add_scalar("train/loss", loss.item(), epoch * len_train_loader + idx)
            correct += output.argmax(dim=1).eq(target).sum().item()
            mean_loss = torch.mean(torch.Tensor([mean_loss, loss.item()])).item()
            loss.backward()
            myoptimizer.step()
            if idx % 10 == 0:
                print(
                    "Train Epoch: [{}], batch: [{}], loss: [{:.6f}]".format(
                        epoch, idx, loss.item()
                    )
                )
        tb.add_scalar("train/Accuracy", correct / len_train, epoch)
        tb.add_scalar("train/MeanLoss", mean_loss, epoch)
        tb.add_scalar("train/TotalLoss", train_loss, epoch)
        print(
            "Train Epoch: [{}], trainLoss: [{:.6f}], correct: [{}/{}]".format(
                epoch, train_loss, correct, len_train
            )
        )

        # val
        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        len_val = len(valloader.dataset)  # type:ignore
        with torch.no_grad():
            # for data, target in valloader:
            for data, target in valloader:
                data, target = data.to(device), target.to(device)
                output: torch.Tensor = model(data)  # type:ignore
                loss: torch.Tensor = CRITERIA(output, target.long())  # type:ignore
                val_loss += loss.item()
                _, best = output.max(dim=1)
                val_correct += best.eq(target.view_as(best)).sum().item()
            tb.add_scalar("val/Accuracy", val_correct / len_val, epoch)
            tb.add_scalar("val/Loss", val_loss, epoch)
            print(
                "Epoch: [{}], val loss: [{:6f}], correct: [{}/{}]".format(
                    epoch, val_loss, val_correct, len_val
                )
            )

    # conventionally, there should be a validate dataset
    # and a test dataset, but test dataset is omitted here
    # testloader = DataLoader(
    #     dataset=MpDatasetCNN(DS_TEST, transforms=transforms),
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=1,
    #     pin_memory=False,
    # )

    # test_corr = 0.0
    # with torch.no_grad():
    #     for data, target in testloader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         _, best = output.max(dim=1)
    #         test_corr += best.eq(target.view_as(best)).sum().item()
    #     print(
    #         "Finished! test accuracy: [{}/{}, {:6f}]".format(
    #             test_corr, len(testloader), test_corr / len(testloader.dataset)  # type:ignore
    #         )
    #     )

    torch.save(model.state_dict(), CHECK_POINTS + "/net" + COMMENT + ".pth")

    tb.close()


def val():
    # saveDir = Path("repeatSave/cnnSave")
    saveDir = Path("repeatSave/cnn1000Save")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DS_TEST = "data/dataset/datasetSplit430/test"
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(
            mean=(0.5, ),
            std=(0.5, ),
        )
    ])
    ckptPath = Path(CHECK_POINTS)
    model = CNN(in_channel=1, out_class=120, dropout=0.2)
    model.to(device=DEVICE)
    testloader = DataLoader(
        dataset=MpDatasetCNN(DS_TEST, transforms=transforms),
        batch_size=128,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
    )
    # ckpts = ckptPath.glob("*CNN*-LR_0.0001*-DS_1+2.430.120*_CMT_LRN_REPEAT_*.pth")
    ckpts = ckptPath.glob(
        "*CNN*-LR_0.0001*-DS_1+2.1000.120_norm0.5std0.5_CMT_LRN_REPEAT*.pth"
    )
    for i, ckpt in enumerate(ckpts):
        print(f"DEBUG: [{ckpt}]")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        saveRep = saveDir / f"{i+1}"
        if not saveRep.exists():
            saveRep.mkdir(parents=True)
        ds = "test"
        Y_pred = []
        Y_val = []
        Y_score = []
        with torch.no_grad():
            for data, label in testloader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                output: torch.Tensor = F.softmax(model(data), dim=1)
                # print(output)
                _, best = output.max(dim=1)
                Y_pred.extend(best.tolist())
                Y_val.extend(label.tolist())
                Y_score.extend(output.tolist())
        with open(saveRep / f"valpred_{ds}.json", "w", encoding="utf-8") as f:
            valpred = {
                "truelabels": Y_val,
                "preds": Y_pred,
                "proba": Y_score,
            }
            json.dump(valpred, f, indent=4)

        with open(saveRep / f"valreport_{ds}.json", "w", encoding="utf-8") as f:
            json.dump(
                metrics.classification_report(Y_val, Y_pred, output_dict=True),
                f,
                indent=4,
            )
            print(f"Saved classification report for [{ds}]...")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", "-m", type=str, default="val", help="Mode to run, train or val"
    )
    args = parser.parse_args()
    assert args.mode in ["train", "val"], "Mode must be train or val"
    if args.mode == "train":
        main()
    elif args.mode == "val":
        val()
