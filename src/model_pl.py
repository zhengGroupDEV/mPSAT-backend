"""
Description: models for pytorch-lightning training
Author: Rainyl
License: Apache License 2.0
"""
import copy
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import classification_report

from .config import ConfigBase, ConfigCnn, ConfigCnn2D
from .model import CNN, CNN2D


class MpSaveCkptOnShutdown(Callback):
    def __init__(self, saveto: str = "ckpts/interrupt.ckpt") -> None:
        super().__init__()
        self.saveto = saveto

    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        trainer.save_checkpoint(self.saveto)
        print(f"Exception detected, saved to {self.saveto}")


class StepLrWithWarmup(optim.lr_scheduler.StepLR):
    def __init__(
        self,
        optimizer,
        step_size: int,
        warmup_step: int,
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
    ) -> None:
        self.warmup_step = warmup_step
        self.init_lr_groups = copy.deepcopy(optimizer.param_groups)
        super(StepLrWithWarmup, self).__init__(optimizer, step_size, gamma, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:  # type: ignore
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self._step_count <= self.warmup_step:  # type: ignore
            lr_scale = min(1.0, float(self._step_count) / self.warmup_step)  # type: ignore
            lr = [group["lr"] * lr_scale for group in self.init_lr_groups]
            return lr
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):  # type: ignore
            lr = [group["lr"] for group in self.optimizer.param_groups]  # type: ignore
            return lr
        lr = [group["lr"] * self.gamma for group in self.optimizer.param_groups]  # type: ignore
        return lr


class MpModelLightBase(pl.LightningModule):
    def __init__(
        self,
        conf: ConfigBase,
        optimizer: str = "adam",
        lr: float = 1e-4,
        sc_step: int = 100,
        sc_gamma: float = 0.5,
        warmup_step: int = 0,
        save_dir: str = "repeatSave/",
    ) -> None:
        super(MpModelLightBase, self).__init__()
        self.model: nn.Module = nn.Identity()
        self.optim = optimizer
        self.lr = lr
        self.sc_step = sc_step
        self.sc_gamma = sc_gamma
        self.warmup_step = warmup_step
        self.save_dir = Path(save_dir) / conf.model

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        # t = torch.randint(0, 1000, (conf.num_class, 3600))
        # self.example_input_array = [t, torch.arange(conf.num_class)]

        # metrics
        self.accuracy_score = torchmetrics.Accuracy(
            num_classes=conf.num_class,
            average="macro",
        )
        self.precision_score = torchmetrics.Precision(
            num_classes=conf.num_class,
            average="macro",
        )
        self.recall_score = torchmetrics.Recall(
            num_classes=conf.num_class,
            average="macro",
        )
        # self.pr_curve = torchmetrics.PrecisionRecallCurve(num_classes=conf.num_class)

        # trues and preds to save
        self.all_trues = []
        self.all_preds = []
        self.all_scores = []

        self.predict_trues = []
        self.predict_preds = []

    def forward(self, src, labels):
        logits = self.model(src)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return preds

    def training_step(self, src, labels):
        logits = self.model(src)
        loss = F.cross_entropy(logits, labels)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, src, labels):
        logits = self.model(src)
        loss = F.cross_entropy(logits, labels)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        self.accuracy_score(probs, labels)
        self.precision_score(preds, labels)
        self.recall_score(preds, labels)
        # self.pr_curve(F.softmax(logits, dim=-1), labels)

        self.log("val/loss", loss)
        self.log("val/accuracy", self.accuracy_score)
        self.log("val/precision", self.precision_score)
        self.log("val/recall", self.recall_score)

        return loss

    def on_validation_end(self) -> None:
        ...

    def test_step(self, src, labels):
        logits = self.model(src)
        loss = F.cross_entropy(logits, labels)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        self.all_trues.extend(labels.tolist())
        self.all_preds.extend(preds.tolist())
        self.all_scores.extend(probs.tolist())
        self.accuracy_score(probs, labels)
        self.precision_score(preds, labels)
        self.recall_score(preds, labels)
        # self.pr_curve(F.softmax(logits, dim=-1), labels)

        self.log("test/loss", loss)
        self.log("test/accuracy", self.accuracy_score)
        self.log("test/precision", self.precision_score)
        self.log("test/recall", self.recall_score)
        return loss

    def on_test_end(self) -> None:
        report: Dict[str, Any] = classification_report(
            self.all_trues, self.all_preds, output_dict=True
        )  # type: ignore
        report["trues"] = self.all_trues
        report["preds"] = self.all_preds
        report["scores"] = self.all_scores
        saveto: Path = Path(self.save_dir)
        # find the next folder
        for i in range(100):
            saveto = self.save_dir / f"{i}"
            if not saveto.exists():
                saveto.mkdir(parents=True)
                break
        with open(saveto / "report.json", "w") as f:
            json.dump(report, f, indent=4)

    def predict_step(self, src, labels):
        logits = self.model(src)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        self.predict_preds.extend(preds.tolist())
        self.predict_trues.extend(labels.tolist())

    def on_predict_end(self) -> None:
        report: Dict[str, Any] = classification_report(
            self.predict_trues,
            self.predict_preds,
            output_dict=True,
        )  # type: ignore
        report["trues"] = self.predict_trues
        report["preds"] = self.predict_preds
        saveto = self.save_dir / f"predict"
        if not saveto.exists():
            saveto.mkdir(parents=True)
        with open(saveto / "report.json", "w") as f:
            json.dump(report, f, indent=4)

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"optimizer {self.optim} not implemented")
        scheduler = {
            "scheduler": StepLrWithWarmup(
                optimizer,
                step_size=self.sc_step,
                gamma=self.sc_gamma,
                warmup_step=self.warmup_step,
            ),
            "name": "lr_scheduler",
            # "monitor": "train/loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]


class MpModelCnnLight(MpModelLightBase):
    def __init__(
        self,
        conf: ConfigCnn,
        optimizer: str = "adam",
        lr: float = 0.0001,
        sc_step: int = 100,
        sc_gamma: float = 0.5,
        warmup_step: int = 0,
        save_dir: str = "repeatSave/",
    ) -> None:
        super(MpModelCnnLight, self).__init__(
            conf,
            optimizer,
            lr,
            sc_step,
            sc_gamma,
            warmup_step,
            save_dir,
        )
        self.model = CNN(
            in_channel=conf.in_channel,
            out_class=conf.num_class,
            dropout=conf.dropout,
        )

    def forward(self, batch, batch_idx):
        src, labels = batch
        src = src.unsqueeze(1)
        return super().forward(src, labels)

    def training_step(self, batch, batch_idx):
        src, labels = batch
        src = src.unsqueeze(1)
        return super().training_step(src, labels)

    def validation_step(self, batch, batch_idx):
        src, labels = batch
        src = src.unsqueeze(1)
        return super().validation_step(src, labels)

    def test_step(self, batch, batch_idx):
        src, labels = batch
        src = src.unsqueeze(1)
        return super().test_step(src, labels)

    def predict_step(self, batch, batch_idx):
        src, labels = batch
        src = src.unsqueeze(1)
        return super().predict_step(src, labels)


class MpModelCnn2DLight(MpModelLightBase):
    def __init__(
        self,
        conf: ConfigCnn2D,
        optimizer: str = "adam",
        lr: float = 0.0001,
        sc_step: int = 100,
        sc_gamma: float = 0.5,
        warmup_step: int = 0,
        save_dir: str = "repeatSave/",
    ) -> None:
        super(MpModelCnn2DLight, self).__init__(
            conf,
            optimizer,
            lr,
            sc_step,
            sc_gamma,
            warmup_step,
            save_dir,
        )
        self.model = CNN2D(
            in_channel=conf.in_channel,
            out_class=conf.num_class,
            dropout=conf.dropout,
        )

    def forward(self, batch, batch_idx):
        src, labels = batch
        return super().forward(src, labels)

    def training_step(self, batch, batch_idx):
        src, labels = batch
        return super().training_step(src, labels)

    def test_step(self, batch, batch_idx):
        src, labels = batch
        return super().test_step(src, labels)

    def validation_step(self, batch, batch_idx):
        src, labels = batch
        return super().validation_step(src, labels)

    def predict_step(self, batch, batch_idx):
        src, labels = batch
        return super().predict_step(src, labels)
