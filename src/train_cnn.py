"""
Description: train models built by Pytorch
Author: Rainyl
Date: 2022-09-08 16:11:28
"""
import glob
import os
import random
import warnings
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .config import (
    ConfigBase,
    ConfigCnn,
    ConfigCnn2D,
    __supported_device__,
    __supported_models__,
)
from .dataset import MpDatasetCNN, MpDatasetCnn2D
from .model_pl import (
    MpModelCnn2DLight,  # MpModelCnnNextLight,
    MpModelCnnLight,
    MpSaveCkptOnShutdown,
)


def seed_everything(seed: int):
    print(f"set global seed to {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def load_data_set(
    path: str, conf: ConfigBase, args: Namespace, shuffle=True
) -> DataLoader:
    model = args.model
    assert model in __supported_models__
    transforms = None
    if model in ["cnn", "cnext"]:
        ds = MpDatasetCNN(
            path=path,
            fmt="ftr",
            transforms=transforms,
            seq_len=conf.seq_len,
        )
    elif model == "cnn2d":
        ds = MpDatasetCnn2D(
            path=path,
            transforms=transforms,
            fmt="png",
        )
    else:
        raise ValueError(
            f"model {model} not supported, supported models are {__supported_models__}"
        )
    loader = DataLoader(
        dataset=ds,
        batch_size=conf.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        # collate_fn=collate_fn_seq,
    )
    return loader


def main(args: Namespace):
    # config
    if args.model == "cnn":
        conf = ConfigCnn(args.conf, model=args.model)
        model = MpModelCnnLight(
            conf=conf,
            optimizer=args.optim,
            lr=args.lr,
            sc_step=args.sc_step,
            sc_gamma=args.sc_gamma,
            warmup_step=args.warmup,
            save_dir=args.save_dir,
        )
    elif args.model == "cnn2d":
        conf = ConfigCnn2D(args.conf, model=args.model)
        model = MpModelCnn2DLight(
            conf=conf,
            optimizer=args.optim,
            lr=args.lr,
            sc_step=args.sc_step,
            sc_gamma=args.sc_gamma,
            warmup_step=args.warmup,
            save_dir=args.save_dir,
        )
    else:
        raise ValueError()
    if args.overwrite_config:
        conf.save_init()
    conf.load(args.conf)
    time_now = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    print(
        f"Training dataset: {args.train_set}\n"
        f"Validation dataset: {args.val_set}\n"
        f"Test dataset: {args.test_set}\n"
        f"MCCV dataset: {args.mccv_set}"
    )
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    version_info = (
        f"model_{args.model}-lr_{args.lr}"
        f"-seed_{args.seed}-warmup_{args.warmup}"
        f"-extra_{args.extra}-time_{time_stamp}"
    )
    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.model,
        # log_graph=True,
        version=version_info,
    )

    # callbacks
    # dev_monitor = DeviceStatsMonitor()
    early_stop_ = EarlyStopping(
        monitor="val/accuracy",
        min_delta=0.00,
        patience=args.early_stop_epoch,
        verbose=False,
        mode="max",
    )
    save_on_shutdown = MpSaveCkptOnShutdown()
    grad_accum = GradientAccumulationScheduler({0: args.accum_step})
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_name = (
        f"mp={args.extra}-seed={args.seed}"
        "-epoch={epoch}-step={step}-val_loss={val/loss:.3f}"
    )
    ckpt_callback = ModelCheckpoint(
        dirpath=f"ckpts/{args.model}/{time_now}",
        filename=ckpt_name,
        # monitor="val/loss",
        # mode="min",
        every_n_epochs=args.save_step,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )

    # trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            # dev_monitor,
            grad_accum,
            lr_monitor,
            ckpt_callback,
            save_on_shutdown,
            early_stop_,  # NOTE: comment when MCCV
        ],
        accelerator=args.device,
        devices=1,
        gradient_clip_val=1,
        max_epochs=args.epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        val_check_interval=0.5,
        precision=32,
        # auto_lr_find=True,
        num_sanity_val_steps=1,
        # strategy="ddp",
    )

    if args.predict_only:
        predict_loader = load_data_set(
            path=args.predict_set,
            conf=conf,
            args=args,
            shuffle=False,
        )
        trainer.predict(
            model=model,
            dataloaders=predict_loader,
            ckpt_path=args.ckpt,
        )
    else:
        # dataset and dataloader
        train_loader = load_data_set(
            path=args.train_set,
            conf=conf,
            args=args,
        )
        val_loader = load_data_set(
            path=args.val_set,
            conf=conf,
            args=args,
            shuffle=False,
        )
        test_loader = load_data_set(
            path=args.test_set,
            conf=conf,
            args=args,
            shuffle=False,
        )

        if not args.test_only:
            # fit
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                # ckpt_path="ckpts/vit/20221107_092942/mp=cb-nr-vit-seed=42-epoch=17-step=9187-val_loss=0.355.ckpt",
            )
            trainer.test(dataloaders=test_loader, model=model, ckpt_path=None)
        else:
            trainer.test(dataloaders=test_loader, model=model, ckpt_path=args.ckpt)


if __name__ == "__main__":
    parser = ArgumentParser()
    # common
    parser.add_argument("--train-set", dest="train_set", type=str, required=True)
    parser.add_argument("--test-set", dest="test_set", type=str, required=True)
    parser.add_argument("--val-set", dest="val_set", type=str, required=True)
    parser.add_argument("--predict-set", dest="predict_set", type=str, required=False)
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--save-dir", dest="save_dir", type=str, default="repeatSave/")
    parser.add_argument("--checkpoint", dest="ckpt", type=str, default="best")
    parser.add_argument(
        "--overwrite-config",
        dest="overwrite_config",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        action="store_true",
    )
    parser.add_argument(
        "--predict-only",
        dest="predict_only",
        action="store_true",
    )
    # model specified
    parser.add_argument("--optim", dest="optim", type=str, default="adam")
    parser.add_argument("--config", dest="conf", type=str, required=True)
    parser.add_argument("--model", dest="model", type=str, default="cnn")
    parser.add_argument("--extra", dest="extra", type=str, default="")
    parser.add_argument("--epochs", dest="epochs", type=int, default=30)
    parser.add_argument(
        "--early-stop-epoch",
        dest="early_stop_epoch",
        type=int,
        default=5,
    )
    parser.add_argument("--lr", dest="lr", type=float, default=0.0001)
    parser.add_argument("--save-step", dest="save_step", type=int, default=2)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--accum-step", dest="accum_step", type=int, default=16)
    parser.add_argument("--scheculer-step", dest="sc_step", type=int, default=100)
    parser.add_argument("--warmup", dest="warmup", type=int, default=1000)
    parser.add_argument("--scheculer-gamma", dest="sc_gamma", type=float, default=0.5)
    parser.add_argument("--seed", dest="seed", type=int, default=42)  # currently no use

    args: Namespace = parser.parse_args(
        [
            "--train-set",
            "",
            "--val-set",
            "",
            "--test-set",
            "",
            "--device",
            "cuda",
            # "cpu",
            "--optim",
            "adam",
            "--save-step",
            "1",
            "--scheculer-gamma",
            "0.5",
            # "--seed",
            # "20",
            "--overwrite-config",
            "--num-workers",
            "10",
            "--save-dir",
            "repeatSave/",
            # "repeatSave/1000/",
            "--early-stop-epoch",
            "10",
            ##################### CNN ##########################
            "--config",
            "config/config_cnn.json",
            # "--checkpoint",
            # "ckpts/cnn_used",
            "--model",
            "cnn",
            "--epochs",
            "50",
            "--lr",
            "1e-3",
            "--accum-step",
            "1",
            "--scheculer-gamma",
            "0.8",
            "--scheculer-step",
            "10",
            "--warmup",
            "5",
            "--save-dir",
            "repeatSave/1000",
            "--extra",
            "cb-nrpb-nopos",
            #################### CNN2D ##########################
            "--config",
            "config/config_cnn2d.json",
            "--model",
            "cnn2d",
            "--epochs",
            "50",
            "--lr",
            "1e-4",
            "--accum-step",
            "1",
            "--scheculer-step",
            "2",
            "--scheculer-gamma",
            "0.8",
            "--warmup",
            "1",
            "--extra",
            "cb-nrpb",
            "--save-dir",
            "repeatSave/1000",
        ]
    )  # type: ignore

    assert args.model in __supported_models__, f"model {args.model} not supported"
    assert args.device in __supported_device__, f"device {args.device} not supported"

    p = Path(args.save_dir) / f"{args.model}"
    # if p.exists():
    #     new_dir = p.parent / f"{args.model}.bak"
    #     if new_dir.exists():
    #         shutil.rmtree(new_dir)
    #     shutil.move(p, new_dir)

    for global_count, seed in enumerate(
        [
            0,
            21,
            42,
            63,
            84,
            100,
            221,
            442,
            663,
            884,
        ]
    ):
        if (p / f"{global_count}").exists():
            print(f"Skip {global_count}")
            continue
        ckpt = args.ckpt
        if args.test_only:
            ckpts_seed = glob.glob(args.ckpt + f"/*-seed={seed}-*.ckpt")
            assert (
                len(ckpts_seed) == 1
            ), f"found {len(ckpts_seed)} ckpts for seed {seed}, check it"
            ckpt = ckpts_seed[0]
        args.ckpt = ckpt
        seed_everything(seed)
        args.seed = seed
        # args.global_count = global_count

        main(args)
