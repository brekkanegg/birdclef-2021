import random
import os, sys
import time
import numpy as np
import torch
import logging
from pathlib import Path

import pandas as pd
from contextlib import contextmanager
from typing import Optional


def set_seed(seed=52):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(log_file="train.log"):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def get_stats(submission):
    nocall_submit = submission.query("birds=='nocall'")
    yescall_submit = submission.query("birds!='nocall'")
    bird_num = 0
    for _, r in yescall_submit.iterrows():
        bird_num += len(r["birds"].split(" "))

    nocall_num = len(nocall_submit)
    nocall_r = len(nocall_submit) / len(submission)
    bird_per_yescall = bird_num / (len(yescall_submit) + 1e-4)

    return nocall_num, nocall_r, bird_per_yescall


# Helper functions


def get_cv(submission):
    def get_metrics(s_true, s_pred):
        s_true = set(s_true.split(" "))
        s_pred = set(s_pred.split(" "))
        tp, gt, pred = len(s_true.intersection(s_pred)), len(s_true), len(s_pred)
        fp = pred - tp
        fn = gt - tp

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # TEST_AUDIO_ROOT = Path("/data2/minki/kaggle/birdclef-2021/train_soundscapes")
    TARGET_PATH = Path("/data2/minki/kaggle/birdclef-2021/train_soundscape_labels.csv")

    # if TARGET_PATH:
    sub_target = pd.read_csv(TARGET_PATH)
    sub_target = sub_target.merge(submission, how="left", on="row_id")

    #     print(sub_target["birds_x"].notnull().sum(), sub_target["birds_x"].notnull().sum())
    assert sub_target["birds_x"].notnull().all()
    assert sub_target["birds_y"].notnull().all()

    df_metrics = pd.DataFrame(
        [
            get_metrics(s_true, s_pred)
            for s_true, s_pred in zip(sub_target.birds_x, sub_target.birds_y)
        ]
    )

    tot_tp = df_metrics["tp"].sum()
    tot_fp = df_metrics["fp"].sum()
    tot_fn = df_metrics["fn"].sum()

    micro_f1 = 2 * tot_tp / (2 * tot_tp + tot_fp + tot_fn + 1e-4)

    return micro_f1
