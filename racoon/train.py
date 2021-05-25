"""
https://www.kaggle.com/hidehisaarai1213/pytorch-training-birdclef2021-starter/data

# TODO: 
# spectogram move to data
# check dataset shape ----- 


# Add noises, +a : data.py
# Add Mix-up
# amp
# use rating ?
# secondary label
# add checkpoint callback

"""

import gc
import os

# import math
import random
import warnings

import albumentations as A

# import cv2
# import librosa
import numpy as np
import pandas as pd

# import soundfile as sf
# import timm
import torch

# import torch.optim as optim
# import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from pathlib import Path
from typing import List

# from albumentations.pytorch import ToTensorV2
# from albumentations.core.transforms_interface import ImageOnlyTransform
# from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import Runner, SupervisedRunner
from sklearn import model_selection

from config import CFG


# from sklearn import metrics
# from timm.models.layers import SelectAdaptivePool2d
# from torch.optim.optimizer import Optimizer
# from torchlibrosa.stft import LogmelFilterBank, Spectrogram
# from torchlibrosa.augmentation import SpecAugmentation

from opts import get_optimizer, get_scheduler

if CFG.use == 1:
    from data import WaveformDataset  # , get_transforms
    from model import TimmSED
elif CFG.use == 2:
    from data2 import WaveformDataset  # , get_transforms
    from model2 import TimmSED

from losses import get_criterion
from callbacks import get_callbacks


# this notebook is by default run on debug mode (only train one epoch).
# If you'd like to get the results on par with that of inference notebook, you'll need to train the model around 30 epochs


def set_seed(seed=52):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_runner(device: torch.device):
    return SupervisedRunner(
        device=device, input_key="image", input_target_key="targets"
    )


warnings.filterwarnings("ignore")

logdir = Path("out")
logdir.mkdir(exist_ok=True, parents=True)
if (logdir / "train.log").exists():
    os.remove(logdir / "train.log")
logger = init_logger(log_file=logdir / "train.log")

# environment
set_seed(CFG.seed)
device = get_device()

# validation
splitter = getattr(model_selection, CFG.split)(**CFG.split_params)

# data
train = pd.read_csv(CFG.train_csv)

# main loop
for i, (trn_idx, val_idx) in enumerate(splitter.split(train, y=train["primary_label"])):
    if i == CFG.fold:
        break

logger.info("=" * 120)
logger.info(f"Fold {i} Training")
logger.info("=" * 120)

trn_df = train.loc[trn_idx, :].reset_index(drop=True)
val_df = train.loc[val_idx, :].reset_index(drop=True)

loaders = {
    phase: torchdata.DataLoader(
        WaveformDataset(
            df_,
            CFG.train_datadir,
            # img_size=CFG.img_size,
            # waveform_transforms=get_transforms(phase),
            period=CFG.period,
            validation=(phase == "valid"),
        ),
        **CFG.loader_params[phase],
    )  # type: ignore
    for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
}

model = TimmSED(
    base_model_name=CFG.base_model_name,
    pretrained=CFG.pretrained,
    num_classes=CFG.num_classes,
    in_channels=CFG.in_channels,
)
criterion = get_criterion()
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer)
callbacks = get_callbacks()
runner = get_runner(device)
runner.train(
    model=model,
    criterion=criterion,
    loaders=loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=CFG.epochs,
    verbose=True,
    logdir=logdir / f"{CFG.name}/{CFG.fold}",  # FIXME:
    callbacks=callbacks,
    main_metric=CFG.main_metric,
    minimize_metric=CFG.minimize_metric,
)
