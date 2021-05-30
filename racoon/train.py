import os
import random
import warnings
import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as torchdata
from catalyst.dl import Runner, SupervisedRunner
from sklearn import model_selection
from pathlib import Path
import os
import yaml
import argparse

from config import CFG

from utils import set_seed, init_logger
from opts import get_optimizer, get_scheduler

from losses import get_criterion
from callbacks import get_callbacks


if CFG.use == 1:
    from inputs.data import WaveformDataset  # baseline
    from models.model import TimmSED
elif CFG.use == 2:
    from inputs.data2 import WaveformDataset  # , get_transforms
    from models.model2 import TimmSED


print("==========")
print(f"Method: {CFG.use}, Model: {CFG.base_model_name}, Name: {CFG.name}")
print("==========")


warnings.filterwarnings("ignore")

logdir = CFG.logdir / CFG.name
logdir.mkdir(exist_ok=True, parents=True)

with open(logdir / "cfg.yaml", "w") as f:
    yaml.dump(vars(CFG), f, sort_keys=False)

if (logdir / "train.log").exists():
    os.remove(logdir / "train.log")
logger = init_logger(log_file=logdir / "train.log")

# environment
set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# validation
splitter = getattr(model_selection, CFG.split)(**CFG.split_params)

# data
train = pd.read_csv(CFG.train_csv)

# add rating cut
train = train.query(f"rating>={CFG.rating}").reset_index()


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

# if CFG.mixup:
#     from catalyst.dl.callbacks import MixupCallback

#     callbacks += [MixupCallback]

runner = SupervisedRunner(device=device, input_key="image", input_target_key="targets")
runner.train(
    model=model,
    criterion=criterion,
    loaders=loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=CFG.epochs,
    verbose=True,
    logdir=logdir,  # FIXME:
    callbacks=callbacks,
    main_metric=CFG.main_metric,
    minimize_metric=CFG.minimize_metric,
    resume=CFG.resume_dir,
    check=CFG.check,
    timeit=CFG.timeit,
    overfit=CFG.overfit,
    load_best_on_end=CFG.load_best_on_end,
)
