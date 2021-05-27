import os
import random
import warnings
import albumentations as A

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# import torch.utils.data as torchdata
from torch.utils.data import DataLoader
import time
from pathlib import Path
import tqdm
import soundfile as sf

from config import CFG
from utils import set_seed, get_logger, timer

if CFG.use == 1:
    from inputst.testdata import TestDataset  # baseline
    from models.model import TimmSED
elif CFG.use == 2:
    from inputs.testdata2 import TestDataset  # , get_transforms
    from models.model2 import TimmSED

warnings.filterwarnings("ignore")

set_seed(CFG.seed)


# main loop
logger = get_logger("main.log")

logger.info("=" * 120)
logger.info(f"Inference")
logger.info("=" * 120)

# Data
TARGET_SR = 32000
DATADIR = CFG.test_datadir
TEST = len(list(DATADIR.glob("*.ogg"))) != 0
if not TEST:
    # FIXME:
    DATADIR = Path("/data2/minki/kaggle/ramdisk/train_soundscapes")

test_audios = list(DATADIR.glob("*.ogg"))
test_audio_ids = ["_".join(audio_id.name.split("_")[:2]) for audio_id in test_audios]
submission_df = pd.DataFrame({"row_id": test_audio_ids})


def prediction_for_clip(test_df: pd.DataFrame, clip: np.ndarray, model, threshold=0.5):

    dataset = TestDataset(df=test_df, clip=clip)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction_dict = {}
    for image, row_id in loader:
        row_id = row_id[0]
        image = image.to(device)

        with torch.no_grad():
            prediction = model(image)
            proba = prediction["clipwise_output"].detach().cpu().numpy().reshape(-1)

        events = proba >= threshold
        labels = np.argwhere(events).reshape(-1).tolist()

        if len(labels) == 0:
            prediction_dict[row_id] = "nocall"
        else:
            labels_str_list = list(map(lambda x: CFG.target_columns[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string

    return prediction_dict


# Prediction function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimmSED(
    base_model_name=CFG.base_model_name,
    pretrained=False,
    num_classes=CFG.num_classes,
    in_channels=CFG.in_channels,
)

ckpt = torch.load(CFG.test_weights_path)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()


warnings.filterwarnings("ignore")
prediction_dfs = []
for audio_path in test_audios:
    with timer(f"Loading {str(audio_path)}", logger):
        clip, _ = sf.read(audio_path)

    seconds = []
    row_ids = []
    # FIXME: 605?

    tot_len = int(np.ceil(len(clip) / 32000))
    if tot_len != 600:
        print("Warning: time is not 10 min", tot_len)

    for second in range(5, tot_len + 5, 5):
        row_id = "_".join(audio_path.name.split("_")[:2]) + f"_{second}"
        seconds.append(second)
        row_ids.append(row_id)

    test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
    with timer(f"Prediction on {audio_path}", logger):
        prediction_dict = prediction_for_clip(
            test_df, clip=clip, model=model, threshold=CFG.test_threshold
        )
    row_id = list(prediction_dict.keys())
    birds = list(prediction_dict.values())
    prediction_df = pd.DataFrame({"row_id": row_id, "birds": birds})
    prediction_dfs.append(prediction_df)

submission = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)

submission.to_csv(
    f"/home/minki/kaggle/birdclef-2021/racoon/submissions/{CFG.name}_submission.csv",
    index=False,
)
