import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import torch.utils.data as torchdata
from torch.utils.data import DataLoader
import time
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_threshold", "--lt", type=float, default=0.15)
parser.add_argument("--global_threshold", "--gt", type=float, default=0.25)
parser.add_argument("--global_length", "--gl", type=int, default=30)
parser.add_argument("--gpu", "--g", type=str, default="5")

CFG, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = CFG.gpu


from inputs.testdata2 import TestDataset
from models.model2 import TimmSED
from utils import get_stats, get_cv

###################### Model Load
# load model
def load_models(ckpts, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = []
    for w in ckpts:
        m = TimmSED(
            base_model_name=model_name,
            pretrained=False,
            num_classes=397,
            in_channels=1,
        )
        m.load_state_dict(torch.load(w)["model_state_dict"])
        m = m.to(device)
        model_list.append(m)

    return model_list


##################### Predict
@torch.no_grad()
def get_outputs(data, models, chunks_len=[5], tta=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dict = {}

    for chunk in chunks_len:
        image = data[f"{chunk}sec_mel"].to(device)
        probsum = None
        for model in models:
            model.eval()
            # FIXME: tta
            if tta > 0:
                pred = model(image.permute(1, 0, 2, 3))
                pred["clipwise_output"] = torch.mean(pred["clipwise_output"], dim=0)

            else:
                pred = model(image)

            prob = pred["clipwise_output"].detach().cpu().numpy().reshape(-1)

            if probsum is None:
                probsum = prob
            else:
                probsum += prob

        probavg = probsum / len(models)
        out_dict[f"{chunk}sec_probavg"] = probavg

    return out_dict


####################### Get predictions (logits)
def get_prediction_dict(test_audios, model_dict, tta):
    prediction_dict = {}
    for audio_path in tqdm(test_audios):
        clip, _ = sf.read(audio_path)

        seconds = []
        row_ids = []

        tot_len = int(np.ceil(len(clip) / 32000))
        if tot_len != 600:
            print("Warning: time is not 10 min", tot_len)

        for second in range(5, tot_len + 5, 5):
            row_id = "_".join(audio_path.name.split("_")[:2]) + f"_{second}"
            seconds.append(second)
            row_ids.append(row_id)

        test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
        dataset = TestDataset(df=test_df, clip=clip, chunks_len=[5, 30, 20], tta=tta)
        loader = torchdata.DataLoader(dataset, batch_size=1, shuffle=False)

        for data in loader:
            row_id = data["row_id"][0]
            prediction_dict[row_id] = {}

            # rst101_outputs = get_outputs(
            #     data, model_dict["rst101"], chunks_len=[5], tta=tta
            # )
            # prediction_dict[row_id]["rst101_5"] = rst101_outputs["5sec_probavg"]

            effb0_outputs = get_outputs(
                data, model_dict["effb0"], chunks_len=[5, 30], tta=tta
            )
            prediction_dict[row_id]["effb0_5"] = effb0_outputs["5sec_probavg"]
            prediction_dict[row_id]["effb0_30"] = effb0_outputs["30sec_probavg"]

            # rxt50_outputs = get_outputs(
            #     data, model_dict["rxt50"], chunks_len=[5, 20], tta=tta
            # )
            # prediction_dict[row_id]["rxt50_5"] = rxt50_outputs["5sec_probavg"]
            # prediction_dict[row_id]["rxt50_20"] = rxt50_outputs["20sec_probavg"]

    return prediction_dict


################### Make submission file with Threshold cuts
def make_submission_file(
    prediction_dict, global_threshold, local_threshold, target_columns
):

    row_ids = prediction_dict.keys()
    prediction_df = pd.DataFrame({"row_id": row_ids, "birds": None})

    for idx, row in prediction_df.iterrows():
        rid = row["row_id"]

        # ResNeSt 101
        rst101_prob_5 = prediction_dict[rid]["rst101_5"]

        # EFFB0
        effb0_prob_5 = prediction_dict[rid]["effb0_5"]
        effb0_prob_30 = prediction_dict[rid]["effb0_30"]
        effb0_prob_5 = effb0_prob_5 * (effb0_prob_30 > global_threshold)

        # ResNext50
        rxt50_prob_5 = prediction_dict[rid]["rxt50_5"]
        rxt50_prob_20 = prediction_dict[rid]["rxt50_20"]
        rxt50_prob_5 = rxt50_prob_5 * (rxt50_prob_20 > global_threshold)

        # Ensemble
        # proba = effb0_prob_5
        proba = (rst101_prob_5 + effb0_prob_5 + rxt50_prob_5) / 3

        # Top-k
        save_index = np.zeros_like(proba)
        top_k_index = proba.argsort()[::-1][:3]
        save_index[top_k_index] = 1
        proba = proba * save_index

        events = proba >= local_threshold

        labels = np.argwhere(events).reshape(-1).tolist()
        if len(labels) == 0:
            label_string = "nocall"
        else:
            labels_str_list = list(map(lambda x: target_columns[x], labels))
            label_string = " ".join(labels_str_list)

        prediction_df.loc[idx, "birds"] = label_string

    return prediction_df


###################### Main
TRAIN_META = pd.read_csv("/data2/minki/kaggle/birdclef-2021/train_metadata.csv")
TARGET_COLUMNS = sorted(set(TRAIN_META["primary_label"]))

# TARGET_SR = 32000
DATADIR = Path("/data2/minki/kaggle/birdclef-2021/test_soundscapes")
TEST = len(list(DATADIR.glob("*.ogg"))) != 0
if not TEST:
    DATADIR = Path("/data2/minki/kaggle/birdclef-2021/train_soundscapes")

TEST_AUDIOS = list(DATADIR.glob("*.ogg"))
TEST_AUDIO_IDS = ["_".join(audio_id.name.split("_")[:2]) for audio_id in TEST_AUDIOS]
SUBMISSION_DF = pd.DataFrame({"row_id": TEST_AUDIO_IDS})

TTA = 5

# RST101_MODELS = load_models([f"../input/birdclef2021racoonmod2fold1bestckpt/f{i}_rst101_sec10_2.pth" for i in range(5)], 'resnest101e')

model_dict = {}

# model_dict["rst101"] = load_models(
#     [
#         f"../input/birdclef2021racoonmod2fold1bestckpt/f{i}_rst101_sec10_2.pth"
#         for i in range(5)
#     ],
#     "resnest101e",
# )

# model_dict["effb0"] = load_models(
#     [
#         f"../input/birdclef2021racoonmod2fold1bestckpt/f{i}_effb0_sec30_2.pth"
#         for i in range(5)
#     ],
#     "efficientnet_b0",
# )
# model_dict["rxt50"] = load_models(
#     [
#         f"../input/birdclef2021racoonmod2fold1bestckpt/f{i}_rxt50_sec20_2.pth"
#         for i in range(5)
#     ],
#     "resnext50_32x4d",
# )

model_dict["effb0"] = load_models(
    [
        f"/data2/minki/kaggle/birdclef-2021/ckpt/0527_rf{i}v2/checkpoints/best.pth"
        for i in range(5)
    ],
    "efficientnet_b0",
)

prediction_dict = get_prediction_dict(
    test_audios=TEST_AUDIOS, model_dict=model_dict, tta=TTA
)

# CHECK
gth = [0.25]  # 0 is ignore
lth = [0.2, 0.15, 0.1, 0.05]

hpo_df = []
for gt in gth:
    for lt in lth:
        tempf = make_submission_file(prediction_dict, gt, lt, TARGET_COLUMNS)
        nocall_num, nocall_r, bird_per_yescall = get_stats(tempf)
        cv = get_cv(tempf)
        hpo_df.append([gt, lt, nocall_num, nocall_r, bird_per_yescall, cv])

hpo_df = pd.DataFrame(
    hpo_df, columns=["gth", "lth", "nocall", "nocall_r", "bird/yescall", "cv"]
)
hpo_df
