import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from audiomentations import *

import torch.utils.data as torchdata

from config import CFG

import time


class WaveformDataset(torchdata.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        datadir: Path,
        # img_size=224,
        # waveform_transforms=None,
        period=10,
        validation=False,
    ):
        self.df = df
        self.datadir = datadir
        # self.img_size = img_size
        # self.waveform_transforms = waveform_transforms

        self.period = period
        self.validation = validation
        self.waveform_transforms = (
            self.get_wav_transforms() if not self.validation else None
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        t0 = time.time()

        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        ebird_code = sample["primary_label"]

        y, sr = sf.read(self.datadir / ebird_code / wav_name)

        t1 = time.time()
        # print('//reading time: ', t1-t0)

        # Crop or Pad
        len_y = len(y)
        effective_length = sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            if not self.validation:
                start = np.random.randint(effective_length - len_y)
            else:
                start = 0
            new_y[start : start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            if not self.validation:
                start = np.random.randint(len_y - effective_length)
            else:
                start = 0
            y = y[start : start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        y = np.nan_to_num(y)

        if not self.validation:
            y = self.waveform_transforms(y, sr)
            t2 = time.time()
            # print('//aug time: ', t2-t1)

        y = np.nan_to_num(y)

        y = normalize(y)
        y = np.nan_to_num(y)

        t3 = time.time()
        # print('//norm time: ', t3-t2)

        labels = np.zeros(len(CFG.target_columns), dtype=float)

        # FIXME: secondary label
        labels[CFG.target_columns.index(ebird_code)] = 1.0

        return {"image": y, "targets": labels}

    # # FIXME: Add augmentations
    def get_wav_transforms(self):
        transforms = CFG.transforms
        if self.validation:
            phase = "valid"
        else:
            phase = "train"

        if transforms is None:
            return None
        else:
            if transforms[phase] is None:
                return None

            trns_list = []
            for trns_conf in transforms[phase]:
                trns_name = trns_conf["name"]
                trns_params = (
                    {} if trns_conf.get("params") is None else trns_conf["params"]
                )
                if globals().get(trns_name) is not None:
                    trns_cls = globals()[trns_name]
                    trns_list.append(trns_cls(**trns_params))

            if len(trns_list) > 0:
                return Compose(trns_list)
            else:
                return None

    # def get_wav_transforms(self):
    #     """
    #     Returns the transformation to apply on waveforms
    #     Returns:
    #         Audiomentations transform -- Transforms
    #     """
    #     transforms = Compose(
    #         [
    #             AddGaussianSNR(max_SNR=0.5, p=0.5),
    #             AddBackgroundNoise(
    #                 sounds_path=CFG.background_datadir,
    #                 min_snr_in_db=0,
    #                 max_snr_in_db=2,
    #                 p=0.5,
    #             ),
    #         ]
    #     )
    #     return transforms


def normalize(y: np.ndarray):
    max_vol = np.abs(y).max()
    y_vol = y * 1 / max_vol
    return np.asfortranarray(y_vol)


# class Normalize:
#     def __call__(self, y: np.ndarray):
#         max_vol = np.abs(y).max()
#         y_vol = y * 1 / max_vol
#         return np.asfortranarray(y_vol)


# class Compose:
#     def __init__(self, transforms: list):
#         self.transforms = transforms

#     def __call__(self, y: np.ndarray):
#         for trns in self.transforms:
#             y = trns(y)
#         return y
