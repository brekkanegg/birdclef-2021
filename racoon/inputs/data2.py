import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
import librosa

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

        self.period = period
        self.validation = validation
        self.wav_transfos = get_wav_transforms() if not self.validation else None
        self.spec_transfos = get_specaug_transforms() if not self.validation else None

        self.sample_len = CFG.period * CFG.sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        ebird_code = sample["primary_label"]

        y, sr = sf.read(self.datadir / ebird_code / wav_name)

        y = crop_or_pad(
            y,
            self.sample_len,
            sr=CFG.sample_rate,
            train=not self.validation,
            probs=None,
        )

        if self.wav_transfos is not None:
            y = self.wav_transfos(y, CFG.sample_rate)

        melspec = compute_melspec(y, CFG)
        melspec = (melspec - melspec.mean()) / (melspec.std() + 1e-6)
        melspec = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-6)

        if self.spec_transfos is not None:
            melspec = self.spec_transfos(melspec)

        image = melspec[np.newaxis, ...]

        labels = np.zeros(len(CFG.target_columns), dtype=float)

        # FIXME: secondary label
        labels[CFG.target_columns.index(ebird_code)] = 1.0

        if CFG.use_secondary_label:
            ebird_code_second = sample["secondary_labels"]
            if len(ebird_code_second) > 2:
                ebird_code_second = ebird_code_second.replace("'", "")[1:-1].split(", ")
                for c in ebird_code_second:
                    labels[CFG.target_columns.index(c)] = 0.5

        return {"image": image, "targets": labels}


def crop_or_pad(y, length, sr, train=True, probs=None):
    """
    Crops an array to a chosen length
    Arguments:
        y {1D np array} -- Array to crop
        length {int} -- Length of the crop
        sr {int} -- Sampling rate
    Keyword Arguments:
        train {bool} -- Whether we are at train time. If so, crop randomly, else return the beginning of y (default: {True})
        probs {None or numpy array} -- Probabilities to use to chose where to crop (default: {None})
    Returns:
        1D np array -- Cropped array
    """
    # Pad
    if len(y) <= length:
        y = np.concatenate([y, np.zeros(length - len(y))])

    # Crop
    else:
        if not train:
            start = 0
        elif probs is None:
            start = np.random.randint(len(y) - length)
        else:
            start = (
                np.random.choice(np.arange(len(probs)), p=probs) + np.random.random()
            )
            start = int(sr * (start))

        y = y[start : start + length]

    return y.astype(np.float32)


def get_wav_transforms():
    """
    Returns the transformation to apply on waveforms
    Returns:
        Audiomentations transform -- Transforms
    """
    transforms = Compose(
        [
            AddGaussianSNR(max_SNR=0.5, p=0.5),
            AddBackgroundNoise(
                sounds_path=CFG.background_datadir,
                min_snr_in_db=0,
                max_snr_in_db=2,
                p=0.5,
            ),
        ]
    )
    return transforms


def compute_melspec(y, params):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = librosa.feature.melspectrogram(
        y,
        sr=params.sample_rate,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=params.fmax,
    )

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec


def get_specaug_transforms():
    """
    Returns the transformation to apply on waveforms
    Returns:
        Audiomentations transform -- Transforms
    """
    transforms = SpecFrequencyMask(
        min_mask_fraction=0.03,
        max_mask_fraction=0.25,
        fill_mode="constant",
        fill_constant=0.0,
        p=0.5,
    )
    return transforms
