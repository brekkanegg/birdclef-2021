import torch.utils.data as torchdata
import numpy as np
import pandas as pd
import librosa


from config import CFG

# FIXME: 10 --> 5
class TestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, clip: np.ndarray):
        self.df = df
        self.clip = clip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        SR = CFG.sample_rate
        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)

        start_index = SR * start_seconds
        end_index = SR * end_seconds

        y = self.clip[start_index:end_index].astype(np.float32)

        melspec = compute_melspec(y, CFG)
        melspec = (melspec - melspec.mean()) / (melspec.std() + 1e-6)
        melspec = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-6)

        image = melspec[np.newaxis, ...]

        return image, row_id


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
