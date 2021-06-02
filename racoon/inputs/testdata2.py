import torch.utils.data as torchdata
import numpy as np
import pandas as pd
import librosa
from audiomentations import *


# from config import CFG
# Modified

# # TODO: TTA
# class TestDataset(torchdata.Dataset):
#     def __init__(self, df: pd.DataFrame, clip: np.ndarray, chunks_len=[5, 30, 20]):
#         self.df = df
#         self.clip = clip
#         self.chunks_len = chunks_len

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx: int):

#         item_dict = {}

#         SR = 32000
#         sample = self.df.loc[idx, :]
#         row_id = sample.row_id
#         item_dict["row_id"] = row_id

#         for chunk_len in self.chunks_len:
#             chunk_h = (chunk_len - 5) // 2
#             end_seconds_c = min(int(sample.seconds + chunk_h + 1), 600)
#             start_seconds_c = max(int(end_seconds_c - 5 - chunk_h), 0)
#             start_index_c = SR * start_seconds_c
#             end_index_c = SR * end_seconds_c

#             y_c = self.clip[start_index_c:end_index_c].astype(np.float32)

#             # if self.wav_transfos is not None:
#             #     y_c = self.wav_transfos(y_c, 32000)

#             melspec_c = self.compute_melspec(y_c)
#             melspec_c = (melspec_c - melspec_c.mean()) / (melspec_c.std() + 1e-6)
#             melspec_c = (melspec_c - melspec_c.min()) / (
#                 melspec_c.max() - melspec_c.min() + 1e-6
#             )

#             image_c = melspec_c[np.newaxis, ...]
#             item_dict[f"{chunk_len}sec_mel"] = image_c

#         return item_dict

#     def compute_melspec(self, y):
#         """
#         Computes a mel-spectrogram and puts it at decibel scale
#         Arguments:
#             y {np array} -- signal
#             params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
#         Returns:
#             np array -- Mel-spectrogram
#         """
#         melspec = librosa.feature.melspectrogram(
#             y, sr=32000, n_mels=128, fmin=20, fmax=16000
#         )

#         melspec = librosa.power_to_db(melspec).astype(np.float32)
#         return melspec


class TestDataset(torchdata.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        clip: np.ndarray,
        chunks_len=[5, 30, 20],
        tta=10,
        background_datadir="/data2/minki/kaggle/birdclef-2021/background_soundscape",
    ):
        self.df = df
        self.clip = clip
        self.chunks_len = chunks_len
        self.tta = tta
        if self.tta > 0:
            self.wav_transfos = self.get_wav_transforms(background_datadir)
            self.spec_transfos = self.get_specaug_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        item_dict = {}

        SR = 32000
        sample = self.df.loc[idx, :]
        row_id = sample.row_id
        item_dict["row_id"] = row_id

        for chunk_len in self.chunks_len:
            chunk_h = (chunk_len - 5) // 2
            end_seconds_c = min(int(sample.seconds + chunk_h + 1), 600)
            start_seconds_c = max(int(end_seconds_c - 5 - chunk_h), 0)
            start_index_c = SR * start_seconds_c
            end_index_c = SR * end_seconds_c

            y_c = self.clip[start_index_c:end_index_c].astype(np.float32)

            if self.tta > 0:
                image_cs = []
                for _ in range(self.tta):
                    y_c_m = self.wav_transfos(y_c, 32000)
                    melspec_c = self.compute_melspec(y_c_m)
                    melspec_c = (melspec_c - melspec_c.mean()) / (
                        melspec_c.std() + 1e-6
                    )
                    melspec_c = (melspec_c - melspec_c.min()) / (
                        melspec_c.max() - melspec_c.min() + 1e-6
                    )
                    melspec_c = self.spec_transfos(melspec_c)

                    image_c = melspec_c[np.newaxis, ...]
                    image_cs.append(image_c)

                image_cs = np.concatenate(image_cs, axis=0)

                item_dict[f"{chunk_len}sec_mel"] = image_cs

            else:
                melspec_c = self.compute_melspec(y_c)
                melspec_c = (melspec_c - melspec_c.mean()) / (melspec_c.std() + 1e-6)
                melspec_c = (melspec_c - melspec_c.min()) / (
                    melspec_c.max() - melspec_c.min() + 1e-6
                )

                image_c = melspec_c[np.newaxis, ...]
                item_dict[f"{chunk_len}sec_mel"] = image_c

        return item_dict

    def compute_melspec(self, y):
        """
        Computes a mel-spectrogram and puts it at decibel scale
        Arguments:
            y {np array} -- signal
            params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
        Returns:
            np array -- Mel-spectrogram
        """
        melspec = librosa.feature.melspectrogram(
            y, sr=32000, n_mels=128, fmin=20, fmax=16000
        )

        melspec = librosa.power_to_db(melspec).astype(np.float32)
        return melspec

    def get_wav_transforms(self, background_datadir):
        """
        Returns the transformation to apply on waveforms
        Returns:
            Audiomentations transform -- Transforms
        """
        transforms = Compose(
            [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                AddGaussianSNR(max_SNR=0.5, p=0.5),
                AddBackgroundNoise(
                    sounds_path=background_datadir,
                    min_snr_in_db=0,
                    max_snr_in_db=2,
                    p=0.5,
                ),
                FrequencyMask(min_frequency_band=0.0, max_frequency_band=0.5, p=0.5),
                Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.5),
            ]
        )
        return transforms

    def get_specaug_transforms(self):
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
