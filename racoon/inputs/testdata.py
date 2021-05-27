# FIXME:
class TestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, clip: np.ndarray, waveform_transforms=None):
        self.df = df
        self.clip = clip
        self.waveform_transforms = waveform_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        SR = 32000
        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)

        start_index = SR * start_seconds
        end_index = SR * end_seconds

        y = self.clip[start_index:end_index].astype(np.float32)

        y = np.nan_to_num(y)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        y = np.nan_to_num(y)

        return y, row_id
