import h5py
import numpy as np
import pandas as pd
import torch
from ..datasets import _BaseDataset

ECG_LEADS = 12
ECG_SAMPLES = 5000


def _normalize_signal(sig):
    # per-lead 标准化,和你 DrFuse 里一致
    mean = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True) + 1e-8
    return (sig - mean) / std


def _pad_or_crop(sig, target=ECG_SAMPLES):
    if sig.shape[1] >= target:
        return sig[:, :target]
    pad = np.zeros((sig.shape[0], target - sig.shape[1]), dtype=sig.dtype)
    return np.concatenate([sig, pad], axis=1)


class ECGDataset(_BaseDataset):
    """
    从 h5 按 study_id 读波形 (12,5000)。
    study_id 为 NaN 时返回零张量,mask=False。
    """
    def __init__(self, dataframe, hdf5_path, return_mask=False):
        super().__init__(dataframe, return_mask)
        self.hdf5_path = hdf5_path

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        if not pd.isna(sample.ECG) and not pd.isna(sample.study_id):
            try:
                with h5py.File(self.hdf5_path, "r") as hf:
                    sig = hf["waveforms"][str(int(sample.study_id))]["signal"][:]
                sig = _pad_or_crop(sig, ECG_SAMPLES)
                sig = _normalize_signal(sig)
                if np.isnan(sig).any() or np.isinf(sig).any():
                    out, mask = torch.zeros(ECG_LEADS, ECG_SAMPLES), False
                else:
                    out, mask = torch.from_numpy(sig).float(), True
            except Exception:
                out, mask = torch.zeros(ECG_LEADS, ECG_SAMPLES), False
        else:
            out, mask = torch.zeros(ECG_LEADS, ECG_SAMPLES), False
        return (out, mask) if self.return_mask else out
