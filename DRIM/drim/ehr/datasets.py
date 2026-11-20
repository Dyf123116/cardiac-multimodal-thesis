import pickle
import numpy as np
import pandas as pd
import torch
from ..datasets import _BaseDataset


class EHRDataset(_BaseDataset):
    """
    从预加载的 EHR pkl 按 stay_id 读 X(48,68)。
    EHR 恒存在,mask 始终 True。
    """
    _cache = {}  # 类级缓存,避免每个 split 重复加载 pkl

    def __init__(self, dataframe, ehr_dir, return_mask=False):
        super().__init__(dataframe, return_mask)
        self.stay2X = self._load_all(ehr_dir)

    @classmethod
    def _load_all(cls, ehr_dir):
        if ehr_dir in cls._cache:
            return cls._cache[ehr_dir]
        stay2X = {}
        for split in ["train", "val", "test"]:
            with open(f"{ehr_dir}/ehr_phenotyping_48h_{split}.pkl", "rb") as f:
                for r in pickle.load(f):
                    stay2X[int(r["stay_id"])] = r["X"].astype(np.float32)
        cls._cache[ehr_dir] = stay2X
        return stay2X

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        out = torch.from_numpy(self.stay2X[int(sample.stay_id)]).float()  # (48,68)
        mask = True
        return (out, mask) if self.return_mask else out
