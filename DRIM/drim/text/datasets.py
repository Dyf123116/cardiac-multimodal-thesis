import numpy as np
import pandas as pd
import torch
from ..datasets import _BaseDataset

TEXT_DIM = 768


class TextDataset(_BaseDataset):
    """
    从预加载的 text_embeddings.npy 按 text_row 读 (768,)。
    text_row 为 NaN 时返回零张量,mask=False。
    嵌入做 L2 归一化(和你 DrFuse 里一致)。
    """
    _cache = {}

    def __init__(self, dataframe, emb_path, return_mask=False):
        super().__init__(dataframe, return_mask)
        self.emb = self._load(emb_path)

    @classmethod
    def _load(cls, emb_path):
        if emb_path in cls._cache:
            return cls._cache[emb_path]
        emb = np.load(emb_path, allow_pickle=True).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        emb = emb / norms
        cls._cache[emb_path] = emb
        return emb

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        if not pd.isna(sample.TEXT) and not pd.isna(sample.text_row):
            out = torch.from_numpy(self.emb[int(sample.text_row)]).float()
            mask = True
        else:
            out = torch.zeros(TEXT_DIM).float()
            mask = False
        return (out, mask) if self.return_mask else out
