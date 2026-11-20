"""
drfuse_data_adapter.py
======================
将 cardiac_cohort_pipeline.py 的输出适配为 DrFuse 所需的输入格式。

现有输出 → DrFuse 需要的格式：

  chart_timeseries.parquet          ──►  ehr_phenotyping_48h_{split}.pkl
  （稀疏长表，有值的小时才有行）         （密集矩阵，每个 stay 一个 (48,34) 数组）

  ecg_waveforms.h5                  ──►  ECGHdf5Dataset（PyTorch Dataset）
  （原始信号 float32 (12,T)）            （归一化后 tensor (12,5000)，接口同 CXR Dataset）

用法：
  # 1. 生成 EHR pkl（只需运行一次）
  python drfuse_data_adapter.py \\
      --chart_parquet ./output/chart_timeseries.parquet \\
      --static_csv    ./output/static_features.csv \\
      --splits_csv    ./output/subject_splits.csv \\
      --cardiac_csv   ./output/cardiac_subjects.csv \\
      --out_dir       ./output/drfuse_input

  # 2. ECGHdf5Dataset 在训练脚本里直接 import 使用（见文件末尾示例）

依赖：
  pip install pandas numpy torch h5py pyarrow
"""

import argparse
import logging
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 标签定义：9个心脏病亚型（替换 DrFuse 原版 25 disease phenotypes）合并成 6个心脏病亚型
# ─────────────────────────────────────────────────────────────────────────────

CARDIAC_LABEL_GROUPS = {
    "Ischemic HD":      ["I20","I21","I22","I23","I24","I25",
                         "410","411","412","413","414"],
    "Heart Failure":    ["I50","428"],
    "Arrhythmia":       ["I44","I45","I46","I47","I48","I49",
                         "426","427"],
    "Cardiomyopathy":   ["I42","425"],
    "Endocarditis":     ["I33","421"],
    "Inflammatory & Other": ["I30","I31","I32","I34","I35","I36",
                              "I37","I38","I40","I41","I51",
                              "420","421","422","423","424","429"],
}
LABEL_NAMES = list(CARDIAC_LABEL_GROUPS.keys())
N_LABELS    = len(LABEL_NAMES)


def icd_to_label_vector(icd_code: str) -> np.ndarray:
    """将一个 ICD 代码映射为 N_LABELS 维多热标签向量。"""
    vec  = np.zeros(N_LABELS, dtype=np.float32)
    code = str(icd_code).strip().upper()
    for i, prefixes in enumerate(CARDIAC_LABEL_GROUPS.values()):
        if any(code.startswith(p) for p in prefixes):
            vec[i] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Part 1  EHR 适配
# parquet（稀疏长表）→ pkl（密集矩阵列表），格式与 MedFuse/DrFuse 兼容
# ─────────────────────────────────────────────────────────────────────────────

EHR_FEATURE_COLS = [
    "Heart Rate", "Arterial BP systolic", "Arterial BP diastolic",
    "Arterial BP mean", "Non-invasive BP systolic", "Non-invasive BP diastolic",
    "Non-invasive BP mean", "Respiratory Rate", "Temperature (F)", "SpO2",
    "pCO2", "pO2", "Hemoglobin", "WBC", "Chloride (serum)",
    "Magnesium", "Sodium (serum)",
]
N_FEATURES = len(EHR_FEATURE_COLS)   # 17
N_HOURS    = 48                       # DrFuse 默认窗口
# 每个特征的归一化范围（与 PHYS_BOUNDS 对应）
EHR_NORM_BOUNDS = [
    (0, 300), (0, 300), (0, 200), (0, 250),
    (0, 300), (0, 200), (0, 250), (0, 60),
    (70, 120), (0, 100), (0, 200), (0, 800),
    (0, 25), (0, 1000), (0, 200), (0, 10), (100, 200),
]
# 实验室变量（17个，与 lab_timeseries.parquet 列名对应）
LAB_FEATURE_COLS = [
    "Anion Gap", "Bicarbonate", "Chloride", "Creatinine", "Glucose",
    "Hematocrit", "Hemoglobin", "MCH", "MCHC", "MCV",
    "Platelet Count", "Potassium", "RBC", "Sodium", "Urea Nitrogen",
    "WBC", "pH",
]
N_LAB_FEATURES = len(LAB_FEATURE_COLS)  # 17

LAB_NORM_BOUNDS = [
    (0, 40),    # Anion Gap
    (0, 50),    # Bicarbonate
    (70, 130),  # Chloride
    (0, 15),    # Creatinine
    (0, 600),   # Glucose
    (0, 60),    # Hematocrit
    (0, 25),    # Hemoglobin
    (15, 40),   # MCH
    (25, 40),   # MCHC
    (60, 120),  # MCV
    (0, 800),   # Platelet Count
    (0, 10),    # Potassium
    (0, 8),     # RBC
    (100, 180), # Sodium
    (0, 200),   # Urea Nitrogen
    (0, 100),   # WBC
    (6.5, 8.0), # pH
]

def _sparse_to_dense(stay_df: pd.DataFrame, lab_df: pd.DataFrame = None):
    # chart 部分（原来的逻辑）
    chart_values = np.zeros((N_HOURS, N_FEATURES), dtype=np.float32)
    chart_mask   = np.zeros((N_HOURS, N_FEATURES), dtype=np.float32)
    for _, row in stay_df.iterrows():
        h = int(row["hour"])
        if not (0 <= h < N_HOURS):
            continue
        for j, feat in enumerate(EHR_FEATURE_COLS):
            v = row.get(feat, np.nan)
            if pd.notna(v):
                lo, hi = EHR_NORM_BOUNDS[j]
                v_norm = np.clip((float(v) - lo) / (hi - lo + 1e-8), 0.0, 1.0)
                chart_values[h, j] = v_norm
                chart_mask[h, j]   = 1.0

    # lab 部分（新增）
    lab_values = np.zeros((N_HOURS, N_LAB_FEATURES), dtype=np.float32)
    lab_mask   = np.zeros((N_HOURS, N_LAB_FEATURES), dtype=np.float32)
    if lab_df is not None:
        for _, row in lab_df.iterrows():
            h = int(row["hour"])
            if not (0 <= h < N_HOURS):
                continue
            for j, feat in enumerate(LAB_FEATURE_COLS):
                v = row.get(feat, np.nan)
                if pd.notna(v):
                    lo, hi = LAB_NORM_BOUNDS[j]
                    v_norm = np.clip((float(v) - lo) / (hi - lo + 1e-8), 0.0, 1.0)
                    lab_values[h, j] = v_norm
                    lab_mask[h, j]   = 1.0

    # 拼接：[chart值, lab值, chart_mask, lab_mask] = (48, 68)
    X = np.concatenate([chart_values, lab_values, chart_mask, lab_mask], axis=1)
    ts = np.arange(N_HOURS, dtype=np.int64)
    return X, ts

def convert_ehr_to_pkl(
    chart_parquet: Path,
    lab_parquet:   Path,
    static_csv:    Path,
    splits_csv:    Path,
    cardiac_csv:   Path,
    out_dir:       Path,
):
    """
    生成三个 pkl 文件：
      ehr_phenotyping_48h_train.pkl
      ehr_phenotyping_48h_val.pkl
      ehr_phenotyping_48h_test.pkl

    每个文件是一个 list of dict：
      {
        "stay_id" : int,
        "X"       : ndarray (48, 34)      float32  值 + mask
        "ts"      : ndarray (48,)         int64
        "label"   : ndarray (N_LABELS,)   float32  多热标签
        "name"    : str  "{subject_id}_{stay_id}"  DrFuse 用于索引
      }
    """
    log.info("读取输入文件 ...")
    chart   = pd.read_parquet(chart_parquet)
    lab = pd.read_parquet(lab_parquet)
    static  = pd.read_csv(static_csv)
    splits  = pd.read_csv(splits_csv)
    cardiac = pd.read_csv(cardiac_csv, dtype={"primary_cardiac_icd": str})

    # stay_id → label 向量
    stay_label = (
        cardiac
        .merge(static[["subject_id", "hadm_id", "stay_id"]], on=["subject_id", "hadm_id"])
        [["stay_id", "primary_cardiac_icd"]]
        .drop_duplicates("stay_id")
    )
    stay_to_label = {
        int(r["stay_id"]): icd_to_label_vector(r["primary_cardiac_icd"])
        for _, r in stay_label.iterrows()
    }

    # subject_id → split
    sid_to_split = dict(zip(splits["subject_id"].astype(int), splits["split"]))

    # stay_id → subject_id
    stay_to_subject = dict(
        zip(static["stay_id"].astype(int), static["subject_id"].astype(int))
    )

    buckets = {"train": [], "val": [], "test": []}
    n_missing_label = 0
    stay_to_hadm = dict(
        zip(static["stay_id"].astype(int), static["hadm_id"].astype(int))
    )
    # 按 hadm_id 分组 lab，加速查找
    lab_by_hadm = {hid: g for hid, g in lab.groupby("hadm_id")}

    log.info(f"转换 {chart['stay_id'].nunique():,} 个 stay ...")
    for stay_id, grp in chart.groupby("stay_id"):
        stay_id    = int(stay_id)
        subject_id = stay_to_subject.get(stay_id)
        if subject_id is None:
            continue
        split = sid_to_split.get(subject_id)
        if split not in buckets:
            continue
        label = stay_to_label.get(stay_id)
        if label is None:
            n_missing_label += 1
            continue

        hadm_id = stay_to_hadm.get(stay_id)
        lab_grp = lab_by_hadm.get(hadm_id) if hadm_id is not None else None
        X, ts = _sparse_to_dense(grp, lab_grp)
        buckets[split].append({
            "stay_id": stay_id,
            "X":       X,
            "ts":      ts,
            "label":   label,
            "name":    f"{subject_id}_{stay_id}",
        })

    if n_missing_label:
        log.warning(f"  {n_missing_label} 个 stay 没有对应标签，已跳过")

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, records in buckets.items():
        out_path = out_dir / f"ehr_phenotyping_48h_{split}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(records, f, protocol=4)
        log.info(f"→ {out_path.name}  ({len(records):,} stays)")

    # 保存标签名称（DrFuse 用 label_names 初始化分类头）
    with open(out_dir / "label_names.pkl", "wb") as f:
        pickle.dump(LABEL_NAMES, f)
    log.info(f"→ label_names.pkl  {LABEL_NAMES}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2  ECG 适配
# ECGHdf5Dataset：替换 DrFuse 原版的 CXR Dataset
# ─────────────────────────────────────────────────────────────────────────────

ECG_LEADS   = 12
ECG_SAMPLES = 5000   # 10s @ 500Hz，MIMIC-IV-ECG 标准长度


def _normalize_signal(sig: np.ndarray) -> np.ndarray:
    """
    Per-lead 零均值单位方差标准化。
    sig : (12, T)  float32
    返回: (12, T)  float32
    """
    mean = sig.mean(axis=1, keepdims=True)          # (12, 1)
    std  = sig.std(axis=1, keepdims=True) + 1e-8    # 防止除零
    return ((sig - mean) / std).astype(np.float32)


def _pad_or_crop(sig: np.ndarray, target: int) -> np.ndarray:
    """
    将信号长度统一到 target 采样点。
    sig : (12, T)
    返回: (12, target)
    """
    T = sig.shape[1]
    if T >= target:
        return sig[:, :target]
    # 右侧补零
    pad = np.zeros((sig.shape[0], target - T), dtype=np.float32)
    return np.concatenate([sig, pad], axis=1)


class ECGHdf5Dataset(Dataset):
    """
    从 ecg_waveforms.h5 读取 ECG 波形。

    __getitem__ 返回格式与 DrFuse 原版 CXR Dataset 完全对齐：
      (ecg_tensor, label_tensor, has_ecg, name)

      ecg_tensor  : torch.FloatTensor  (12, 5000)   归一化后的 12 导联信号
      label_tensor: torch.FloatTensor  (N_LABELS,)  多热标签
      has_ecg     : int  1=有ECG，0=缺失（缺失时 ecg_tensor 全零）
      name        : str  "{subject_id}_{stay_id}"

    DrFuse 原版对应字段：
      (image_tensor (3,224,224), label_tensor, has_cxr, name)

    所以在 DrFuse 的 models/drfuse.py 里只需把
      CXREncoder(输入 3×224×224) → ECGEncoder(输入 12×5000)
      其余融合逻辑、missing modality 处理完全不变。

    参数
    ----
    hdf5_path    : ecg_waveforms.h5 路径
    ehr_pkl_path : ehr_phenotyping_48h_{split}.pkl 路径
                   用于对齐顺序，保证 EHR 和 ECG 的索引一一对应
    aug_missing_ratio : 训练时随机丢弃 ECG 的比例（DrFuse missing augmentation）
    is_train     : True 时启用 missing augmentation
    """

    CLASSES = LABEL_NAMES
    def __init__(
        self,
        hdf5_path:         Path,
        ehr_pkl_path:      Path,
        aligned_csv:       Path,        # ← 新增，传入 ecg_icu_aligned.csv 路径
        text_emb_path:     Path  = None,   # ← 新增
        text_index_path:   Path  = None,
        aug_missing_ratio: float = 0.0,
        is_train:          bool  = False,
        paired_only:       bool  = False,
        ehr_feat_missing_ratio: float = 0.0,   # ← EHR 特征级缺失增强
    ):
        self.hdf5_path         = Path(hdf5_path)
        self.aug_missing_ratio = aug_missing_ratio
        self.is_train          = is_train
        self.ehr_feat_missing_ratio = ehr_feat_missing_ratio

        with open(ehr_pkl_path, "rb") as f:
            ehr_records = pickle.load(f)

    # 从 ecg_icu_aligned.csv 建立 stay_id → study_id 映射
        self.stay_to_study: dict = {}
        if Path(aligned_csv).exists():
            aligned = pd.read_csv(aligned_csv)
            for _, row in aligned.iterrows():
                self.stay_to_study[int(row["stay_id"])] = str(int(row["study_id"]))
                # stay_id → hadm_id：必须覆盖全部样本，不能只用 aligned_csv
        # （aligned_csv 只含有 ECG 的记录，会导致「无 ECG ⇒ 无文本」的伪关联）
        self.stay_to_hadm = {}
        _drim_csv = Path('/home/featurize/data/output/cardiac_drim.csv')
        if _drim_csv.exists():
            _d = pd.read_csv(_drim_csv, usecols=['stay_id', 'hadm_id'])
            self.stay_to_hadm = dict(zip(_d['stay_id'].astype(int),
                                         _d['hadm_id'].astype(int)))
        elif Path(aligned_csv).exists():   # 兜底，保持旧行为
            for _, row in aligned.iterrows():
                self.stay_to_hadm[int(row["stay_id"])] = int(row["hadm_id"])

                
    # 加载文本嵌入（hadm_id → 768维向量）
        self.text_emb      = None
        self.hadm_to_textidx = {}
        if text_emb_path is not None and Path(text_emb_path).exists():
            self.text_emb = np.load(text_emb_path, allow_pickle=True)  # (N, 768)
            _norms = np.linalg.norm(self.text_emb, axis=1, keepdims=True)
            _norms[_norms == 0] = 1e-6
            self.text_emb = (self.text_emb / _norms).astype(np.float32)
            text_idx = pd.read_csv(text_index_path)
            for i, hid in enumerate(text_idx['hadm_id']):
                self.hadm_to_textidx[int(hid)] = i
            log.info(f"  文本嵌入加载完成: {self.text_emb.shape}")

    # paired_only：只保留有 ECG 的 stay
        if paired_only:
            ehr_records = [r for r in ehr_records
                           if r["stay_id"] in self.stay_to_study]

        self.ehr_records = ehr_records
        self.stay_ids    = [r["stay_id"] for r in ehr_records]

        n_paired = sum(1 for s in self.stay_ids if s in self.stay_to_study)
        log.info(
            f"ECGHdf5Dataset [{ehr_pkl_path.stem}]  "
            f"{'paired_only ' if paired_only else ''}"
            f"total={len(self.stay_ids):,}  "
            f"paired(有ECG)={n_paired:,}  "
            f"partial(无ECG)={len(self.stay_ids)-n_paired:,}"
        )
    def __len__(self):
        return len(self.stay_ids)

    def __getitem__(self, idx: int):
        ehr_rec  = self.ehr_records[idx]
        stay_id  = ehr_rec["stay_id"]
        name     = ehr_rec["name"]

    # EHR 数据（保持 numpy，和原版一致）
        x          = ehr_rec["X"]           # ndarray (48, 68): 17ICU值+17化验值+17ICU mask+17化验mask

        # ── EHR 特征级缺失增强（仅训练时）────────────────────────────
        # 在【已观测】位置随机置零（数值与 mask 同时置 0），每次取样重新随机，
        # 使模型每个 epoch 见到不同的特征缺失模式。
        # 依据 NAIM 原文的正则化设计；DrFuse 原有的 aug_missing_ratio 只作用于
        # ECG/Text，EHR 特征级此前无任何增强。
        if self.is_train and self.ehr_feat_missing_ratio > 0:
            x = x.copy()
            _n = x.shape[1] // 2
            _obs = x[:, _n:] > 0.5
            _drop = _obs & (np.random.rand(*_obs.shape) < self.ehr_feat_missing_ratio)
            x[:, :_n][_drop] = 0.0
            x[:, _n:][_drop] = 0.0
        y          = ehr_rec["label"]       # ndarray (N_LABELS,)
        seq_length = int(ehr_rec["ts"][-1]) + 1  # 原版用 seq_length 表示有效时间步

        study_id = self.stay_to_study.get(stay_id)
        if self.is_train and study_id is not None:
            if np.random.rand() < self.aug_missing_ratio:
                study_id = None

        if study_id is None:
            ecg_tensor = torch.zeros(ECG_LEADS, ECG_SAMPLES, dtype=torch.float32)
            ecg_avail  = 0
        else:
            try:
                with h5py.File(self.hdf5_path, "r") as hf:
                    sig = hf["waveforms"][study_id]["signal"][:]
            except Exception:
                ecg_tensor = torch.zeros(ECG_LEADS, ECG_SAMPLES, dtype=torch.float32)
                ecg_avail  = 0
            else:
                sig = _pad_or_crop(sig, ECG_SAMPLES)
                sig = _normalize_signal(sig)
                if np.isnan(sig).any() or np.isinf(sig).any():
                    ecg_tensor = torch.zeros(ECG_LEADS, ECG_SAMPLES, dtype=torch.float32)
                    ecg_avail  = 0
                else:
                    ecg_tensor = torch.FloatTensor(sig)
                    ecg_avail  = 1
        # ── Text ─────────────────────────────────────────────────────────────
        hadm_id = self.stay_to_hadm.get(stay_id, -1)
        text_idx = self.hadm_to_textidx.get(hadm_id)
        if self.is_train and text_idx is not None:
            if np.random.rand() < self.aug_missing_ratio:
                text_idx = None

        if text_idx is None or self.text_emb is None:
            text_tensor = torch.zeros(768, dtype=torch.float32)
            text_avail  = 0
        else:
            text_tensor = torch.FloatTensor(self.text_emb[text_idx])
            text_avail  = 1

        # avail 向量：[EHR=1, ECG=0/1, Text=0/1]
        avail = np.array([1, ecg_avail, text_avail], dtype=np.float32)

        return x, ecg_tensor, text_tensor, y, seq_length, avail

# ─────────────────────────────────────────────────────────────────────────────
# Part 3  修改 DrFuse main.py 的最小示例
# ─────────────────────────────────────────────────────────────────────────────

def ecg_collate_fn(batch):
    xs, ecgs, texts, ys, seq_lengths, avails = zip(*batch)
    return (
        np.stack(xs),                               # (B, 48, 34)
        torch.stack(ecgs),                          # (B, 12, 5000)
        torch.stack(texts),                         # (B, 768)
        np.stack(ys),                               # (B, N_LABELS)
        np.array(seq_lengths, dtype=np.int64),      # (B,)
        np.stack(avails).astype(np.float32),        # (B, 3)
    )

DRFUSE_MAIN_PATCH = """
# ── 在 drfuse/main.py 中替换 CXR 相关部分 ──────────────────────────────────

# 原版（删除）：
# from utils import EHRDiscretizer, EHRNormalizer, get_ehr_datasets, load_cxr_ehr
# dataloaders = load_cxr_ehr(...)

# 替换为：
from drfuse_data_adapter import ECGHdf5Dataset, LABEL_NAMES
from torch.utils.data import DataLoader

data_root    = Path(args.data_dir)      # 指向 ./output/drfuse_input/
hdf5_path    = Path(args.hdf5_path)     # 指向 ./output/ecg_waveforms.h5

# EHR Dataset（DrFuse 原版，读取 pkl，不变）
ehr_datasets = get_ehr_datasets(
    discretizer, normalizer,
    ehr_data_dir       = data_root,
    ehr_pkl_fpath_train= data_root / "ehr_phenotyping_48h_train.pkl",
    ehr_pkl_fpath_val  = data_root / "ehr_phenotyping_48h_val.pkl",
    ehr_pkl_fpath_test = data_root / "ehr_phenotyping_48h_test.pkl",
)

# ECG Dataset（替换 CXR）
ecg_train = ECGHdf5Dataset(
    hdf5_path         = hdf5_path,
    ehr_pkl_path      = data_root / "ehr_phenotyping_48h_train.pkl",
    aug_missing_ratio = args.aug_missing_ratio,
    is_train          = True,
)
ecg_val  = ECGHdf5Dataset(hdf5_path, data_root / "ehr_phenotyping_48h_val.pkl")
ecg_test = ECGHdf5Dataset(hdf5_path, data_root / "ehr_phenotyping_48h_test.pkl")

train_dl = DataLoader(ecg_train, batch_size=args.batch_size,
                      shuffle=True,  num_workers=args.num_workers)
val_dl   = DataLoader(ecg_val,   batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers)
test_dl  = DataLoader(ecg_test,  batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers)

# ── 在 drfuse/models/drfuse.py 中替换 CXREncoder ───────────────────────────

# 原版 CXREncoder（删除，ResNet50 输入 3×224×224）：
# class CXREncoder(nn.Module): ...

# 替换为 1D-CNN ECG 编码器（输入 12×5000，输出 hidden_size）：
class ECGEncoder(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12,  64, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128,256, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256,256, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(256, hidden_size)

    def forward(self, x):        # x: (B, 12, 5000)
        return self.proj(self.net(x).squeeze(-1))   # → (B, hidden_size)

# DrFuse 初始化时把 self.cxr_encoder = CXREncoder(...) 改为：
# self.cxr_encoder = ECGEncoder(hidden_size=args.hidden_size)
# 其余所有融合代码（disentangle、disease-wise attention、logit pooling）完全不动
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DrFuse 数据适配器：EHR parquet → pkl，ECG HDF5 Dataset"
    )
    parser.add_argument("--chart_parquet", required=True,
                        help="chart_timeseries.parquet 路径")
    parser.add_argument("--lab_parquet",   required=True,
                        help="lab_timeseries.parquet 路径")
    parser.add_argument("--static_csv",    required=True,
                        help="static_features.csv 路径")
    parser.add_argument("--splits_csv",    required=True,
                        help="subject_splits.csv 路径")
    parser.add_argument("--cardiac_csv",   required=True,
                        help="cardiac_subjects.csv 路径")
    parser.add_argument("--out_dir",       required=True,
                        help="输出目录（存放生成的 pkl 文件）")
    args = parser.parse_args()

    convert_ehr_to_pkl(
        chart_parquet = Path(args.chart_parquet),
        lab_parquet   = Path(args.lab_parquet),
        static_csv    = Path(args.static_csv),
        splits_csv    = Path(args.splits_csv),
        cardiac_csv   = Path(args.cardiac_csv),
        out_dir       = Path(args.out_dir),
    )

    print("\n" + "=" * 60)
    print("EHR pkl 生成完成。")
    print("ECG 侧：在训练脚本里直接 import ECGHdf5Dataset 使用。")
    print("修改 DrFuse 代码参考：")
    print(DRFUSE_MAIN_PATCH)


if __name__ == "__main__":
    main()