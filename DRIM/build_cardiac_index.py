"""
生成 DRIM 用的索引表 cardiac_drim.csv
每行一个 stay,列含各模态可用标记 + 6维标签 + split
"""
import pickle
import numpy as np
import pandas as pd

OUT = "/home/featurize/data/output"

# ── 1. 从三个 split 的 EHR pkl 收集 stay_id / label / split ──────────
rows = []
for split in ["train", "val", "test"]:
    with open(f"{OUT}/drfuse_input/ehr_phenotyping_48h_{split}.pkl", "rb") as f:
        recs = pickle.load(f)
    for r in recs:
        rows.append({
            "stay_id": int(r["stay_id"]),
            "name":    r["name"],
            "split":   split,
            **{f"label_{i}": float(r["label"][i]) for i in range(6)},
        })
df = pd.DataFrame(rows)
print(f"EHR 总样本: {len(df)}  (train/val/test = "
      f"{(df.split=='train').sum()}/{(df.split=='val').sum()}/{(df.split=='test').sum()})")

# ── 2. stay_id → hadm_id(从 static)────────────────────────────────
static = pd.read_csv(f"{OUT}/static_features.csv")
stay2hadm = dict(zip(static["stay_id"].astype(int), static["hadm_id"].astype(int)))
df["hadm_id"] = df["stay_id"].map(stay2hadm)

# ── 3. stay_id → study_id(从 ECG 对齐表,一个stay可能多条ECG取第一条)─
aligned = pd.read_csv(f"{OUT}/ecg_icu_aligned.csv")
stay2study = (aligned.dropna(subset=["stay_id", "study_id"])
              .drop_duplicates("stay_id")
              .set_index("stay_id")["study_id"].astype(int).to_dict())
df["study_id"] = df["stay_id"].map(stay2study)

# ── 4. 文本可用性(hadm_id 是否在 text_index 里)────────────────────
tidx = pd.read_csv(f"{OUT}/text_index.csv")
text_hadms = set(tidx["hadm_id"].astype(int))
# 同时建立 hadm_id → 行号(用于后续读 npy)
hadm2textrow = {int(h): i for i, h in enumerate(tidx["hadm_id"])}
df["text_row"] = df["hadm_id"].map(hadm2textrow)

# ── 5. 模态可用标记 ──────────────────────────────────────────────
df["EHR"]  = 1                                    # EHR 恒有
df["ECG"]  = df["study_id"].notna().astype("Int64")
df["TEXT"] = df["text_row"].notna().astype("Int64")

# ECG/TEXT 缺失的标记成 NaN(符合 DRIM 约定:NaN=缺失)
df.loc[df["ECG"]  == 0, "ECG"]  = np.nan
df.loc[df["TEXT"] == 0, "TEXT"] = np.nan

# ── 统计覆盖 ─────────────────────────────────────────────────────
print(f"ECG  可用: {df['ECG'].notna().sum()} ({df['ECG'].notna().mean():.1%})")
print(f"TEXT 可用: {df['TEXT'].notna().sum()} ({df['TEXT'].notna().mean():.1%})")
print(f"三模态全有: {((df['ECG'].notna()) & (df['TEXT'].notna())).sum()}")

# ── 保存 ─────────────────────────────────────────────────────────
out_path = f"{OUT}/cardiac_drim.csv"
df.to_csv(out_path, index=False)
print(f"→ 已保存 {out_path}  shape={df.shape}")
print(df.head(3).to_string())
