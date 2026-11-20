"""
cardiac_cohort_pipeline.py
==========================
心脏病队列提取 + EHR & ECG 多模态预处理流水线
基于 MIMIC-IV 2.2 (EHR) + MIMIC-IV-ECG 1.0 + MedFuse 提取方法

假设数据已在本地，目录结构:
  MIMIC_IV_ROOT/
    hosp/
      diagnoses_icd.csv.gz
      admissions.csv.gz
      patients.csv.gz
      labevents.csv.gz
    icu/
      icustays.csv.gz
      chartevents.csv.gz

  MIMIC_ECG_ROOT/
    machine_measurements.csv   (或 .csv.gz)
    record_list.csv
    files/
      p{subject_id_prefix}/p{subject_id}/s{study_id}/*.dat / *.hea

用法:
  python cardiac_cohort_pipeline.py \
      --mimic_root /data/mimiciv/2.2 \
      --ecg_root   /data/mimic-iv-ecg/1.0 \
      --output_dir ./output

  # 同时提取时序特征（chartevents + labevents，较慢）：
  python cardiac_cohort_pipeline.py ... --extract_timeseries

  # 同时提取原始 ECG 波形统计（需要 wfdb）：
  python cardiac_cohort_pipeline.py ... --extract_waveform_stats

  # 流式提取原始 ECG 波形并写入 HDF5（零内存累积，需要 wfdb + h5py）：
  python cardiac_cohort_pipeline.py ... --extract_waveforms_hdf5

  # 断点续传（跳过 HDF5 中已写入的 study_id）：
  python cardiac_cohort_pipeline.py ... --extract_waveforms_hdf5 --resume

依赖:
  pip install pandas numpy tqdm pyarrow
  pip install wfdb h5py   # 仅波形相关步骤需要
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  ICD 心脏病代码定义（ICD-9 + ICD-10）
# ─────────────────────────────────────────────────────────────────────────────

ICD9_CARDIAC_PREFIXES = [
    "410", "411", "412", "413", "414",  # 缺血性心脏病 / AMI
    "420", "421", "422", "423", "424",  # 心包/心内膜/心肌炎
    "425",                               # 心肌病
    "426", "427",                        # 传导障碍 / 心律失常
    "428",                               # 心力衰竭
    "429",                               # 心脏病其他
]

ICD10_CARDIAC_PREFIXES = [
    "I20", "I21", "I22", "I23", "I24", "I25",  # 缺血性心脏病
    "I30", "I31", "I32", "I33",                  # 心包/心内膜炎
    "I34", "I35", "I36", "I37", "I38",          # 瓣膜病
    "I40", "I41", "I42",                         # 心肌炎 / 心肌病
    "I44", "I45", "I46", "I47", "I48", "I49",  # 传导障碍 / 心律失常
    "I50",                                       # 心力衰竭
    "I51",                                       # 心脏病并发症
]


def icd_is_cardiac(code: str, version: str) -> bool:
    code = str(code).strip().upper()
    prefixes = ICD9_CARDIAC_PREFIXES if str(version) == "9" else ICD10_CARDIAC_PREFIXES
    return any(code.startswith(p) for p in prefixes)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  辅助：安全读取 csv / csv.gz
# ─────────────────────────────────────────────────────────────────────────────

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """自动处理 .csv 与 .csv.gz，文件不存在时给出明确报错。"""
    if not path.exists():
        # 尝试互换后缀
        alt = path.with_suffix(".csv.gz") if path.suffix == ".csv" else path.with_suffix(".csv")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"找不到文件（已尝试 .csv / .csv.gz）: {path}\n"
                f"请确认 --mimic_root / --ecg_root 路径正确。"
            )
    log.info(f"读取 {path} ...")
    return pd.read_csv(path, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  提取心脏病 subject_id
# ─────────────────────────────────────────────────────────────────────────────

def extract_cardiac_subjects(mimic_root: Path) -> pd.DataFrame:
    """
    读取 diagnoses_icd.csv.gz，筛选心脏病患者。
    每个 (subject_id, hadm_id) 只保留 seq_num 最小的心脏病诊断（主诊断）。

    返回列: [subject_id, hadm_id, primary_cardiac_icd, icd_version]
    """
    diag = read_csv(
        mimic_root / "hosp" / "diagnoses_icd.csv.gz",
        dtype={"icd_code": str, "icd_version": str},
    )
    log.info(f"  diagnoses 原始行数: {len(diag):,}")

    diag["is_cardiac"] = diag.apply(
        lambda r: icd_is_cardiac(r["icd_code"], r["icd_version"]), axis=1
    )
    cardiac = diag[diag["is_cardiac"]].copy()
    log.info(f"  心脏病诊断行数: {len(cardiac):,}")
    log.info(f"  唯一 subject_id 数: {cardiac['subject_id'].nunique():,}")

    primary = (
        cardiac.sort_values("seq_num")
        .groupby(["subject_id", "hadm_id"])
        .first()
        .reset_index()[["subject_id", "hadm_id", "icd_code", "icd_version"]]
        .rename(columns={"icd_code": "primary_cardiac_icd"})
    )
    return primary


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EHR 静态特征
# ─────────────────────────────────────────────────────────────────────────────

def build_static_features(
    mimic_root: Path,
    cardiac_df: pd.DataFrame,
) -> tuple:
    """
    合并 patients + admissions + icustays + cardiac_df，生成静态特征表。
    同时返回 icu_stays 供后续时序/ECG 对齐使用。

    返回: (static_df, icu_stays_df)
    """
    subject_ids = set(cardiac_df["subject_id"])
    hadm_ids    = set(cardiac_df["hadm_id"])

    patients = read_csv(mimic_root / "hosp" / "patients.csv.gz")
    patients = patients[patients["subject_id"].isin(subject_ids)]

    admissions = read_csv(
        mimic_root / "hosp" / "admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    admissions = admissions[admissions["hadm_id"].isin(hadm_ids)]

    icu = read_csv(
        mimic_root / "icu" / "icustays.csv.gz",
        parse_dates=["intime", "outtime"],
    )
    icu = icu[icu["subject_id"].isin(subject_ids)]
    log.info(f"  心脏病 ICU stays: {len(icu):,}")

    # race 字段在 MIMIC-IV 2.2 中为 race，旧版为 ethnicity，做兼容处理
    race_col = "race" if "race" in admissions.columns else "ethnicity"

    static = (
        patients[["subject_id", "gender", "anchor_age", "dod"]]
        .merge(
            admissions[[
                "subject_id", "hadm_id", "admittime", "dischtime",
                "deathtime", "admission_type", "insurance",
                "marital_status", race_col,
            ]],
            on="subject_id",
        )
        .merge(
            icu[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"]],
            on=["subject_id", "hadm_id"],
        )
        .merge(cardiac_df, on=["subject_id", "hadm_id"])
    )

    # 住院死亡标签
    static["in_hospital_mortality"] = (
        static["deathtime"].notna()
        & (static["deathtime"] <= static["dischtime"])
    ).astype(int)

    log.info(f"  静态特征行数: {len(static):,}")
    return static, icu


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EHR 时序特征（DrFuse / MedFuse 17 变量）
# ─────────────────────────────────────────────────────────────────────────────

CHART_FEATURES = {
    220045: "Heart Rate",
    220050: "Arterial BP systolic",
    220051: "Arterial BP diastolic",
    220052: "Arterial BP mean",
    220179: "Non-invasive BP systolic",
    220180: "Non-invasive BP diastolic",
    220181: "Non-invasive BP mean",
    220210: "Respiratory Rate",
    223761: "Temperature (F)",
    220277: "SpO2",
    220235: "pCO2",
    220224: "pO2",
    220228: "Hemoglobin",
    220546: "WBC",
    220602: "Chloride (serum)",
    220635: "Magnesium",
    220645: "Sodium (serum)",
}

PHYS_BOUNDS = {
    "Heart Rate":                (0,   300),
    "Arterial BP systolic":      (0,   300),
    "Arterial BP diastolic":     (0,   200),
    "Arterial BP mean":          (0,   250),
    "Non-invasive BP systolic":  (0,   300),
    "Non-invasive BP diastolic": (0,   200),
    "Non-invasive BP mean":      (0,   250),
    "Respiratory Rate":          (0,    60),
    "Temperature (F)":           (70,  120),
    "SpO2":                      (0,   100),
    "pCO2":                      (0,   200),
    "pO2":                       (0,   800),
    "Hemoglobin":                (0,    25),
    "WBC":                       (0,  1000),
    "Chloride (serum)":          (0,   200),
    "Magnesium":                 (0,    10),
    "Sodium (serum)":            (100, 200),
}

LAB_FEATURES = {
    50912: "Creatinine",
    50971: "Potassium",
    50983: "Sodium",
    50902: "Chloride",
    50882: "Bicarbonate",
    51221: "Hematocrit",
    51222: "Hemoglobin",
    51248: "MCH",
    51249: "MCHC",
    51250: "MCV",
    51265: "Platelet Count",
    51279: "RBC",
    51301: "WBC",
    50868: "Anion Gap",
    50931: "Glucose",
    51006: "Urea Nitrogen",
    50820: "pH",
}


def extract_chart_timeseries(
    mimic_root: Path, icu_stays: pd.DataFrame, hours: int = 48
) -> pd.DataFrame:
    """
    从 chartevents 提取前 `hours` 小时的 17 项时序特征。
    按小时分桶取均值，并过滤生理范围外的异常值。

    返回: DataFrame [stay_id, hour, feature_1, ..., feature_17]
    """
    stay_ids    = set(icu_stays["stay_id"])
    feature_ids = set(CHART_FEATURES.keys())
    intime_map  = icu_stays.set_index("stay_id")["intime"].to_dict()

    chunks = []
    for chunk in pd.read_csv(
        mimic_root / "icu" / "chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        dtype={"itemid": int},
        chunksize=5_000_000,
    ):
        sub = chunk[chunk["stay_id"].isin(stay_ids) & chunk["itemid"].isin(feature_ids)]
        if len(sub):
            chunks.append(sub)

    if not chunks:
        log.warning("chartevents 中未找到匹配数据，跳过。")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["feature"] = df["itemid"].map(CHART_FEATURES)
    df["intime"]  = df["stay_id"].map(intime_map)
    df["hour"]    = ((df["charttime"] - df["intime"]).dt.total_seconds() / 3600).astype(int)
    df = df[(df["hour"] >= 0) & (df["hour"] < hours)]

    # 生理范围过滤
    valid = pd.concat(
        [df[df["feature"].eq(feat) & df["valuenum"].between(lo, hi)]
         for feat, (lo, hi) in PHYS_BOUNDS.items()],
        ignore_index=True,
    )

    result = (
        valid.groupby(["stay_id", "hour", "feature"])["valuenum"]
        .mean()
        .unstack("feature")
        .reset_index()
    )
    log.info(f"  chart 时序 shape: {result.shape}")
    return result


def extract_lab_timeseries(
    mimic_root: Path, icu_stays: pd.DataFrame, hours: int = 48
) -> pd.DataFrame:
    """
    从 labevents 提取前 `hours` 小时的 lab 特征，以 ICU intime 为零点。

    返回: DataFrame [hadm_id, hour, lab_1, ..., lab_17]
    """
    hadm_ids    = set(icu_stays["hadm_id"].dropna().astype(int))
    feature_ids = set(LAB_FEATURES.keys())
    intime_map  = (
        icu_stays.dropna(subset=["hadm_id"])
        .assign(hadm_id=lambda d: d["hadm_id"].astype(int))
        .set_index("hadm_id")["intime"]
        .to_dict()
    )

    chunks = []
    for chunk in pd.read_csv(
        mimic_root / "hosp" / "labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        dtype={"itemid": int},
        chunksize=5_000_000,
    ):
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        sub = chunk[chunk["hadm_id"].isin(hadm_ids) & chunk["itemid"].isin(feature_ids)]
        if len(sub):
            chunks.append(sub)

    if not chunks:
        log.warning("labevents 中未找到匹配数据，跳过。")
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["feature"] = df["itemid"].map(LAB_FEATURES)
    df["intime"]  = df["hadm_id"].map(intime_map)
    df = df.dropna(subset=["intime"])
    df["hour"] = ((df["charttime"] - df["intime"]).dt.total_seconds() / 3600).astype(int)
    df = df[(df["hour"] >= 0) & (df["hour"] < hours) & df["valuenum"].ge(0)]

    result = (
        df.groupby(["hadm_id", "hour", "feature"])["valuenum"]
        .mean()
        .unstack("feature")
        .reset_index()
    )
    log.info(f"  lab 时序 shape: {result.shape}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ECG 维度
# ─────────────────────────────────────────────────────────────────────────────

def load_ecg_records(ecg_root: Path, subject_ids: set) -> pd.DataFrame:
    """读取 record_list，过滤至心脏病队列。"""
    records = read_csv(ecg_root / "record_list.csv")
    records["subject_id"] = records["subject_id"].astype(int)
    records = records[records["subject_id"].isin(subject_ids)].copy()
    log.info(f"  ECG records（心脏病队列）: {len(records):,}")
    return records


def load_ecg_measurements(ecg_root: Path, subject_ids: set) -> pd.DataFrame:
    """
    读取仪器自动测量结果（QRS、PR、QT、QTc 等结构化特征）。
    可作为轻量 ECG 特征直接输入 DrFuse ECG 编码器。
    """
    meas = read_csv(ecg_root / "machine_measurements.csv")
    meas["subject_id"] = meas["subject_id"].astype(int)
    meas = meas[meas["subject_id"].isin(subject_ids)].copy()
    log.info(f"  ECG machine_measurements shape: {meas.shape}")
    return meas


def align_ecg_to_icu(icu_stays: pd.DataFrame, ecg_records: pd.DataFrame) -> pd.DataFrame:
    """
    将 ECG 检查时间与 ICU stay 对齐。
    保留 ecg_time 落在 [intime - 24h, outtime + 24h] 范围内的记录。
    每个 stay_id 只取时间偏移绝对值最小的一条（DrFuse 策略：最近一张）。

    返回列: [subject_id, stay_id, hadm_id, study_id, ecg_time, hour_offset]
    """
    log.info("对齐 ECG records 与 ICU stays ...")
    ecg_records = ecg_records.copy()
    ecg_records["ecg_time"] = pd.to_datetime(ecg_records["ecg_time"])

    merged = ecg_records.merge(
        icu_stays[["subject_id", "stay_id", "hadm_id", "intime", "outtime"]],
        on="subject_id",
        how="inner",
    )
    mask = (
        (merged["ecg_time"] >= merged["intime"] - pd.Timedelta(hours=24))
        & (merged["ecg_time"] <= merged["outtime"] + pd.Timedelta(hours=24))
    )
    merged = merged[mask].copy()
    merged["hour_offset"] = (
        (merged["ecg_time"] - merged["intime"]).dt.total_seconds() / 3600
    ).round(1)

    # 每个 stay_id 取最近的一条 ECG
    merged = (
        merged.assign(abs_offset=merged["hour_offset"].abs())
        .sort_values("abs_offset")
        .drop_duplicates(subset=["stay_id"])
        .drop(columns="abs_offset")
        .reset_index(drop=True)
    )

    log.info(f"  对齐后 ECG-ICU records: {len(merged):,}  "
             f"（{merged['subject_id'].nunique():,} 个患者）")
    return merged


def extract_ecg_waveform_stats(
    ecg_root: Path, records: pd.DataFrame, max_records: int = None
) -> pd.DataFrame:
    """
    可选：读取原始 .dat/.hea 波形，提取 12 导联基础统计特征（均值/标准差/min/max）。
    大规模训练时建议改用深度嵌入（如 ECG-FM / ST-MEM）。

    需要: pip install wfdb
    """
    try:
        import wfdb
    except ImportError:
        log.warning("未安装 wfdb，跳过波形读取。请执行: pip install wfdb")
        return pd.DataFrame()

    if max_records:
        records = records.head(max_records)

    results = []
    for _, row in tqdm(records.iterrows(), total=len(records), desc="读取 ECG 波形"):
        sid    = str(int(row["subject_id"]))
        study  = str(int(row["study_id"]))
        prefix = sid[:4]
        # MIMIC-IV-ECG 路径: files/p{prefix5}/p{subject_id}/s{study_id}/{filename}
        filename  = row.get("path", study)
        full_path = ecg_root / "files" / f"p{prefix}" / f"p{sid}" / f"s{study}" / filename

        try:
            rec = wfdb.rdrecord(str(full_path))
            sig = rec.p_signal  # shape: (n_samples, n_leads)
            if sig is None:
                continue
            stat = {"subject_id": int(sid), "study_id": int(study)}
            for i, lead in enumerate(rec.sig_name):
                s = sig[:, i]
                stat[f"{lead}_mean"] = float(np.nanmean(s))
                stat[f"{lead}_std"]  = float(np.nanstd(s))
                stat[f"{lead}_min"]  = float(np.nanmin(s))
                stat[f"{lead}_max"]  = float(np.nanmax(s))
            results.append(stat)
        except Exception as e:
            log.debug(f"跳过 {full_path}: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  流式提取原始 ECG 波形 → HDF5（零内存累积）
# ─────────────────────────────────────────────────────────────────────────────
#
# HDF5 结构（ecg_waveforms.h5）:
#
#   /waveforms/{study_id}/signal      float32 (12, T)  12导联原始信号，T≈5000
#   /waveforms/{study_id}/lead_names  bytes   (12,)    导联名称
#   /waveforms/{study_id}.attrs[fs]   int              采样率（通常500Hz）
#   /waveforms/{study_id}.attrs[subject_id / stay_id]  关联键
#
#   /index/{列名}  向量数据集，记录每条成功写入的元信息
#     列: study_id, subject_id, stay_id, ecg_time, hour_offset, n_samples, fs
#
# 关键内存策略：
#   - 每次只读一条波形（wfdb.rdrecord），写完立即 del sig / del rec
#   - 不在任何 list/dict 中累积信号数组
#   - gzip 压缩（level=4），磁盘占用约为原始大小的 40-60%
#   - --resume 时跳过 HDF5 中已存在的 study_id，支持断点续传

def extract_waveforms_to_hdf5(
    ecg_root: Path,
    aligned: pd.DataFrame,
    out_path: Path,
    resume: bool = False,
) -> Path:
    """
    流式读取原始 ECG 波形，逐条写入 HDF5，全程不在内存中累积信号数组。

    参数
    ----
    ecg_root : MIMIC-IV-ECG 根目录
    aligned  : align_ecg_to_icu() 的输出，含 study_id/stay_id/subject_id 等列
    out_path : 输出 HDF5 文件路径（如 output/ecg_waveforms.h5）
    resume   : True 时跳过 HDF5 中已写入的 study_id，支持中断后续传
    """
    try:
        import wfdb
        import h5py
    except ImportError as e:
        raise ImportError(
            "请先安装依赖: pip install wfdb h5py\n"
            f"缺少模块: {e}"
        )

    n_total  = len(aligned)
    n_ok     = 0
    n_skip   = 0
    n_resume = 0
    index_rows = []   # 只存元数据（整数/字符串），不存信号

    # 断点续传：读取已写入的 study_id 集合
    existing_ids: set = set()
    if resume and out_path.exists():
        import h5py as _h5
        with _h5.File(out_path, "r") as _f:
            if "waveforms" in _f:
                existing_ids = set(_f["waveforms"].keys())
        log.info(f"  断点续传：发现已有 {len(existing_ids):,} 条记录，将跳过")

    mode = "a" if (resume and out_path.exists()) else "w"

    with h5py.File(out_path, mode) as hf:
        wf_grp = hf.require_group("waveforms")

        for _, row in tqdm(aligned.iterrows(), total=n_total, desc="写入 ECG → HDF5"):
            study_id   = str(int(row["study_id"]))
            subject_id = str(int(row["subject_id"]))
            stay_id = int(row["stay_id"]) if "stay_id" in row.index and pd.notna(row.get("stay_id")) else -1

            # 断点续传跳过
            if study_id in existing_ids:
                n_resume += 1
                continue

            # MIMIC-IV-ECG 路径: files/p{prefix5}/p{subject_id}/s{study_id}/{filename}
            prefix    = subject_id[:4]
            filename  = row.get("path", study_id)
            full_path = ecg_root / filename

            # ── 读取波形（单条，读完立即释放）────────────────────────────
            try:
                rec = wfdb.rdrecord(str(full_path))
            except Exception as e:
                log.debug(f"读取失败，跳过 study={study_id}: {e}")
                n_skip += 1
                continue

            sig = rec.p_signal   # ndarray (n_samples, n_leads) 或 None
            if sig is None or sig.ndim != 2 or sig.shape[1] == 0:
                n_skip += 1
                del rec
                continue

            n_samples, n_leads = sig.shape
            fs = int(rec.fs) if rec.fs else 500

            # ── 写入 HDF5（转置为 (n_leads, n_samples)，gzip 压缩）──────
            if study_id in wf_grp:
                n_resume += 1
                continue
            grp = wf_grp.create_group(study_id)
            grp.create_dataset(
                "signal",
                data=sig.T.astype(np.float32),          # (12, T)
                compression="gzip",
                compression_opts=4,
                chunks=(n_leads, min(1000, n_samples)),
            )
            grp.create_dataset(
                "lead_names",
                data=np.array(
                    [ln.encode("utf-8") for ln in rec.sig_name],
                    dtype=h5py.special_dtype(vlen=bytes),
                ),
            )
            grp.attrs["fs"]         = fs
            grp.attrs["subject_id"] = int(subject_id)
            grp.attrs["stay_id"]    = stay_id

            # ── 只把元数据（小对象）加入列表，信号立即释放 ───────────────
            index_rows.append({
                "study_id":    int(study_id),
                "subject_id":  int(subject_id),
                "stay_id":     stay_id,
                "ecg_time":    str(row.get("ecg_time", "")),
                "hour_offset": float(row.get("hour_offset", 0.0)),
                "n_samples":   n_samples,
                "n_leads":     n_leads,
                "fs":          fs,
            })

            del sig, rec   # 显式释放，尽早触发 GC
            n_ok += 1

        # ── 写入索引表（向量数据集，方便后续快速查询）───────────────────
        if index_rows:
            idx_df = pd.DataFrame(index_rows)

            # 断点续传时合并旧索引
            if resume and "index" in hf:
                old = {k: hf["index"][k][:] for k in hf["index"].keys()}
                old_df = pd.DataFrame(old)
                # bytes 列转回 str
                for c in old_df.columns:
                    if old_df[c].dtype.kind == "S":
                        old_df[c] = old_df[c].str.decode("utf-8")
                idx_df = pd.concat([old_df, idx_df], ignore_index=True)
                del hf["index"]

            idx_grp = hf.create_group("index")
            for col in idx_df.columns:
                arr = idx_df[col].values
                if arr.dtype == object or str(arr.dtype) == 'object':
                    arr = np.array([str(x).encode("utf-8") for x in arr])  # 强制转 bytes
                elif arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
            idx_grp.create_dataset(col, data=arr)

    size_gb = out_path.stat().st_size / 1024 ** 3
    log.info(
        f"→ ecg_waveforms.h5  "
        f"写入={n_ok:,} / 失败/跳过={n_skip:,} / 续传跳过={n_resume:,} / "
        f"文件大小={size_gb:.2f} GB"
    )
    return out_path



# ─────────────────────────────────────────────────────────────────────────────
# 6.  数据集划分（按 subject_id，8:1:1，与 DrFuse/MedFuse 一致）
# ─────────────────────────────────────────────────────────────────────────────

def split_subjects(
    subject_ids: set, train: float = 0.8, val: float = 0.1, seed: int = 42
) -> tuple:
    rng  = np.random.default_rng(seed)
    sids = np.array(sorted(subject_ids))
    rng.shuffle(sids)
    n       = len(sids)
    n_train = int(n * train)
    n_val   = int(n * val)
    return (
        set(sids[:n_train]),
        set(sids[n_train: n_train + n_val]),
        set(sids[n_train + n_val:]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  主流程
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    mimic_root = Path(args.mimic_root)
    ecg_root   = Path(args.ecg_root)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. 心脏病队列 ──────────────────────────────────────────────────────
    cardiac_df  = extract_cardiac_subjects(mimic_root)
    subject_ids = set(cardiac_df["subject_id"])
    hadm_ids    = set(cardiac_df["hadm_id"])

    cardiac_df.to_csv(out_dir / "cardiac_subjects.csv", index=False)
    log.info(f"→ cardiac_subjects.csv  "
             f"(subjects={len(subject_ids):,}, hadm={len(hadm_ids):,})")

    # ── 3. 静态 EHR 特征 ───────────────────────────────────────────────────
    static, icu_stays = build_static_features(mimic_root, cardiac_df)
    static.to_csv(out_dir / "static_features.csv", index=False)
    log.info(f"→ static_features.csv  (rows={len(static):,})")

    # ── 4. 时序 EHR 特征（可选）──────────────────────────────────────────
    if args.extract_timeseries:
        chart_ts = extract_chart_timeseries(mimic_root, icu_stays, hours=args.hours)
        if len(chart_ts):
            chart_ts.to_parquet(out_dir / "chart_timeseries.parquet", index=False)
            log.info(f"→ chart_timeseries.parquet  {chart_ts.shape}")

        lab_ts = extract_lab_timeseries(mimic_root, icu_stays, hours=args.hours)
        if len(lab_ts):
            lab_ts.to_parquet(out_dir / "lab_timeseries.parquet", index=False)
            log.info(f"→ lab_timeseries.parquet  {lab_ts.shape}")

    # ── 5. ECG 维度 ────────────────────────────────────────────────────────
    ecg_records = load_ecg_records(ecg_root, subject_ids)
    ecg_records.to_csv(out_dir / "ecg_records.csv", index=False)
    log.info(f"→ ecg_records.csv  (rows={len(ecg_records):,})")

    ecg_meas = load_ecg_measurements(ecg_root, subject_ids)
    if len(ecg_meas):
        ecg_meas.to_csv(out_dir / "ecg_measurements.csv", index=False)
        log.info(f"→ ecg_measurements.csv  {ecg_meas.shape}")

    # ECG ↔ ICU stay 时间对齐（每个 stay 取最近一条 ECG）
    aligned = align_ecg_to_icu(icu_stays, ecg_records)
    aligned.to_csv(out_dir / "ecg_icu_aligned.csv", index=False)
    log.info(f"→ ecg_icu_aligned.csv  (rows={len(aligned):,})")

    # 可选：流式提取原始 ECG 波形 → HDF5（零内存累积）
    if args.extract_waveforms_hdf5:
        extract_waveforms_to_hdf5(
            ecg_root = ecg_root,
            aligned  = ecg_records,
            out_path = out_dir / "ecg_waveforms.h5",
            resume   = args.resume,
        )

    # 可选：原始波形统计特征（轻量版，只存统计数字）
    if args.extract_waveform_stats:
        wf = extract_ecg_waveform_stats(
            ecg_root, ecg_records, max_records=args.waveform_max
        )
        if len(wf):
            wf.to_csv(out_dir / "ecg_waveform_stats.csv", index=False)
            log.info(f"→ ecg_waveform_stats.csv  {wf.shape}")

    # ── 6. 数据集划分 ──────────────────────────────────────────────────────
    train_ids, val_ids, test_ids = split_subjects(subject_ids)
    splits = pd.DataFrame([
        {
            "subject_id": sid,
            "split": "train" if sid in train_ids else ("val" if sid in val_ids else "test"),
        }
        for sid in subject_ids
    ])
    splits.to_csv(out_dir / "subject_splits.csv", index=False)
    log.info(
        f"→ subject_splits.csv  "
        f"train={len(train_ids):,} / val={len(val_ids):,} / test={len(test_ids):,}"
    )

    # ── 摘要 ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("预处理完成！输出文件:")
    for f in sorted(out_dir.glob("*")):
        log.info(f"  {f.name:<40} {f.stat().st_size / 1024:>8.1f} KB")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIMIC-IV 心脏病队列 EHR+ECG 预处理（本地数据处理，不含下载）"
    )
    parser.add_argument(
        "--mimic_root", required=True,
        help="MIMIC-IV 2.2 根目录，包含 hosp/ 和 icu/ 子目录",
    )
    parser.add_argument(
        "--ecg_root", required=True,
        help="MIMIC-IV-ECG 1.0 根目录，包含 record_list.csv 和 files/",
    )
    parser.add_argument(
        "--output_dir", default="./output",
        help="输出目录（默认: ./output）",
    )
    parser.add_argument(
        "--hours", type=int, default=48,
        help="提取 ICU 入院后前 N 小时的时序数据（默认 48，与 DrFuse/MedFuse 一致）",
    )
    parser.add_argument(
        "--extract_timeseries", action="store_true",
        help="提取 chartevents + labevents 时序特征（文件较大，耗时较长）",
    )
    parser.add_argument(
        "--extract_waveform_stats", action="store_true",
        help="读取原始 ECG 波形并提取 12 导联统计特征（需安装 wfdb）",
    )
    parser.add_argument(
        "--waveform_max", type=int, default=None,
        help="波形读取数量上限，调试时使用（默认全量）",
    )
    parser.add_argument(
        "--extract_waveforms_hdf5", action="store_true",
        help="流式提取原始 ECG 波形并写入 HDF5（零内存累积，需安装 wfdb + h5py）",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="HDF5 断点续传：跳过已写入的 study_id（与 --extract_waveforms_hdf5 配合使用）",
    )
    args = parser.parse_args()
    main(args)