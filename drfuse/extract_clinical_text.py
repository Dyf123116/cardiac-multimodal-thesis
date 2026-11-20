"""
extract_clinical_text.py
========================
从 MIMIC-IV-Note 提取心脏病队列的安全文本特征：
  - 出院小结：Chief Complaint + History of Present Illness
              + Past Medical History + Family History + Physical Exam
  - 放射报告：FINDINGS 部分（去掉 IMPRESSION）

输出: text_features.parquet
  列: hadm_id, subject_id, discharge_text, radiology_text, combined_text
"""

import re
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 安全章节定义
# ─────────────────────────────────────────────────────────────────────────────

# 出院小结：允许提取的章节（不含诊断结论）
SAFE_DISCHARGE_SECTIONS = [
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
    "Family History",
    "Physical Exam",
]

# 出院小结：禁止提取的章节（含诊断或治疗结论）
BLOCKED_DISCHARGE_SECTIONS = [
    "Major Surgical or Invasive Procedure",
    "Brief Hospital Course",
    "Hospital Course",
    "Discharge Diagnosis",
    "Discharge Diagnoses",
    "Discharge Condition",
    "Discharge Instructions",
    "Discharge Medications",
    "Pertinent Results",
    "Medications on Admission",
    "Followup Instructions",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. 出院小结提取函数
# ─────────────────────────────────────────────────────────────────────────────

def extract_discharge_text(text: str) -> str:
    """
    从出院小结里只提取安全章节的内容。

    章节格式：
      章节名:\n内容\n\n下一章节名:
    或
      章节名:\n内容\n\n[空行]

    策略：
      1. 找到所有章节的起止位置
      2. 只保留 SAFE_DISCHARGE_SECTIONS 里的章节内容
      3. 拼接成一段文本
    """
    if not isinstance(text, str):
        return ""

    # 构建所有已知章节名的正则（用于定位章节边界）
    all_sections = SAFE_DISCHARGE_SECTIONS + BLOCKED_DISCHARGE_SECTIONS
    # 转义特殊字符，构建匹配模式
    section_pattern = "|".join(
        re.escape(s) for s in sorted(all_sections, key=len, reverse=True)
    )
    # 找到所有章节标题的位置
    header_re = re.compile(
        rf"^({section_pattern})\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    matches = list(header_re.finditer(text))
    if not matches:
        # 没找到任何章节标题，尝试简单关键词匹配
        return _fallback_extract(text)

    # 逐章节提取
    extracted = []
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # 判断是否是安全章节
        is_safe = any(
            section_name.lower() == s.lower()
            for s in SAFE_DISCHARGE_SECTIONS
        )
        if is_safe and content:
            extracted.append(f"{section_name}: {content}")

    return "\n\n".join(extracted)


def _fallback_extract(text: str) -> str:
    """
    章节标题格式不规范时的备用提取方案：
    直接用关键词匹配，截取到第一个危险章节之前。
    """
    # 找到第一个危险章节的位置
    danger_keywords = [
        r"Brief Hospital Course",
        r"Hospital Course",
        r"Discharge Diagnos",
        r"Discharge Medications",
        r"Pertinent Results",
    ]
    cutoff = len(text)
    for kw in danger_keywords:
        m = re.search(kw, text, re.IGNORECASE)
        if m and m.start() < cutoff:
            cutoff = m.start()

    safe_text = text[:cutoff].strip()

    # 去掉头部的患者信息行（Name、Unit No 等）
    safe_text = re.sub(
        r"(Name|Unit No|Admission Date|Discharge Date|Date of Birth|Sex|"
        r"Service|Allergies|Attending)[:\s]+[^\n]*\n",
        "",
        safe_text,
        flags=re.IGNORECASE,
    )
    return safe_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. 放射报告提取函数
# ─────────────────────────────────────────────────────────────────────────────

def extract_radiology_text(text: str) -> str:
    """
    从放射报告里只提取 FINDINGS 部分，去掉 IMPRESSION。

    格式通常是：
      EXAMINATION: ...
      FINDINGS: ...
      IMPRESSION: ...   ← 去掉
    """
    if not isinstance(text, str):
        return ""

    # 尝试提取 FINDINGS 章节
    findings_match = re.search(
        r"FINDINGS?\s*:?\s*(.*?)(?=IMPRESSION|CONCLUSION|RECOMMENDATION|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if findings_match:
        findings = findings_match.group(1).strip()
        # 同时保留 EXAMINATION 行（说明是什么检查）
        exam_match = re.search(
            r"EXAMINATION\s*:?\s*([^\n]+)", text, re.IGNORECASE
        )
        exam = exam_match.group(1).strip() if exam_match else ""
        if exam:
            return f"EXAMINATION: {exam}\nFINDINGS: {findings}"
        return f"FINDINGS: {findings}"

    # 没有明确的 FINDINGS 标题，去掉 IMPRESSION 之后的内容
    impression_match = re.search(
        r"IMPRESSION\s*:", text, re.IGNORECASE
    )
    if impression_match:
        return text[:impression_match.start()].strip()

    # 都没有找到，返回全文（放射报告通常没有诊断结论风险）
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_features(
    note_root: Path,
    output_dir: Path,
    cardiac_csv: Path,
) -> None:
    """
    读取 MIMIC-IV-Note，提取心脏病队列的安全文本特征。

    输出: output_dir/text_features.parquet
      列: subject_id, hadm_id, discharge_text, radiology_text, combined_text
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取心脏病队列
    cardiac = pd.read_csv(cardiac_csv, dtype={"hadm_id": "Int64"})
    hadm_ids    = set(cardiac["hadm_id"].dropna().astype(int))
    subject_ids = set(cardiac["subject_id"].astype(int))
    log.info(f"心脏病队列: {len(hadm_ids):,} 个 hadm_id")

    # ── 出院小结 ──────────────────────────────────────────────────────────
    log.info("处理出院小结 discharge.csv.gz ...")
    dis_chunks = []
    for chunk in pd.read_csv(
        note_root / "discharge.csv.gz",
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
        dtype={"hadm_id": "Int64", "subject_id": int},
        parse_dates=["charttime"],
        chunksize=50_000,
    ):
        sub = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if len(sub):
            dis_chunks.append(sub)

    if dis_chunks:
        discharge_df = pd.concat(dis_chunks, ignore_index=True)
        log.info(f"  找到出院小结: {len(discharge_df):,} 条")

        tqdm.pandas(desc="提取出院小结安全章节")
        discharge_df["discharge_text"] = discharge_df["text"].progress_apply(
            extract_discharge_text
        )
        # 每个 hadm_id 只保留最新的一条出院小结
        discharge_df = (
            discharge_df.sort_values("charttime")
            .groupby("hadm_id")
            .last()
            .reset_index()[["hadm_id", "subject_id", "discharge_text"]]
        )
    else:
        log.warning("未找到心脏病队列的出院小结")
        discharge_df = pd.DataFrame(
            columns=["hadm_id", "subject_id", "discharge_text"]
        )

    # ── 放射报告 ──────────────────────────────────────────────────────────
    log.info("处理放射报告 radiology.csv.gz ...")
    rad_chunks = []
    for chunk in pd.read_csv(
        note_root / "radiology.csv.gz",
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
        dtype={"hadm_id": "Int64", "subject_id": int},
        parse_dates=["charttime"],
        chunksize=100_000,
    ):
        sub = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if len(sub):
            rad_chunks.append(sub)

    if rad_chunks:
        radiology_df = pd.concat(rad_chunks, ignore_index=True)
        log.info(f"  找到放射报告: {len(radiology_df):,} 条")

        tqdm.pandas(desc="提取放射报告 Findings")
        radiology_df["radiology_text"] = radiology_df["text"].progress_apply(
            extract_radiology_text
        )
        # 每个 hadm_id 可能有多条放射报告，全部拼接（用换行分隔）
        radiology_df = (
            radiology_df.sort_values("charttime")
            .groupby("hadm_id")["radiology_text"]
            .apply(lambda x: " [SEP] ".join(x.dropna()))
            .reset_index()
        )
    else:
        log.warning("未找到心脏病队列的放射报告")
        radiology_df = pd.DataFrame(
            columns=["hadm_id", "radiology_text"]
        )

    # ── 合并两种文本 ──────────────────────────────────────────────────────
    log.info("合并文本特征 ...")
    text_df = discharge_df.merge(
        radiology_df, on="hadm_id", how="outer"
    )
    text_df["hadm_id"] = text_df["hadm_id"].astype(int)

    # 拼接两种文本，[SEP] 分隔，供 ClinicalBERT 使用
    def combine_texts(row):
        parts = []
        if pd.notna(row.get("discharge_text")) and row["discharge_text"]:
            parts.append(str(row["discharge_text"]))
        if pd.notna(row.get("radiology_text")) and row["radiology_text"]:
            parts.append(str(row["radiology_text"]))
        return " [SEP] ".join(parts)

    text_df["combined_text"] = text_df.apply(combine_texts, axis=1)

    # 统计覆盖情况
    n_total = len(hadm_ids)
    n_discharge = text_df["discharge_text"].notna().sum()
    n_radiology = text_df["radiology_text"].notna().sum()
    n_both = (
        text_df["discharge_text"].notna()
        & text_df["radiology_text"].notna()
    ).sum()
    log.info(f"  覆盖情况（共 {n_total:,} 个 hadm_id）:")
    log.info(f"    有出院小结: {n_discharge:,} ({n_discharge/n_total:.1%})")
    log.info(f"    有放射报告: {n_radiology:,} ({n_radiology/n_total:.1%})")
    log.info(f"    两者都有:   {n_both:,}     ({n_both/n_total:.1%})")

    # ── 保存 ──────────────────────────────────────────────────────────────
    out_path = output_dir / "text_features.parquet"
    text_df.to_parquet(out_path, index=False)
    log.info(f"→ text_features.parquet  shape={text_df.shape}")

    # 验证：打印一条样本看提取效果
    sample = text_df[text_df["combined_text"].str.len() > 100].iloc[0]
    log.info(f"\n样本 hadm_id={sample['hadm_id']}:")
    log.info(f"  discharge_text 前200字:\n{str(sample.get('discharge_text',''))[:200]}")
    log.info(f"  radiology_text 前200字:\n{str(sample.get('radiology_text',''))[:200]}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提取心脏病队列的安全临床文本特征")
    parser.add_argument("--note_root",   required=True,
                        help="MIMIC-IV-Note 目录（含 discharge.csv.gz）")
    parser.add_argument("--output_dir",  required=True,
                        help="输出目录")
    parser.add_argument("--cardiac_csv", required=True,
                        help="cardiac_subjects.csv 路径")
    args = parser.parse_args()

    extract_text_features(
        note_root   = Path(args.note_root),
        output_dir  = Path(args.output_dir),
        cardiac_csv = Path(args.cardiac_csv),
    )