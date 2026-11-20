import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DISCHARGE_MAX_TOKENS = 300
RADIOLOGY_MAX_TOKENS = 200

class ClinicalTextDataset(Dataset):
    def __init__(self, df):
        self.hadm_ids        = df["hadm_id"].tolist()
        self.discharge_texts = df["discharge_text"].fillna("").tolist()
        self.radiology_texts = df["radiology_text"].fillna("").tolist()

    def __len__(self):
        return len(self.hadm_ids)

    def __getitem__(self, idx):
        return {
            "hadm_id":        self.hadm_ids[idx],
            "discharge_text": self.discharge_texts[idx],
            "radiology_text": self.radiology_texts[idx],
        }

def make_collate_fn(tokenizer, discharge_max, radiology_max):
    def collate_fn(batch):
        hadm_ids = [b["hadm_id"] for b in batch]
        dis_enc = tokenizer(
            [b["discharge_text"] for b in batch],
            max_length=discharge_max, truncation=True,
            padding=False, add_special_tokens=False,
        )
        rad_enc = tokenizer(
            [b["radiology_text"] for b in batch],
            max_length=radiology_max, truncation=True,
            padding=False, add_special_tokens=False,
        )
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        input_ids_list, attention_mask_list = [], []
        for i in range(len(batch)):
            ids  = [cls_id] + dis_enc["input_ids"][i] + [sep_id] + rad_enc["input_ids"][i] + [sep_id]
            mask = [1] * len(ids)
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id  = tokenizer.pad_token_id
        input_ids_padded      = [ids  + [pad_id] * (max_len - len(ids))  for ids  in input_ids_list]
        attention_mask_padded = [mask + [0]      * (max_len - len(mask)) for mask in attention_mask_list]
        return {
            "hadm_ids":       hadm_ids,
            "input_ids":      torch.tensor(input_ids_padded,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
        }
    return collate_fn

def encode_texts(text_parquet, output_dir, batch_size=64, device="cuda", resume=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path   = output_dir / "text_embeddings.npy"
    index_path = output_dir / "text_index.csv"

    log.info(f"读取 {text_parquet} ...")
    df = pd.read_parquet(text_parquet)
    df = df[df["combined_text"].fillna("").str.len() > 0].copy()
    df["hadm_id"] = df["hadm_id"].astype(int)
    log.info(f"  有效文本记录: {len(df):,}")

    if resume and emb_path.exists() and index_path.exists():
        done_ids = set(pd.read_csv(index_path)["hadm_id"].astype(int))
        df = df[~df["hadm_id"].isin(done_ids)].copy()
        log.info(f"  断点续传：跳过 {len(done_ids):,}，剩余 {len(df):,}")

    if len(df) == 0:
        log.info("所有记录已编码完成。")
        return

    log.info(f"加载 {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    model.to(device)
    log.info(f"  模型加载完成，使用设备: {device}")

    dataset    = ClinicalTextDataset(df)
    collate_fn = make_collate_fn(tokenizer, DISCHARGE_MAX_TOKENS, RADIOLOGY_MAX_TOKENS)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=(device=="cuda"))

    all_embeddings, all_hadm_ids = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="编码临床文本"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_emb.cpu().float().numpy())
            all_hadm_ids.extend(batch["hadm_ids"])

    embeddings = np.concatenate(all_embeddings, axis=0)
    log.info(f"  编码完成，shape: {embeddings.shape}")

    if resume and emb_path.exists() and index_path.exists():
        old_emb   = np.load(emb_path)
        old_index = pd.read_csv(index_path)
        embeddings   = np.concatenate([old_emb, embeddings], axis=0)
        all_hadm_ids = old_index["hadm_id"].tolist() + all_hadm_ids

    np.save(emb_path, embeddings)
    pd.DataFrame({"hadm_id": all_hadm_ids}).to_csv(index_path, index=False)
    log.info(f"-> text_embeddings.npy  {embeddings.shape}")
    log.info(f"-> text_index.csv       {len(all_hadm_ids):,} 条")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_parquet", required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--resume",       action="store_true")
    args = parser.parse_args()
    encode_texts(
        text_parquet = Path(args.text_parquet),
        output_dir   = Path(args.output_dir),
        batch_size   = args.batch_size,
        device       = args.device,
        resume       = args.resume,
    )
