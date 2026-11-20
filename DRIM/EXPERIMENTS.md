
## 2026-10-08 DRIM + disease-wise attention
- 基线: preds/drim_test_seed1999.npz / models/drim_seed1999.pt (test PRAUC 0.4560, AUROC 0.7950)
- 诊断: 无ECG患者 AUROC 0.7537 vs 有ECG 0.8036；遮掉ECG仅 -0.0074 → 约15%信息损失、85%人群差异
- 诊断: 去掉ECG后 缺血性心脏病 -0.030 / 心衰 -0.003 / 心内膜炎 +0.035 → 模态贡献因病而异
- DW 实验: DRIM_DW=1 DRIM_TAG=_dw seed 1999, 新模块 drim/disease_wise.py (约53万参数)
  4路token: fusion_s输出 + EHR/ECG/文本 unique；辅助损失 lambda_pred=1, lambda_attn=1
  输出: preds/drim_test_seed1999_dw.npz (含 attn), models/drim_seed1999_dw.pt
