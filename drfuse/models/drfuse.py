import math

import torch
from torch import nn
from torch.nn import functional as F

from .ehr_transformer import EHRTransformer


# ─────────────────────────────────────────────────────────────────────────────
# ResNet1D ECG 编码器
# 输入: (B, 12, 5000)   12导联 × 5000采样点
# 输出: (B, hidden_size)
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """带残差连接的 1D 卷积块，kernel_size=7 适合 ECG 长序列"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2        = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """
    1D ResNet ECG 编码器（完整独立网络，shared 和 distinct 各用一个）
    输入 : (B, 12, 5000)
    输出 : (B, hidden_size)
    """
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        # stem：降采样 4×
        self.stem = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        # 4 个残差层，逐步升通道 + 降时序长度
        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.proj    = nn.Linear(512, hidden_size)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [ResBlock1D(in_ch, out_ch, stride, downsample)]
        for _ in range(1, n_blocks):
            layers.append(ResBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):              # x : (B, 12, 5000)
        x = self.stem(x)               # (B, 64,  1250)
        x = self.layer1(x)             # (B, 64,  1250)
        x = self.layer2(x)             # (B, 128,  625)
        x = self.layer3(x)             # (B, 256,  313)
        x = self.layer4(x)             # (B, 512,  157)
        x = self.avgpool(x).squeeze(-1)# (B, 512)
        return self.proj(x)            # (B, hidden_size)


# ─────────────────────────────────────────────────────────────────────────────
# DrFuse 主模型
# ─────────────────────────────────────────────────────────────────────────────

class DrFuseModel(nn.Module):
    def __init__(self, hidden_size, num_classes, ehr_dropout, ehr_n_layers, ehr_n_head,
                 cxr_model='resnet1d', logit_average=False):
        super().__init__()
        self.num_classes  = num_classes
        self.hidden_size  = hidden_size
        self.logit_average = logit_average

        # ── EHR 编码器（Transformer）──────────────────────────────────────
        self.ehr_model = EHRTransformer(
            input_size=68,              # 17 ICU值 + 17 化验值 + 17 ICU mask + 17 化验mask
            num_classes=num_classes,
            d_model=hidden_size,
            n_head=ehr_n_head,
            n_layers_feat=1,
            n_layers_shared=ehr_n_layers,
            n_layers_distinct=ehr_n_layers,
            dropout=ehr_dropout,
            use_naim=False,
            split_encoder=True,         # ← shared/distinct 独立底层编码器              # ← NAIM 特征级缺失
            naim_d_feat=64,
            naim_n_head=4,
        )

        # ── ECG 编码器（ResNet1D，shared 和 distinct 各一个独立网络）──────
        self.cxr_model_shared_net = ResNet1D(hidden_size)   # → feat_cxr_shared
        self.cxr_model_spec_net   = ResNet1D(hidden_size)   # → feat_cxr_distinct
        # 文本编码器：768维 ClinicalBERT 嵌入 → hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
       # 文本 shared 和 distinct 投影（两路独立）
        self.text_shared_proj  = nn.Linear(hidden_size, hidden_size)
        self.text_distinct_proj = nn.Linear(hidden_size, hidden_size)

        # 文本预测头
        self.text_model_linear = nn.Linear(hidden_size, num_classes)
  
        # ── 融合模块（与原版完全一致）────────────────────────────────────
        self.shared_project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.ehr_model_linear  = nn.Linear(hidden_size, num_classes)
        self.cxr_model_linear  = nn.Linear(hidden_size, num_classes)
        self.fuse_model_shared = nn.Linear(hidden_size, num_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.shared_fusion_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_proj = nn.Linear(hidden_size, (2 + num_classes) * hidden_size)
        self.final_pred_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, img, text, seq_lengths, avail, grl_lambda):
        # avail: (B, 3)  [EHR=1, ECG=0/1, Text=0/1]
        ecg_avail  = avail[:, 1].unsqueeze(1)   # (B, 1)
        text_avail = avail[:, 2].unsqueeze(1)   # (B, 1)

        # ── EHR 编码 ──────────────────────────────────────────────────────────
        feat_ehr_shared, feat_ehr_distinct, pred_ehr = self.ehr_model(x, seq_lengths)

        # ── ECG 编码 ──────────────────────────────────────────────────────────
        feat_ecg_shared   = self.cxr_model_shared_net(img)   # (B, hidden_size)
        feat_ecg_distinct = self.cxr_model_spec_net(img)     # (B, hidden_size)

        # ── Text 编码 ─────────────────────────────────────────────────────────
        feat_text_base     = self.text_proj(text)                        # (B, hidden_size)
        feat_text_shared   = self.text_shared_proj(feat_text_base)       # (B, hidden_size)
        feat_text_distinct = self.text_distinct_proj(feat_text_base)     # (B, hidden_size)

        # 文本预测头
        pred_text = self.text_model_linear(feat_text_distinct).sigmoid()

        # ── Shared 特征对齐 ───────────────────────────────────────────────────
        feat_ehr_shared  = self.shared_project(feat_ehr_shared)
        feat_ecg_shared  = self.shared_project(feat_ecg_shared)
        feat_text_shared = self.shared_project(feat_text_shared)

        # ── Logit Pooling（动态根据可用模态融合）─────────────────────────────
        # 三路 logsumexp，缺失的模态贡献被 avail 权重抑制
        h_ehr  = feat_ehr_shared
        h_ecg  = feat_ecg_shared
        h_text = feat_text_shared

        numerator   = h_ehr
        denominator = torch.ones_like(ecg_avail)

        numerator   = numerator + ecg_avail  * h_ecg
        denominator = denominator + ecg_avail

        numerator   = numerator + text_avail * h_text
        denominator = denominator + text_avail

        feat_avg_shared = numerator / denominator   # (B, hidden_size)
        pred_shared = self.fuse_model_shared(feat_avg_shared).sigmoid()

        # ── Disease-wise Attention Fusion（4路：EHR + shared + ECG + Text）────
        attn_input = torch.stack(
            [feat_ehr_distinct, feat_avg_shared, feat_ecg_distinct, feat_text_distinct], dim=1
        )  # (B, 4, hidden_size)

        qkvs = self.attn_proj(attn_input)  # (B, 4, (2+num_classes)*hidden_size)
        q, v, *k = qkvs.chunk(2 + self.num_classes, dim=-1)
        # q: (B, 4, hidden_size)
        # v: (B, 4, hidden_size)
        # k: num_classes 个，每个 (B, 4, hidden_size)

        # query 动态平均（根据可用模态）
        q_ehr    = q[:, 0, :]
        q_shared = q[:, 1, :]
        q_ecg    = q[:, 2, :]
        q_text   = q[:, 3, :]

        q_sum = q_ehr + q_shared
        q_cnt = torch.ones(q.shape[0], 1, device=q.device) * 2
        q_sum = q_sum + ecg_avail  * q_ecg
        q_cnt = q_cnt + ecg_avail
        q_sum = q_sum + text_avail * q_text
        q_cnt = q_cnt + text_avail
        q_mean = q_sum / q_cnt   # (B, hidden_size)

        ks = torch.stack(k, dim=1)   # (B, num_classes, 4, hidden_size)
        attn_logits = torch.einsum('bd,bnkd->bnk', q_mean, ks)  # (B, num_classes, 4)
        attn_logits = attn_logits / math.sqrt(q.shape[-1])

        # mask 缺失模态（最后一维：0=EHR, 1=shared, 2=ECG, 3=Text）
        attn_mask = torch.ones_like(attn_logits)
        attn_mask[ecg_avail.squeeze()  == 0, :, 2] = 0
        attn_mask[text_avail.squeeze() == 0, :, 3] = 0
        attn_logits  = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, num_classes, 4)

        feat_final = torch.matmul(attn_weights, v)     # (B, num_classes, hidden_size)
        pred_final = self.final_pred_fc(feat_final)
        pred_final = torch.diagonal(pred_final, dim1=1, dim2=2).sigmoid()

        return {
            'feat_ehr_shared':   feat_ehr_shared,
            'feat_cxr_shared':   feat_ecg_shared,      # ← 用 feat_cxr_shared 兼容 trainer
            'feat_ehr_distinct': feat_ehr_distinct,
            'feat_cxr_distinct': feat_ecg_distinct,    # ← 用 feat_cxr_distinct 兼容 trainer
            'feat_text_shared':  feat_text_shared,     # ← 新增
            'feat_text_distinct':feat_text_distinct,   # ← 新增
            'feat_final':        feat_final,
            'pred_final':        pred_final,
            'pred_shared':       pred_shared,
            'pred_ehr':          pred_ehr,
            'pred_cxr':          pred_text,            # ← pred_cxr 键名保留，内容是 text 预测
            'attn_weights':      attn_weights,
        }