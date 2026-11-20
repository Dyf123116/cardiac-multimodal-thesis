import math
import torch
from torch import nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.rand(1, max_len, d_model))
        self.pe.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # x: (batch_size, seq_len, embedding_dim)
        return self.dropout(x)


class NAIMFeatureEncoder(nn.Module):
    """NAIM / MARIA 风格的特征级掩码注意力，作用在【每个时间步内部】的特征轴上。

    替代原来的 self.emb = nn.Linear(input_size, d_model)。
    输入输出签名完全一致：(B, T, input_size) -> (B, T, d_model)，
    所以 EHRTransformer.forward 以及下游 drfuse.py 一行都不用改。

    输入的 input_size = 2 * n_feat，前 n_feat 列是数值，后 n_feat 列是 mask。
    对本项目：n_feat = 34（17 ICU监护 + 17 化验），input_size = 68。

    注意力公式（NAIM）：ReLU(softmax(QK^T/sqrt(d) + M) + M^T) V
      - M   把【缺失的 key】屏蔽掉  -> 没人能 attend 到缺失特征
      - M^T 把【缺失的 query】置零 -> 缺失特征自己不产生输出向量
    实现上用有限大负数 + masked_fill 代替 -inf，避免 amp / 全缺失行产生 NaN。
    """

    def __init__(self, n_feat, d_model, d_feat=64, n_head=4, dropout=0.1,
                 mask_is_observed=True):
        super().__init__()
        assert d_feat % n_head == 0, "d_feat 必须能被 n_head 整除"
        self.n_feat = n_feat
        self.d_feat = d_feat
        self.n_head = n_head
        self.d_head = d_feat // n_head
        self.mask_is_observed = mask_is_observed

        # 每个特征一套自己的数值嵌入： token_j = v_j * w_j + b_j
        self.w = nn.Parameter(torch.randn(n_feat, d_feat) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_feat, d_feat))
        # 特征身份嵌入：v_j 被置零后，token 仍然知道"我是第 j 个变量"
        self.feat_emb = nn.Parameter(torch.randn(n_feat, d_feat) * 0.02)

        self.q_proj = nn.Linear(d_feat, d_feat)
        self.k_proj = nn.Linear(d_feat, d_feat)
        self.v_proj = nn.Linear(d_feat, d_feat)
        self.out_proj = nn.Linear(d_feat, d_feat)
        self.norm = nn.LayerNorm(d_feat)
        self.attn_drop = nn.Dropout(dropout)
        self.to_d_model = nn.Linear(d_feat, d_model)

        self.NEG = -1e4  # 有限负数，代替 -inf

    def forward(self, x):
        B, T, _ = x.shape
        n, d = self.n_feat, self.d_feat

        val = x[..., :n]                                   # (B,T,n) 数值
        m_raw = x[..., n:2 * n]                            # (B,T,n) 掩码列
        observed = m_raw if self.mask_is_observed else (1.0 - m_raw)
        observed = observed.clamp(0.0, 1.0)                # (B,T,n) 1=观测到

        # ---- 特征 token 化：缺失位置只保留身份，不带数值 ----
        tok = (val * observed).unsqueeze(-1) * self.w + self.b + self.feat_emb
        tok = tok.reshape(B * T, n, d)

        miss = (observed.reshape(B * T, n) < 0.5)          # (BT,n) True=缺失

        q = self.q_proj(tok).view(B * T, n, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(tok).view(B * T, n, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(tok).view(B * T, n, self.n_head, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)   # (BT,h,n,n)

        # + M：屏蔽缺失的 key（列）
        scores = scores.masked_fill(miss[:, None, None, :], self.NEG)
        attn = scores.softmax(dim=-1)

        # + M^T 再 ReLU：缺失的 query（行）整行置零
        attn = F.relu(attn.masked_fill(miss[:, None, :, None], -1.0))
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * T, n, d)
        out = self.out_proj(out)
        out = self.norm(out + tok * (~miss).unsqueeze(-1).to(out.dtype))

        # LayerNorm 有 bias，缺失 token 会被拉出非零，这里显式重新置零
        out = out.reshape(B, T, n, d) * observed.unsqueeze(-1)

        # 特征轴上对【已观测】的 token 做 masked mean -> 每个时间步一个向量
        denom = observed.sum(dim=-1, keepdim=True).clamp(min=1.0)      # (B,T,1)
        pooled = out.sum(dim=2) / denom                                # (B,T,d_feat)
        return self.to_d_model(pooled)                                 # (B,T,d_model)


class EHRTransformer(nn.Module):
    def __init__(self, input_size, num_classes,
                 d_model=256, n_head=8, n_layers_feat=1,
                 n_layers_shared=1, n_layers_distinct=1,
                 dropout=0.3, max_len=350,
                 # ---- NAIM 开关，默认 False 时行为与原版逐位一致 ----
                 use_naim=False, naim_d_feat=64, naim_n_head=4,
                 naim_dropout=0.1, naim_mask_is_observed=True,
                 # ---- shared/distinct 是否各自拥有独立的底层编码器 ----
                 split_encoder=False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_naim = use_naim

        if use_naim:
            assert input_size % 2 == 0, \
                f"NAIM 要求 input_size 为偶数（数值列+掩码列），当前 {input_size}"
            self.emb = NAIMFeatureEncoder(
                n_feat=input_size // 2, d_model=d_model,
                d_feat=naim_d_feat, n_head=naim_n_head,
                dropout=naim_dropout, mask_is_observed=naim_mask_is_observed)
        else:
            self.emb = nn.Linear(input_size, d_model)

        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_feat = nn.TransformerEncoder(layer, num_layers=n_layers_feat)

        # split_encoder=True 时，distinct 分支拥有自己独立的
        # emb / pos_encoder / model_feat，与 shared 分支完全不共享底层。
        # 对齐 DRIM 的「每模态两个独立编码器」设计。
        self.split_encoder = split_encoder
        if split_encoder:
            if use_naim:
                self.emb_d = NAIMFeatureEncoder(
                    n_feat=input_size // 2, d_model=d_model,
                    d_feat=naim_d_feat, n_head=naim_n_head,
                    dropout=naim_dropout, mask_is_observed=naim_mask_is_observed)
            else:
                self.emb_d = nn.Linear(input_size, d_model)
            self.pos_encoder_d = LearnablePositionalEncoding(d_model, dropout=0, max_len=max_len)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
            self.model_feat_d = nn.TransformerEncoder(layer, num_layers=n_layers_feat)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_shared = nn.TransformerEncoder(layer, num_layers=n_layers_shared)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout)
        self.model_distinct = nn.TransformerEncoder(layer, num_layers=n_layers_distinct)

        self.fc_distinct = nn.Linear(d_model, num_classes)

    def forward(self, x, seq_lengths):
        attn_mask = torch.stack([torch.cat([torch.zeros(len_, device=x.device),
                                 float('-inf')*torch.ones(max(seq_lengths)-len_, device=x.device)])
                                for len_ in seq_lengths])
        x_in = x
        x = self.emb(x_in)  # NAIM 或 Linear，签名相同
        x = self.pos_encoder(x)
        feat = self.model_feat(x, src_key_padding_mask=attn_mask)
        h_shared = self.model_shared(feat, src_key_padding_mask=attn_mask)

        if self.split_encoder:
            xd = self.emb_d(x_in)
            xd = self.pos_encoder_d(xd)
            feat_d = self.model_feat_d(xd, src_key_padding_mask=attn_mask)
            h_distinct = self.model_distinct(feat_d, src_key_padding_mask=attn_mask)
        else:
            h_distinct = self.model_distinct(feat, src_key_padding_mask=attn_mask)

        padding_mask = torch.ones_like(attn_mask).unsqueeze(2)
        padding_mask[attn_mask == float('-inf')] = 0
        rep_shared = (padding_mask * h_shared).sum(dim=1) / padding_mask.sum(dim=1)
        rep_distinct = (padding_mask * h_distinct).sum(dim=1) / padding_mask.sum(dim=1)

        pred_distinct = self.fc_distinct(rep_distinct).sigmoid()
        return rep_shared, rep_distinct, pred_distinct