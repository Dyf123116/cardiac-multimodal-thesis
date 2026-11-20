"""按疾病的模态注意力（移植自 DrFuse 的 disease-wise attention）。
token 顺序: 0 = shared（fusion_s 输出）, 1..M = 各模态 unique（顺序同 modalities）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseWiseHead(nn.Module):
    def __init__(self, dim, num_classes, modalities):
        super().__init__()
        self.mods = list(modalities)
        self.C, self.d = num_classes, dim
        self.n_tok = 1 + len(self.mods)
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, num_classes * dim)   # 每个 token 对每个疾病一个 key
        self.v_proj = nn.Linear(dim, dim)
        self.cls_w = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.cls_w)
        self.cls_b = nn.Parameter(torch.zeros(num_classes))
        self.tok_heads = nn.ModuleList([nn.Linear(dim, num_classes) for _ in range(self.n_tok)])

    def forward(self, emb_sh, emb_u, mask, fusion_s):
        dev = self.cls_w.device
        mask = {k: v.to(dev).bool() for k, v in mask.items()}
        sh = fusion_s({k: emb_sh[k].to(dev) for k in self.mods}, mask)
        if isinstance(sh, tuple):
            sh = sh[0]
        T = self.norm(torch.stack([sh] + [emb_u[k].to(dev) for k in self.mods], 1))  # (B,K,d)
        B, K, d = T.shape
        avail = torch.stack([torch.ones(B, dtype=torch.bool, device=dev)]
                            + [mask[k] for k in self.mods], 1)                       # (B,K)

        q = self.q_proj((T * avail.unsqueeze(-1)).sum(1) / avail.sum(1, keepdim=True))  # (B,d)
        k = self.k_proj(T).view(B, K, self.C, d)                                        # (B,K,C,d)
        v = self.v_proj(T)                                                              # (B,K,d)
        s = torch.einsum('bd,bkcd->bck', q, k) / math.sqrt(d)                          # (B,C,K)
        s = s.masked_fill(~avail.unsqueeze(1), -1e4)
        attn = s.softmax(-1)                                                            # (B,C,K)
        feat = torch.einsum('bck,bkd->bcd', attn, v)                                    # (B,C,d)
        out = (feat * self.cls_w).sum(-1) + self.cls_b                                  # (B,C)
        tok_logits = torch.stack([h(T[:, i]) for i, h in enumerate(self.tok_heads)], 1) # (B,K,C)
        return out, {'attn': attn, 'tok_logits': tok_logits, 'avail': avail}


def dw_aux_loss(aux, y, lam_pred=1.0, lam_attn=1.0):
    tl, attn, avail = aux['tok_logits'], aux['attn'], aux['avail'].float()
    bce = F.binary_cross_entropy_with_logits(tl, y.unsqueeze(1).expand_as(tl), reduction='none')  # (B,K,C)
    pred_loss = (bce.mean(-1) * avail).sum() / avail.sum()

    # 每个疾病上：哪一路预测损失更低，就该获得更高注意力
    L = bce.detach().transpose(1, 2)                                   # (B,C,K)
    K = L.shape[-1]
    rank = torch.zeros((), device=tl.device)
    for i in range(K):
        for j in range(i + 1, K):
            both = (avail[:, i] * avail[:, j]).unsqueeze(1).expand_as(L[..., i])
            target = torch.sign(L[..., j] - L[..., i])
            r = F.margin_ranking_loss(attn[..., i], attn[..., j], target, reduction='none')
            rank = rank + (r * both).sum() / both.sum().clamp(min=1)
    return lam_pred * pred_loss + lam_attn * rank
