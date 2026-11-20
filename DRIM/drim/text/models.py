import torch
from torch import nn


class TextEncoder(nn.Module):
    """
    DRIM 风格的文本编码器。
    输入是预编码好的 BiomedBERT 嵌入 (B, 768)，
    通过投影层映射到 embedding_dim。
    shared/unique 的分离由 DRIM 框架完成。

    输入: (B, 768)
    输出: (B, embedding_dim)
    """
    def __init__(self, embedding_dim: int, dropout: float = 0.3,
                 text_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 768)
        return self.proj(x)   # (B, embedding_dim)
