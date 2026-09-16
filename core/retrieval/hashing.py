from __future__ import annotations

import hashlib

import jieba
import numpy as np


def tokenize(text: str) -> list[str]:
    tokens = [t.strip() for t in jieba.lcut(text.lower()) if t.strip()]
    return tokens or [w for w in text if w.strip()]


class HashingEmbedder:
    """词袋哈希向量：MD5 哈希桶 + TF 加权，embed 输出已 L2 归一。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec
