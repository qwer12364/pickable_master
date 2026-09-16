from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class KnowledgeChunk:
    doc: str          # 文档名（rules/technique/equipment/strategy）
    section: str      # 小节标题
    text: str
    idx: int


@dataclass
class RetrievedHit:
    chunk: KnowledgeChunk
    fused_score: float
    bm25_rank: int
    vec_rank: int


class Embedder(Protocol):
    """文本 → 向量。实现必须保证向量已 L2 归一（检索层直接用点积当余弦）。"""

    def embed(self, text: str) -> np.ndarray: ...


class Retriever(Protocol):
    """查询接口：按领域过滤、取 top_k 命中。"""

    def retrieve(self, query: str, *, kind: str | None = None,
                 top_k: int | None = None) -> list[RetrievedHit]: ...


class IndexableRetriever(Protocol):
    """索引接口：加载器（KnowledgeStore）切好块后注入。"""

    def index(self, chunks: list[KnowledgeChunk]) -> None: ...
