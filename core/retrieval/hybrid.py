from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.retrieval.base import (
    Embedder,
    IndexableRetriever,
    KnowledgeChunk,
    Retriever,
    RetrievedHit,
)
from app.core.retrieval.hashing import HashingEmbedder, tokenize

# RRF 融合中向量路的最小余弦阈值：低于该值视为哈希桶碰撞噪声，不参与排序
VEC_FLOOR = 0.05


class HybridRetriever:
    """BM25 + 哈希向量 + RRF 融合的检索实现（Embedder 可注入替换）。"""

    def __init__(self, embedder: Embedder | None = None, *,
                 dim: int = 4096, top_k: int = 4,
                 vec_floor: float = VEC_FLOOR) -> None:
        self.embedder = embedder or HashingEmbedder(dim=dim)
        self.top_k = top_k
        self.vec_floor = vec_floor
        self.chunks: list[KnowledgeChunk] = []
        self._bm25: dict[str, BM25Okapi] = {}
        self._vectors: np.ndarray | None = None
        # 全局 chunk 序号 → 所属文档内语料位置（BM25 按文档建索引，
        # get_scores 返回的是文档内部顺序的分数，需要映射取分）
        self._doc_positions: dict[int, int] = {}

    @property
    def bm25(self) -> dict[str, BM25Okapi]:
        """按文档名分组的 BM25 索引（兼容旧 KnowledgeStore._bm25 的检查）。"""
        return self._bm25

    def index(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks
        by_doc: dict[str, list[str]] = {}
        for global_idx, chunk in enumerate(chunks):
            texts = by_doc.setdefault(chunk.doc, [])
            self._doc_positions[global_idx] = len(texts)
            texts.append(chunk.text)
        self._bm25 = {doc: BM25Okapi([tokenize(t) for t in texts])
                      for doc, texts in by_doc.items()}
        # (n, dim) 矩阵：所有向量 L2 归一后，@ 查询向量即一次算出全部余弦相似度
        self._vectors = np.vstack([self.embedder.embed(c.text) for c in chunks])

    def retrieve(self, query: str, *, kind: str | None = None,
                 top_k: int | None = None) -> list[RetrievedHit]:
        if not self.chunks or self._vectors is None:
            return []
        k = top_k or self.top_k
        q_tokens = tokenize(query)

        # 路 1：BM25 词面匹配（按所属文档的索引取分）
        bm25_scores: list[tuple[int, float]] = []
        for i, chunk in enumerate(self.chunks):
            if kind and chunk.doc != kind:
                continue
            scores = self._bm25[chunk.doc].get_scores(q_tokens)
            bm25_scores.append((i, float(scores[self._doc_positions[i]])))
        # 路 2：哈希向量余弦相似度（一次矩阵乘法）
        q_vec = self.embedder.embed(query)
        sims = self._vectors @ q_vec

        def _rank(pairs: list[tuple[int, float]]) -> dict[int, int]:
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
            return {i: r + 1 for r, (i, _) in enumerate(pairs_sorted)}

        bm25_rank = _rank([(i, s) for i, s in bm25_scores if s > 0])
        # 哈希向量仅做同词命中，余弦低于阈值视为桶碰撞噪声
        vec_rank = _rank([(i, float(sims[i])) for i, _ in bm25_scores
                          if sims[i] >= self.vec_floor])

        # RRF 融合：score = Σ 1/(60 + rank)
        fused: dict[int, float] = {}
        for i, _ in bm25_scores:
            score = 0.0
            if i in bm25_rank:
                score += 1.0 / (60 + bm25_rank[i])
            if i in vec_rank:
                score += 1.0 / (60 + vec_rank[i])
            if score > 0:
                fused[i] = score

        hits = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            RetrievedHit(chunk=self.chunks[i], fused_score=s,
                         bm25_rank=bm25_rank.get(i, 0), vec_rank=vec_rank.get(i, 0))
            for i, s in hits
        ]


def format_hits(hits: list[RetrievedHit], *, query: str) -> str:
    """把命中格式化为带来源标注的文本（注入 prompt 用）。"""
    if not hits:
        return f"知识库中未检索到与「{query}」相关的内容。"
    lines = []
    for hit in hits:
        snippet = re.sub(r"\s+", " ", hit.chunk.text)[:400]
        lines.append(f"【{hit.chunk.doc}/{hit.chunk.section}】{snippet}")
    return "\n\n".join(lines)


def merge_hits(groups: list[list[RetrievedHit]], *, top_k: int) -> list[RetrievedHit]:
    """合并多个 kind 的检索结果：按融合分降序、按 (doc, idx) 去重、截断 top_k。"""
    seen: set[tuple[str, int]] = set()
    merged: list[RetrievedHit] = []
    for hit in sorted((h for g in groups for h in g),
                      key=lambda h: h.fused_score, reverse=True):
        key = (hit.chunk.doc, hit.chunk.idx)
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
        if len(merged) >= top_k:
            break
    return merged
