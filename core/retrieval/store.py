from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.core.retrieval.base import KnowledgeChunk, Retriever, RetrievedHit
from app.core.retrieval.hybrid import HybridRetriever, format_hits


def _split_markdown(path: Path) -> list[KnowledgeChunk]:
    text = path.read_text(encoding="utf-8")
    doc = path.stem
    sections: list[tuple[str, str]] = []
    current_title, current_body = "总览", []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title, current_body = line[3:].strip(), []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_title, "\n".join(current_body).strip()))

    chunks: list[KnowledgeChunk] = []
    for title, body in sections:
        if len(body) <= 500:
            chunks.append(KnowledgeChunk(doc=doc, section=title, text=body, idx=0))
            continue
        buf: list[str] = []
        size = 0
        for para in body.split("\n"):
            buf.append(para)
            size += len(para)
            if size >= 400:
                chunks.append(KnowledgeChunk(doc=doc, section=title,
                                             text="\n".join(buf).strip(), idx=0))
                buf, size = [], 0
        if buf:
            chunks.append(KnowledgeChunk(doc=doc, section=title,
                                         text="\n".join(buf).strip(), idx=0))
    for i, chunk in enumerate(chunks):
        chunk.idx = i
    return chunks


class KnowledgeStore:

    def __init__(self, knowledge_dir: Path | None = None,
                 retriever: Retriever | None = None) -> None:
        settings = get_settings()
        self.dir = knowledge_dir or settings.knowledge_path
        self.top_k = settings.retrieval_top_k
        self._retriever = retriever or HybridRetriever(
            dim=settings.embedding_dim, top_k=self.top_k)
        self.chunks: list[KnowledgeChunk] = []

    def load(self) -> "KnowledgeStore":
        for md in sorted(self.dir.glob("*.md")):
            self.chunks.extend(_split_markdown(md))
        if hasattr(self._retriever, "index"):
            self._retriever.index(self.chunks)  # type: ignore[attr-defined]
        return self

    def retrieve(self, query: str, *, kind: str | None = None,
                 top_k: int | None = None) -> list[RetrievedHit]:
        return self._retriever.retrieve(query, kind=kind, top_k=top_k)

    def search_formatted(self, query: str, *, kind: str | None = None,
                         top_k: int | None = None) -> str:
        hits = self.retrieve(query, kind=kind, top_k=top_k)
        return format_hits(hits, query=query)

    @property
    def _bm25(self):
        return getattr(self._retriever, "bm25", {})


@lru_cache(maxsize=1)
def get_knowledge_store() -> KnowledgeStore:
    return KnowledgeStore().load()



