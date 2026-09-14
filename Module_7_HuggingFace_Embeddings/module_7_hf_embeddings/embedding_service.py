from __future__ import annotations

from typing import Sequence
from embedding_model import HuggingFaceEmbeddingModel


class EmbeddingService:
    """Public Module 7 interface for the complete RAG system."""

    def __init__(self, **model_kwargs) -> None:
        self._embedder = HuggingFaceEmbeddingModel(**model_kwargs)

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedder.embedding_dimension

    def embed_documents(self, chunks: Sequence[str]) -> list[list[float]]:
        return self._embedder.embed_documents(chunks)

    def embed_query(self, query: str) -> list[float]:
        return self._embedder.embed_query(query)

    def embed_text(self, text: str) -> list[float]:
        return self._embedder.embed_text(text)
