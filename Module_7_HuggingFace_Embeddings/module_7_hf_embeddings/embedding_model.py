from __future__ import annotations

from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    BATCH_SIZE,
    DEVICE,
    MODEL_NAME,
    NORMALIZE_EMBEDDINGS,
    SHOW_PROGRESS,
)
from exceptions import (
    EmbeddingDimensionError,
    InvalidEmbeddingInputError,
    ModelLoadError,
)


class HuggingFaceEmbeddingModel:
    """Reusable Hugging Face Sentence Transformer wrapper."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE,
        normalize_embeddings: bool = NORMALIZE_EMBEDDINGS,
        show_progress: bool = SHOW_PROGRESS,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress = show_progress
        self.device = self._resolve_device(device)
        self._model = None
        self._dimension = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        device = device.lower()

        if device in {"cpu", "cuda"}:
            return device

        if device != "auto":
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model(self) -> SentenceTransformer:
        """Load the model only when first needed."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                self._dimension = self._model.get_sentence_embedding_dimension()

                if not self._dimension:
                    raise ModelLoadError(
                        "The model did not report a valid embedding dimension."
                    )

            except Exception as exc:
                raise ModelLoadError(
                    f"Could not load embedding model '{self.model_name}'. "
                    f"Check internet/model availability and dependencies. "
                    f"Original error: {exc}"
                ) from exc

        return self._model

    @property
    def embedding_dimension(self) -> int:
        _ = self.model
        return self._dimension

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, Sequence):
            raise InvalidEmbeddingInputError(
                "Input must be a string or a sequence of strings."
            )

        cleaned = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise InvalidEmbeddingInputError(
                    f"Item {index} is not a string."
                )

            text = text.strip()

            if not text:
                raise InvalidEmbeddingInputError(
                    f"Item {index} is empty."
                )

            cleaned.append(text)

        if not cleaned:
            raise InvalidEmbeddingInputError(
                "At least one text must be supplied."
            )

        return cleaned

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate document embeddings in batches."""
        texts = self._validate_texts(texts)

        try:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
        except Exception as exc:
            raise EmbeddingDimensionError(
                f"Document embedding generation failed: {exc}"
            ) from exc

        array = np.asarray(vectors, dtype=np.float32)

        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)

        self._validate_dimension(array)
        return array.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate one query embedding using the same model."""
        texts = self._validate_texts([query])

        try:
            vector = self.model.encode(
                texts[0],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
        except Exception as exc:
            raise EmbeddingDimensionError(
                f"Query embedding generation failed: {exc}"
            ) from exc

        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        self._validate_dimension(array)
        return array.tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _validate_dimension(self, vectors: np.ndarray) -> None:
        expected = self.embedding_dimension
        actual = vectors.shape[-1]

        if actual != expected:
            raise EmbeddingDimensionError(
                f"Embedding dimension mismatch: expected {expected}, "
                f"received {actual}."
            )
