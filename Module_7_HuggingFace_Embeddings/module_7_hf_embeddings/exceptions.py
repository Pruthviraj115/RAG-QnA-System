class EmbeddingError(Exception):
    """Base exception for Module 7."""


class ModelLoadError(EmbeddingError):
    """Raised when the embedding model cannot be loaded."""


class InvalidEmbeddingInputError(EmbeddingError):
    """Raised when embedding input is invalid."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when an unexpected embedding dimension is detected."""
