import os

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").lower()

NORMALIZE_EMBEDDINGS = os.getenv(
    "NORMALIZE_EMBEDDINGS", "true"
).lower() in {"1", "true", "yes", "y", "on"}

SHOW_PROGRESS = os.getenv(
    "EMBEDDING_SHOW_PROGRESS", "true"
).lower() in {"1", "true", "yes", "y", "on"}
