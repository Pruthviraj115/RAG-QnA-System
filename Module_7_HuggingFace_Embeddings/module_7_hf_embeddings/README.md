# Module 7 — Hugging Face Embedding Generation

RAG pipeline:
Clean Text Chunk -> Hugging Face Sentence Transformer -> Numerical Vector -> Module 8 / ChromaDB

Default model: `sentence-transformers/all-MiniLM-L6-v2`

Features:
- Lazy model loading
- CPU/GPU auto detection
- Batch document embedding
- Query embedding
- Optional L2 normalization
- Input validation
- Embedding dimension validation
- Reusable interface for Module 8 / ChromaDB

## Install

```powershell
pip install -r requirements.txt
```

The first run downloads the Hugging Face model.

## Test

```powershell
python test_embeddings.py
```

## Module 8 usage

```python
from embedding_service import EmbeddingService

service = EmbeddingService()

chunks = [
    "Artificial intelligence is used in healthcare.",
    "Machine learning models learn patterns from data."
]

vectors = service.embed_documents(chunks)
query_vector = service.embed_query("How is AI used in healthcare?")
```

Pass `vectors` to ChromaDB as document embeddings and `query_vector` for query retrieval.

Use the same model/configuration for documents and queries.
