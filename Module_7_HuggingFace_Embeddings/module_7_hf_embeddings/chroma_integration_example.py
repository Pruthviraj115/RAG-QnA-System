from embedding_service import EmbeddingService


def prepare_for_chromadb(chunks: list[str]):
    """Prepare Module 7 output for Module 8 / ChromaDB."""
    service = EmbeddingService()

    embeddings = service.embed_documents(chunks)
    ids = [f"chunk_{i + 1}" for i in range(len(chunks))]

    return ids, chunks, embeddings


if __name__ == "__main__":
    chunks = [
        "The university provides internship opportunities.",
        "Students must satisfy the required eligibility criteria.",
    ]

    ids, documents, embeddings = prepare_for_chromadb(chunks)

    print("IDs:", ids)
    print("Documents:", documents)
    print("Embedding dimension:", len(embeddings[0]))
