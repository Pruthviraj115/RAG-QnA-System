from embedding_service import EmbeddingService


def main() -> None:
    service = EmbeddingService()

    documents = [
        "Artificial intelligence is used in healthcare.",
        "Machine learning models learn patterns from data.",
        "Natural language processing helps computers understand text.",
    ]

    print("=" * 60)
    print("MODULE 7 - HUGGING FACE EMBEDDING TEST")
    print("=" * 60)
    print(f"Model       : {service.model_name}")
    print(f"Dimension   : {service.embedding_dimension}")
    print()

    print("Generating document embeddings...")
    document_vectors = service.embed_documents(documents)

    print(f"Documents   : {len(document_vectors)}")
    print(f"Vector size : {len(document_vectors[0])}")
    print(f"First 5     : {document_vectors[0][:5]}")
    print()

    query = "How is AI used in medical applications?"
    print(f"Query       : {query}")

    query_vector = service.embed_query(query)

    print(f"Query size  : {len(query_vector)}")
    print(f"First 5     : {query_vector[:5]}")
    print()

    assert len(document_vectors) == len(documents)
    assert all(
        len(vector) == service.embedding_dimension
        for vector in document_vectors
    )
    assert len(query_vector) == service.embedding_dimension

    print("SUCCESS: Module 7 embedding pipeline is working.")


if __name__ == "__main__":
    main()
