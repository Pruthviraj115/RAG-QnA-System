from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# ==========================================
# Load PDF Document
# ==========================================

def load_documents(pdf_path):
    """
    Load a PDF document.
    """

    loader = PyPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    return documents


# ==========================================
# Clean Text
# ==========================================

def clean_text(text):
    """
    Clean unnecessary whitespace from text.
    """

    text = " ".join(
        text.split()
    )

    return text.strip()


# ==========================================
# Create Chunks
# ==========================================

def create_chunks(documents):
    """
    Clean documents and split them
    into smaller chunks.
    """

    # Clean document text
    for document in documents:

        document.page_content = clean_text(
            document.page_content
        )

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Split documents
    chunks = text_splitter.split_documents(
        documents
    )

    return chunks