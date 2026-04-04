from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from loguru import logger

from load_data import chunk_documents
from enhance import summarise_chunks

from load_data import PROJECT_ROOT

DB_FOLDER = PROJECT_ROOT / "db"

EMBEDDINGS_MODEL = "nomic-embed-text"


def create_vector_database(documents, db_name="chroma_db_v1"):
    persist_directory = DB_FOLDER / db_name
    """Create and persist ChromaDB vector database"""
    logger.info(
        "Building vector store | document_count={} persist_directory={!r}",
        len(documents),
        persist_directory,
    )

    embedding_model = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

    logger.debug("Creating Chroma database | metric=cosine")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    logger.success(
        "Vector store ready | persist_directory={!r}",
        persist_directory,
    )
    return db


def process_pdfs():
    chunks = chunk_documents()
    summarised_chunks = summarise_chunks(chunks)
    return summarised_chunks


def load_database(db_name="chroma_db_v1"):
    persist_directory = DB_FOLDER / db_name
    try:
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=OllamaEmbeddings(model=EMBEDDINGS_MODEL),
        )
    except Exception:
        logger.exception(
            "Failed to load database from persist_directory={!r}", persist_directory
        )
        raise
    logger.success("Database loaded | persist_directory={!r}", persist_directory)
    return db



if __name__ == "__main__":
    summarised_chunks = process_pdfs()
    create_vector_database(summarised_chunks, db_name="chroma_db_v1")

