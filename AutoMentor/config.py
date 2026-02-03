from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent

# Data Paths
SOURCE_DOCS_DIR = BASE_DIR / "data" / "source_docs"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index"

# Ollama Model Configuration
OLLAMA_MODEL = "llama3:latest"

# Embeddings Model
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
