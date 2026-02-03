import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import config

def load_documents(source_dir: Path) -> list:
    """Loads all documents from the source directory."""
    
    # Define loader for different file types
    pdf_loader = DirectoryLoader(str(source_dir), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    csv_loader = DirectoryLoader(str(source_dir), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True)
    txt_loader = DirectoryLoader(str(source_dir), glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    
    # Load documents from all loaders
    all_loaders = [pdf_loader, csv_loader, txt_loader]
    documents = []
    for loader in all_loaders:
        documents.extend(loader.load())
        
    return documents

def split_text_into_chunks(documents: list) -> list:
    """Splits documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_and_save_embeddings(chunks: list, save_path: Path):
    """Generates embeddings and saves them to a FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL)
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(str(save_path))

def run_ingestion_pipeline():
    """Executes the full document ingestion pipeline."""
    print("Starting document ingestion...")
    
    # 1. Load documents
    print(f"Loading documents from {config.SOURCE_DOCS_DIR}...")
    documents = load_documents(config.SOURCE_DOCS_DIR)
    if not documents:
        print("No documents found. Please add documents to the 'data/source_docs' directory.")
        return
    print(f"Loaded {len(documents)} documents.")

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    chunks = split_text_into_chunks(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Create and save embeddings
    print("Generating embeddings and creating FAISS index...")
    create_and_save_embeddings(chunks, config.FAISS_INDEX_PATH)
    print(f"FAISS index created and saved at {config.FAISS_INDEX_PATH}")
    print("Ingestion complete.")

if __name__ == '__main__':
    # For testing the ingestion module directly
    config.SOURCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # Create a dummy file for testing
    (config.SOURCE_DOCS_DIR / "test.txt").write_text("This is a test document for AutoMentor.")
    run_ingestion_pipeline()
