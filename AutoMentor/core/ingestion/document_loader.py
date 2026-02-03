import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, DirectoryLoader

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

def split_text_into_chunks(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    """Splits documents into smaller chunks for embedding."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == '__main__':
    # For testing the document_loader module directly
    from AutoMentor import config
    config.SOURCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # Create a dummy file for testing
    (config.SOURCE_DOCS_DIR / "test_loader.txt").write_text("This is a test document for AutoMentor loader.")
    
    print(f"Loading documents from {config.SOURCE_DOCS_DIR}...")
    docs = load_documents(config.SOURCE_DOCS_DIR)
    print(f"Loaded {len(docs)} documents.")
    
    chunks = split_text_into_chunks(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks.")
