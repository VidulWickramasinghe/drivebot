from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from AutoMentor import config # Import config at the top level

def create_and_save_embeddings(chunks: list, save_path: Path): # Removed embeddings_model_name argument
    """Generates embeddings and saves them to a FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL) # Use config directly
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    # Ensure the parent directory exists before saving
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(save_path))

if __name__ == '__main__':
    # For testing the embedder module directly
    from langchain.schema import Document
    
    # Create dummy chunks
    dummy_chunks = [
        Document(page_content="This is the first part of a test document."),
        Document(page_content="This is the second part, which talks about cars."),
        Document(page_content="Automotive knowledge is very important.")
    ]
    
    print(f"Creating and saving embeddings to {config.FAISS_INDEX_PATH}...")
    create_and_save_embeddings(dummy_chunks, config.FAISS_INDEX_PATH) # Removed embeddings_model_name argument
    print("Embeddings created and saved successfully.")
    
    # Verify by loading
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL)
    loaded_vector_store = FAISS.load_local(str(config.FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    print(f"Loaded vector store with {len(loaded_vector_store.index_to_docstore_id)} documents.")