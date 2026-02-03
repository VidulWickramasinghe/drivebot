from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from rich.console import Console
from core.rag import load_rag_chain
from core.ingestion.document_loader import split_text_into_chunks
from core.ingestion.embedder import create_and_save_embeddings
from langchain.schema import Document
import uvicorn
import asyncio
import config
import shutil
from pathlib import Path
from typing import List
import tempfile

app = FastAPI(
    title="AutoMentor API",
    description="Local automotive knowledge AI assistant API.",
    version="1.0.0"
)

# Use a global variable to store the RAG chain
# It will be loaded once when the application starts
rag_chain = None
console = Console()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.on_event("startup")
async def startup_event():
    """Load the RAG chain when the FastAPI application starts."""
    global rag_chain
    console.print("[bold green]Loading AutoMentor RAG chain for API...[/bold green]")
    try:
        rag_chain = load_rag_chain()
        console.print("[bold green]AutoMentor API Ready![/bold green]")
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        console.print("[bold yellow]Please run the 'python main.py ingest' command first to create the FAISS index.[/bold yellow]")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        console.print(f"[bold red]Failed to load RAG chain:[/bold red] {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize AutoMentor.")


@app.post("/query", response_model=QueryResponse)
async def query_automotive(request: QueryRequest):
    """
    Query the AutoMentor AI for automotive knowledge.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="AutoMentor is not initialized. Please check server logs.")
    
    try:
        # FastAPI runs handlers in a separate thread pool by default for sync functions.
        # For async, we can just call it directly.
        # Langchain's invoke is synchronous, so we run it in a thread pool.
        result = await asyncio.to_thread(rag_chain.invoke, {"question": request.question})
        return QueryResponse(answer=result['answer'])
    except Exception as e:
        console.print(f"[bold red]Error during query:[/bold red] {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your query.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "AutoMentor_initialized": rag_chain is not None}

@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Ingest new documents into the AutoMentor knowledge base.
    Accepts PDF, TXT, and CSV files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    console.print(f"[bold green]Received {len(files)} file(s) for ingestion[/bold green]")
    
    # Create a temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Save uploaded files to temp directory
            for file in files:
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in ['.pdf', '.txt', '.csv']:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file type: {file.filename}. Only PDF, TXT, and CSV are allowed."
                    )
                
                file_path = temp_path / file.filename
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                console.print(f"Saved {file.filename} to temp directory")
            
            # Load documents from temp directory
            from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader, PyPDFLoader
            
            documents = []
            for file_path in temp_path.iterdir():
                try:
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == '.csv':
                        loader = CSVLoader(str(file_path))
                    elif file_path.suffix.lower() == '.txt':
                        loader = TextLoader(str(file_path))
                    else:
                        continue
                    
                    docs = loader.load()
                    documents.extend(docs)
                    console.print(f"Loaded {len(docs)} chunks from {file_path.name}")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load {file_path.name}: {e}[/yellow]")
            
            if not documents:
                raise HTTPException(status_code=400, detail="No valid documents could be loaded from the uploaded files.")
            
            console.print(f"[bold blue]Total documents loaded: {len(documents)}[/bold blue]")
            
            # Split documents into chunks
            console.print("Splitting documents into chunks...")
            chunks = await asyncio.to_thread(
                split_text_into_chunks, 
                documents, 
                config.CHUNK_SIZE, 
                config.CHUNK_OVERLAP
            )
            console.print(f"[bold blue]Created {len(chunks)} text chunks[/bold blue]")
            
            # Create and save embeddings
            console.print("Generating embeddings and updating FAISS index...")
            await asyncio.to_thread(
                create_and_save_embeddings,
                chunks,
                config.FAISS_INDEX_PATH
            )
            console.print(f"[bold green]FAISS index updated at {config.FAISS_INDEX_PATH}[/bold green]")
            
            # Reload the RAG chain with updated index
            global rag_chain
            console.print("[bold yellow]Reloading RAG chain with new documents...[/bold yellow]")
            rag_chain = await asyncio.to_thread(load_rag_chain)
            console.print("[bold green]RAG chain reloaded successfully![/bold green]")
            
            return {
                "message": f"Successfully ingested {len(files)} file(s) with {len(chunks)} chunks.",
                "files_processed": [f.filename for f in files],
                "total_chunks": len(chunks)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            console.print(f"[bold red]Error during ingestion:[/bold red] {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred during ingestion: {str(e)}")

if __name__ == "__main__":
    # To run the API server, use: uvicorn api.server:app --reload --port 8000
    # For demonstration purposes, we'll start it programmatically
    # Note: When running directly via `python api/server.py`, `startup_event` might not be called by uvicorn.
    # It's better to run using `uvicorn api.server:app --host 0.0.0.0 --port 8000`
    
    console.print("[bold yellow]To start the API, run: uvicorn api.server:app --host 0.0.0.0 --port 8000[/bold yellow]")
    console.print("[bold yellow]Make sure to run 'python main.py ingest' first to create the FAISS index.[/bold yellow]")
    # For programmatic start, which might not correctly trigger startup_event in some environments
    # uvicorn.run(app, host="0.0.0.0", port=8000)
