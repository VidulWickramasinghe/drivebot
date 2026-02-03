from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rich.console import Console
from core.rag import load_rag_chain
import uvicorn
import asyncio

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

if __name__ == "__main__":
    # To run the API server, use: uvicorn api.server:app --reload --port 8000
    # For demonstration purposes, we'll start it programmatically
    # Note: When running directly via `python api/server.py`, `startup_event` might not be called by uvicorn.
    # It's better to run using `uvicorn api.server:app --host 0.0.0.0 --port 8000`
    
    console.print("[bold yellow]To start the API, run: uvicorn api.server:app --host 0.0.0.0 --port 8000[/bold yellow]")
    console.print("[bold yellow]Make sure to run 'python main.py ingest' first to create the FAISS index.[/bold yellow]")
    # For programmatic start, which might not correctly trigger startup_event in some environments
    # uvicorn.run(app, host="0.0.0.0", port=8000)
