import typer
from rich.console import Console
from rich.markdown import Markdown

from core.ingestion.document_loader import load_documents, split_text_into_chunks
from core.ingestion.embedder import create_and_save_embeddings
from core.rag.rag import load_rag_chain # Explicitly import rag.py's function
import config

app = typer.Typer()
console = Console()

@app.command()
def ingest():
    """
    Run the document ingestion pipeline.
    Loads documents, creates embeddings, and saves them to the FAISS vector store.
    """
    console.print("[bold green]Starting document ingestion...[/bold green]")
    
    # 1. Load documents
    console.print(f"Loading documents from [yellow]{config.SOURCE_DOCS_DIR}[/yellow]...")
    documents = load_documents(config.SOURCE_DOCS_DIR)
    if not documents:
        console.print("[bold red]No documents found.[/bold red] Please add documents to the 'data/source_docs' directory.")
        return
    console.print(f"Loaded [bold blue]{len(documents)}[/bold blue] documents.")

    # 2. Split documents into chunks
    console.print("Splitting documents into chunks...")
    chunks = split_text_into_chunks(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    console.print(f"Created [bold blue]{len(chunks)}[/bold blue] text chunks.")

    # 3. Create and save embeddings
    console.print("Generating embeddings and creating FAISS index...")
    create_and_save_embeddings(chunks, config.FAISS_INDEX_PATH)
    console.print(f"FAISS index created and saved at [yellow]{config.FAISS_INDEX_PATH}[/yellow]")
    console.print("[bold green]Ingestion complete.[/bold green]")

@app.command()
def query():
    """
    Start an interactive query session with AutoMentor.
    Type 'exit' or 'quit' to end the session.
    """
    console.print("[bold cyan]Welcome to AutoMentor AI![/bold cyan]")
    console.print("Loading the RAG chain... This might take a moment.")
    
    try:
        rag_chain = load_rag_chain()
        console.print("[bold green]Ready to assist. Ask me anything about your vehicles.[/bold green]")
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        console.print("Please run the 'ingest' command first.")
        raise typer.Exit()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred while loading RAG chain:[/bold red] {e}")
        raise typer.Exit()

    while True:
        try:
            question = console.input("[bold]You: [/bold]")
            if question.lower() in ["exit", "quit"]:
                console.print("[bold cyan]Goodbye![/bold cyan]")
                break
            
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                result = rag_chain.invoke({"question": question})
                answer_md = Markdown(result['answer'])

            console.print("\n[bold]AutoMentor:[/bold]")
            console.print(answer_md)
            console.print("-" * 20)

        except KeyboardInterrupt:
            console.print("\n[bold cyan]Goodbye![/bold cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred during query:[/bold red] {e}")

if __name__ == "__main__":
    # Ensure data directories exist
    config.SOURCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    app()