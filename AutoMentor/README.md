# AutoMentor

**Created by:** stevyOP / ggDevLabs  
**Project:** AutoMentor - A Local Automotive Knowledge AI Assistant

## 1. Project Overview

AutoMentor is a powerful, fully offline, and domain-specific AI assistant designed to provide comprehensive automotive knowledge. Leveraging local Ollama LLM models and a Retrieval-Augmented Generation (RAG) pipeline, AutoMentor delivers accurate answers to questions regarding vehicle specifications, maintenance procedures, troubleshooting, and automotive best practices.

The system is built for complete autonomy, operating without any internet connectivity for its core functions. It efficiently ingests various local document types—such as PDF manuals, CSV specification sheets, and plain text guides—to construct a rich, searchable knowledge base. This base then allows AutoMentor to provide precise, context-driven responses to user queries, ensuring reliability and relevance.

## 2. Key Features

-   **Fully Offline Operation:** All functionalities, including the language model and vector store, run entirely on your local machine without requiring internet access.
-   **Local LLM Integration:** Seamlessly integrates with Ollama to utilize local models like `llama3:latest`, ensuring data privacy and reducing latency.
-   **Retrieval-Augmented Generation (RAG):** Enhances answer accuracy and mitigates hallucinations by grounding responses in the provided, relevant documents.
-   **Multi-Document Ingestion:** Capable of processing and incorporating knowledge from PDF, CSV, and TXT files into a unified and queryable knowledge base.
-   **Efficient Vector Storage:** Utilizes FAISS (Facebook AI Similarity Search) for high-performance local vector indexing and rapid similarity searches.
-   **Multi-Turn Conversation Memory:** Maintains context throughout a conversation session, enabling natural follow-up questions and more coherent interactions.
-   **Flexible Interface Options:** Offers both a Command-Line Interface (CLI) for direct interaction and an optional FastAPI backend for integration into other applications or services.

## 3. Project Folder Structure

```
AutoMentor/
├── README.md
├── requirements.txt
├── main.py                     # CLI entry point (ingest & query commands)
├── config.py                   # Centralized configuration settings
├── core/
│   ├── __init__.py
│   ├── ingestion.py            # Document loading, chunking, embedding, FAISS index creation
│   ├── rag.py                  # RAG chain setup (retriever, LLM, prompt, memory)
│
├── data/
│   ├── source_docs/            # Place your car manuals (PDF, CSV, TXT) here
│   └── vector_store/           # FAISS vector store will be saved here (e.g., faiss_index)
│
├── api/
│   └── server.py               # Optional FastAPI backend for local API
│
└── examples/
    ├── query1.txt
    └── query2.txt
```

## 4. Installation

**Prerequisites:**
-   **Python 3.9+:** Ensure Python is installed on your system.
-   **Ollama:** Download and install Ollama from [ollama.ai](https://ollama.ai/).

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AutoMentor
    ```

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the local LLM via Ollama:**
    Ensure your Ollama server is running. Then, pull the desired model. The default configured in `config.py` is `llama3:latest`.
    ```bash
    ollama pull llama3:latest
    ```
    *You can change `OLLAMA_MODEL` in `config.py` to any other model available on your local Ollama server.*

## 5. Usage

### 5.1. Adding Knowledge Documents

Place all your automotive documents (PDF manuals, CSV specification sheets, plain `.txt` files) into the `data/source_docs/` directory.

### 5.2. Ingesting Documents (Building the Knowledge Base)

Before you can query AutoMentor, you need to ingest your documents to build the searchable knowledge base (FAISS index).
This command loads documents, splits them into chunks, generates embeddings, and saves the FAISS index to `data/vector_store/`.

```bash
python main.py ingest
```
You should see progress messages in the console indicating document loading, chunking, and index creation.

### 5.3. Querying AutoMentor via CLI

You can interact with AutoMentor through the command-line interface.

**For an interactive chat session with conversational memory:**

```bash
python main.py query
```
Once the RAG chain loads, you'll be prompted to ask questions. Type `exit` or `quit` to end the session.

```
Welcome to AutoMentor AI!
Loading the RAG chain... This might take a moment.
AutoMentor Ready! Ask me anything about your vehicles.
You: What is the battery capacity of the Nissan Leaf 40kWh?
AutoMentor: The Nissan Leaf with the 40kWh designation has a battery capacity of 40 kilowatt-hours.
--------------------
You: How long does it take to charge?
AutoMentor: (AutoMentor provides charging times based on the context of the 40kWh Nissan Leaf and retrieved documents)
--------------------
You: exit
Goodbye!
```

### 5.4. Optional: Running the FastAPI Backend

If you wish to expose AutoMentor's capabilities via a local API, you can run the FastAPI server.

1.  **Ensure the FAISS index is already created** by running `python main.py ingest`.
2.  **Start the API server:**
    ```bash
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
    ```
    The `--reload` flag is optional but useful during development.
3.  **Access the API:**
    -   **Interactive Docs:** Open your browser to `http://localhost:8000/docs` to see the OpenAPI/Swagger UI.
    -   **Health Check:** `http://localhost:8000/health`
    -   **Query Endpoint:** `POST` requests to `http://localhost:8000/query` with a JSON body like:
        ```json
        {
          "question": "How do I troubleshoot engine misfire in a Toyota Corolla?"
        }
        ```

## 6. Example Queries

Here are some example questions you can ask AutoMentor:

-   `"What is the battery capacity of Nissan Leaf 40kWh?"`
-   `"How to troubleshoot engine misfire in Toyota Corolla?"`
-   `"Compare charging times for 2023 Tesla Model 3 vs 2023 Nissan Leaf."`
-   `"What is the recommended tire pressure for a 2020 Honda Civic?"`
-   `"Explain the function of an EGR valve."`
-   `"What are the common causes of excessive oil consumption in older vehicles?"`

## 7. Developer Notes

-   **How to Add New Car Manuals/Documents:**
    Simply place any new `.pdf`, `.csv`, or `.txt` files containing automotive information into the `data/source_docs/` directory. AutoMentor's ingestion pipeline is configured to automatically detect and process these file types.

-   **How to Retrain/Update Embeddings:**
    After adding new documents to `data/source_docs/`, you must re-run the ingestion pipeline to update the knowledge base:
    ```bash
    python main.py ingest
    ```
    This process will regenerate the embeddings and update the FAISS index with the new information. The current implementation rebuilds the index entirely. For very large datasets, an incremental update strategy could be implemented, but for a prototype, this full rebuild is simpler and effective.

-   **Customizing the LLM:**
    You can easily switch to a different local Ollama model by modifying the `OLLAMA_MODEL` variable in `config.py` (e.g., to `codellama:latest` or another compatible model you have pulled).

-   **Adjusting RAG Parameters:**
    Parameters such as `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py` can be tuned to optimize document splitting for better retrieval performance. Experimenting with `search_kwargs={"k": 5}` in `core/rag.py` (which controls how many document chunks are retrieved) can also impact answer quality.
