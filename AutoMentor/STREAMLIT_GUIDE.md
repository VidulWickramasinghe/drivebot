# AutoMentor Quick Start Guide

## Prerequisites
1. Install Python 3.9+
2. Install Ollama from https://ollama.ai/
3. Pull the LLM model: `ollama pull llama3:latest`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running AutoMentor with Streamlit UI

### Step 1: Ingest Documents (First Time Setup)
Place your car manuals (PDF, TXT, CSV) in `data/source_docs/` folder, then run:
```bash
python main.py ingest
```

### Step 2: Start the API Server
Open a terminal and run:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Step 3: Start the Streamlit Frontend
Open another terminal and run:
```bash
streamlit run app.py
```

The Streamlit app will open in your browser at http://localhost:8501

## Features

### 💬 Query Knowledge Base
- Chat with AutoMentor about automotive topics
- Conversation history maintained in session
- Real-time responses from local LLM

### 📁 Upload Documents
- Upload new documents directly from the UI
- Supports PDF, TXT, and CSV files
- Automatic ingestion and index updates

## Troubleshooting

If the API shows as disconnected:
1. Make sure Ollama is running: `ollama serve`
2. Ensure the API server is running on port 8000
3. Check that you've run the initial ingestion: `python main.py ingest`

## Alternative: CLI Mode
You can still use the CLI without the UI:
```bash
python main.py query
```
