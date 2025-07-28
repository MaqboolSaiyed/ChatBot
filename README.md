# Changi Airport RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides information about Changi Airport and Jewel Changi. The chatbot uses local LLMs through Ollama, FAISS for vector search, and FastAPI for the web interface.

## Features

- **Local LLM Integration**: Uses Ollama to run models locally
- **Efficient Retrieval**: FAISS for fast similarity search
- **Fallback Mechanism**: Multiple models for reliability
- **Web Interface**: Simple chat UI with FastAPI
- **Open Source**: No paid APIs required

## Tech Stack

- **Backend**: Python 3.9+
- **LLM**: Ollama with Mistral/Llama2
- **Vector Store**: FAISS
- **Embeddings**: all-minilm (local via Ollama)
- **Web Framework**: FastAPI
- **Frontend**: Simple HTML/JS chat interface

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download required Ollama models:
   ```bash
   ollama pull mistral
   ollama pull llama2
   ollama pull all-minilm
   ```

3. Run the web scraper (if needed):
   ```bash
   python webscrape.py
   ```

4. Ingest the data:
   ```bash
   python ingest.py
   ```

5. Start the server:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

6. Open `http://localhost:8000` in your browser

## Project Structure

- `app.py` - Main FastAPI application
- `ingest.py` - Data ingestion and vector store creation
- `webscrape.py` - Web scraper for Changi Airport data
- `requirements.txt` - Python dependencies
- `faiss_index/` - Vector store (created after ingestion)
- `static/` - Frontend assets

## Troubleshooting

- **Model not found**: Ensure you've pulled the required Ollama models
- **Vector store issues**: Delete the `faiss_index` directory and re-run `ingest.py`
- **Port in use**: Change the port in the uvicorn command

## License

MIT
