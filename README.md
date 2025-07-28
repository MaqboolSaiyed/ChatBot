# Changi Airport RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides information about Changi Airport and Jewel Changi. The chatbot uses Gemini API for LLM and embeddings, FAISS for vector search, and FastAPI for the web interface.

## Features

- **Cloud LLM Integration**: Uses Gemini API for generation
- **Efficient Retrieval**: FAISS for fast similarity search
- **Web Interface**: Simple chat UI with FastAPI
- **Open Source**: No paid APIs required for vector search

## Tech Stack

- **Backend**: Python 3.9+
- **LLM**: Gemini Pro (Google Generative AI)
- **Vector Store**: FAISS
- **Embeddings**: Gemini Embeddings (cloud via Gemini API)
- **Web Framework**: FastAPI
- **Frontend**: Simple HTML/JS chat interface

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Gemini API key (get it from Google AI Studio):
   - Create a `.env` file with:
     ```
     GEMINI_API_KEY=your_api_key_here
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
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. Open `http://localhost:8000` in your browser

## Project Structure

- `main.py` - Main FastAPI application
- `ingest.py` - Data ingestion and vector store creation
- `webscrape.py` - Web scraper for Changi Airport data
- `requirements.txt` - Python dependencies
- `faiss_index/` - Vector store (created after ingestion)
- `static/` - Frontend assets

## Troubleshooting

- **API key issues**: Ensure your `.env` file is set up and the key is valid
- **Vector store issues**: Delete the `faiss_index` directory and re-run `ingest.py`
- **Port in use**: Change the port in the uvicorn command

## License

MIT
