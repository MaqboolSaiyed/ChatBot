"""
Main entry point for Changi Airport RAG Chatbot.
This file is required by Google Cloud Run to automatically detect and run the application.
"""

import uvicorn
from app import app

if __name__ == "__main__":
    # Run with production settings
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(8080),
        workers=1,
        log_level="info"
    )
