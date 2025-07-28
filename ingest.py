"""Ingest scraped JSONL, create embeddings, and store in local FAISS index.
Run:
    python ingest.py --data scraped_data.jsonl --out faiss_index
"""
import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def embed_with_gemini(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title="Embedding"
    )
    return result['embedding']

class GeminiEmbeddings:
    def embed_query(self, text):
        return embed_with_gemini(text)
    def __call__(self, text):
        return self.embed_query(text)


def load_jsonl(path: Path):
    """Load records from a JSONL file and convert them to LangChain `Document`s."""
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text:
                continue
            source = obj.get("url", "")
            docs.append(Document(page_content=text, metadata={"source": source}))
    return docs


def build_faiss_index(documents, out_dir: Path):
    print(f"Splitting {len(documents)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Use the same embedding model as in app.py
    print("Initializing embedding model...")
    embedder = GeminiEmbeddings()
    
    # Test the embedding model
    try:
        test_embedding = embedder.embed_query("test")
        print(f"Embedding model initialized successfully. Vector dimensions: {len(test_embedding)}")
    except Exception as e:
        print(f"Error testing embedding model: {e}")
        raise
    
    print("Creating FAISS index...")
    try:
        vectordb = FAISS.from_documents(chunks, embedder)
        print(f"FAISS index created with {vectordb.index.ntotal} vectors")
        
        # Save the index
        vectordb.save_local(str(out_dir))
        print(f"FAISS index saved to {out_dir}")
        
        # Print index info
        print("\n=== FAISS Index Information ===")
        print(f"Number of vectors: {vectordb.index.ntotal}")
        print(f"Vector dimensions: {vectordb.index.d}")
        print("=" * 30 + "\n")
        
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create FAISS vector index from scraped data.")
    parser.add_argument("--data", default="scraped_data.jsonl", help="Path to JSONL file with scraped data.")
    parser.add_argument("--out", default="faiss_index", help="Directory to store FAISS index.")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    documents = load_jsonl(data_path)
    if not documents:
        raise ValueError("No valid documents found in the supplied JSONL file.")

    print(f"Loaded {len(documents)} raw documents. Splitting and embedding…")
    build_faiss_index(documents, out_path)
    print(f"FAISS index saved to {out_path.absolute()}")
