"""FastAPI RAG chatbot serving answers about Changi / Jewel Changi Airport.

Usage:
    uvicorn app:app --reload --port 8000
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Model configuration
class ModelType(str, Enum):
    PRIMARY = "mistral"
    FALLBACK = "llama2"
    SUMMARIZER = "mistral"
    EMBEDDING = "all-minilm"  # Using all-minilm which is more commonly available

# Response quality thresholds
MIN_ANSWER_LENGTH = 30
MIN_ANSWER_QUALITY = 0.5  # Subjective quality threshold (0-1)

# Load environment variables
load_dotenv()

# Configuration
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", ModelType.EMBEDDING)
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", ModelType.PRIMARY)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", ModelType.FALLBACK)
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", ModelType.SUMMARIZER)

# Configure Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model for generation
gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

# Embedding function using Gemini
# (Gemini embedding API: model='models/embedding-001')
def embed_with_gemini(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title="Embedding"
    )
    return result['embedding']

# Replace OllamaEmbeddings with Gemini embedding wrapper
class GeminiEmbeddings:
    def embed_query(self, text):
        return embed_with_gemini(text)
    def __call__(self, text):
        return self.embed_query(text)
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

embeddings = GeminiEmbeddings()

# Load FAISS index
index_path = Path(INDEX_DIR)
if index_path.exists():
    try:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded FAISS index with {vectorstore.index.ntotal} vectors")
        
        # Print index diagnostics
        print("\n=== FAISS Index Diagnostics ===")
        print(f"Index type: {type(vectorstore.index).__name__}")
        print(f"Number of dimensions: {vectorstore.index.d}")
        print(f"Number of vectors: {vectorstore.index.ntotal}")
        
        # Check embedding dimensions
        if vectorstore.index.ntotal > 0:
            try:
                sample_vector = vectorstore.index.reconstruct(0)
                print(f"Sample vector shape: {len(sample_vector) if sample_vector is not None else 'None'}")
                print(f"Sample vector first 5 dims: {sample_vector[:5] if sample_vector is not None else 'None'}")
            except Exception as e:
                print(f"Error checking vector dimensions: {e}")
        
        print("=" * 30 + "\n")
        
    except Exception as e:
        print(f"\n!!! ERROR LOADING FAISS INDEX !!!")
        print(f"Error: {str(e)}")
        print("Please make sure you've run ingest.py with the correct embedding model.")
        print("Trying to delete and recreate the index...")
        import shutil
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
        print("Please run ingest.py to recreate the index with the correct embedding model.")
        raise
else:
    raise FileNotFoundError(f"\n!!! FAISS index not found at {INDEX_DIR} !!!\nPlease run ingest.py first to create the index.")

# Set up retrievers with different search strategies
print("\nSetting up retrievers...")
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 15, "lambda_mult": 0.5}
)
print("MMR retriever configured")

retriever_similarity = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)
print("Similarity retriever configured")

# Initialize FastAPI
app = FastAPI(title="Changi RAG Chatbot", version="1.0")
templates = Jinja2Templates(directory="templates")

# Serve static files
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to load chat interface: {str(e)}"}
        )

# -------------------------
# Response Handling
# -------------------------
DEFAULT_RESPONSE = (
    "I couldn't find specific information about that in Changi Airport's resources. "
    "Please check the official website or contact Changi Airport directly for the most accurate details."
)

COMMON_QUESTIONS = {
    "hi": "Hello! I'm here to help with information about Changi Airport and Jewel Changi. What would you like to know?",
    "hello": "Hi there! I can help you find information about Changi Airport facilities, services, and more. What would you like to know?",
    "thanks": "You're welcome! Is there anything else you'd like to know about Changi Airport?",
    "thank you": "You're welcome! Feel free to ask if you have more questions about Changi Airport.",
    "help": "I can help you find information about Changi Airport's facilities, services, shopping, dining, and more. Just ask me a question!"
}

# Summarization prompt
SUMMARIZE_PROMPT = """
Please rewrite the following response to be more natural and conversational, while keeping all key information:

Original: {text}

Rewritten in a friendly, helpful tone:"""

def preprocess_question(q: str) -> str:
    """Clean and normalize the user's question."""
    q = q.lower().strip()
    q = re.sub(r'\s+', ' ', q)  # Normalize whitespace
    q = re.sub(r'[?.,!;]*$', '', q)  # Remove trailing punctuation
    if not q.endswith('?'):
        q += '?'
    return q

def format_answer(answer: str) -> str:
    """Clean up the model's response."""
    if not answer or not answer.strip():
        return DEFAULT_RESPONSE
    
    # Clean up common issues
    answer = re.sub(r'(?i)\b(?:as an ai language model|i am an ai|i don\'t have real-time information)[^.]*\.?', '', answer)
    answer = answer.strip()
    
    # Ensure proper punctuation
    if not answer.endswith(('.', '!', '?')):
        answer = answer.rstrip('.,;:') + '.'
    
    # Capitalize first letter
    if len(answer) > 1:
        answer = answer[0].upper() + answer[1:]
    
    return answer


# -------------------------
# API Models
# -------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., example="What free rest areas are available at Changi Airport?")

class Source(BaseModel):
    url: str
    text_snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []

# -------------------------
# Chat Endpoint
# -------------------------
async def generate_with_fallback(question: str, context: str) -> dict:
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = gemini_model.generate_content(prompt)
    answer = response.text.strip() if hasattr(response, 'text') else str(response)
    return {"answer": answer, "source_docs": [], "model": "gemini-pro"}

async def summarize_response(text: str) -> str:
    """Improve response quality using summarization model."""
    if not text or len(text) < 50:  # Don't summarize very short texts
        return text
        
    try:
        print("\n--- Attempting to improve answer quality ---")
        prompt = f"""Please improve the following response to be more natural, concise and helpful:
        
        Original response: {text}
        
        Improved response:"""
        
        # Generate the improved response
        response = gemini_model.generate_content(prompt)
        
        if response and hasattr(response, 'text') and response.text:
            improved_text = response.text.strip()
            print(f"Improved response: {improved_text[:200]}...")
            return improved_text
        
        print("No valid response from summarization model")
        return text
        
    except Exception as e:
        print(f"Error in summarization: {e}")
        return text  # Return original text if summarization fails

def is_high_quality(answer: str) -> bool:
    """Check if the answer meets quality thresholds."""
    if not answer or len(answer) < MIN_ANSWER_LENGTH:
        return False
    
    # Check for common low-quality indicators
    low_quality_phrases = [
        "i don't know", 
        "no information", 
        "i couldn't find", 
        "i don't have",
        "i don't understand"
    ]
    
    if any(phrase in answer.lower() for phrase in low_quality_phrases):
        return False
        
    return True

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    print(f"\n=== New Request ===")
    print(f"Question: {req.question}")
    
    # Validate input
    if not req.question or not req.question.strip():
        print("Error: Empty question")
        return ChatResponse(answer="Please enter a question about Changi Airport.")
    
    # Handle common greetings
    question = preprocess_question(req.question)
    print(f"Processed question: {question}")
    
    if question.lower() in (q.lower() for q in COMMON_QUESTIONS):
        print("Matched common question")
        return ChatResponse(answer=COMMON_QUESTIONS[question.lower()])
    
    try:
        print("\n=== Starting Document Retrieval ===")
        print(f"Question: {question}")
        
        # Test direct embedding of the question
        try:
            question_embedding = embeddings.embed_query(question)
            print(f"Question embedding dimensions: {len(question_embedding)}")
            print(f"First 5 dimensions: {question_embedding[:5]}")
        except Exception as e:
            print(f"Error creating question embedding: {e}")
        
        # Test MMR retriever
        print("\nTesting MMR retriever...")
        try:
            docs_mmr = retriever_mmr.invoke(question)
            print(f"Retrieved {len(docs_mmr)} documents from MMR retriever")
            if docs_mmr:
                print(f"First document content: {docs_mmr[0].page_content[:200]}...")
                print(f"First document metadata: {docs_mmr[0].metadata}")
        except Exception as e:
            print(f"Error in MMR retriever: {e}")
            docs_mmr = []
        
        # Test similarity retriever
        print("\nTesting similarity retriever...")
        try:
            docs_similarity = retriever_similarity.invoke(question)
            print(f"Retrieved {len(docs_similarity)} documents from similarity retriever")
            if docs_similarity:
                print(f"First document content: {docs_similarity[0].page_content[:200]}...")
                print(f"First document metadata: {docs_similarity[0].metadata}")
        except Exception as e:
            print(f"Error in similarity retriever: {e}")
            docs_similarity = []
        
        # Combine and deduplicate documents
        all_docs = []
        seen = set()
        for doc in docs_mmr + docs_similarity:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen:
                seen.add(doc_hash)
                all_docs.append(doc)
                
        print(f"\nTotal unique documents after deduplication: {len(all_docs)}")
        
        if not all_docs:
            print("\n!!! WARNING: No documents retrieved !!!")
            print("Possible causes:")
            print("1. The vector database might be empty")
            print("2. The embedding model might not match the one used to create the index")
            print("3. The retrievers might be too strict with similarity thresholds")
            return ChatResponse(answer=DEFAULT_RESPONSE)
        
        # Log first document for debugging
        if all_docs:
            print(f"First document content: {all_docs[0].page_content[:200]}...")
            print(f"First document metadata: {all_docs[0].metadata}")
        
        # Create a dictionary to store unique documents
        unique_docs = {}
        for doc in all_docs:
            if doc.page_content:  # Only include non-empty documents
                unique_docs[doc.page_content[:200]] = doc  # Use start of content as key
        
        if not unique_docs:
            print("No valid documents after filtering")
            return ChatResponse(answer=DEFAULT_RESPONSE)
        
        # 2. Generate answer with fallback
        context = "\n".join([doc.page_content for doc in unique_docs.values()])
        print(f"Context length: {len(context)} characters")
        
        print("Generating answer...")
        # Get the answer as a string, not a coroutine
        answer_obj = await generate_with_fallback(question, context)
        answer = answer_obj if isinstance(answer_obj, str) else answer_obj.get("answer", "")
        print(f"Generated answer: {answer[:200]}...")
        
        # 3. Improve answer quality if needed
        if not is_high_quality(answer):
            print("Answer quality low, attempting to improve...")
            improved_answer = await summarize_response(answer)
            if improved_answer:
                answer = improved_answer
                print(f"Improved answer: {answer[:200]}...")
            else:
                print("Could not improve answer quality")
        
        # 4. Format response with sources
        sources = [
            Source(
                url=doc.metadata.get('source', 'Unknown'),
                text_snippet=doc.page_content[:200] + "..."
            )
            for doc in unique_docs.values()
        ]
        
        # 5. Final formatting
        improved_answer = format_answer(answer) if answer else "I couldn't generate a proper response. Please try again."
        print(f"Final answer: {improved_answer[:200]}...")
        
        return ChatResponse(answer=improved_answer, sources=sources)
    except Exception as e:
        import traceback
        error_msg = f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}"
        print("\n!!! ERROR !!!")
        print(error_msg)
        print("\n")
        return ChatResponse(
            answer="I'm having trouble processing your request. The error has been logged. Please try again with a different question.",
            sources=[]
        )


@app.get("/chat-ui", response_class=HTMLResponse)
async def chat_page():
    # Minimal inlined HTML/JS chat interface
    return """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='utf-8'>
        <title>Changi RAG Chatbot</title>
        <style>
            body{font-family:Arial,Helvetica,sans-serif;background:#f5f5f5;margin:0;padding:1rem}
            #chat{max-width:800px;margin:0 auto;background:#fff;border:1px solid #ddd;border-radius:6px;padding:1rem;height:70vh;overflow-y:auto}
            .msg{margin:0.5rem 0;padding:0.5rem;border-radius:4px}
            .user{background:#d1e7ff;align-self:flex-end}
            .bot{background:#e2e2e2}
            #inputBox{width:80%;padding:0.5rem}
            #sendBtn{padding:0.55rem 1rem}
        </style>
    </head>
    <body>
        <h2>Changi & Jewel Chatbot</h2>
        <div id='chat'></div>
        <div style='margin-top:1rem;'>
            <input id='inputBox' placeholder='Type your question...' />
            <button id='sendBtn'>Send</button>
        </div>
        <script>
            const chatDiv=document.getElementById('chat');
            const input=document.getElementById('inputBox');
            const btn=document.getElementById('sendBtn');
            function addMsg(text,cls){
                const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;chatDiv.appendChild(d);chatDiv.scrollTop=chatDiv.scrollHeight;
            }
            async function send(){
                const q=input.value.trim();if(!q)return;addMsg(q,'user');input.value='';
                try{
                    const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
                    const data=await res.json();addMsg(data.answer,'bot');
                }catch(e){addMsg('Error: '+e,'bot');}
            }
            btn.onclick=send;input.addEventListener('keydown',e=>{if(e.key==='Enter'){send();}});
        </script>
    </body>
    </html>
    """



