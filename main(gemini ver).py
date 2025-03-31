from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
from typing import Dict
import time
import google.generativeai as genai
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    try:
        initialize_model()
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
    yield
    # Shutdown code (if any)
    
# Initialize FastAPI app
app = FastAPI(title="Animal Disease RAG API", lifespan=lifespan)

# Pydantic models
class Query(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    metrics: Dict

# Global variables
vector_store = None
rag_chain = None

def load_documents(csv_path: str):
    """Load documents from CSV file"""
    try:
        loader = CSVLoader(file_path=csv_path, encoding='utf-8')
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        raise

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def setup_vector_store(documents, save_path: str = "FAISS"):
    """Initialize or load the vector store using FAISS"""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Create a new vector store if the directory doesn't exist
    if not os.path.exists(save_path):
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model
        )
        vector_store.save_local(save_path)
    else:
        # Load existing vector store
        vector_store = FAISS.load_local(
            folder_path=save_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    
    return vector_store

def setup_gemini_llm():
    """Setup Google Gemini LLM"""
    # Get API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", 
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=512
    )
    
    return llm

def create_rag_chain(vector_store, llm):
    """Create RAG chain with Gemini LLM and the vector store"""
    rag_prompt = PromptTemplate.from_template(
        """Answer the question only if it is explicitly about veterinary diseases or animal health. Follow these steps:  
        1. Check Scope:
           - If the question is unrelated to veterinary topics, say: "I don't know. My expertise is limited to veterinary diseases and animal health."  
           - Do not use the context for non-veterinary questions.
        2. Veterinary Answers:  
           - If veterinary-related, answer directly and concisely using the provided context.  
           - Do not add self-generated questions, hypothetical scenarios, or unrelated topics.
           - Only supplement with veterinary knowledge if the context is insufficient. 
        Context: {context} 
        Question: {question}"""
    )
    
    # Create the RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True
    )
    
    return chain

def initialize_model():
    """Initialize the RAG model with all components"""
    global vector_store, rag_chain
    
    # Check if data exists and load it
    csv_path = "Animal disease spreadsheet - Sheet1.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    # Load and process documents
    documents = load_documents(csv_path)
    split_docs = split_documents(documents)
    
    # Setup vector store
    vector_store = setup_vector_store(split_docs)
    
    # Setup Gemini LLM
    llm = setup_gemini_llm()
    
    # Create RAG chain
    rag_chain = create_rag_chain(vector_store, llm)



@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {
        "message": "Animal Disease RAG API is running",
        "status": "healthy" if rag_chain is not None else "model not initialized"
    }

@app.post("/ask", response_model=RAGResponse)
async def ask_question(query: Query):
    """Endpoint to answer veterinary questions"""
    if not rag_chain:
        try:
            initialize_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
    
    try:
        # Measure performance
        start_time = time.time()
        
        # Get response from RAG chain
        result = rag_chain.invoke({"query": query.question})
        
        # Calculate metrics
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Prepare metrics (no token count since we're using API)
        metrics = {
            "time_taken": round(time_taken, 2),
            "model": "gemini-2.0-flash-lite"
        }
        
        return RAGResponse(
            answer=result["result"],
            metrics=metrics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main(gemini ver):app", host="localhost", port=8000, reload=True)