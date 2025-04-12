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
    source_type: str  # Added to indicate whether response is from RAG or general knowledge

# Global variables
vector_store = None
rag_chain = None
llm = None  # Added to store the Gemini LLM for direct access

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
        temperature=0.4,
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
    global vector_store, rag_chain, llm
    
    # Check if data exists and load it
    csv_path = "Animal_disease.csv"
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

def is_disease_specific_question(question: str) -> bool:
    """
    Determine if a question is specifically about animal diseases (RAG dataset)
    or a general veterinary question.
    
    Args:
        question: The user's question
        
    Returns:
        bool: True if disease-specific, False if general veterinary question
    """
    # Define disease-related keywords that would indicate we should use RAG
    disease_keywords = [
        "disease", "infection", "syndrome", "pathogen", "bacteria", "virus", 
        "fungal", "parasite", "symptom", "treatment", "prognosis", "outbreak",
        "epidemic", "transmission", "contagious", "vaccine", "mortality",
        "lesion", "diagnosis", "antibiotic", "antiviral", "fever", "infection",
        "contamination", "pathology", "incubation"
    ]
    
    # Convert question to lowercase for case-insensitive matching
    question_lower = question.lower()
    
    # Check if any disease keywords are in the question
    if any(keyword in question_lower for keyword in disease_keywords):
        return True
    
    return False

async def get_general_veterinary_answer(question: str):
    """
    Handle general veterinary questions using Gemini directly
    
    Args:
        question: The user's question
        
    Returns:
        str: The answer from Gemini
    """
    global llm
    
    # If LLM not initialized, do it now
    if llm is None:
        llm = setup_gemini_llm()
    
    # Create a prompt template for general veterinary questions
    general_vet_prompt = PromptTemplate.from_template(
        """ You are a veterinary assistant providing information about animals and animal care.

        FIRST, explicitly evaluate if the question is about animals or veterinary topics:
        
        Step 1: SCOPE CHECK 
        - Is this question about animals, animal care, veterinary medicine, pet ownership, livestock management, wildlife, or any animal-related topic?
        - If NO, respond ONLY with: "I'm sorry, I can only answer questions related to animals and veterinary topics."
        - If YES, proceed to Step 2.
        
        Step 2: ANSWER FORMATION (only if Step 1 is YES)
        - Provide a clear, concise, and helpful answer about the animal-related topic.
        - Focus on factual information and best practices in veterinary care.
        - Avoid speculating or giving definitive medical advice that should come from a licensed veterinarian.
        - For serious medical conditions, recommend consulting a veterinarian.
        
        Question: {question}
        
        Answer:"""
    )
    
    # Format the prompt with the question
    formatted_prompt = general_vet_prompt.format(question=question)
    
    # Get response from Gemini
    response = await llm.ainvoke(formatted_prompt)
    
    return response.content

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
    if not rag_chain or not llm:
        try:
            initialize_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")
    
    try:
        # Measure performance
        start_time = time.time()
        
        # Determine if this is a disease-specific question or general veterinary question
        if is_disease_specific_question(query.question):
            # Get response from RAG chain for disease-specific questions
            result = rag_chain.invoke({"query": query.question})
            answer = result["result"]
            source_type = "rag_database"
        else:
            # Get response from general veterinary knowledge
            answer = await get_general_veterinary_answer(query.question)
            source_type = "general_knowledge"
        
        # Calculate metrics
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Prepare metrics
        metrics = {
            "time_taken": round(time_taken, 2),
            "model": "gemini-2.0-flash-lite"
        }
        
        return RAGResponse(
            answer=answer,
            metrics=metrics,
            source_type=source_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)