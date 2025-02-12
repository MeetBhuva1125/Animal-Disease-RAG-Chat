from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import uvicorn
from typing import Dict
import os

# Initialize FastAPI app
app = FastAPI(title="Animal Disease RAG API")

# Simplified Pydantic models
class Query(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    metrics: Dict

# Initialize global variables
vector_store = None
rag_chain = None
tokenizer = None

def load_and_process_data(csv_path: str):
    loader = CSVLoader(file_path=csv_path, encoding='utf-8')
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def initialize_vector_store(documents, save_path: str = "FAISS"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    vector_store = FAISS.from_documents(
        documents,
        embedding_model
    )
    vector_store.save_local(save_path)
    return vector_store

def initialize_model():
    global vector_store, rag_chain, tokenizer
    
    # Check if data exists and load it
    csv_path = "Animal disease spreadsheet - Sheet1.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    # Process data and initialize vector store if it doesn't exist
    if not os.path.exists("FAISS"):
        documents = load_and_process_data(csv_path)
        vector_store = initialize_vector_store(documents)
    else:
        # Load existing vector store
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vector_store = FAISS.load_local(
            folder_path='FAISS',
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    
    # Initialize model
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 8-bit configuration
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    # Initialize pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        do_sample=True,
        return_full_text=False
    )
    
    # Initialize LLM and RAG chain
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
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
        Question: {question} 
        <|assitant|>"""
    )
    
    global rag_chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True
    )

@app.on_event("startup")
async def startup_event():
    try:
        initialize_model()
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Animal Disease RAG API is running",
        "status": "healthy" if rag_chain is not None else "model not initialized"
    }

@app.post("/ask", response_model=RAGResponse)
async def ask_question(query: Query):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        import time
        start_time = time.time()
        
        # Get response from RAG chain
        result = rag_chain.invoke({"query": query.question})
        
        # Calculate metrics
        end_time = time.time()
        time_taken = end_time - start_time
        tokens = tokenizer(result["result"], return_tensors="pt").input_ids.shape[1]
        tokens_per_second = tokens / time_taken
        
        # Prepare metrics
        metrics = {
            "time_taken": round(time_taken, 2),
            "tokens_generated": tokens,
            "tokens_per_second": round(tokens_per_second, 2)
        }
        
        return RAGResponse(
            answer=result["result"],
            metrics=metrics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)