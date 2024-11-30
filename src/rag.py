from datetime import datetime
import PyPDF2
from sentence_transformers import SentenceTransformer
from db import db
import google.generativeai as genai
import numpy as np
import io
import os
import uuid
import requests
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    embeddings = model.encode(chunks)  
    return embeddings

def create_pdf_id():
    pdf_id = str(uuid.uuid4())[:8]
    return pdf_id
    
def store_embeddings(embeddings_list, chunks):
    pdf_id = create_pdf_id()
    for embedding, chunk in zip(embeddings_list, chunks):
        response = db.table("embeddings").insert({
            "embedding": embedding,
            "metadata": pdf_id,
            "documents": chunk
        }).execute()
    return pdf_id

def retrieve_relevant_chunks(query, pdf_id, top_k=3):
    try:
        query_embedding = model.encode([query])
        query_list = query_embedding.tolist()[0]  # Extract the vector as a list

        query_embedding_str = str(query_list).replace('[', '{').replace(']', '}')
        response=db.rpc("sim_search",{
            "query_embedding": query_embedding_str,
            "pdf_id_param" : pdf_id,
            "top_k":top_k
        }).execute()

        if not response.data:
            raise Exception("No results found.")
        relevant_chunks = [row['documents'] for row in response.data]
        return relevant_chunks
    except Exception as e:
        raise Exception({"error": str(e)})

def generate_response(query,context,API_KEY):
    genai.configure(api_key=str(API_KEY))
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    context = " ".join(context)  # Ensure context is a string
    prompt = f"User query: {query}\nContext: {context}\nAnswer:"
    
    response = model.generate_content(prompt) 
    return response.text

def saveToDB(text,user_id,title):
    try:
        data={
            "text":text,
            "user_id": user_id,
            "title": title
        }
        response=db.table("documents").insert(data).execute()
    except Exception as e:
        raise Exception({"error": str(e)})