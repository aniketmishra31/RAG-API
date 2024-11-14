from datetime import datetime
import PyPDF2
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import io
import chromadb
import uuid

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()

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

def create_collection(client):
    pdf_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_name = f"pdf_{pdf_id}_{timestamp}"
    try:
        collection = client.get_or_create_collection(collection_name)
    except Exception as e:
        print(f"Error creating or connecting to collection: {e}")
        collection = client.create_collection(collection_name)
    return collection_name,pdf_id
    
def store_embeddings_in_chroma(embeddings, chunks):
    collection_name , pdf_id = create_collection(client)
    ids = [str(i) for i in range(len(chunks))]  # Generate unique IDs for each chunk
    collection=client.get_or_create_collection(name=collection_name)
    collection.add(
        documents=chunks,
        metadatas=[{"pdf_id": pdf_id}] * len(chunks),
        ids=ids,
        embeddings=embeddings
    )
    return collection_name
def retrieve_relevant_chunks(query, collection_name, top_k=3):
    query_embedding = model.encode([query])
    collection=client.get_collection(collection_name)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    documents = [item for sublist in results['documents'] for item in sublist]  # Flatten the list
    return documents

def generate_response(query,context,API_KEY):
    genai.configure(api_key=str(API_KEY))
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    context = " ".join(context)  # Ensure context is a string
    prompt = f"User query: {query}\nContext: {context}\nAnswer:"
    
    response = model.generate_content(prompt)
    return response.text