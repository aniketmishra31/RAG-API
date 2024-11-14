from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rag import extract_text_from_pdf,chunk_text,embed_chunks,store_embeddings_in_chroma,retrieve_relevant_chunks,generate_response
import io
import os

load_dotenv()

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS']={'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload-pdf', methods=['POST'])
def upload_and_load():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}) , 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not file and not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        pdf_file=io.BytesIO(file.read())
        text=extract_text_from_pdf(pdf_file)
        
        chunks=chunk_text(text)
        embeddings=embed_chunks(chunks)
        
        collection_name=store_embeddings_in_chroma(embeddings,chunks)
        
        return jsonify({"collection_name": collection_name,"message":"File embedded successfully"}), 201
    except Exception as e:
        return jsonify({"error": e}) , 500
    
@app.route("/ask-pdf",methods=['POST'])
def rag_generate():
    try:
        query=request.json.get('query')
        collection_name=request.headers.get('X-Collection')
        
        if not collection_name:
            return jsonify({"error": "Collection header missing"}) , 403
        if not query:
            return jsonify({"error": "No query found in body"}) , 400
        
        api_key = os.getenv("API_KEY")

        if not api_key:
            return jsonify({"error": "No API key found"}), 403
        
        context=retrieve_relevant_chunks(query=query,collection_name=collection_name)
        response=generate_response(query=query,context=context,API_KEY=api_key)
        
        if len(response)==0:
            return jsonify({"error":"Error while generating request"}) , 500
        
        return jsonify({"response": response}) , 201
    except Exception as e:
        return jsonify({"error": str(e)}) , 500
    
if __name__ == '__main__':
    app.run(debug=True)