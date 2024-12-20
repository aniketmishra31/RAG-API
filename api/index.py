from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from rag import checkApiKey,saveToDB,extract_text_from_pdf,chunk_text,embed_chunks,store_embeddings,retrieve_relevant_chunks,generate_response
import io
import os

load_dotenv()

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "https://rag-liard.vercel.app"}}, 
     methods=["GET", "POST", "PUT", "PATCH", "DELETE"], 
     allow_headers=["Content-Type", "Authorization", "X-apiKey"],
     supports_credentials=True)

app.config['ALLOWED_EXTENSIONS']={'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload-pdf', methods=['POST'])
def upload_and_load():
    try:
        user_id = request.form.get('user_id')
        title = request.form.get('title')
        if not user_id:
            return jsonify({"error": "Missing user_id in form data"}), 400
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}) , 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not file and not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        pdf_file=io.BytesIO(file.read())
        text=extract_text_from_pdf(pdf_file)
        
        document_id=saveToDB(text,user_id,title)
        
        chunks=chunk_text(text)
        embeddings=embed_chunks(chunks)
        embeddings_list=embeddings.tolist()
        pdf_id=store_embeddings(embeddings_list,chunks,document_id)
        
        return jsonify({"pdf_id": str(pdf_id),"message":"File embedded successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}) , 500
    
@app.route("/ask-pdf/<string:doc_id>/<string:pdf_id>",methods=['POST'])
def rag_generate(doc_id,pdf_id):
    try:
        query=request.json.get('query')
        ask_pdf_api_key=request.headers.get('X-apiKey')
        
        if not query:
            return jsonify({"error": "No query found in body"}) , 400
        if not ask_pdf_api_key:
            return jsonify({"error": "No API key found in request"}) , 400
        
        api_key = os.getenv("API_KEY")
        
        match=checkApiKey(doc_id=doc_id,api_key=ask_pdf_api_key)
        
        if match==False:
            return jsonify({"error": "Invalid API key in headers"}), 403
        
        if not api_key:
            return jsonify({"error": "No API key found"}), 403
        
        context=retrieve_relevant_chunks(query=query,pdf_id=pdf_id)
        response=generate_response(query=query,context=context,API_KEY=api_key)
        
        if len(response)==0:
            return jsonify({"error":"Error while generating request"}) , 500
        
        return jsonify({"response": response}) , 201
    except Exception as e:
        return jsonify({"error": str(e)}) , 500
    
if __name__ == '__main__':
    app.run(debug=True)