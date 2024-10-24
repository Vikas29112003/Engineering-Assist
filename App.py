import os
from flask_cors import CORS
from flask import Flask, request, jsonify
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from pymongo import MongoClient
import gc
import threading
from functools import partial

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})

# Environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URL = os.environ.get("MONGO_URL")

if not GROQ_API_KEY or not MONGO_URL:
    raise ValueError("Missing required environment variables")

# Global variables with better memory management
class AppState:
    def __init__(self):
        self.collection_name = None
        self.temperature = 0
        self.db_name = "User"
        self._client = None
        self._embeddings = None
        self._llm = None
        self.lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            self._client = MongoClient(MONGO_URL)
        return self._client

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings()
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatGroq(
                model_name="llama-3.1-70b-versatile",
                temperature=self.temperature,
                groq_api_key=GROQ_API_KEY
            )
        return self._llm

    def reset_llm(self):
        self._llm = None

state = AppState()

def process_pdf_in_batches(file_path, batch_size=5):
    """Process PDF documents in smaller batches to manage memory"""
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Reduced chunk size
    texts = text_splitter.split_documents(documents)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        yield batch
        gc.collect()  # Force garbage collection after each batch

@app.route('/db_collection', methods=['POST'])
def db_collection():
    data = request.json
    if not data or 'collection_name' not in data:
        return jsonify({'error': 'No collection_name provided'}), 400
    
    with state.lock:
        state.collection_name = data['collection_name']
    
    return jsonify({'message': f'Database and collection set to {state.db_name} and {state.collection_name}'})

@app.route('/settemperature', methods=['POST'])
def settemperature():
    data = request.json
    if not data or 'temperature' not in data:
        return jsonify({'error': 'No temperature provided'}), 400
    
    with state.lock:
        state.temperature = float(data['temperature'])
        state.reset_llm()
    
    return jsonify({'message': f'Temperature set to {state.temperature}'})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    if not state.collection_name:
        return jsonify({'error': 'Collection name not set. Please call /db_collection first'}), 400
    
    try:
        chat_history = data.get('history', [])
        db = state.client[state.db_name]
        collection = db[state.collection_name]

        # Use smaller retrieval batch size
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=state.embeddings,
            index_name="default"
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Limit number of retrieved documents
        qa_chain = RetrievalQA.from_chain_type(
            llm=state.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Limit chat history to last 5 exchanges
        recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
        formatted_history = "\n".join([
            f"User: {exchange['human']}\nAssistant: {exchange['assistant']}"
            for exchange in recent_history
        ])
        
        context = f"Previous conversation:\n{formatted_history}\n\nCurrent question: " if formatted_history else ""
        full_query = context + data['question']

        response = qa_chain.invoke({"query": full_query})
        gc.collect()  # Force garbage collection after query
        
        return jsonify({
            'answer': response['result'],
            'collection': state.collection_name
        })
        
    except Exception as e:
        print({'error': f'Query failed: {str(e)}'})
        return jsonify({
            'error': f'Query failed: {str(e)}',
            'collection_name': state.collection_name
        }), 500

@app.route('/upload', methods=['POST'])
def upload():
    type = int(request.form.get('type', '0'))
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    temp_path = os.path.join('/tmp', file.filename)
    try:
        file.save(temp_path)
        collection_name = str(file.filename)
        if collection_name.endswith('.pdf'):
            collection_name = collection_name[:-4]

        db = state.client[state.db_name]
        
        # Handle collection creation/deletion
        if type == 0:
            if collection_name in db.list_collection_names():
                db.drop_collection(collection_name)
            db.create_collection(collection_name)
        else:
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
        
        collection = db[collection_name]
        
        # Create or update index
        if "vector_index" in collection.list_search_indexes():
            collection.drop_index("vector_index")
        collection.create_search_index(
            {"definition": {
                "mappings": {"dynamic": True, "fields": {
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": 768,
                        "similarity": "cosine"
                    }}}},
             "name": "default"
            }
        )

        # Process PDF in batches
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=state.embeddings,
            index_name="vector_index"
        )

        for batch in process_pdf_in_batches(temp_path):
            vectorstore.add_documents(batch)
            gc.collect()  # Force garbage collection after each batch
        
        return jsonify({
            'message': 'File processed successfully',
            'collection_name': collection_name
        })
        
    except Exception as e:
        print({'error': f'Upload failed: {str(e)}'})
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)