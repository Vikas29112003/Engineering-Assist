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
import tempfile
import nltk
import speech_recognition as sr
import shutil

# Before any NLTK imports or operations
import os
import nltk

# Set NLTK data path - make sure this happens BEFORE any NLTK operations
nltk_data_path = "/tmp/nltk_data"
os.environ["NLTK_DATA"] = nltk_data_path
os.makedirs(nltk_data_path, exist_ok=True)

# Configure NLTK paths - modify this section to be more explicit
nltk.data.path = [nltk_data_path]  # Replace the paths instead of inserting

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})

# Get MongoDB URL from environment, with a fallback for testing
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")

collection_name = None
temperature = 0
db_name = "User"
client = MongoClient(MONGO_URL)
embeddings = HuggingFaceEmbeddings()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=temperature,
    groq_api_key=os.environ["GROQ_API_KEY"]
)

@app.route('/db_collection', methods=['POST'])
def db_collection():
    global collection_name
    data = request.json
    if not data or 'collection_name' not in data:
        return jsonify({'error': 'No collection_name provided'}), 400
    collection_name = data['collection_name']
    return jsonify({'message': f'Database and collection set to {db_name} and {collection_name}'})


@app.route('/voicetotext', methods=['POST'])
def voicetotext():
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file_path)
        r = sr.Recognizer()
        with sr.AudioFile(temp_file_path) as source:
            audio = r.record(source)
        os.remove(temp_file_path)
        return jsonify({'text': r.recognize_google(audio)})
    except Exception as e:
        print({'error': f'Voice to text failed: {str(e)}'})
        return jsonify({'error': f'Voice to text failed: {str(e)}'}), 500


@app.route('/settemperature', methods=['POST'])
def settemperature():
    global temperature, llm
    data = request.json
    if not data or 'temperature' not in data:
        return jsonify({'error': 'No temperature provided'}), 400
    
    temperature = float(data['temperature'])
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=temperature,
        groq_api_key=os.environ["GROQ_API_KEY"]
    )
    return jsonify({'message': f'Temperature set to {temperature}'})

@app.route('/send', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})

@app.route('/query', methods=['POST'])
def query():
    global collection_name, client, embeddings, llm
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    if not collection_name:
        return jsonify({'error': 'Collection name not set. Please call /db_collection first'}), 400
    
    try:
        query = data['question']
        if len(query) > 1000:
            error = "Query length should be less than 1000 characters"
            return jsonify({'error': error}), 400
        
        chat_history = data.get('history', [])
        db = client[db_name]
        collection = db[collection_name]
        index_name = collection_name + "_vector_index"
        
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=index_name
        )
        
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        formatted_history = "\n".join([
            f"User: {exchange['human']}\nAssistant: {exchange['assistant']}"
            for exchange in chat_history
        ])
        context = ""
        if formatted_history:
            context = f"Previous conversation:\n{formatted_history}\n\nCurrent question: "
        
        full_query = context + data['question']

        response = qa_chain.invoke({"query": full_query})
        ans = response['result']
        source_doc = response["source_documents"][0].metadata["source"]
        ans = ans + f"\n\nSource: {source_doc[5:-4]}"
        return jsonify({
            'answer': ans
        })
        
    except Exception as e:
        print({'error': f'Query failed: {str(e)}'})
        return jsonify({
            'error': f'Query failed: {str(e)}',
            'collection_name': collection_name
        }), 500


@app.route('/upload', methods=['POST'])
def upload():
    global client, db_name, embeddings, collection_name
    type = int(request.form.get('type', '0'))
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Initialize db at the beginning to avoid UnboundLocalError in except block
    db = client[db_name]
    temp_file_path = None
    custom_filename = None
    
    try:
        # Save uploaded file to a temporary writable location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file_path)
        
        # Use a writable path for the custom filename
        custom_filename = f"/tmp/{file.filename}"
        shutil.copy(temp_file_path, custom_filename)
        
        # Set collection name before trying to load documents
        collection_name = str(file.filename)
        if collection_name.endswith('.pdf'):
            collection_name = collection_name[:-4]
            
        # Configure loader to work without NLTK tokenizers
        # The "fast" strategy relies less on NLTK
        loader = UnstructuredPDFLoader(
            custom_filename,
            mode="elements",
            strategy="fast",
            # Skip NLTK-dependent processing
            skip_header_footer=True,
            paragraph_grouper=False
        )
        
        # Load documents
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        if type == 0:
            if collection_name in db.list_collection_names():
                db.drop_collection(collection_name)
            db.create_collection(collection_name)
        else:
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
        
        collection = db[collection_name]
        index_name = collection_name + "_vector_index"
        
        # Create search index if it doesn't exist
        if index_name not in [idx["name"] for idx in collection.list_search_indexes()]:
            collection.create_search_index(
                {
                    "definition": {
                        "mappings": {
                            "dynamic": True,
                            "fields": {
                                "embedding": {
                                    "type": "knnVector",
                                    "dimensions": 768,
                                    "similarity": "cosine"
                                }
                            }
                        }
                    },
                    "name": index_name
                }
            )
        
        # Add documents to vector store
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=index_name
        )
        vectorstore.add_documents(texts)
        
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if custom_filename and os.path.exists(custom_filename):
            os.remove(custom_filename)
        
        return jsonify({
            'message': 'File processed successfully',
            'collection_name': collection_name
        })
        
    except Exception as e:
        print({'error': f'Upload failed: {str(e)}'})
        # Only attempt to drop collection if it was created
        if collection_name and collection_name in db.list_collection_names():
            db.drop_collection(collection_name)
        # Clean up temporary files on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if custom_filename and os.path.exists(custom_filename):
            os.remove(custom_filename)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)