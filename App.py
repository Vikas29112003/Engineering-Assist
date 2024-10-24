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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MONGO_URL = os.environ.get("MONGO_URL")

if not GROQ_API_KEY or not MONGO_URL:
    raise ValueError("Missing required environment variables")

collection_name = None
temperature = 0
db_name = "User"
client = MongoClient(MONGO_URL)
embeddings = HuggingFaceEmbeddings()

llm = ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=temperature,
    groq_api_key=GROQ_API_KEY
)

# Your existing routes remain the same
@app.route('/db_collection', methods=['POST'])
def db_collection():
    global collection_name
    data = request.json
    if not data or 'collection_name' not in data:
        return jsonify({'error': 'No collection_name provided'}), 400
    collection_name = data['collection_name']
    return jsonify({'message': f'Database and collection set to {db_name} and {collection_name}'})

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
        groq_api_key=GROQ_API_KEY
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
        chat_history = data.get('history', [])
        db = client[db_name]
        collection = db[collection_name]

        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="default"
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
        
        return jsonify({
            'answer': ans,
            'collection': collection_name
        })
        
    except Exception as e:
        print({'error': f'Query failed: {str(e)}'})
        return jsonify({
            'error': f'Query failed: {str(e)}',
            'collection_name': collection_name
        }), 500

@app.route('/upload', methods=['POST'])
def upload():
    global client, db_name, embeddings
    type = int(request.form.get('type', '0'))
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        temp_path = os.path.join('/tmp', file.filename)
        file.save(temp_path)
        loader = UnstructuredPDFLoader(temp_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        db = client[db_name]
        collection_name = str(file.filename)
        if collection_name.endswith('.pdf'):
            collection_name = collection_name[:-4]
        if type == 0:
            if collection_name in db.list_collection_names():
                db.drop_collection(collection_name)
            db.create_collection(collection_name)
            collection = db[collection_name]
        else:
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
            collection = db[collection_name]
        
        if "vector_index" in collection.list_search_indexes():
            collection.drop_index("vector_index")
        collection.create_search_index(
            {"definition":
                {"mappings": {"dynamic": True, "fields": {
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": 768,
                        "similarity": "cosine"
                        }}}},
            "name": "default"
            }
        )

        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index"
        )
        vectorstore.add_documents(texts)
        
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'message': 'File processed successfully',
            'collection_name': collection_name
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print({'error': f'Upload failed: {str(e)}'})
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)