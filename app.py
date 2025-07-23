from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rags.data_retriever import DataRetriever
from dotenv import load_dotenv
import os

# Import VectorDB integration
from memory.pinecone.api_integration import register_vectordb_routes
from memory.pinecone.vectordb_manager import VectorDBManager

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# List of supported providers
SUPPORTED_PROVIDERS = {
    "openai": "OpenAI embedding model (text-embedding-ada-002)",
    "huggingface": "Hugging Face embedding models (sentence-transformers/all-mpnet-base-v2)",
    "gemini": "Google Gemini embedding models (embedding-001)"
}

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Sales Maker API",
        "endpoints": {
            "/embedding-model": "Get embedding model based on provider name",
            "/providers": "Get list of supported providers",
            "/health": "Check API health status",
            "/client": "Web interface to interact with the API",
            "/api/vectordb/health": "Check vector database health",
            "/api/vectordb/add-documents": "Add documents to vector database",
            "/api/vectordb/query": "Query the vector database",
            "/api/vectordb/delete-index": "Delete the vector database index"
        }
    })

@app.route('/client')
def client():
    return render_template('api_client.html')

@app.route('/embedding-model', methods=['GET'])
def get_embedding_model():
    provider_name = request.args.get('provider', '')
    
    if not provider_name:
        return jsonify({
            "error": "Missing 'provider' parameter"
        }), 400
    
    data_retriever = DataRetriever(provider_name)
    embedding_model = data_retriever.get_embedding_model()
    has_api_key = data_retriever.has_valid_api_key()
    
    response = {
        "provider": provider_name,
        "embedding_model": embedding_model,
        "api_key_configured": has_api_key
    }
    
    # Add a warning message if no API key is configured
    if not has_api_key and provider_name.lower() in ["openai", "huggingface", "gemini"]:
        response["warning"] = f"No API key configured for {provider_name}. Please set the appropriate environment variable."
    
    return jsonify(response)

@app.route('/providers', methods=['GET'])
def get_providers():
    return jsonify({
        "providers": SUPPORTED_PROVIDERS
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    })

# Initialize VectorDBManager
vector_db_manager = None
try:
    default_provider = os.environ.get('DEFAULT_EMBEDDING_PROVIDER', 'openai')
    vector_db_manager = VectorDBManager(provider_name=default_provider)
    print(f"Initialized VectorDBManager with {default_provider} provider")
except Exception as e:
    print(f"Warning: Failed to initialize VectorDBManager: {str(e)}")
    print("Vector database endpoints will return errors until configuration is fixed.")

# Register vector database routes
register_vectordb_routes(app)

if __name__ == '__main__':
    # Get configuration from environment variables
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get("DEBUG", "True").lower() == "true"
    
    # Make sure the templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    # Print configuration information
    print(f"Starting Sales Maker API on port {port}")
    print(f"Debug mode: {debug_mode}")
    print(f"Environment variables loaded from .env file")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=debug_mode)