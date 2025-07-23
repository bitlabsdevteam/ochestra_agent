from flask import Flask, request, jsonify, render_template, Response, stream_template
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from rags.data_retriever import DataRetriever
from dotenv import load_dotenv
import os
import json
import time
from typing import Generator

# Import VectorDB integration
from memory.pinecode.vectordb_manager import VectorDBManager

# Import OrchestraAgent
from agents.orchestra_agent import OrchestraAgent

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Swagger UI
SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI
API_URL = '/static/swagger.json'  # Our API url (can be a local file or url)

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "AI Agent API Documentation"
    }
)

# Register Swagger UI blueprint with Flask app
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# List of supported providers
SUPPORTED_PROVIDERS = {
    "openai": "OpenAI embedding model (text-embedding-ada-002)",
    "huggingface": "Hugging Face embedding models (sentence-transformers/all-mpnet-base-v2)",
    "gemini": "Google Gemini embedding models (embedding-001)"
}

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the AI Agent API",
        "endpoints": {
            "/api/docs": "GET - Interactive API documentation with Swagger UI",
            "/embedding/models": "GET - List available embedding models",
            "/embedding/providers": "GET - List available LLM providers",
            "/health": "GET - Health check",
            "/api/agent/chat": "POST - Chat with Orchestra Agent (non-streaming)",
            "/api/agent/chat/stream": "POST - Chat with Orchestra Agent (streaming)",
            "/api/agent/travel": "POST - Travel planning with Orchestra Agent (non-streaming)",
            "/api/agent/travel/stream": "POST - Travel planning with Orchestra Agent (streaming)",
            "/vectordb/create": "POST - Create vector database",
            "/vectordb/add": "POST - Add documents to vector database",
            "/vectordb/search": "POST - Search vector database",
            "/vectordb/delete": "DELETE - Delete vector database",
            "/client": "GET - Web interface to interact with the API"
        }
    })

@app.route('/client')
def client():
    return render_template('api_client.html')

# The /api/docs endpoint is now handled by the Swagger UI blueprint

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

# Orchestra Agent endpoints
@app.route('/api/agent/chat', methods=['POST'])
def chat_with_agent():
    """Non-streaming chat endpoint for Orchestra Agent."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        message = data['message']
        llm_provider = data.get('llm_provider', 'openai')
        temperature = data.get('temperature', 0.7)
        system_prompt = data.get('system_prompt')
        
        # Initialize agent
        agent = OrchestraAgent(
            llm_prefix=llm_provider,
            system_prompt=system_prompt,
            temperature=temperature,
            use_tools=False
        )
        
        # Process query
        response = agent.process_query(message, use_travel_agent=False)
        
        return jsonify({
            "response": response,
            "conversation_history": agent.get_conversation_history()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agent/travel', methods=['POST'])
def travel_with_agent():
    """Non-streaming travel planning endpoint for Orchestra Agent."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        message = data['message']
        llm_provider = data.get('llm_provider', 'openai')
        temperature = data.get('temperature', 0.7)
        system_prompt = data.get('system_prompt')
        
        # Initialize agent with tools
        agent = OrchestraAgent(
            llm_prefix=llm_provider,
            system_prompt=system_prompt,
            temperature=temperature,
            use_tools=True
        )
        
        # Process query with travel agent
        response = agent.process_query(message, use_travel_agent=True)
        
        return jsonify({
            "response": response,
            "conversation_history": agent.get_conversation_history(),
            "available_tools": [tool.name for tool in agent.get_available_tools()]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_streaming_response(agent: OrchestraAgent, message: str, use_travel_agent: bool = False) -> Generator[str, None, None]:
    """Generate streaming response from Orchestra Agent."""
    try:
        # Add user message to history
        agent.conversation_history.append({"role": "user", "content": message})
        
        if use_travel_agent and hasattr(agent, 'travel_agent_executor'):
            # For travel agent, we'll simulate streaming by chunking the response
            response = agent.travel_agent_executor.invoke({"input": message})
            raw_response = response.get("output", "I couldn't process that request.")
            
            try:
                parsed_response = agent._parse_travel_agent_response(raw_response)
                agent.conversation_history.append({"role": "assistant", "content": parsed_response["response"]})
                
                # Stream the parsed response
                yield f"data: {json.dumps({'type': 'thinking', 'content': parsed_response.get('thinking', '')})}\n\n"
                time.sleep(0.1)
                
                # Stream function calls if any
                for func_call in parsed_response.get('function_calls', []):
                    yield f"data: {json.dumps({'type': 'function_call', 'content': {'name': func_call.name, 'arguments': func_call.arguments}})}\n\n"
                    time.sleep(0.1)
                
                # Stream the final response in chunks
                response_text = parsed_response["response"]
                chunk_size = 20
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i+chunk_size]
                    yield f"data: {json.dumps({'type': 'response', 'content': chunk})}\n\n"
                    time.sleep(0.05)
                    
            except Exception as e:
                agent.conversation_history.append({"role": "assistant", "content": raw_response})
                # Stream raw response in chunks
                chunk_size = 20
                for i in range(0, len(raw_response), chunk_size):
                    chunk = raw_response[i:i+chunk_size]
                    yield f"data: {json.dumps({'type': 'response', 'content': chunk})}\n\n"
                    time.sleep(0.05)
        else:
            # For standard conversation, use LLM streaming if available
            messages = agent._convert_history_to_messages()
            
            # Check if LLM supports streaming
            if hasattr(agent.llm, 'stream'):
                response_content = ""
                for chunk in agent.llm.stream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        response_content += chunk.content
                        yield f"data: {json.dumps({'type': 'response', 'content': chunk.content})}\n\n"
                        time.sleep(0.01)
                agent.conversation_history.append({"role": "assistant", "content": response_content})
            else:
                # Fallback: simulate streaming by chunking the response
                response = agent.llm.invoke(messages)
                response_content = response.content
                agent.conversation_history.append({"role": "assistant", "content": response_content})
                
                chunk_size = 20
                for i in range(0, len(response_content), chunk_size):
                    chunk = response_content[i:i+chunk_size]
                    yield f"data: {json.dumps({'type': 'response', 'content': chunk})}\n\n"
                    time.sleep(0.05)
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

@app.route('/api/agent/chat/stream', methods=['POST'])
def stream_chat_with_agent():
    """Streaming chat endpoint for Orchestra Agent."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        message = data['message']
        llm_provider = data.get('llm_provider', 'openai')
        temperature = data.get('temperature', 0.7)
        system_prompt = data.get('system_prompt')
        
        # Initialize agent
        agent = OrchestraAgent(
            llm_prefix=llm_provider,
            system_prompt=system_prompt,
            temperature=temperature,
            use_tools=False
        )
        
        return Response(
            generate_streaming_response(agent, message, use_travel_agent=False),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agent/travel/stream', methods=['POST'])
def stream_travel_with_agent():
    """Streaming travel planning endpoint for Orchestra Agent."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        message = data['message']
        llm_provider = data.get('llm_provider', 'openai')
        temperature = data.get('temperature', 0.7)
        system_prompt = data.get('system_prompt')
        
        # Initialize agent with tools
        agent = OrchestraAgent(
            llm_prefix=llm_provider,
            system_prompt=system_prompt,
            temperature=temperature,
            use_tools=True
        )
        
        return Response(
            generate_streaming_response(agent, message, use_travel_agent=True),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize VectorDBManager
vector_db_manager = None
try:
    default_provider = os.environ.get('DEFAULT_EMBEDDING_PROVIDER', 'openai')
    vector_db_manager = VectorDBManager(provider_name=default_provider)
    print(f"Initialized VectorDBManager with {default_provider} provider")
except Exception as e:
    print(f"Warning: Failed to initialize VectorDBManager: {str(e)}")
    print("Vector database endpoints will return errors until configuration is fixed.")

# Vector database routes would be registered here if needed

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