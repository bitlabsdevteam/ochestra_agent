# Project Setup Guide

## Create Project Folder and Environment Setup

```bash
# Create a new project folder
mkdir <project_folder_name>

# Move into the project folder
cd <project_folder_name>

# Open the folder in VS Code
code .

# Create a new Conda environment with Python 3.10
conda create -p <env_name> python=3.10 -y

# Activate the environment (use full path to the environment)
conda activate <path_of_the_env>

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Initialize Git
git init

# Stage all files
git add .

# Commit changes
git commit -m "<write your commit message>"

# Push to remote (after adding remote origin)
git push

# Cloning the repository
git clone https://github.com/sunnysavita10/document_portal.git
```
## Minimum Requirements for the Project

### LLM Models
- **Groq** (Free)
- **OpenAI** (Paid)
- **Gemini** (15 Days Free Access)
- **Claude** (Paid)
- **Hugging Face** (Free)
- **Ollama** (Local Setup)

### Embedding Models
- **OpenAI**
- **Hugging Face**
- **Gemini**

### Vector Databases
- **In-Memory**
- **On-Disk**
- **Cloud-Based**
  - **Pinecone**: Vector database for similarity search

## API Keys

### GROQ API Key
- [Get your API Key](https://console.groq.com/keys)  
- [Groq Documentation](https://console.groq.com/docs/overview)

### Gemini API Key
- [Get your API Key](https://aistudio.google.com/apikey)  
- [Gemini Documentation](https://ai.google.dev/gemini-api/docs/models)

### Pinecone API Key
- [Get your API Key](https://app.pinecone.io/)  
- [Pinecone Documentation](https://docs.pinecone.io/docs/overview)

## API Usage

The project includes a REST API for interacting with the system. Here's how to use it:

### Environment Configuration

The application uses environment variables for configuration. Create a `.env` file in the project root directory based on the provided `.env.example` file:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your configuration
vim .env  # or use any text editor
```

Key environment variables include:
- `PORT`: The port on which the API server will run (default: 8080)
- `DEBUG`: Enable/disable debug mode (default: True)
- `OPENAI_API_KEY`: Your OpenAI API key for embedding models
- `GEMINI_API_KEY`: Your Google Gemini API key
- `HUGGINGFACE_API_KEY`: Your Hugging Face API key
- `WEATHERAPI_KEY`: Your WeatherAPI.com API key for weather data
- `PINECONE_API_KEY`: Your Pinecone API key for vector database
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (default: gcp-starter)
- `PINECONE_INDEX_NAME`: Name of your Pinecone index (default: sales-maker-index)

### Starting the API Server

```bash
# Install the required dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```

The server will start on http://localhost:8080 (or the port specified in your .env file)

### API Endpoints

#### 1. Home Endpoint

- **URL**: `/`
- **Method**: GET
- **Description**: Returns information about available endpoints
- **Example**:
  ```bash
  curl http://localhost:8080/
  ```

#### 2. Embedding Model Endpoint

- **URL**: `/embedding-model`
- **Method**: GET
- **Parameters**: `provider` (optional) - The name of the provider (e.g., 'openAI', 'huggingface', 'gemini')
- **Description**: Returns the appropriate embedding model based on the provider name
- **Example**:
  ```bash
  curl http://localhost:8080/embedding-model?provider=openAI
  ```

#### 3. Providers Endpoint

- **URL**: `/providers`
- **Method**: GET
- **Description**: Returns a list of all supported providers and their descriptions
- **Example**:
  ```bash
  curl http://localhost:8080/providers
  ```

#### 4. Health Endpoint

- **URL**: `/health`
- **Method**: GET
- **Description**: Returns the health status of the API
- **Example**:
  ```bash
  curl http://localhost:8080/health
  ```

#### 5. Web Client Interface

- **URL**: `/client`
- **Method**: GET
- **Description**: Provides a web-based interface to interact with the API
- **Usage**: Open http://localhost:8080/client in your web browser

### Testing the API

A test script is provided to demonstrate how to use the API:

```bash
python test_api.py
```

This script will test all available endpoints and display the results.

Alternatively, you can use the web client interface by navigating to http://localhost:8080/client in your web browser, which provides an interactive way to test all API endpoints.

## Vector Database Integration

The project includes a Pinecone vector database integration for storing and retrieving embeddings.

### VectorDBManager

The `VectorDBManager` class in `memory/pinecode/vectordb_manager.py` provides the following functionality:

- **Initialization**: Connect to Pinecone using your API key
- **Index Creation**: Create a new Pinecone index with appropriate dimensions
- **Document Addition**: Add documents to the vector database
- **Retrieval**: Query the vector database and retrieve relevant documents

### Example Usage

```python
from memory.pinecone.vectordb_manager import VectorDBManager
from langchain.schema.document import Document

# Initialize with your preferred embedding provider
manager = VectorDBManager(provider_name="openai")

# Create an index
manager.create_index()

# Add documents
documents = [
    Document(page_content="Your text here", metadata={"source": "example"})
]
manager.add_documents(documents, namespace="example")

# Query the database
results = manager.query("Your query here", namespace="example", top_k=4)

# Get a retriever for use with LangChain
retriever = manager.get_retriever(namespace="example")
```

A complete example is available in `memory/pinecode/example_usage.py`.


