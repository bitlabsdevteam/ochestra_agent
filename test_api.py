import requests
import json
import sys

def test_api():
    # Base URL for the API
    base_url = "http://localhost:8080"  # Updated port to 8080
    
    try:
        # Test the home endpoint
        print("Testing home endpoint...")
        response = requests.get(base_url, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("\n" + "-"*50 + "\n")
        
        # Test the embedding-model endpoint with OpenAI provider
        print("Testing embedding-model endpoint with OpenAI provider...")
        response = requests.get(f"{base_url}/embedding-model", params={"provider": "openAI"}, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Check if API key is configured
        if response.json().get("api_key_configured"):
            print("✅ API key is configured for OpenAI")
        else:
            print("⚠️ API key is not configured for OpenAI")
            
        print("\n" + "-"*50 + "\n")
        
        # Test the embedding-model endpoint with Hugging Face provider
        print("Testing embedding-model endpoint with Hugging Face provider...")
        response = requests.get(f"{base_url}/embedding-model", params={"provider": "huggingface"}, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Check if API key is configured
        if response.json().get("api_key_configured"):
            print("✅ API key is configured for Hugging Face")
        else:
            print("⚠️ API key is not configured for Hugging Face")
            
        print("\n" + "-"*50 + "\n")
        
        # Test the embedding-model endpoint with Gemini provider
        print("Testing embedding-model endpoint with Gemini provider...")
        response = requests.get(f"{base_url}/embedding-model", params={"provider": "gemini"}, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Check if API key is configured
        if response.json().get("api_key_configured"):
            print("✅ API key is configured for Gemini")
        else:
            print("⚠️ API key is not configured for Gemini")
            
        print("\n" + "-"*50 + "\n")
        
        # Test the embedding-model endpoint with a non-supported provider
        print("Testing embedding-model endpoint with a non-supported provider...")
        response = requests.get(f"{base_url}/embedding-model", params={"provider": "other_provider"}, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("\n" + "-"*50 + "\n")
        
        # Test the embedding-model endpoint without provider parameter
        print("Testing embedding-model endpoint without provider parameter...")
        response = requests.get(f"{base_url}/embedding-model", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("\n" + "-"*50 + "\n")
        
        # Test the providers endpoint
        print("Testing providers endpoint...")
        response = requests.get(f"{base_url}/providers", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("\n" + "-"*50 + "\n")
        
        # Test the health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return True
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: Could not connect to the server at {base_url}")
        print("Make sure the server is running by executing: python app.py")
        return False
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    print("This script tests the Sales Maker API.")
    print("Make sure the API server is running before executing this script.")
    print("You can start the server by running: python app.py")
    print("\n" + "-"*50 + "\n")
    
    # Ask for confirmation before proceeding
    proceed = input("Do you want to proceed with the API test? (y/n): ")
    if proceed.lower() == 'y':
        success = test_api()
        if not success:
            sys.exit(1)
    else:
        print("Test aborted.")