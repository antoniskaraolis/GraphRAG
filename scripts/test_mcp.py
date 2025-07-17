#!/usr/bin/env python3
# scripts/test_mcp.py
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", payload=None):
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=payload)
        
        response.raise_for_status()
        print(f"✅ {endpoint}: Success")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"{endpoint}: Failed - {str(e)}")
        return False

if __name__ == "__main__":
    # Test graph endpoints
    test_endpoint("/graph/stats")
    test_endpoint("/graph/search", "POST", {"query": "quantum computing", "top_k": 2})
    
    # Test RAG endpoints
    test_endpoint("/rag/query", "POST", {"question": "What are recent advances in quantum error correction?"})
    test_endpoint("/rag/context?node_id=paper_1234.56789")
    
    # Test cluster endpoints
    test_endpoint("/clusters/list")
    test_endpoint("/clusters/analyze")
