import urllib.request
import json

try:
    response = urllib.request.urlopen('http://localhost:8000/api/ml/models/')
    data = response.read().decode('utf-8')
    parsed = json.loads(data)
    
    print(f"Status: {response.getcode()}")
    print(f"Content-Type: {response.getheader('Content-Type')}")
    print(f"Data type: {type(parsed)}")
    print(f"Data length: {len(parsed) if isinstance(parsed, list) else 'Not a list'}")
    print(f"First few characters: {data[:200]}...")
    
    if isinstance(parsed, list):
        print(f"Found {len(parsed)} models:")
        for model in parsed:
            print(f"  - {model.get('id')}: {model.get('name')} ({model.get('model_type')})")
    else:
        print("Response is not a list:", parsed)
        
except Exception as e:
    print(f"Error: {e}")
