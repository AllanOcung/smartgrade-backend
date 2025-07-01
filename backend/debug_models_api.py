#!/usr/bin/env python
import os
import sys
import django
import json

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

django.setup()

from predictions.models import MLModel
from predictions.serializers import MLModelSerializer

def debug_ml_models():
    """Debug ML models API response"""
    
    print("🔍 ML Models Debug Check")
    print("=" * 40)
    
    # Check database directly
    models = MLModel.objects.all()
    print(f"📊 Models in database: {models.count()}")
    
    if models.count() == 0:
        print("❌ No models found in database!")
        return
    
    print("\n📋 Raw Database Models:")
    for model in models:
        print(f"   ID: {model.id}")
        print(f"   Name: {model.name}")
        print(f"   Type: {model.model_type}")
        print(f"   Status: {model.training_status}")
        print(f"   Active: {model.is_active}")
        print(f"   Created: {model.created_at}")
        print("   ---")
    
    # Check serialized response (what the API returns)
    print("\n📤 Serialized API Response:")
    serializer = MLModelSerializer(models, many=True)
    serialized_data = serializer.data
    
    print(json.dumps(serialized_data, indent=2, default=str))
    
    # Check for any issues
    print(f"\n✅ API would return {len(serialized_data)} models")
    
    # Test the actual API endpoint simulation
    from django.test import Client
    client = Client()
    
    try:
        response = client.get('/api/ml/models/')
        print(f"\n🌐 API Endpoint Test:")
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response Length: {len(data) if isinstance(data, list) else 'Not a list'}")
            print(f"   First Model: {data[0]['name'] if data and len(data) > 0 else 'None'}")
        else:
            print(f"   Error: {response.content}")
    except Exception as e:
        print(f"   API Test Failed: {str(e)}")

if __name__ == "__main__":
    debug_ml_models()
