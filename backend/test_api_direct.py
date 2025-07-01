#!/usr/bin/env python3
"""
Direct API test script to debug the /api/ml/models/ endpoint
"""
import os
import sys
import django
from django.conf import settings
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from predictions.models import MLModel
from predictions.serializers import MLModelSerializer
from django.test import Client
from django.urls import reverse

def test_database():
    """Test what's in the database"""
    print("=== DATABASE TEST ===")
    models = MLModel.objects.all()
    print(f"Found {models.count()} models in database:")
    for model in models:
        print(f"  - ID: {model.id}, Name: {model.name}, Type: {model.model_type}, Status: {model.training_status}")
        print(f"    Accuracy: {model.accuracy}, F1: {model.f1_score}, Active: {model.is_active}")
    print()

def test_serializer():
    """Test the serializer directly"""
    print("=== SERIALIZER TEST ===")
    models = MLModel.objects.all()
    serializer = MLModelSerializer(models, many=True)
    data = serializer.data
    print(f"Serialized {len(data)} models:")
    for model_data in data:
        print(f"  - ID: {model_data['id']}, Name: {model_data['name']}, Type: {model_data['model_type']}")
        print(f"    Status: {model_data['training_status']}, Accuracy: {model_data.get('accuracy')}")
    print(f"Full serialized data: {json.dumps(data, indent=2)}")
    print()

def test_api_endpoint():
    """Test the API endpoint using Django test client"""
    print("=== API ENDPOINT TEST (Django Test Client) ===")
    try:
        client = Client()
        response = client.get('/api/ml/models/')
        print(f"Status Code: {response.status_code}")
        print(f"Content Type: {response.get('Content-Type')}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response contains {len(data)} models")
            for model in data:
                print(f"  - ID: {model.get('id')}, Name: {model.get('name')}, Type: {model.get('model_type')}")
                print(f"    Status: {model.get('training_status')}, Accuracy: {model.get('accuracy')}")
            print(f"Full response: {json.dumps(data, indent=2)}")
        else:
            print(f"Error response: {response.content.decode()}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()

def test_viewset_directly():
    """Test the viewset directly"""
    print("=== VIEWSET TEST ===")
    try:
        from predictions.views import MLModelViewSet
        from django.http import HttpRequest
        from rest_framework.request import Request
        
        # Create a mock request
        http_request = HttpRequest()
        http_request.method = 'GET'
        request = Request(http_request)
        
        # Create viewset instance
        viewset = MLModelViewSet()
        viewset.request = request
        viewset.format_kwarg = None
        
        # Get queryset and serializer
        queryset = viewset.get_queryset()
        print(f"Queryset contains {queryset.count()} models")
        
        for model in queryset:
            print(f"  - ID: {model.id}, Name: {model.name}, Status: {model.training_status}")
            
        # Test list method
        response = viewset.list(request)
        print(f"Viewset response status: {response.status_code}")
        print(f"Viewset response data: {response.data}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print()

if __name__ == "__main__":
    test_database()
    test_serializer()
    test_api_endpoint()
    test_viewset_directly()
