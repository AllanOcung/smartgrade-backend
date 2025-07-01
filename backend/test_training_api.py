#!/usr/bin/env python3
"""
Test the model training API endpoint directly
"""
import os
import sys
import django
from django.conf import settings
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from django.test import Client
from predictions.models import MLModel

def test_training_api():
    """Test the training endpoint"""
    print("=== TRAINING API TEST ===")
    
    # Get available models
    models = MLModel.objects.all()
    print(f"Available models: {models.count()}")
    for model in models:
        print(f"  - ID: {model.id}, Name: {model.name}, Status: {model.training_status}")
    
    if models.count() == 0:
        print("No models found!")
        return
    
    # Test training API for the first model
    client = Client()
    model_to_train = models.first()
    
    print(f"\n=== Testing training for model {model_to_train.id}: {model_to_train.name} ===")
    
    # Test with minimal payload
    payload = {
        'use_grid_search': True
    }
    
    print(f"Sending POST to /api/ml/models/{model_to_train.id}/train/ with payload: {payload}")
    
    response = client.post(
        f'/api/ml/models/{model_to_train.id}/train/',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    print(f"Response status: {response.status_code}")
    print(f"Response content: {response.content.decode()}")
    
    if response.status_code != 200:
        print(f"ERROR: Training failed with status {response.status_code}")
        if hasattr(response, 'json'):
            try:
                error_data = json.loads(response.content.decode())
                print(f"Error details: {error_data}")
            except:
                pass
    else:
        print("SUCCESS: Training request accepted")

if __name__ == "__main__":
    test_training_api()
