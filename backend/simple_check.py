#!/usr/bin/env python3
"""
Simple debug script to check models
"""
import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from predictions.models import MLModel

def check_models():
    """Check models in database"""
    models = MLModel.objects.all()
    print(f"Found {models.count()} models:")
    for model in models:
        print(f"  ID: {model.id} | Name: {model.name} | Type: {model.model_type} | Status: {model.training_status}")

if __name__ == "__main__":
    check_models()
