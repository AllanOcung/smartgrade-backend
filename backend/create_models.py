#!/usr/bin/env python
import os
import sys
import django

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

django.setup()

from predictions.models import MLModel
from students.models import Student

def create_default_models():
    """Create default ML models for the system"""
    
    print("🤖 Creating Default ML Models...")
    print("=" * 40)
    
    # Check if models already exist
    existing_models = MLModel.objects.count()
    if existing_models > 0:
        print(f"ℹ️  Found {existing_models} existing models")
        for model in MLModel.objects.all():
            print(f"   - {model.name} ({model.training_status})")
        return
    
    default_models = [
        {
            'name': 'Random Forest Classifier',
            'model_type': 'random_forest',
            'description': 'High-performance ensemble method with feature importance',
            'is_active': True,
        },
        {
            'name': 'Logistic Regression Classifier',
            'model_type': 'logistic_regression',
            'description': 'Fast and interpretable model for dropout prediction',
            'is_active': True,
        },
        {
            'name': 'Gradient Boosting Classifier',
            'model_type': 'gradient_boosting',
            'description': 'Advanced boosting algorithm for complex patterns',
            'is_active': True,
        },
        {
            'name': 'Neural Network Classifier',
            'model_type': 'neural_network',
            'description': 'Deep learning model for complex pattern recognition',
            'is_active': True,
        }
    ]
    
    created_count = 0
    
    for model_data in default_models:
        model, created = MLModel.objects.get_or_create(
            name=model_data['name'],
            model_type=model_data['model_type'],
            defaults={
                'description': model_data['description'],
                'is_active': model_data['is_active'],
                'training_status': 'untrained',
                'training_data_count': 0,
            }
        )
        
        if created:
            print(f"✅ Created: {model.name}")
            created_count += 1
        else:
            print(f"ℹ️  Exists: {model.name}")
    
    print(f"\n🎉 Setup complete! Created {created_count} new models.")
    
    # Check student data
    student_count = Student.objects.count()
    print(f"👥 Students in database: {student_count}")
    
    if student_count == 0:
        print("⚠️  No student data found. Please upload CSV data first!")
    else:
        print("✅ Student data available for training.")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Go to the Models tab in the frontend")
    print("2. Click 'Train Model' on any model (recommend Random Forest first)")
    print("3. Wait for training to complete")
    print("4. Generate predictions in the Predictions tab")

if __name__ == "__main__":
    create_default_models()
