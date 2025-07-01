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

def check_ml_status():
    """Check the current status of ML models and student data"""
    
    print("🔍 SmartGrade ML System Status Check")
    print("=" * 50)
    
    # Check students
    student_count = Student.objects.count()
    print(f"👥 Students in database: {student_count}")
    
    if student_count > 0:
        print("   Sample student data:")
        sample_student = Student.objects.first()
        print(f"   - ID: {sample_student.sn}, Gender: {sample_student.gender}")
        print(f"   - Courses: CSC1201={sample_student.csc1201}, CSC1202={sample_student.csc1202}")
    
    print()
    
    # Check models
    models = MLModel.objects.all()
    print(f"🤖 ML Models in database: {models.count()}")
    
    if models.count() == 0:
        print("   ❌ No models found! Running setup...")
        from setup_ml import setup_default_models
        setup_default_models()
        models = MLModel.objects.all()
        print(f"   ✅ Created {models.count()} default models")
    
    print("\n📋 Model Status:")
    for model in models:
        status_icon = "✅" if model.training_status == 'trained' else "⏳" if model.training_status == 'training' else "❌"
        accuracy_text = f"({(model.accuracy*100):.1f}% accuracy)" if model.accuracy else "(no accuracy data)"
        print(f"   {status_icon} {model.name} - {model.training_status} {accuracy_text}")
    
    print()
    
    # Training recommendations
    trained_models = models.filter(training_status='trained')
    untrained_models = models.filter(training_status='untrained')
    
    if trained_models.count() == 0:
        print("🎯 NEXT STEPS:")
        print("   1. Go to Models tab in the frontend")
        print("   2. Click 'Train Model' on any untrained model")
        print("   3. Wait for training to complete")
        print("   4. Then go to Predictions tab to generate predictions")
    else:
        print(f"✅ READY FOR PREDICTIONS!")
        print(f"   - {trained_models.count()} trained models available")
        print(f"   - {student_count} students ready for prediction")
        print("   - Go to Predictions tab to generate predictions")
    
    if untrained_models.count() > 0:
        print(f"\n🔧 OPTIONAL: {untrained_models.count()} more models available for training")

if __name__ == "__main__":
    check_ml_status()
