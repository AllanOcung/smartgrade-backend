#!/usr/bin/env python
import os
import sys
import django

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

django.setup()

from predictions.models import MLModel
from predictions.ml_service import ml_service

def setup_default_models():
    """Create default ML models for testing"""
    
    default_models = [
        {
            'name': 'Logistic Regression Classifier',
            'model_type': 'logistic_regression',
            'description': 'Fast and interpretable model for dropout prediction',
            'is_active': True,
        },
        {
            'name': 'Random Forest Classifier',
            'model_type': 'random_forest',
            'description': 'High-performance ensemble method with feature importance',
            'is_active': True,
        },
        {
            'name': 'Gradient Boosting Classifier',
            'model_type': 'gradient_boosting',
            'description': 'Advanced boosting algorithm for complex patterns',
            'is_active': False,
        },
        {
            'name': 'Neural Network Classifier',
            'model_type': 'neural_network',
            'description': 'Deep learning model for complex pattern recognition',
            'is_active': False,
        }
    ]
    
    created_models = []
    
    for model_data in default_models:
        model, created = MLModel.objects.get_or_create(
            name=model_data['name'],
            model_type=model_data['model_type'],
            defaults={
                'description': model_data['description'],
                'is_active': model_data['is_active'],
                'training_status': 'not_trained',
                'version': '1.0'
            }
        )
        
        if created:
            created_models.append(model)
            print(f"✓ Created model: {model.name}")
        else:
            print(f"→ Model already exists: {model.name}")
    
    return created_models

def test_ml_service():
    """Test the ML service functionality"""
    print("\n🧪 Testing ML Service...")
    
    # Check if we have any trained models
    from student_data.models import StudentRecord
    
    student_count = StudentRecord.objects.count()
    print(f"✓ Found {student_count} student records for training")
    
    if student_count == 0:
        print("⚠️  No student data available for training. Please upload some CSV data first.")
        return False
    
    # Get first model for testing
    first_model = MLModel.objects.filter(is_active=True).first()
    if not first_model:
        print("❌ No active models found")
        return False
    
    print(f"✓ Testing with model: {first_model.name}")
    
    # Test training (this might take a while)
    if first_model.training_status != 'trained':
        print(f"🚀 Starting training for {first_model.name}...")
        result = ml_service.train_model(first_model.id, use_grid_search=False)  # Faster without grid search
        
        if result['success']:
            print(f"✅ Training completed successfully!")
            print(f"   - Accuracy: {result['accuracy']:.3f}")
            print(f"   - F1 Score: {result['f1_score']:.3f}")
        else:
            print(f"❌ Training failed: {result['error']}")
            return False
    else:
        print(f"✓ Model already trained (accuracy: {first_model.accuracy:.3f})")
    
    # Test prediction
    first_student = StudentRecord.objects.first()
    if first_student:
        print(f"🔮 Testing prediction for student {first_student.sn}...")
        prediction_result = ml_service.predict_student_risk(first_student.id, first_model.id)
        
        if prediction_result['success']:
            pred = prediction_result['prediction']
            print(f"✅ Prediction completed!")
            print(f"   - Risk Level: {pred['risk_level']}")
            print(f"   - Dropout Probability: {pred['dropout_probability']:.3f}")
            print(f"   - Confidence: {pred['confidence']:.3f}")
        else:
            print(f"❌ Prediction failed: {prediction_result['error']}")
            return False
    
    return True

if __name__ == '__main__':
    print("🚀 Setting up SmartGrade ML System...")
    print("=" * 50)
    
    # Create default models
    print("\n📊 Creating default ML models...")
    created_models = setup_default_models()
    
    # Test ML service
    try:
        success = test_ml_service()
        if success:
            print("\n🎉 ML System setup completed successfully!")
            print("\nNext steps:")
            print("1. Start the Django server: python manage.py runserver")
            print("2. Open the frontend and navigate to ML Model Management")
            print("3. Train additional models and run predictions")
        else:
            print("\n⚠️  ML System setup completed with some issues")
            print("Check the error messages above and ensure you have student data uploaded")
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
