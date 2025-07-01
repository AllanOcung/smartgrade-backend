#!/usr/bin/env python
import os
import sys
import django

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

django.setup()

# Test imports
try:
    from predictions.models import MLModel
    print("✓ Successfully imported MLModel")
    
    from predictions.views import MLModelViewSet
    print("✓ Successfully imported MLModelViewSet")
    
    from predictions.ml_service import ml_service
    print("✓ Successfully imported ml_service")
    
    print("\n🎉 All ML components imported successfully!")
    print("Now run: python manage.py makemigrations predictions")
    print("Then run: python manage.py migrate")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
