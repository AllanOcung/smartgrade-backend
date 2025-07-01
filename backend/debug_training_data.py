#!/usr/bin/env python3
"""
Debug the ML training data preparation
"""
import os
import sys
import django
import pandas as pd

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

from student_data.models import StudentRecord
from predictions.ml_service import MLService

def debug_training_data():
    """Debug the training data preparation"""
    print("=== STUDENT DATA ANALYSIS ===")
    
    # Get student records
    students_qs = StudentRecord.objects.all()
    students_data = list(students_qs.values())
    students_df = pd.DataFrame(students_data)
    
    print(f"Total student records: {len(students_df)}")
    
    if students_df.empty:
        print("No student data found!")
        return
    
    print(f"Columns in student data: {list(students_df.columns)}")
    print(f"Data types:\n{students_df.dtypes}")
    
    # Check for required columns
    required_columns = [
        'retakes', 'csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201',
        'gender', 'sponsorship', 'session', 'remarks', 'dropped'
    ]
    
    missing_columns = [col for col in required_columns if col not in students_df.columns]
    print(f"Missing required columns: {missing_columns}")
    
    if missing_columns:
        print("ERROR: Cannot train models due to missing columns!")
        return
    
    # Check data quality
    print("\n=== DATA QUALITY CHECK ===")
    for col in required_columns:
        print(f"{col}:")
        print(f"  - Null values: {students_df[col].isnull().sum()}")
        print(f"  - Unique values: {students_df[col].nunique()}")
        if students_df[col].dtype == 'object':
            print(f"  - Sample values: {students_df[col].unique()[:5]}")
        print()
    
    # Test ML service preparation
    print("=== TESTING ML SERVICE ===")
    try:
        ml_service = MLService()
        X, y = ml_service.prepare_features(students_df)
        print(f"Features prepared successfully!")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Target distribution: {y.value_counts()}")
        print(f"Feature columns: {list(X.columns)}")
    except Exception as e:
        print(f"ERROR in prepare_features: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training_data()
