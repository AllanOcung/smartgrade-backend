#!/usr/bin/env python
import os
import sys
import pandas as pd
import io

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')

import django
django.setup()

def test_csv_parsing():
    csv_file_path = r'c:\Users\User\Desktop\Ryeko AI-Internship\SmartGrade\test_upload.csv'
    
    print("Testing CSV parsing...")
    print(f"Reading file: {csv_file_path}")
    
    # Read the file as the Django view would
    with open(csv_file_path, 'rb') as f:
        file_content = f.read().decode('utf-8')
    
    print("Raw file content (first 200 chars):")
    print(repr(file_content[:200]))
    print()
    
    # Parse with pandas as tab-separated
    csv_data = pd.read_csv(io.StringIO(file_content), sep='\t')
    
    print("Parsed columns:")
    print(csv_data.columns.tolist())
    print()
    
    print("First few rows:")
    print(csv_data.head())
    print()
    
    # Check for required columns
    basic_required_columns = ['SN', 'Gender', 'Sponsorship', 'Session', 'Retakes', 'Remarks', 'Dropped']
    missing_columns = [col for col in basic_required_columns if col not in csv_data.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
    else:
        print("All required columns found!")
    
    # Test the extract_course_score function
    def extract_course_score(row, course_prefix):
        """Extract the final numeric score for a course from multiple columns"""
        course_columns = [col for col in csv_data.columns if col.startswith(course_prefix)]
        
        print(f"Columns for {course_prefix}: {course_columns}")
        
        if not course_columns:
            return 0.0
        
        # Try to find the main score column (usually the 3rd column for each course)
        if len(course_columns) >= 3:
            try:
                score_value = row[course_columns[2]]
                print(f"Score value from column {course_columns[2]}: {score_value}")
                if pd.isna(score_value) or score_value == '':
                    return 0.0
                return float(score_value)
            except (ValueError, TypeError) as e:
                print(f"Error parsing score: {e}")
                return 0.0
        
        return 0.0
    
    if not csv_data.empty:
        first_row = csv_data.iloc[0]
        print("\nTesting score extraction for first row:")
        for course in ['CSC1201', 'CSC1202', 'CSC1203', 'BSM1201', 'ICT1201']:
            score = extract_course_score(first_row, course)
            print(f"{course}: {score}")

if __name__ == '__main__':
    test_csv_parsing()
