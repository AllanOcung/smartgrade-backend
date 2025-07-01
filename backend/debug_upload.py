#!/usr/bin/env python
"""Debug script to test CSV upload functionality"""
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
django.setup()

import requests
import json

def test_upload():
    """Test the CSV upload endpoint"""
    url = 'http://localhost:8000/api/students/upload_csv/'
    
    # Path to test CSV file
    csv_file_path = '../../test_upload.csv'
    
    if not os.path.exists(csv_file_path):
        print(f"Error: Test CSV file not found at {csv_file_path}")
        return
    
    try:
        with open(csv_file_path, 'rb') as f:
            files = {'file': ('test_upload.csv', f, 'text/csv')}
            
            print("Sending request to:", url)
            response = requests.post(url, files=files)
            
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            
            try:
                response_data = response.json()
                print(f"Response Data: {json.dumps(response_data, indent=2)}")
            except:
                print(f"Response Text: {response.text}")
                
    except Exception as e:
        print(f"Error making request: {e}")

if __name__ == '__main__':
    test_upload()
