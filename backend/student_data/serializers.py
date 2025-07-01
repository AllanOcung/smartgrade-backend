from rest_framework import serializers
from .models import StudentRecord, DataUploadBatch


class StudentRecordSerializer(serializers.ModelSerializer):
    average_score = serializers.ReadOnlyField()
    is_at_risk = serializers.ReadOnlyField()
    course_scores_dict = serializers.ReadOnlyField()
    
    class Meta:
        model = StudentRecord
        fields = [
            'id', 'sn', 'gender', 'sponsorship', 'session', 'retakes',
            'csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201',
            'remarks', 'dropped', 'average_score', 'is_at_risk',
            'course_scores_dict', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class StudentRecordCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating student records from CSV data"""
    
    class Meta:
        model = StudentRecord
        fields = [
            'sn', 'gender', 'sponsorship', 'session', 'retakes',
            'csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201',
            'remarks', 'dropped'
        ]
    
    def validate_sn(self, value):
        """Ensure SN is unique"""
        if StudentRecord.objects.filter(sn=value).exists():
            raise serializers.ValidationError(f"Student with SN {value} already exists.")
        return value


class DataUploadBatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataUploadBatch
        fields = [
            'id', 'filename', 'uploaded_by', 'uploaded_at',
            'total_records', 'successful_records', 'failed_records',
            'processing_status', 'validation_errors'
        ]
        read_only_fields = ['uploaded_at', 'uploaded_by']


class StudentSummarySerializer(serializers.Serializer):
    """Serializer for dashboard statistics"""
    total_students = serializers.IntegerField()
    at_risk_students = serializers.IntegerField()
    average_gpa = serializers.FloatField()
    dropout_rate = serializers.FloatField()
    gender_distribution = serializers.DictField()
    session_distribution = serializers.DictField()
    performance_distribution = serializers.DictField()


class CSVUploadSerializer(serializers.Serializer):
    """Serializer for CSV file upload"""
    file = serializers.FileField()
    
    def validate_file(self, value):
        """Validate that the uploaded file is a CSV or TSV"""
        # Accept various text file extensions
        allowed_extensions = ['.csv', '.tsv', '.txt']
        file_extension = None
        
        for ext in allowed_extensions:
            if value.name.lower().endswith(ext):
                file_extension = ext
                break
        
        if not file_extension:
            raise serializers.ValidationError(
                f"File must be a CSV, TSV, or TXT file. Allowed extensions: {', '.join(allowed_extensions)}"
            )
        
        # Check file size (10MB limit)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size must be less than 10MB.")
        
        return value
