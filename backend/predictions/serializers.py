from rest_framework import serializers
from .models import MLModel, StudentPrediction, ModelTrainingJob, PredictionBatch
from student_data.serializers import StudentRecordSerializer


class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = [
            'id', 'name', 'model_type', 'version', 'description',
            'accuracy', 'precision', 'recall', 'f1_score',
            'is_active', 'training_status', 'training_data_count',
            'training_started_at', 'training_completed_at',
            'hyperparameters', 'feature_importance',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at', 'training_started_at', 'training_completed_at']


class StudentPredictionSerializer(serializers.ModelSerializer):
    student = StudentRecordSerializer(read_only=True)
    model = MLModelSerializer(read_only=True)
    
    class Meta:
        model = StudentPrediction
        fields = [
            'id', 'student', 'model', 'dropout_probability',
            'predicted_gpa', 'risk_category', 'performance_category',
            'confidence_score', 'contributing_factors',
            'predicted_at', 'is_latest'
        ]
        read_only_fields = ['predicted_at']


class StudentPredictionCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = StudentPrediction
        fields = [
            'student', 'model', 'dropout_probability',
            'predicted_gpa', 'risk_category', 'performance_category',
            'confidence_score', 'contributing_factors', 'is_latest'
        ]


class ModelTrainingJobSerializer(serializers.ModelSerializer):
    model = MLModelSerializer(read_only=True)
    duration = serializers.ReadOnlyField()
    
    class Meta:
        model = ModelTrainingJob
        fields = [
            'id', 'model', 'started_by', 'training_config',
            'dataset_filter', 'status', 'progress_percentage',
            'logs', 'error_message', 'started_at', 'completed_at', 'duration'
        ]
        read_only_fields = ['started_at', 'completed_at', 'duration']


class PredictionBatchSerializer(serializers.ModelSerializer):
    model = MLModelSerializer(read_only=True)
    
    class Meta:
        model = PredictionBatch
        fields = [
            'id', 'model', 'created_by', 'total_students',
            'processed_students', 'failed_predictions',
            'status', 'created_at', 'completed_at'
        ]
        read_only_fields = ['created_at', 'completed_at']


class ModelTrainingRequestSerializer(serializers.Serializer):
    """Serializer for model training requests"""
    model_id = serializers.IntegerField(required=False)  # Optional since model comes from URL
    use_grid_search = serializers.BooleanField(default=True)
    training_config = serializers.DictField(required=False, default=dict)
    dataset_filter = serializers.DictField(required=False, default=dict)


class PredictionRequestSerializer(serializers.Serializer):
    """Serializer for single prediction requests"""
    model_id = serializers.IntegerField()
    student_id = serializers.IntegerField()


class BatchPredictionRequestSerializer(serializers.Serializer):
    """Serializer for batch prediction requests"""
    model_id = serializers.IntegerField()
    student_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        help_text="List of student IDs to predict. If empty, predict for all students."
    )
