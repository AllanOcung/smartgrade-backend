from django.contrib import admin
from .models import MLModel, StudentPrediction, ModelTrainingJob, PredictionBatch


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'version', 'is_active', 'training_status', 'accuracy', 'created_at']
    list_filter = ['model_type', 'is_active', 'training_status']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']


@admin.register(StudentPrediction)
class StudentPredictionAdmin(admin.ModelAdmin):
    list_display = ['student', 'model', 'risk_category', 'dropout_probability', 'confidence_score', 'is_latest', 'predicted_at']
    list_filter = ['risk_category', 'is_latest', 'model', 'predicted_at']
    search_fields = ['student__sn']
    readonly_fields = ['predicted_at']
    ordering = ['-predicted_at']


@admin.register(ModelTrainingJob)
class ModelTrainingJobAdmin(admin.ModelAdmin):
    list_display = ['model', 'status', 'progress_percentage', 'started_by', 'started_at', 'duration']
    list_filter = ['status', 'started_at']
    readonly_fields = ['started_at', 'completed_at', 'duration']
    ordering = ['-started_at']


@admin.register(PredictionBatch)
class PredictionBatchAdmin(admin.ModelAdmin):
    list_display = ['model', 'total_students', 'processed_students', 'failed_predictions', 'status', 'created_at']
    list_filter = ['status', 'created_at']
    readonly_fields = ['created_at', 'completed_at']
    ordering = ['-created_at']
