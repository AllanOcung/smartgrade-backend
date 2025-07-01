from django.db import models
from django.contrib.auth.models import User
from student_data.models import StudentRecord
import json


class MLModel(models.Model):
    """
    Model representing a machine learning model for predictions
    """
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50, choices=[
        ('logistic_regression', 'Logistic Regression'),
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('neural_network', 'Neural Network'),
    ])
    version = models.CharField(max_length=20, default='1.0')
    description = models.TextField(blank=True)
    
    # Performance Metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # Model Status
    is_active = models.BooleanField(default=False)
    training_status = models.CharField(max_length=20, choices=[
        ('not_trained', 'Not Trained'),
        ('training', 'Training'),
        ('trained', 'Trained'),
        ('failed', 'Failed'),
    ], default='not_trained')
    
    # Training Information
    training_data_count = models.IntegerField(default=0)
    training_started_at = models.DateTimeField(null=True, blank=True)
    training_completed_at = models.DateTimeField(null=True, blank=True)
    trained_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Model Configuration
    hyperparameters = models.JSONField(default=dict, blank=True)
    feature_importance = models.JSONField(default=dict, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'ml_models'
        verbose_name = "ML Model"
        verbose_name_plural = "ML Models"
        ordering = ['-created_at']
        unique_together = ['name', 'version']
    
    def __str__(self):
        return f"{self.name} v{self.version} ({self.model_type})"


class StudentPrediction(models.Model):
    """
    Model storing predictions for individual students
    """
    student = models.ForeignKey(StudentRecord, on_delete=models.CASCADE, related_name='predictions')
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='predictions')
    
    # Prediction Results
    dropout_probability = models.FloatField(help_text="Probability of dropping out (0-1)")
    predicted_gpa = models.FloatField(null=True, blank=True, help_text="Predicted GPA")
    risk_category = models.CharField(max_length=20, choices=[
        ('low', 'Low Risk'),
        ('medium', 'Medium Risk'),
        ('high', 'High Risk'),
    ])
    performance_category = models.CharField(max_length=10, choices=[
        ('NP', 'Not Passed'),
        ('PP', 'Passed'),
        ('AB', 'Absent'),
    ], null=True, blank=True)
    
    # Confidence and Features
    confidence_score = models.FloatField(help_text="Model confidence (0-1)")
    contributing_factors = models.JSONField(default=dict, blank=True, 
                                          help_text="Factors contributing to prediction")
    
    # Metadata
    predicted_at = models.DateTimeField(auto_now_add=True)
    is_latest = models.BooleanField(default=True, help_text="Is this the latest prediction for this student")
    
    class Meta:
        db_table = 'student_predictions'
        verbose_name = "Student Prediction"
        verbose_name_plural = "Student Predictions"
        ordering = ['-predicted_at']
        indexes = [
            models.Index(fields=['risk_category']),
            models.Index(fields=['is_latest']),
            models.Index(fields=['predicted_at']),
        ]
    
    def __str__(self):
        return f"Prediction for Student {self.student.sn} - {self.risk_category}"
    
    def save(self, *args, **kwargs):
        # Ensure only one latest prediction per student-model combination
        if self.is_latest:
            StudentPrediction.objects.filter(
                student=self.student, 
                model=self.model,
                is_latest=True
            ).update(is_latest=False)
        super().save(*args, **kwargs)


class ModelTrainingJob(models.Model):
    """
    Model to track model training jobs and their progress
    """
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='training_jobs')
    started_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Job Configuration
    training_config = models.JSONField(default=dict, help_text="Training configuration parameters")
    dataset_filter = models.JSONField(default=dict, blank=True, help_text="Filters applied to training dataset")
    
    # Job Status
    status = models.CharField(max_length=20, choices=[
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ], default='queued')
    
    # Progress and Results
    progress_percentage = models.IntegerField(default=0)
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    
    # Timing
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'model_training_jobs'
        verbose_name = "Model Training Job"
        verbose_name_plural = "Model Training Jobs"
        ordering = ['-started_at']
    
    def __str__(self):
        return f"Training {self.model.name} - {self.status}"
    
    @property
    def duration(self):
        """Calculate job duration"""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return None


class PredictionBatch(models.Model):
    """
    Model to track batch prediction jobs
    """
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Batch Information
    total_students = models.IntegerField(default=0)
    processed_students = models.IntegerField(default=0)
    failed_predictions = models.IntegerField(default=0)
    
    # Status
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    
    # Timing
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'prediction_batches'
        verbose_name = "Prediction Batch"
        verbose_name_plural = "Prediction Batches"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Batch prediction - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
