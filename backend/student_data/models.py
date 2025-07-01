from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User


class StudentRecord(models.Model):
    """
    Model representing a student's academic record matching CS_data.csv structure
    """
    
    # Basic Information
    sn = models.IntegerField("Serial Number", unique=True)
    gender = models.CharField(max_length=10, choices=[
        ('M', 'Male'),
        ('F', 'Female'),
    ])
    sponsorship = models.CharField(max_length=20, choices=[
        ('Gov', 'Government'),
        ('Private', 'Private'),
        ('Scholarship', 'Scholarship'),
    ])
    session = models.CharField(max_length=20, help_text="Academic session")
    retakes = models.IntegerField(default=0, validators=[MinValueValidator(0)])
    
    # Course Scores (0-100 scale)
    csc1201 = models.FloatField("CSC1201 Score", default=0.0, validators=[
        MinValueValidator(0.0), MaxValueValidator(100.0)
    ])
    csc1202 = models.FloatField("CSC1202 Score", default=0.0, validators=[
        MinValueValidator(0.0), MaxValueValidator(100.0)
    ])
    csc1203 = models.FloatField("CSC1203 Score", default=0.0, validators=[
        MinValueValidator(0.0), MaxValueValidator(100.0)
    ])
    bsm1201 = models.FloatField("BSM1201 Score", default=0.0, validators=[
        MinValueValidator(0.0), MaxValueValidator(100.0)
    ])
    ict1201 = models.FloatField("ICT1201 Score", default=0.0, validators=[
        MinValueValidator(0.0), MaxValueValidator(100.0)
    ])
    
    # Outcome Variables
    remarks = models.CharField(max_length=10, choices=[
        ('NP', 'Not Passed'),
        ('PP', 'Passed'),
        ('AB', 'Absent'),
    ])
    dropped = models.CharField(max_length=1, choices=[
        ('Y', 'Yes'),
        ('N', 'No'),
    ])
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
    class Meta:
        db_table = 'student_records'
        verbose_name = "Student Record"
        verbose_name_plural = "Student Records"
        ordering = ['sn']
        indexes = [
            models.Index(fields=['session']),
            models.Index(fields=['remarks']),
            models.Index(fields=['dropped']),
            models.Index(fields=['gender']),
        ]
    
    def __str__(self):
        return f"Student {self.sn} - {self.session}"
    
    @property
    def average_score(self):
        """Calculate average score across all courses"""
        scores = [self.csc1201, self.csc1202, self.csc1203, self.bsm1201, self.ict1201]
        # Filter out None values
        valid_scores = [score for score in scores if score is not None]
        if not valid_scores:
            return 0.0
        return sum(valid_scores) / len(valid_scores)
    
    @property
    def is_at_risk(self):
        """Determine if student is at risk based on performance"""
        return self.average_score < 50 or self.dropped == 'Y'
    
    @property
    def course_scores_dict(self):
        """Return course scores as dictionary for easy access"""
        return {
            'CSC1201': self.csc1201,
            'CSC1202': self.csc1202,
            'CSC1203': self.csc1203,
            'BSM1201': self.bsm1201,
            'ICT1201': self.ict1201,
        }


class DataUploadBatch(models.Model):
    """
    Model to track CSV file uploads and batch processing
    """
    filename = models.CharField(max_length=255)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    total_records = models.IntegerField(default=0)
    successful_records = models.IntegerField(default=0)
    failed_records = models.IntegerField(default=0)
    processing_status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    validation_errors = models.JSONField(default=list, blank=True)
    
    class Meta:
        db_table = 'data_upload_batches'
        verbose_name = "Data Upload Batch"
        verbose_name_plural = "Data Upload Batches"
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.filename} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
