from django.contrib import admin
from .models import StudentRecord, DataUploadBatch


@admin.register(StudentRecord)
class StudentRecordAdmin(admin.ModelAdmin):
    list_display = ['sn', 'gender', 'session', 'average_score', 'remarks', 'dropped', 'created_at']
    list_filter = ['gender', 'session', 'remarks', 'dropped', 'sponsorship']
    search_fields = ['sn', 'session']
    readonly_fields = ['created_at', 'updated_at', 'average_score', 'is_at_risk']
    ordering = ['sn']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('sn', 'gender', 'sponsorship', 'session', 'retakes', 'uploaded_by')
        }),
        ('Course Scores', {
            'fields': ('csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201')
        }),
        ('Outcomes', {
            'fields': ('remarks', 'dropped')
        }),
        ('Calculated Fields', {
            'fields': ('average_score', 'is_at_risk'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(DataUploadBatch)
class DataUploadBatchAdmin(admin.ModelAdmin):
    list_display = ['filename', 'uploaded_by', 'total_records', 'successful_records', 'failed_records', 'processing_status', 'uploaded_at']
    list_filter = ['processing_status', 'uploaded_at']
    readonly_fields = ['uploaded_at']
    ordering = ['-uploaded_at']
