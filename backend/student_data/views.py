from django.db.models import Avg, Count, Q
from django.db import transaction
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
import pandas as pd
import io
from .models import StudentRecord, DataUploadBatch
from .serializers import (
    StudentRecordSerializer, StudentRecordCreateSerializer,
    DataUploadBatchSerializer, StudentSummarySerializer, CSVUploadSerializer
)


class StudentRecordViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing student records
    """
    queryset = StudentRecord.objects.all()
    serializer_class = StudentRecordSerializer
    
    def get_serializer_class(self):
        if self.action == 'create':
            return StudentRecordCreateSerializer
        return StudentRecordSerializer
    
    def perform_create(self, serializer):
        # Handle anonymous users for development
        user = self.request.user if self.request.user.is_authenticated else None
        serializer.save(uploaded_by=user)
    
    @action(detail=False, methods=['get'])
    def dashboard_stats(self, request):
        """Get dashboard statistics"""
        students = StudentRecord.objects.all()
        
        total_students = students.count()
        at_risk_students = students.filter(
            Q(dropped='Y') | Q(csc1201__lt=50) | Q(csc1202__lt=50) | 
            Q(csc1203__lt=50) | Q(bsm1201__lt=50) | Q(ict1201__lt=50)
        ).count()
        
        # Calculate average GPA equivalent (simplified)
        avg_scores = students.aggregate(
            avg_csc1201=Avg('csc1201'),
            avg_csc1202=Avg('csc1202'),
            avg_csc1203=Avg('csc1203'),
            avg_bsm1201=Avg('bsm1201'),
            avg_ict1201=Avg('ict1201')
        )
        
        # Handle case when there are no students or no valid scores
        valid_scores = [v for v in avg_scores.values() if v is not None]
        if valid_scores:
            overall_avg = sum(valid_scores) / len(valid_scores)
            average_gpa = overall_avg / 25  # Convert to 4.0 scale approximation
        else:
            overall_avg = 0.0
            average_gpa = 0.0
        
        dropout_rate = (students.filter(dropped='Y').count() / total_students * 100) if total_students > 0 else 0
        
        # Gender distribution
        gender_dist = dict(students.values('gender').annotate(count=Count('gender')).values_list('gender', 'count'))
        
        # Session distribution
        session_dist = dict(students.values('session').annotate(count=Count('session')).values_list('session', 'count'))
        
        # Performance distribution
        performance_dist = dict(students.values('remarks').annotate(count=Count('remarks')).values_list('remarks', 'count'))
        
        stats = {
            'total_students': total_students,
            'at_risk_students': at_risk_students,
            'average_gpa': round(average_gpa, 2),
            'dropout_rate': round(dropout_rate, 1),
            'gender_distribution': gender_dist,
            'session_distribution': session_dist,
            'performance_distribution': performance_dist
        }
        
        serializer = StudentSummarySerializer(stats)
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload_csv(self, request):
        """Upload and process CSV file"""
        serializer = CSVUploadSerializer(data=request.data)
        if not serializer.is_valid():
            # Return detailed validation errors
            return Response({
                'error': 'File validation failed',
                'details': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        csv_file = serializer.validated_data['file']
        
        # Create upload batch record
        user = request.user if request.user.is_authenticated else None
        batch = DataUploadBatch.objects.create(
            filename=csv_file.name,
            uploaded_by=user,
            processing_status='processing'
        )
        
        try:
            # Read CSV file
            file_content = csv_file.read().decode('utf-8')
            
            # Try different separators to auto-detect format
            csv_data = None
            separators = ['\t', ',', ';']
            
            for sep in separators:
                try:
                    csv_data = pd.read_csv(io.StringIO(file_content), sep=sep)
                    if len(csv_data.columns) > 5:  # Reasonable number of columns
                        break
                except Exception:
                    continue
            
            if csv_data is None or csv_data.empty:
                batch.processing_status = 'failed'
                batch.validation_errors = ["Could not parse CSV file"]
                batch.save()
                return Response(
                    {'error': 'Could not parse CSV file'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Clean column names (remove extra whitespace)
            csv_data.columns = csv_data.columns.str.strip()
            
            # Validate basic required columns
            basic_required_columns = ['SN', 'Gender', 'Sponsorship', 'Session', 'Retakes', 'Remarks', 'Dropped']
            
            missing_columns = [col for col in basic_required_columns if col not in csv_data.columns]
            if missing_columns:
                batch.processing_status = 'failed'
                batch.validation_errors = [f"Missing columns: {', '.join(missing_columns)}"]
                batch.save()
                return Response(
                    {'error': f"Missing required columns: {', '.join(missing_columns)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Helper function to extract course score from complex format
            def extract_course_score(row, course_prefix):
                """Extract the final numeric score for a course from multiple columns"""
                course_columns = [col for col in csv_data.columns if col.startswith(course_prefix)]
                
                if not course_columns:
                    return 0.0
                
                # Try to find the main score column (usually the 3rd column for each course)
                # Format appears to be: score1, score2, total_score, letter_grade, grade_point
                if len(course_columns) >= 3:
                    try:
                        # Use the 3rd column as the total score
                        score_value = row[course_columns[2]]
                        if pd.isna(score_value) or score_value == '':
                            return 0.0
                        return float(score_value)
                    except (ValueError, TypeError):
                        return 0.0
                
                # Fallback: try first numeric column
                for col in course_columns:
                    try:
                        score_value = row[col]
                        if pd.notna(score_value) and str(score_value).replace('.', '').isdigit():
                            return float(score_value)
                    except (ValueError, TypeError):
                        continue
                
                return 0.0
            
            # Process records
            successful_records = []
            failed_records = []
            validation_errors = []
            
            # Value mapping for CSV data to model fields
            sponsorship_mapping = {
                'Government': 'Gov',
                'Private': 'Private',
                'Scholarship': 'Scholarship',
                'Gov': 'Gov'  # Handle cases where it's already correct
            }
            
            with transaction.atomic():
                for index, row in csv_data.iterrows():
                    try:
                        # Map CSV values to model values
                        sponsorship_value = row['Sponsorship']
                        if sponsorship_value in sponsorship_mapping:
                            sponsorship_value = sponsorship_mapping[sponsorship_value]
                        
                        # Map CSV columns to model fields using helper function
                        student_data = {
                            'sn': int(row['SN']),
                            'gender': row['Gender'],
                            'sponsorship': sponsorship_value,
                            'session': row['Session'],
                            'retakes': int(row['Retakes']) if pd.notna(row['Retakes']) else 0,
                            'csc1201': extract_course_score(row, 'CSC1201'),
                            'csc1202': extract_course_score(row, 'CSC1202'),
                            'csc1203': extract_course_score(row, 'CSC1203'),
                            'bsm1201': extract_course_score(row, 'BSM1201'),
                            'ict1201': extract_course_score(row, 'ICT1201'),
                            'remarks': row['Remarks'],
                            'dropped': row['Dropped']
                        }
                        
                        # Validate and create record
                        record_serializer = StudentRecordCreateSerializer(data=student_data)
                        if record_serializer.is_valid():
                            # Handle anonymous users for development
                            user = request.user if request.user.is_authenticated else None
                            record_serializer.save(uploaded_by=user)
                            successful_records.append(student_data['sn'])
                        else:
                            failed_records.append(student_data['sn'])
                            validation_errors.append(f"Row {index + 1}: {record_serializer.errors}")
                    
                    except Exception as e:
                        failed_records.append(row.get('SN', f'Row {index + 1}'))
                        validation_errors.append(f"Row {index + 1}: {str(e)}")
            
            # Update batch record
            batch.total_records = len(csv_data)
            batch.successful_records = len(successful_records)
            batch.failed_records = len(failed_records)
            batch.processing_status = 'completed' if len(failed_records) == 0 else 'completed'
            batch.validation_errors = validation_errors
            batch.save()
            
            return Response({
                'message': 'CSV upload completed',
                'batch_id': batch.id,
                'total_records': batch.total_records,
                'successful_records': batch.successful_records,
                'failed_records': batch.failed_records,
                'validation_errors': validation_errors[:10]  # Limit errors shown
            })
        
        except Exception as e:
            batch.processing_status = 'failed'
            batch.validation_errors = [str(e)]
            batch.save()
            return Response(
                {'error': f"Failed to process CSV: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def all_for_analytics(self, request):
        """Get all students for analytics without pagination"""
        students = StudentRecord.objects.all()
        serializer = StudentRecordSerializer(students, many=True)
        return Response(serializer.data)


class DataUploadBatchViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing upload batch history
    """
    queryset = DataUploadBatch.objects.all()
    serializer_class = DataUploadBatchSerializer
    
    def get_queryset(self):
        return DataUploadBatch.objects.filter(uploaded_by=self.request.user).order_by('-uploaded_at')
