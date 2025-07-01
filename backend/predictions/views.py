from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.db.models import Count, Q
from django.utils import timezone
from django.http import JsonResponse
import logging

from .models import MLModel, StudentPrediction, ModelTrainingJob, PredictionBatch
from .serializers import (
    MLModelSerializer, StudentPredictionSerializer, 
    ModelTrainingJobSerializer, PredictionBatchSerializer,
    ModelTrainingRequestSerializer, PredictionRequestSerializer,
    BatchPredictionRequestSerializer
)
from .ml_service import ml_service

logger = logging.getLogger(__name__)


class MLModelViewSet(viewsets.ModelViewSet):
    """ViewSet for ML Model management"""
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    permission_classes = [AllowAny]  # For development
    pagination_class = None  # Disable pagination for ML models
    
    @action(detail=True, methods=['post'])
    def train(self, request, pk=None):
        """Train a specific ML model"""
        try:
            logger.info(f"Training request received for model {pk}")
            logger.info(f"Request data: {request.data}")
            
            model = self.get_object()
            logger.info(f"Model found: {model.name} ({model.model_type})")
            
            # Validate request data
            serializer = ModelTrainingRequestSerializer(data=request.data)
            if not serializer.is_valid():
                logger.error(f"Serializer validation failed: {serializer.errors}")
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            logger.info("Serializer validation passed")
            
            # Check if model is already training
            if model.training_status == 'training':
                logger.warning(f"Model {model.id} is already training")
                return Response(
                    {'error': 'Model is already training'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Start training (this could be moved to a background task)
            use_grid_search = serializer.validated_data.get('use_grid_search', True)
            logger.info(f"Starting training with grid_search={use_grid_search}")
            
            result = ml_service.train_model(model.id, use_grid_search=use_grid_search)
            logger.info(f"Training completed: {result}")
            
            if result['success']:
                return Response({
                    'message': 'Model training completed successfully',
                    'results': result
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': 'Model training failed',
                    'details': result.get('error')
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return Response({
                'error': 'Training failed',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['post'])
    def create_defaults(self, request):
        """Create default ML models"""
        try:
            default_models = [
                {
                    'name': 'Random Forest Classifier',
                    'model_type': 'random_forest',
                    'description': 'High-performance ensemble method with feature importance',
                    'is_active': True,
                },
                {
                    'name': 'Logistic Regression Classifier',
                    'model_type': 'logistic_regression',
                    'description': 'Fast and interpretable model for dropout prediction',
                    'is_active': True,
                },
                {
                    'name': 'Gradient Boosting Classifier',
                    'model_type': 'gradient_boosting',
                    'description': 'Advanced boosting algorithm for complex patterns',
                    'is_active': True,
                },
                {
                    'name': 'Neural Network Classifier',
                    'model_type': 'neural_network',
                    'description': 'Deep learning model for complex pattern recognition',
                    'is_active': True,
                }
            ]
            
            created_models = []
            
            for model_data in default_models:
                model, created = MLModel.objects.get_or_create(
                    name=model_data['name'],
                    model_type=model_data['model_type'],
                    defaults={
                        'description': model_data['description'],
                        'is_active': model_data['is_active'],
                        'training_status': 'untrained',
                        'training_data_count': 0,
                    }
                )
                
                if created:
                    created_models.append(model)
            
            return Response({
                'success': True,
                'message': f'Created {len(created_models)} default models',
                'created_models': [
                    MLModelSerializer(model).data for model in created_models
                ]
            })
            
        except Exception as e:
            logger.error(f"Failed to create default models: {str(e)}")
            return Response(
                {'error': f'Failed to create default models: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def performance(self, request, pk=None):
        """Get model performance metrics"""
        try:
            model = self.get_object()
            result = ml_service.get_model_performance(model.id)
            
            if result['success']:
                return Response(result['metrics'], status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': 'Failed to get performance metrics',
                    'details': result.get('error')
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Performance retrieval failed: {str(e)}")
            return Response({
                'error': 'Failed to get performance metrics',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def predict_single(self, request, pk=None):
        """Make prediction for a single student"""
        try:
            model = self.get_object()
            
            # Validate request
            serializer = PredictionRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            student_id = serializer.validated_data['student_id']
            result = ml_service.predict_student_risk(student_id, model.id)
            
            if result['success']:
                return Response(result['prediction'], status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': 'Prediction failed',
                    'details': result.get('error')
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            return Response({
                'error': 'Prediction failed',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def predict_batch(self, request, pk=None):
        """Make predictions for multiple students"""
        try:
            model = self.get_object()
            
            # Get student IDs from request (optional)
            student_ids = request.data.get('student_ids', None)
            
            result = ml_service.batch_predict(model.id, student_ids)
            
            if result['success']:
                return Response({
                    'message': 'Batch prediction completed',
                    'results': result
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': 'Batch prediction failed',
                    'details': result.get('error')
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return Response({
                'error': 'Batch prediction failed',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def active_models(self, request):
        """Get all active/trained models"""
        active_models = MLModel.objects.filter(
            training_status='trained',
            is_active=True
        )
        serializer = self
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def training_status(self, request):
        """Get training status overview"""
        status_counts = MLModel.objects.values('training_status').annotate(
            count=Count('training_status')
        )
        
        # Recent training jobs
        recent_jobs = ModelTrainingJob.objects.select_related('model').order_by('-started_at')[:10]
        jobs_serializer = ModelTrainingJobSerializer(recent_jobs, many=True)
        
        return Response({
            'status_distribution': list(status_counts),
            'recent_training_jobs': jobs_serializer.data
        })


class StudentPredictionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Student Predictions (read-only)"""
    queryset = StudentPrediction.objects.all()
    serializer_class = StudentPredictionSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        queryset = StudentPrediction.objects.select_related('student', 'model')
        
        # Filter by student
        student_id = self.request.query_params.get('student_id', None)
        if student_id:
            queryset = queryset.filter(student_id=student_id)
        
        # Filter by model
        model_id = self.request.query_params.get('model_id', None)
        if model_id:
            queryset = queryset.filter(model_id=model_id)
        
        # Filter by risk level
        risk_level = self.request.query_params.get('risk_level', None)
        if risk_level:
            queryset = queryset.filter(risk_category=risk_level)
        
        # Only latest predictions by default
        latest_only = self.request.query_params.get('latest_only', 'true').lower() == 'true'
        if latest_only:
            queryset = queryset.filter(is_latest=True)
        
        return queryset.order_by('-predicted_at')
    
    @action(detail=False, methods=['get'])
    def risk_summary(self, request):
        """Get risk level summary statistics"""
        predictions = self.get_queryset()
        
        # Risk distribution
        risk_distribution = predictions.values('risk_category').annotate(
            count=Count('risk_category')
        )
        
        # Model performance comparison
        model_performance = predictions.values('model__name', 'model__id').annotate(
            total_predictions=Count('id'),
            high_risk=Count('id', filter=Q(risk_category='high')),
            medium_risk=Count('id', filter=Q(risk_category='medium')),
            low_risk=Count('id', filter=Q(risk_category='low'))
        )
        
        return Response({
            'risk_distribution': list(risk_distribution),
            'model_performance': list(model_performance),
            'total_predictions': predictions.count()
        })
    
    @action(detail=False, methods=['get'])
    def at_risk_students(self, request):
        """Get students identified as at risk"""
        at_risk_predictions = self.get_queryset().filter(
            risk_category__in=['medium', 'high']
        ).select_related('student')
        
        # Group by risk level
        results = {
            'high_risk': [],
            'medium_risk': []
        }
        
        for prediction in at_risk_predictions:
            student_data = {
                'student_id': prediction.student.id,
                'student_sn': prediction.student.sn,
                'risk_level': prediction.risk_category,
                'dropout_probability': prediction.dropout_probability,
                'confidence': prediction.confidence_score,
                'contributing_factors': prediction.contributing_factors,
                'predicted_at': prediction.predicted_at
            }
            
            if prediction.risk_category == 'high':
                results['high_risk'].append(student_data)
            else:
                results['medium_risk'].append(student_data)
        
        return Response(results)


class ModelTrainingJobViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Model Training Jobs (read-only)"""
    queryset = ModelTrainingJob.objects.all()
    serializer_class = ModelTrainingJobSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        queryset = ModelTrainingJob.objects.select_related('model')
        
        # Filter by model
        model_id = self.request.query_params.get('model_id', None)
        if model_id:
            queryset = queryset.filter(model_id=model_id)
        
        # Filter by status
        status_filter = self.request.query_params.get('status', None)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        return queryset.order_by('-started_at')


class PredictionBatchViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Prediction Batches (read-only)"""
    queryset = PredictionBatch.objects.all()
    serializer_class = PredictionBatchSerializer
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        queryset = PredictionBatch.objects.select_related('model')
        
        # Filter by model
        model_id = self.request.query_params.get('model_id', None)
        if model_id:
            queryset = queryset.filter(model_id=model_id)
        
        return queryset.order_by('-created_at')
