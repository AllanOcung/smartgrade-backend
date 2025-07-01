from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    MLModelViewSet, StudentPredictionViewSet, 
    ModelTrainingJobViewSet, PredictionBatchViewSet
)

# Create router and register viewsets
router = DefaultRouter()
router.register(r'models', MLModelViewSet)
router.register(r'predictions', StudentPredictionViewSet)
router.register(r'training-jobs', ModelTrainingJobViewSet)
router.register(r'batches', PredictionBatchViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
