from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import StudentRecordViewSet, DataUploadBatchViewSet

router = DefaultRouter()
router.register(r'students', StudentRecordViewSet)
router.register(r'uploads', DataUploadBatchViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
]
