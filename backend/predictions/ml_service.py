import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

from django.conf import settings
from django.utils import timezone

from .models import MLModel, StudentPrediction, ModelTrainingJob, PredictionBatch
from student_data.models import StudentRecord

logger = logging.getLogger(__name__)


class MLService:
    """Service class for machine learning operations"""
    
    def __init__(self):
        self.models_dir = os.path.join(settings.MEDIA_ROOT, 'ml_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.algorithms = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'neural_network': MLPClassifier,
        }
        
        self.default_hyperparameters = {
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'max_iter': [1000],
                'random_state': [42]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'random_state': [42]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5],
                'random_state': [42]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'learning_rate': ['adaptive'],
                'max_iter': [1000],
                'random_state': [42]
            }
        }

    def prepare_features(self, students_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from student data"""
        
        # Select numerical features
        feature_columns = [
            'retakes', 'csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201'
        ]
        
        # Add categorical features (encoded)
        categorical_features = ['gender', 'sponsorship', 'session']
        
        # Create feature dataframe
        features_df = students_df[feature_columns].copy()
        
        # Encode categorical features
        for cat_feature in categorical_features:
            if cat_feature in students_df.columns:
                le = LabelEncoder()
                features_df[f'{cat_feature}_encoded'] = le.fit_transform(students_df[cat_feature].fillna('Unknown'))
        
        # Add derived features
        features_df['total_score'] = features_df[['csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201']].sum(axis=1)
        features_df['average_score'] = features_df[['csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201']].mean(axis=1)
        features_df['score_variance'] = features_df[['csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201']].var(axis=1)
        features_df['min_score'] = features_df[['csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201']].min(axis=1)
        features_df['max_score'] = features_df[['csc1201', 'csc1202', 'csc1203', 'bsm1201', 'ict1201']].max(axis=1)
        
        # Target variable (binary: at risk or not)
        # Students with 'NP' remarks or dropped='Y' are considered at risk
        target = ((students_df['remarks'] == 'NP') | (students_df['dropped'] == 'Y')).astype(int)
        
        return features_df, target

    def train_model(self, model_id: int, use_grid_search: bool = True) -> Dict[str, Any]:
        """Train a machine learning model"""
        
        try:
            ml_model = MLModel.objects.get(id=model_id)
            
            # Create training job
            training_job = ModelTrainingJob.objects.create(
                model=ml_model,
                started_by=None,  # Allow None for system-initiated training
                status='running'
            )
            
            # Update model status
            ml_model.training_status = 'training'
            ml_model.training_started_at = timezone.now()
            ml_model.save()
            
            # Load training data
            students_qs = StudentRecord.objects.all()
            students_data = list(students_qs.values())
            students_df = pd.DataFrame(students_data)
            
            if students_df.empty:
                raise ValueError("No training data available")
            
            # Prepare features and target
            X, y = self.prepare_features(students_df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Update training job progress
            training_job.progress_percentage = 25
            training_job.save()
            
            # Get algorithm class
            algorithm_class = self.algorithms[ml_model.model_type]
            
            # Create pipeline with scaling
            scaler = StandardScaler()
            
            if use_grid_search and ml_model.model_type in self.default_hyperparameters:
                # Grid search for best parameters
                param_grid = self.default_hyperparameters[ml_model.model_type]
                
                # Create pipeline
                pipeline = Pipeline([
                    ('scaler', scaler),
                    ('classifier', algorithm_class())
                ])
                
                # Adjust parameter names for pipeline
                pipeline_param_grid = {}
                for param, values in param_grid.items():
                    pipeline_param_grid[f'classifier__{param}'] = values
                
                # Grid search
                grid_search = GridSearchCV(
                    pipeline, 
                    pipeline_param_grid, 
                    cv=5, 
                    scoring='f1',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                # Store best parameters
                ml_model.hyperparameters = grid_search.best_params_
                
            else:
                # Use default parameters
                classifier = algorithm_class(**ml_model.hyperparameters)
                best_model = Pipeline([
                    ('scaler', scaler),
                    ('classifier', classifier)
                ])
                best_model.fit(X_train, y_train)
            
            # Update progress
            training_job.progress_percentage = 75
            training_job.save()
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Update model with metrics
            ml_model.accuracy = accuracy
            ml_model.precision = precision
            ml_model.recall = recall
            ml_model.f1_score = f1
            ml_model.training_data_count = len(students_df)
            ml_model.training_status = 'trained'
            ml_model.training_completed_at = timezone.now()
            
            # Save model to file
            model_filename = f"model_{ml_model.id}_{ml_model.name.replace(' ', '_')}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            # Calculate feature importance
            if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                feature_importance = dict(zip(
                    X.columns, 
                    best_model.named_steps['classifier'].feature_importances_
                ))
                ml_model.feature_importance = feature_importance
            
            ml_model.save()
            
            # Complete training job
            training_job.status = 'completed'
            training_job.progress_percentage = 100
            training_job.completed_at = timezone.now()
            training_job.save()
            
            logger.info(f"Model {ml_model.name} trained successfully with accuracy: {accuracy:.4f}")
            
            return {
                'success': True,
                'model_id': ml_model.id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_samples': len(students_df)
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            
            # Update model and job status
            if 'ml_model' in locals():
                ml_model.training_status = 'failed'
                ml_model.save()
            
            if 'training_job' in locals():
                training_job.status = 'failed'
                training_job.error_message = str(e)
                training_job.save()
            
            return {
                'success': False,
                'error': str(e)
            }

    def load_model(self, model_id: int):
        """Load a trained model from file"""
        try:
            ml_model = MLModel.objects.get(id=model_id)
            
            if ml_model.training_status != 'trained':
                raise ValueError(f"Model {model_id} is not trained")
            
            model_filename = f"model_{ml_model.id}_{ml_model.name.replace(' ', '_')}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            return model, ml_model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise

    def predict_student_risk(self, student_id: int, model_id: int) -> Dict[str, Any]:
        """Predict risk level for a single student"""
        try:
            # Load model and student data
            model, ml_model = self.load_model(model_id)
            student = StudentRecord.objects.get(id=student_id)
            
            # Prepare student data
            student_data = {
                'retakes': student.retakes,
                'csc1201': student.csc1201,
                'csc1202': student.csc1202,
                'csc1203': student.csc1203,
                'bsm1201': student.bsm1201,
                'ict1201': student.ict1201,
                'gender': student.gender,
                'sponsorship': student.sponsorship,
                'session': student.session,
                'remarks': student.remarks,  # Add missing remarks field
                'dropped': student.dropped,  # Add missing dropped field
            }
            
            student_df = pd.DataFrame([student_data])
            X, _ = self.prepare_features(student_df)
            
            # Make prediction
            prediction_proba = model.predict_proba(X)[0]
            prediction = model.predict(X)[0]
            confidence = max(prediction_proba)
            
            # Determine risk level
            if prediction == 1:  # At risk
                if confidence > 0.8:
                    risk_level = 'high'
                elif confidence > 0.6:
                    risk_level = 'medium'
                else:
                    risk_level = 'medium'
            else:  # Not at risk
                risk_level = 'low'
            
            # Get feature importance for this prediction
            feature_importance = {}
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                importance_scores = model.named_steps['classifier'].feature_importances_
                feature_names = X.columns
                feature_importance = dict(zip(feature_names, importance_scores))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
            
            # Create or update prediction record
            student_prediction, created = StudentPrediction.objects.update_or_create(
                student=student,
                model=ml_model,
                defaults={
                    'dropout_probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else float(prediction),
                    'risk_category': risk_level,
                    'confidence_score': float(confidence),
                    'contributing_factors': feature_importance,
                    'is_latest': True
                }
            )
            
            return {
                'success': True,
                'prediction': {
                    'student_id': student_id,
                    'risk_level': risk_level,
                    'dropout_probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else float(prediction),
                    'confidence': float(confidence),
                    'contributing_factors': feature_importance
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for student {student_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def batch_predict(self, model_id: int, student_ids: List[int] = None) -> Dict[str, Any]:
        """Perform batch predictions for multiple students"""
        try:
            ml_model = MLModel.objects.get(id=model_id)
            
            # Create prediction batch record
            batch = PredictionBatch.objects.create(
                model=ml_model,
                created_by=None,  # Set based on request user
                status='processing'
            )
            
            # Get students
            if student_ids:
                students = StudentRecord.objects.filter(id__in=student_ids)
            else:
                students = StudentRecord.objects.all()
            
            batch.total_students = students.count()
            batch.save()
            
            results = []
            processed = 0
            failed = 0
            
            for student in students:
                try:
                    result = self.predict_student_risk(student.id, model_id)
                    if result['success']:
                        results.append(result['prediction'])
                        processed += 1
                    else:
                        failed += 1
                        logger.warning(f"Failed to predict for student {student.id}: {result.get('error')}")
                except Exception as e:
                    failed += 1
                    logger.warning(f"Failed to predict for student {student.id}: {str(e)}")
            
            # Update batch status
            batch.processed_students = processed
            batch.failed_predictions = failed
            batch.status = 'completed'
            batch.completed_at = timezone.now()
            batch.save()
            
            return {
                'success': True,
                'batch_id': batch.id,
                'total_students': batch.total_students,
                'processed': processed,
                'failed': failed,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            if 'batch' in locals():
                batch.status = 'failed'
                batch.save()
            
            return {
                'success': False,
                'error': str(e)
            }

    def get_model_performance(self, model_id: int) -> Dict[str, Any]:
        """Get comprehensive model performance metrics"""
        try:
            ml_model = MLModel.objects.get(id=model_id)
            
            # Basic metrics
            metrics = {
                'model_id': model_id,
                'model_name': ml_model.name,
                'algorithm': ml_model.model_type,
                'status': ml_model.training_status,
                'accuracy': ml_model.accuracy,
                'precision': ml_model.precision,
                'recall': ml_model.recall,
                'f1_score': ml_model.f1_score,
                'training_data_size': ml_model.training_data_count,
                'feature_importance': ml_model.feature_importance,
                'hyperparameters': ml_model.hyperparameters,
                'created_at': ml_model.created_at,
                'last_trained': ml_model.training_completed_at
            }
            
            # Recent predictions summary
            recent_predictions = StudentPrediction.objects.filter(
                model=ml_model,
                is_latest=True
            )
            
            risk_distribution = {}
            for risk_level in ['low', 'medium', 'high']:
                count = recent_predictions.filter(risk_category=risk_level).count()
                risk_distribution[risk_level] = count
            
            metrics['recent_predictions'] = {
                'total': recent_predictions.count(),
                'risk_distribution': risk_distribution
            }
            
            return {
                'success': True,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get model performance: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


# Create singleton instance
ml_service = MLService()
