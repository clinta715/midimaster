#!/usr/bin/env python3
"""
ML Model Training Framework

Purpose:
- Unified framework for training ML models for music analysis
- Support for different model types and training strategies
- Automated model evaluation and validation
- Model persistence and versioning

Features:
- Automated data preprocessing
- Cross-validation support
- Hyperparameter optimization
- Model performance monitoring
- Training pipeline management
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
import os
import json
from datetime import datetime
import warnings

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_insights.feature_extraction import MidiFeatureExtractor


class ModelTrainer:
    """Unified framework for training ML models."""

    def __init__(self, model_dir: str = "ml_insights/models"):
        self.model_dir = model_dir
        self.feature_extractor = MidiFeatureExtractor()
        self.scalers = {}
        self.encoders = {}

        os.makedirs(model_dir, exist_ok=True)

    def train_model(self, model_type: str, X: np.ndarray, y: np.ndarray,
                   model_name: Optional[str] = None, hyperparameters: Optional[Dict[str, Any]] = None,
                   validation_split: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a machine learning model.

        Args:
            model_type: Type of model ('classifier', 'regressor', 'clusterer')
            X: Feature matrix
            y: Target values
            model_name: Name for saving the model
            hyperparameters: Model hyperparameters
            validation_split: Validation set proportion
            cv_folds: Number of cross-validation folds

        Returns:
            Training results and metrics
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}"

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y if model_type == 'classifier' else None
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Initialize model
        model = self._create_model(model_type, hyperparameters)

        # Train model
        print(f"Training {model_type} model...")
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        print("Evaluating model...")
        evaluation_results = self._evaluate_model(
            model, model_type, X_train_scaled, X_val_scaled, y_train, y_val, cv_folds
        )

        # Save model and preprocessing objects
        self._save_model(model, scaler, model_name, evaluation_results)

        results = {
            'model_name': model_name,
            'model_type': model_type,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': X.shape[1],
            'hyperparameters': hyperparameters or {},
            'evaluation': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }

        return results

    def _create_model(self, model_type: str, hyperparameters: Optional[Dict[str, Any]] = None):
        """Create model instance based on type and hyperparameters."""
        if hyperparameters is None:
            hyperparameters = {}

        if model_type == 'classifier':
            model_class = hyperparameters.get('model_class', 'random_forest')
            if model_class == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', None),
                    random_state=42
                )
            elif model_class == 'svm':
                return SVC(
                    C=hyperparameters.get('C', 1.0),
                    kernel=hyperparameters.get('kernel', 'rbf'),
                    random_state=42
                )
            elif model_class == 'logistic':
                return LogisticRegression(
                    C=hyperparameters.get('C', 1.0),
                    random_state=42
                )

        elif model_type == 'regressor':
            model_class = hyperparameters.get('model_class', 'random_forest')
            if model_class == 'random_forest':
                return RandomForestRegressor(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', None),
                    random_state=42
                )
            elif model_class == 'svm':
                return SVR(
                    C=hyperparameters.get('C', 1.0),
                    kernel=hyperparameters.get('kernel', 'rbf')
                )
            elif model_class == 'linear':
                return LinearRegression()

        # Default fallback
        return RandomForestClassifier(random_state=42)

    def _evaluate_model(self, model, model_type: str, X_train: np.ndarray, X_val: np.ndarray,
                       y_train: np.ndarray, y_val: np.ndarray, cv_folds: int) -> Dict[str, Any]:
        """Evaluate trained model performance."""
        evaluation = {}

        # Cross-validation scores
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy' if model_type == 'classifier' else 'r2')
            evaluation['cross_validation'] = {
                'mean_score': float(np.mean(cv_scores)),
                'std_score': float(np.std(cv_scores)),
                'scores': [float(s) for s in cv_scores]
            }
        except Exception as e:
            evaluation['cross_validation'] = {'error': str(e)}

        # Validation set predictions
        y_pred = model.predict(X_val)

        if model_type == 'classifier':
            evaluation['validation_metrics'] = {
                'accuracy': float(accuracy_score(y_val, y_pred)),
                'precision': float(precision_score(y_val, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_val, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_val, y_pred, average='weighted', zero_division=0))
            }
        else:  # regressor
            evaluation['validation_metrics'] = {
                'mse': float(mean_squared_error(y_val, y_pred)),
                'mae': float(mean_absolute_error(y_val, y_pred)),
                'r2_score': float(r2_score(y_val, y_pred))
            }

        # Model-specific metrics
        if hasattr(model, 'feature_importances_'):
            evaluation['feature_importance'] = [float(f) for f in model.feature_importances_]

        return evaluation

    def _save_model(self, model, scaler: StandardScaler, model_name: str, evaluation: Dict[str, Any]):
        """Save model and preprocessing objects."""
        model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")

        # Save model and scaler
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Save metadata
        metadata = {
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'evaluation': evaluation
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Saved model to {model_path}")

    def hyperparameter_tuning(self, model_type: str, X: np.ndarray, y: np.ndarray,
                            param_grid: Dict[str, List], cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.

        Args:
            model_type: Type of model
            X: Feature matrix
            y: Target values
            param_grid: Parameter grid for search
            cv_folds: Number of cross-validation folds

        Returns:
            Best parameters and performance
        """
        # Create base model
        if model_type == 'classifier':
            base_model = RandomForestClassifier(random_state=42)
            scoring = 'accuracy'
        else:
            base_model = RandomForestRegressor(random_state=42)
            scoring = 'r2'

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        print("Performing hyperparameter tuning...")
        grid_search.fit(X_scaled, y)

        results = {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'cv_results': {
                'mean_test_score': [float(s) for s in grid_search.cv_results_['mean_test_score']],
                'std_test_score': [float(s) for s in grid_search.cv_results_['std_test_score']],
                'params': grid_search.cv_results_['params']
            }
        }

        return results

    def train_genre_classifier(self, midi_paths: List[str], genre_labels: List[str]) -> Dict[str, Any]:
        """
        Train a genre classification model from MIDI files.

        Args:
            midi_paths: List of MIDI file paths
            genre_labels: Corresponding genre labels

        Returns:
            Training results
        """
        print(f"Training genre classifier on {len(midi_paths)} samples...")

        # Extract features from MIDI files
        features = []
        valid_labels = []

        for path, label in zip(midi_paths, genre_labels):
            feature_dict = self.feature_extractor.extract_features(path)
            if 'error' not in feature_dict:
                # Convert to feature vector
                feature_vector = self._features_to_vector(feature_dict)
                features.append(feature_vector)
                valid_labels.append(label)

        if not features:
            return {'error': 'No valid features extracted'}

        X = np.array(features)
        y = np.array(valid_labels)

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = np.array(label_encoder.fit_transform(y), dtype=int)

        # Train model
        hyperparameters = {
            'model_class': 'random_forest',
            'n_estimators': 200,
            'max_depth': 10
        }

        results = self.train_model(
            'classifier', X, y_encoded,
            model_name='genre_classifier',
            hyperparameters=hyperparameters
        )

        # Save label encoder
        encoder_path = os.path.join(self.model_dir, "genre_classifier_encoder.pkl")
        joblib.dump(label_encoder, encoder_path)

        results['genres'] = list(label_encoder.classes_)
        return results

    def train_quality_scorer(self, midi_paths: List[str], quality_scores: List[float]) -> Dict[str, Any]:
        """
        Train a quality scoring model from MIDI files.

        Args:
            midi_paths: List of MIDI file paths
            quality_scores: Corresponding quality scores (0-1)

        Returns:
            Training results
        """
        print(f"Training quality scorer on {len(midi_paths)} samples...")

        # Extract features
        features = []
        valid_scores = []

        for path, score in zip(midi_paths, quality_scores):
            feature_dict = self.feature_extractor.extract_features(path)
            if 'error' not in feature_dict:
                feature_vector = self._features_to_vector(feature_dict)
                features.append(feature_vector)
                valid_scores.append(score)

        if not features:
            return {'error': 'No valid features extracted'}

        X = np.array(features)
        y = np.array(valid_scores)

        # Train model
        hyperparameters = {
            'model_class': 'random_forest',
            'n_estimators': 100
        }

        results = self.train_model(
            'regressor', X, y,
            model_name='quality_scorer',
            hyperparameters=hyperparameters
        )

        return results

    def _features_to_vector(self, features: Dict) -> List[float]:
        """Convert feature dictionary to numerical vector."""
        vector = []

        # Basic features
        basic = features.get('basic_info', {})
        vector.extend([
            basic.get('tracks', 1),
            basic.get('ticks_per_beat', 480),
            basic.get('length_seconds', 120)
        ])

        # Temporal features
        temporal = features.get('temporal', {})
        vector.extend([
            temporal.get('tempo_bpm', 120),
            temporal.get('rhythm_regularity', 0.7),
            temporal.get('note_density', 8),
            temporal.get('velocity_mean', 64),
            temporal.get('velocity_std', 20),
            temporal.get('total_duration', 120)
        ])

        # Harmonic features
        harmonic = features.get('harmonic', {})
        vector.extend([
            harmonic.get('detected_key', 0),
            harmonic.get('key_confidence', 0.5),
            harmonic.get('harmonic_complexity', 0.5),
            harmonic.get('pitch_entropy', 0.6),
            harmonic.get('dissonance_index', 0.4),
            harmonic.get('chord_count', 8)
        ])

        # Melodic features
        melodic = features.get('melodic', {})
        vector.extend([
            melodic.get('pitch_range', 24),
            melodic.get('pitch_mean', 60),
            melodic.get('pitch_std', 12),
            melodic.get('contour_complexity', 0.5)
        ])

        # Structural features
        structural = features.get('structural', {})
        vector.extend([
            structural.get('section_count', 4),
            structural.get('structure_complexity', 0.5)
        ])

        # Textural features
        textural = features.get('textural', {})
        vector.extend([
            textural.get('channel_count', 2),
            textural.get('polyphony_mean', 2),
            textural.get('dynamic_range', 40)
        ])

        return vector

    def load_model(self, model_name: str) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
        """
        Load a trained model and its preprocessing objects.

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of (model, scaler, metadata)
        """
        model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")

        if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
            raise FileNotFoundError(f"Model {model_name} not found")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return model, scaler, metadata

    def get_training_summary(self, model_name: str) -> Dict[str, Any]:
        """Get training summary for a model."""
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")

        if not os.path.exists(metadata_path):
            return {'error': f'Model {model_name} metadata not found'}

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata


class ModelEvaluator:
    """Advanced model evaluation and validation tools."""

    def __init__(self, model_dir: str = "ml_insights/models"):
        self.model_dir = model_dir

    def evaluate_model_comprehensive(self, model_name: str, X_test: np.ndarray,
                                   y_test: np.ndarray, model_type: str = 'classifier') -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.

        Args:
            model_name: Name of model to evaluate
            X_test: Test feature matrix
            y_test: Test target values
            model_type: Type of model ('classifier' or 'regressor')

        Returns:
            Comprehensive evaluation results
        """
        # Load model
        trainer = ModelTrainer(self.model_dir)
        model, scaler, metadata = trainer.load_model(model_name)

        # Scale test data
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        evaluation = {
            'model_name': model_name,
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }

        if model_type == 'classifier':
            evaluation.update(self._evaluate_classifier(model, X_test_scaled, y_test, y_pred))
        else:
            evaluation.update(self._evaluate_regressor(model, X_test_scaled, y_test, y_pred))

        # Model interpretability
        if hasattr(model, 'feature_importances_'):
            evaluation['feature_importance'] = {
                'top_features': self._get_top_features(model.feature_importances_),
                'importance_scores': [float(f) for f in model.feature_importances_]
            }

        return evaluation

    def _evaluate_classifier(self, model, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate classifier performance."""
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': self._safe_confusion_matrix(y_test, y_pred)
        }

    def _evaluate_regressor(self, model, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate regressor performance."""
        return {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2_score': float(r2_score(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'prediction_stats': {
                'mean_prediction': float(np.mean(y_pred)),
                'std_prediction': float(np.std(y_pred)),
                'prediction_range': [float(np.min(y_pred)), float(np.max(y_pred))]
            }
        }

    def _safe_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
        """Create confusion matrix safely."""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        return cm.tolist()

    def _get_top_features(self, feature_importances: np.ndarray, top_n: int = 10) -> List[Tuple[int, float]]:
        """Get top N most important features."""
        top_indices = np.argsort(feature_importances)[-top_n:][::-1]
        return [(int(idx), float(feature_importances[idx])) for idx in top_indices]

    def compare_models(self, model_names: List[str], X_test: np.ndarray,
                      y_test: np.ndarray, model_type: str = 'classifier') -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.

        Args:
            model_names: List of model names to compare
            X_test: Test feature matrix
            y_test: Test target values
            model_type: Type of models

        Returns:
            Comparison results
        """
        comparison = {
            'models_compared': model_names,
            'test_samples': len(X_test),
            'model_results': {},
            'best_model': None,
            'ranking': []
        }

        trainer = ModelTrainer(self.model_dir)
        best_score = -float('inf')
        best_model = None

        for model_name in model_names:
            try:
                evaluation = self.evaluate_model_comprehensive(model_name, X_test, y_test, model_type)
                comparison['model_results'][model_name] = evaluation

                # Determine best model
                if model_type == 'classifier':
                    score = evaluation.get('f1_score', 0)
                else:
                    score = evaluation.get('r2_score', -float('inf'))

                if score > best_score:
                    best_score = score
                    best_model = model_name

                comparison['ranking'].append({
                    'model': model_name,
                    'score': score,
                    'metric': 'f1_score' if model_type == 'classifier' else 'r2_score'
                })

            except Exception as e:
                comparison['model_results'][model_name] = {'error': str(e)}

        comparison['best_model'] = best_model
        comparison['ranking'].sort(key=lambda x: x['score'], reverse=True)

        return comparison

    def generate_model_report(self, model_name: str, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report for a trained model.

        Args:
            model_name: Name of the model
            output_path: Path to save the report

        Returns:
            Report content as string
        """
        trainer = ModelTrainer(self.model_dir)

        try:
            model, scaler, metadata = trainer.load_model(model_name)
        except FileNotFoundError:
            return f"Model {model_name} not found."

        report_lines = [
            f"# Model Report: {model_name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Metadata",
            f"- **Model Type**: {metadata.get('evaluation', {}).get('model_type', 'Unknown')}",
            f"- **Training Date**: {metadata.get('saved_at', 'Unknown')}",
            f"- **Training Samples**: {metadata.get('evaluation', {}).get('training_samples', 'Unknown')}",
            "",
            "## Performance Metrics"
        ]

        # Add evaluation metrics
        evaluation = metadata.get('evaluation', {})
        cv = evaluation.get('cross_validation', {})
        val_metrics = evaluation.get('validation_metrics', {})

        if cv:
            report_lines.extend([
                f"- **Cross-Validation Score**: {cv.get('mean_score', 'N/A'):.4f} ± {cv.get('std_score', 'N/A'):.4f}",
            ])

        if val_metrics:
            for metric, value in val_metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.4f}")
                else:
                    report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {value}")

        report_lines.extend([
            "",
            "## Feature Importance"
        ])

        # Add feature importance if available
        if 'feature_importance' in evaluation:
            feature_imp = evaluation['feature_importance']
            for i, importance in enumerate(feature_imp[:10]):  # Top 10
                report_lines.append(f"- Feature {i+1}: {importance:.4f}")

        report = "\n".join(report_lines)

        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_path}")

        return report