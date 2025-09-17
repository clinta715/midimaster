#!/usr/bin/env python3
"""
ML-Based Genre Classification System

Purpose:
- Automatically classify music by genre using ML models
- Provide confidence scores and multi-genre detection
- Support custom genre definitions and training

Features:
- Multi-class genre classification
- Confidence scoring
- Feature importance analysis
- Custom genre model training
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import mido
import os
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_insights.feature_extraction import MidiFeatureExtractor


class GenreClassifier:
    """ML-based genre classification for music."""

    def __init__(self, model_dir: str = "ml_insights/models"):
        self.model_dir = model_dir
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extractor = MidiFeatureExtractor()
        self.ambient_analyzer = None  # Lazy init

        # Genre definitions with characteristic features
        self.genre_definitions = self._get_genre_definitions()

        os.makedirs(model_dir, exist_ok=True)
        self._load_or_train_model()

    def _get_genre_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get genre definitions with characteristic features."""
        return {
            'classical': {
                'tempo_range': (60, 120),
                'harmony_complexity': (0.7, 1.0),
                'polyphony_range': (2, 6),
                'dynamic_range': (0.1, 1.0)
            },
            'electronic': {
                'tempo_range': (100, 160),
                'harmony_complexity': (0.3, 0.7),
                'polyphony_range': (1, 3),
                'dynamic_range': (0.4, 0.8)
            },
            'pop': {
                'tempo_range': (100, 130),
                'harmony_complexity': (0.4, 0.6),
                'polyphony_range': (2, 4),
                'dynamic_range': (0.3, 0.7)
            },
            'rock': {
                'tempo_range': (110, 150),
                'harmony_complexity': (0.3, 0.5),
                'polyphony_range': (2, 4),
                'dynamic_range': (0.5, 0.9)
            },
            'jazz': {
                'tempo_range': (120, 200),
                'harmony_complexity': (0.6, 0.9),
                'polyphony_range': (1, 5),
                'dynamic_range': (0.2, 0.8)
            },
            'hip-hop': {
                'tempo_range': (80, 110),
                'harmony_complexity': (0.2, 0.5),
                'polyphony_range': (1, 3),
                'dynamic_range': (0.4, 0.7)
            },
            'drum-and-bass': {
                'tempo_range': (160, 180),
                'harmony_complexity': (0.2, 0.4),
                'polyphony_range': (1, 3),
                'dynamic_range': (0.6, 1.0)
            }
        }

    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        model_path = os.path.join(self.model_dir, "genre_classifier.pkl")
        scaler_path = os.path.join(self.model_dir, "genre_scaler.pkl")
        le_path = os.path.join(self.model_dir, "genre_label_encoder.pkl")

        if os.path.exists(model_path):
            self.classifier = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(le_path)
            print("Loaded existing genre classifier model.")
        else:
            self._train_genre_model()

    def _train_genre_model(self):
        """Train the genre classification model with synthetic data."""
        # Generate synthetic training data
        training_data = self._generate_genre_training_data()

        if training_data:
            X, y = self._prepare_genre_training_data(training_data)
            if X.size > 0:
                X_scaled = self.scaler.fit_transform(X)

                self.classifier = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )
                self.classifier.fit(X_scaled, y)

                # Fit label encoder
                self.label_encoder.fit(y)

                # Save models
                joblib.dump(self.classifier, os.path.join(self.model_dir, "genre_classifier.pkl"))
                joblib.dump(self.scaler, os.path.join(self.model_dir, "genre_scaler.pkl"))
                joblib.dump(self.label_encoder, os.path.join(self.model_dir, "genre_label_encoder.pkl"))

                print(f"Trained genre classifier on {len(X)} samples")
            else:
                print("No training data available; using default classifier.")
                self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            print("No training data generated; using default classifier.")
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    def _generate_genre_training_data(self) -> List[Dict]:
        """Generate synthetic training data for genre classification."""
        training_data = []
        genres = list(self.genre_definitions.keys())

        for genre in genres:
            for _ in range(50):  # 50 examples per genre
                example = self._generate_genre_example(genre)
                training_data.append(example)

        return training_data

    def _generate_genre_example(self, genre: str) -> Dict[str, Any]:
        """Generate a synthetic example for a specific genre."""
        defn = self.genre_definitions[genre]

        return {
            'genre': genre,
            'tempo': np.random.uniform(*defn['tempo_range']),
            'harmony_complexity': np.random.uniform(*defn['harmony_complexity']),
            'polyphony': np.random.uniform(*defn['polyphony_range']),
            'dynamic_range': np.random.uniform(*defn['dynamic_range']),
            'rhythm_regularity': np.random.uniform(0.5, 0.9),
            'melodic_range': np.random.uniform(10, 30),
            'chord_density': np.random.uniform(0.1, 0.8),
            'instrumentation_diversity': np.random.uniform(1, 10),
            'section_variety': np.random.uniform(0.2, 0.8)
        }

    def _prepare_genre_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for genre classification."""
        features = []
        labels = []

        for item in training_data:
            feature_vector = [
                item.get('tempo', 120),
                item.get('harmony_complexity', 0.5),
                item.get('polyphony', 2),
                item.get('dynamic_range', 0.5),
                item.get('rhythm_regularity', 0.7),
                item.get('melodic_range', 20),
                item.get('chord_density', 0.4),
                item.get('instrumentation_diversity', 5),
                item.get('section_variety', 0.5)
            ]
            features.append(feature_vector)
            labels.append(item['genre'])

        return np.array(features), np.array(labels)

    def classify_genre(self, midi_path: str, top_k: int = 1) -> Dict[str, Any]:
        """Classify the genre of a MIDI file.

        Args:
            midi_path: Path to MIDI file
            top_k: Number of top genres to return (default: 1)
        """
        if self.classifier is None:
            return {'error': 'Model not trained or loaded.'}

        features_data = self.feature_extractor.extract_features(midi_path)

        if 'error' in features_data:
            return {'error': features_data['error']}

        # Extract relevant features for classification
        # Note: This assumes feature_extractor provides compatible features; adjust as needed
        feature_vector = self._extract_classification_features(features_data)
        if feature_vector is None:
            return {'error': 'Unable to extract features for classification.'}

        X_scaled = self.scaler.transform([feature_vector])
        prediction = self.classifier.predict(X_scaled)[0]
        probabilities = self.classifier.predict_proba(X_scaled)[0]

        # Get top_k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_genres = [(self.label_encoder.classes_[i], probabilities[i]) for i in top_indices]

        if top_k == 1:
            genre, confidence = top_genres[0]
            return {
                'predicted_genre': genre,
                'confidence': confidence,
                'probabilities': dict(zip(self.label_encoder.classes_, probabilities))
            }
        else:
            return {
                'top_genres': top_genres,
                'all_probabilities': dict(zip(self.label_encoder.classes_, probabilities))
            }

    def _extract_classification_features(self, features_data: Dict) -> Optional[List[float]]:
        """Extract features from feature_data for genre classification."""
        temporal = features_data.get('temporal', {})
        harmonic = features_data.get('harmonic', {})
        textural = features_data.get('textural', {})

        try:
            return [
                temporal.get('tempo_bpm', 120),
                harmonic.get('harmonic_complexity', 0.5),
                textural.get('polyphony_mean', 2),
                # Dynamic range approximation
                0.5,  # Placeholder; derive from velocity data if available
                temporal.get('rhythm_regularity', 0.7),
                # Melodic range from melodic features
                features_data.get('melodic', {}).get('pitch_range', 20),
                harmonic.get('chord_density', 0.4),
                # Instrumentation diversity approximation
                len(features_data.get('instrumentation', {}).get('unique_instruments', [1])),
                # Section variety
                features_data.get('structural', {}).get('section_count', 4) / 10.0  # Normalized
            ]
        except:
            return None