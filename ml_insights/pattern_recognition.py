#!/usr/bin/env python3
"""
Pattern Recognition Module

Purpose:
- Detect musical hooks and motifs using ML techniques
- Identify arrangement archetypes and musical patterns
- Analyze influencer elements in music

Features:
- Hook detection using sequence analysis
- Motif discovery with clustering
- Arrangement archetype classification
- Influencer analysis for trend identification
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import joblib
import os
from collections import Counter, defaultdict
import math

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_insights.feature_extraction import MidiFeatureExtractor


class HookDetector:
    """ML-based hook and motif detection system."""

    def __init__(self, model_dir: str = "ml_insights/models"):
        self.model_dir = model_dir
        self.pattern_classifier = None
        self.motif_clusterer = None
        self.scaler = StandardScaler()
        self.feature_extractor = MidiFeatureExtractor()

        os.makedirs(model_dir, exist_ok=True)
        self._load_or_train_models()

    def _load_or_train_models(self):
        """Load existing models or train new ones."""
        pattern_model_path = os.path.join(self.model_dir, "pattern_classifier.pkl")
        clusterer_path = os.path.join(self.model_dir, "motif_clusterer.pkl")
        scaler_path = os.path.join(self.model_dir, "pattern_scaler.pkl")

        if os.path.exists(pattern_model_path):
            self.pattern_classifier = joblib.load(pattern_model_path)
            self.motif_clusterer = joblib.load(clusterer_path) if os.path.exists(clusterer_path) else None
            self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else StandardScaler()
        else:
            self._train_models()

    def _train_models(self):
        """Train pattern recognition models."""
        # Generate synthetic pattern training data
        training_data = self._generate_pattern_training_data()

        if training_data:
            # Train pattern classifier
            X, y = self._prepare_pattern_training_data(training_data)
            if X.size > 0:
                X_scaled = self.scaler.fit_transform(X)

                self.pattern_classifier = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )
                self.pattern_classifier.fit(X_scaled, y)

                # Save models
                joblib.dump(self.pattern_classifier, os.path.join(self.model_dir, "pattern_classifier.pkl"))
                joblib.dump(self.scaler, os.path.join(self.model_dir, "pattern_scaler.pkl"))

                print(f"Trained pattern classifier on {len(X)} samples")

    def _generate_pattern_training_data(self) -> List[Dict]:
        """Generate synthetic training data for pattern recognition."""
        training_data = []

        # Generate examples of different pattern types
        pattern_types = ['hook', 'motif', 'riff', 'theme', 'ostinato']

        for pattern_type in pattern_types:
            for _ in range(30):  # 30 examples per type
                example = self._generate_pattern_example(pattern_type)
                training_data.append(example)

        return training_data

    def _generate_pattern_example(self, pattern_type: str) -> Dict[str, Any]:
        """Generate a synthetic example for a pattern type."""
        # Generate features based on pattern type
        if pattern_type == 'hook':
            # Hooks are typically catchy, short, repetitive
            return {
                'pattern_type': pattern_type,
                'repetition_score': np.random.uniform(0.7, 0.95),
                'length_score': np.random.uniform(0.2, 0.4),  # Shorter patterns
                'complexity_score': np.random.uniform(0.3, 0.6),
                'catchiness_score': np.random.uniform(0.7, 0.9),
                'memorability_score': np.random.uniform(0.75, 0.95),
                'contour_variety': np.random.uniform(0.2, 0.8),
                'rhythmic_complexity': np.random.uniform(0.1, 0.7),
                'harmonic_stability': np.random.uniform(0.3, 0.9),
                'pitch_range': np.random.uniform(5, 24),
                'interval_consistency': np.random.uniform(0.4, 0.9)
            }
        elif pattern_type == 'motif':
            # Motifs are developmental, slightly longer
            return {
                'pattern_type': pattern_type,
                'repetition_score': np.random.uniform(0.5, 0.8),
                'length_score': np.random.uniform(0.3, 0.6),
                'complexity_score': np.random.uniform(0.4, 0.7),
                'catchiness_score': np.random.uniform(0.5, 0.8),
                'memorability_score': np.random.uniform(0.6, 0.85),
                'contour_variety': np.random.uniform(0.2, 0.8),
                'rhythmic_complexity': np.random.uniform(0.1, 0.7),
                'harmonic_stability': np.random.uniform(0.3, 0.9),
                'pitch_range': np.random.uniform(5, 24),
                'interval_consistency': np.random.uniform(0.4, 0.9)
            }
        elif pattern_type == 'riff':
            # Riffs are guitar-oriented, rhythmic
            return {
                'pattern_type': pattern_type,
                'repetition_score': np.random.uniform(0.8, 0.95),
                'length_score': np.random.uniform(0.3, 0.5),
                'complexity_score': np.random.uniform(0.5, 0.8),
                'catchiness_score': np.random.uniform(0.6, 0.85),
                'memorability_score': np.random.uniform(0.7, 0.9),
                'contour_variety': np.random.uniform(0.2, 0.8),
                'rhythmic_complexity': np.random.uniform(0.1, 0.7),
                'harmonic_stability': np.random.uniform(0.3, 0.9),
                'pitch_range': np.random.uniform(5, 24),
                'interval_consistency': np.random.uniform(0.4, 0.9)
            }
        elif pattern_type == 'theme':
            # Themes are longer, more developed
            return {
                'pattern_type': pattern_type,
                'repetition_score': np.random.uniform(0.4, 0.7),
                'length_score': np.random.uniform(0.6, 0.9),
                'complexity_score': np.random.uniform(0.6, 0.9),
                'catchiness_score': np.random.uniform(0.4, 0.7),
                'memorability_score': np.random.uniform(0.5, 0.8),
                'contour_variety': np.random.uniform(0.2, 0.8),
                'rhythmic_complexity': np.random.uniform(0.1, 0.7),
                'harmonic_stability': np.random.uniform(0.3, 0.9),
                'pitch_range': np.random.uniform(5, 24),
                'interval_consistency': np.random.uniform(0.4, 0.9)
            }
        else:  # ostinato
            # Ostinatos are repetitive bass/accompaniment patterns
            return {
                'pattern_type': pattern_type,
                'repetition_score': np.random.uniform(0.85, 0.98),
                'length_score': np.random.uniform(0.4, 0.7),
                'complexity_score': np.random.uniform(0.2, 0.5),
                'catchiness_score': np.random.uniform(0.3, 0.6),
                'memorability_score': np.random.uniform(0.4, 0.7),
                'contour_variety': np.random.uniform(0.2, 0.8),
                'rhythmic_complexity': np.random.uniform(0.1, 0.7),
                'harmonic_stability': np.random.uniform(0.3, 0.9),
                'pitch_range': np.random.uniform(5, 24),
                'interval_consistency': np.random.uniform(0.4, 0.9)
            }
        """Generate a synthetic example for a pattern type."""
        example = {'pattern_type': pattern_type}

        # Generate features based on pattern type
        if pattern_type == 'hook':
            # Hooks are typically catchy, short, repetitive
            example.update({
                'repetition_score': np.random.uniform(0.7, 0.95),
                'length_score': np.random.uniform(0.2, 0.4),  # Shorter patterns
                'complexity_score': np.random.uniform(0.3, 0.6),
                'catchiness_score': np.random.uniform(0.7, 0.9),
                'memorability_score': np.random.uniform(0.75, 0.95)
            })
        elif pattern_type == 'motif':
            # Motifs are developmental, slightly longer
            example.update({
                'repetition_score': np.random.uniform(0.5, 0.8),
                'length_score': np.random.uniform(0.3, 0.6),
                'complexity_score': np.random.uniform(0.4, 0.7),
                'catchiness_score': np.random.uniform(0.5, 0.8),
                'memorability_score': np.random.uniform(0.6, 0.85)
            })
        elif pattern_type == 'riff':
            # Riffs are guitar-oriented, rhythmic
            example.update({
                'repetition_score': np.random.uniform(0.8, 0.95),
                'length_score': np.random.uniform(0.3, 0.5),
                'complexity_score': np.random.uniform(0.5, 0.8),
                'catchiness_score': np.random.uniform(0.6, 0.85),
                'memorability_score': np.random.uniform(0.7, 0.9)
            })
        elif pattern_type == 'theme':
            # Themes are longer, more developed
            example.update({
                'repetition_score': np.random.uniform(0.4, 0.7),
                'length_score': np.random.uniform(0.6, 0.9),
                'complexity_score': np.random.uniform(0.6, 0.9),
                'catchiness_score': np.random.uniform(0.4, 0.7),
                'memorability_score': np.random.uniform(0.5, 0.8)
            })
        else:  # ostinato
            # Ostinatos are repetitive bass/accompaniment patterns
            example.update({
                'repetition_score': np.random.uniform(0.85, 0.98),
                'length_score': np.random.uniform(0.4, 0.7),
                'complexity_score': np.random.uniform(0.2, 0.5),
                'catchiness_score': np.random.uniform(0.3, 0.6),
                'memorability_score': np.random.uniform(0.4, 0.7)
            })

        # Add some common features
        example.update({
            'contour_variety': np.random.uniform(0.2, 0.8),
            'rhythmic_complexity': np.random.uniform(0.1, 0.7),
            'harmonic_stability': np.random.uniform(0.3, 0.9),
            'pitch_range': np.random.uniform(5, 24),
            'interval_consistency': np.random.uniform(0.4, 0.9)
        })

        return example

    def _prepare_pattern_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for pattern classification."""
        features = []
        labels = []

        for item in training_data:
            feature_vector = [
                item.get('repetition_score', 0.5),
                item.get('length_score', 0.5),
                item.get('complexity_score', 0.5),
                item.get('catchiness_score', 0.5),
                item.get('memorability_score', 0.5),
                item.get('contour_variety', 0.5),
                item.get('rhythmic_complexity', 0.3),
                item.get('harmonic_stability', 0.6),
                item.get('pitch_range', 12),
                item.get('interval_consistency', 0.6)
            ]
            features.append(feature_vector)
            labels.append(item['pattern_type'])

        return np.array(features), np.array(labels)

    def detect_hooks(self, midi_path: str, min_confidence: float = 0.6) -> List[Dict]:
        """
        Detect hooks in a music file.

        Args:
            midi_path: Path to MIDI file
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected hooks with metadata
        """
        # Extract musical features
        features_data = self.feature_extractor.extract_features(midi_path)

        if 'error' in features_data:
            return [{'error': features_data['error']}]

        # Extract melody and harmony patterns
        hooks = []

        # Look for repetitive melodic patterns (potential hooks)
        melodic_patterns = features_data.get('melodic', {}).get('melodic_patterns', [])

        for pattern in melodic_patterns:
            if self._is_hook_candidate(pattern):
                hook_info = {
                    'type': 'melodic_hook',
                    'pattern': pattern,
                    'confidence': self._calculate_hook_confidence(pattern),
                    'time_range': pattern.get('time_range', [0, 0]),
                    'characteristics': self._analyze_hook_characteristics(pattern)
                }

                if hook_info['confidence'] >= min_confidence:
                    hooks.append(hook_info)

        # Look for harmonic hooks (chord progressions)
        detected_chords = features_data.get('harmonic', {}).get('detected_chords', [])
        if len(detected_chords) >= 4:
            chord_hook = self._detect_chord_hook(detected_chords)
            if chord_hook and chord_hook['confidence'] >= min_confidence:
                hooks.append(chord_hook)

        return hooks

    def _is_hook_candidate(self, pattern: Dict) -> bool:
        """Determine if a pattern is a potential hook."""
        # Check pattern characteristics typical of hooks
        length = pattern.get('length', 0)
        repetitions = pattern.get('repetitions', 0)

        # Hooks are typically short (4-16 notes) and repetitive
        if length < 4 or length > 16:
            return False

        if repetitions < 2:
            return False

        return True

    def _calculate_hook_confidence(self, pattern: Dict) -> float:
        """Calculate confidence score for hook detection."""
        confidence = 0.5  # Base confidence

        # Length factor (hooks are typically 4-12 notes)
        length = pattern.get('length', 8)
        if 4 <= length <= 12:
            confidence += 0.2
        elif length > 16:
            confidence -= 0.1

        # Repetition factor
        repetitions = pattern.get('repetitions', 1)
        confidence += min(0.3, repetitions * 0.1)

        # Interval variety (hooks often have varied but not too complex intervals)
        intervals = pattern.get('intervals', [])
        if intervals:
            unique_intervals = len(set(abs(i) for i in intervals))
            variety_ratio = unique_intervals / len(intervals)
            if 0.3 <= variety_ratio <= 0.7:
                confidence += 0.1

        return min(1.0, max(0.0, confidence))

    def _analyze_hook_characteristics(self, pattern: Dict) -> Dict[str, Any]:
        """Analyze characteristics of a detected hook."""
        characteristics = {}

        pitches = pattern.get('pitches', [])
        if pitches:
            # Pitch range
            characteristics['pitch_range'] = max(pitches) - min(pitches)

            # Contour analysis
            intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
            characteristics['contour_type'] = self._classify_contour(intervals)

            # Rhythmic feel
            characteristics['rhythmic_pattern'] = pattern.get('rhythm', [])

        return characteristics

    def _classify_contour(self, intervals: List[int]) -> str:
        """Classify melodic contour type."""
        if not intervals:
            return 'static'

        # Count direction changes
        directions = []
        for i in range(1, len(intervals)):
            if intervals[i] * intervals[i-1] < 0:
                directions.append('change')
            elif intervals[i] > 0:
                directions.append('up')
            elif intervals[i] < 0:
                directions.append('down')
            else:
                directions.append('static')

        change_ratio = directions.count('change') / len(directions) if directions else 0

        if change_ratio > 0.5:
            return 'zigzag'
        elif intervals[0] > 0 and sum(intervals) > 0:
            return 'ascending'
        elif intervals[0] < 0 and sum(intervals) < 0:
            return 'descending'
        else:
            return 'mixed'

    def _detect_chord_hook(self, chords: List[Dict]) -> Optional[Dict]:
        """Detect chord-based hooks."""
        if len(chords) < 4:
            return None

        # Look for repetitive chord patterns
        chord_sequence = [c.get('root', 0) for c in chords[:8]]  # First 8 chords

        # Check for repetition
        if self._has_repetition(chord_sequence):
            return {
                'type': 'harmonic_hook',
                'pattern': chord_sequence,
                'confidence': 0.75,
                'time_range': [chords[0].get('start_time', 0), chords[-1].get('end_time', 0)],
                'characteristics': {'chord_count': len(chord_sequence), 'repetitive': True}
            }

        return None

    def _has_repetition(self, sequence: List[int], min_repeat: int = 2) -> bool:
        """Check if sequence has repetitive patterns."""
        for length in range(2, len(sequence) // 2 + 1):
            for start in range(len(sequence) - length * 2 + 1):
                pattern = sequence[start:start + length]
                # Look for the pattern later in the sequence
                for match_start in range(start + length, len(sequence) - length + 1):
                    if sequence[match_start:match_start + length] == pattern:
                        return True
        return False


class ArchetypeAnalyzer:
    """ML-based arrangement archetype classification."""

    def __init__(self, model_dir: str = "ml_insights/models"):
        self.model_dir = model_dir
        self.archetype_classifier = None
        self.feature_extractor = MidiFeatureExtractor()

        os.makedirs(model_dir, exist_ok=True)
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Load or train archetype classification model."""
        model_path = os.path.join(self.model_dir, "archetype_classifier.pkl")

        if os.path.exists(model_path):
            self.archetype_classifier = joblib.load(model_path)
        else:
            self._train_model()

    def _train_model(self):
        """Train archetype classification model."""
        # This would use a dataset of known archetypes
        # For now, we'll implement a rule-based system
        print("Using rule-based archetype classification (ML training data not available)")

    def classify_archetype(self, midi_path: str) -> Dict[str, Any]:
        """
        Classify the arrangement archetype of a music file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Archetype classification with confidence
        """
        features_data = self.feature_extractor.extract_features(midi_path)

        if 'error' in features_data:
            return {'error': features_data['error']}

        # Rule-based archetype classification
        archetype = self._classify_archetype_rules(features_data)

        return {
            'primary_archetype': archetype['type'],
            'confidence': archetype['confidence'],
            'characteristics': archetype['characteristics'],
            'description': archetype['description']
        }

    def _classify_archetype_rules(self, features_data: Dict) -> Dict[str, Any]:
        """Rule-based archetype classification."""
        temporal = features_data.get('temporal', {})
        textural = features_data.get('textural', {})
        structural = features_data.get('structural', {})

        # Extract key features
        tempo = temporal.get('tempo_bpm', 120)
        polyphony = textural.get('polyphony_mean', 2)
        section_count = structural.get('section_count', 4)
        rhythm_regularity = temporal.get('rhythm_regularity', 0.7)

        # Classify based on features
        if polyphony > 3 and section_count >= 4:
            return {
                'type': 'classical_symphony',
                'confidence': 0.8,
                'characteristics': ['high_polyphony', 'multiple_sections', 'complex_harmony'],
                'description': 'Classical symphony with complex polyphony and formal structure'
            }
        elif tempo > 160 and rhythm_regularity > 0.8:
            return {
                'type': 'electronic_dance',
                'confidence': 0.85,
                'characteristics': ['fast_tempo', 'strict_rhythm', 'electronic_texture'],
                'description': 'Electronic dance music with driving rhythm and fast tempo'
            }
        elif tempo < 100 and polyphony < 2:
            return {
                'type': 'ballad',
                'confidence': 0.75,
                'characteristics': ['slow_tempo', 'monophonic_melody', 'emotional_focus'],
                'description': 'Ballad with slow tempo and focus on melody and emotion'
            }
        elif tempo > 120 and section_count <= 3:
            return {
                'type': 'pop_song',
                'confidence': 0.8,
                'characteristics': ['verse_chorus_structure', 'medium_tempo', 'catchy_hooks'],
                'description': 'Pop song with verse-chorus structure and catchy elements'
            }
        else:
            return {
                'type': 'generic',
                'confidence': 0.5,
                'characteristics': ['mixed_elements'],
                'description': 'Mixed arrangement without clear archetype'
            }


class InfluencerAnalyzer:
    """Analyze musical influencers and trend patterns."""

    def __init__(self, model_dir: str = "ml_insights/models"):
        self.model_dir = model_dir
        self.feature_extractor = MidiFeatureExtractor()

        # Define musical eras and their characteristics
        self.eras = self._define_musical_eras()

    def _define_musical_eras(self) -> Dict[str, Dict[str, Any]]:
        """Define characteristics of different musical eras."""
        return {
            'baroque': {
                'tempo_range': (60, 120),
                'complexity_focus': 'counterpoint',
                'harmony_style': 'functional',
                'key_features': ['ornamentation', 'terraced_dynamics', 'formal_structure']
            },
            'classical': {
                'tempo_range': (60, 140),
                'complexity_focus': 'form',
                'harmony_style': 'tonal',
                'key_features': ['balanced_phrases', 'clear_melody', 'homophonic_texture']
            },
            'romantic': {
                'tempo_range': (60, 160),
                'complexity_focus': 'expression',
                'harmony_style': 'chromatic',
                'key_features': ['emotional_depth', 'wide_dynamics', 'orchestral_colors']
            },
            'jazz_age': {
                'tempo_range': (120, 200),
                'complexity_focus': 'improvisation',
                'harmony_style': 'extended',
                'key_features': ['swing_rhythm', 'blue_notes', 'collective_improvisation']
            },
            'rock_era': {
                'tempo_range': (100, 180),
                'complexity_focus': 'energy',
                'harmony_style': 'power_chords',
                'key_features': ['strong_beat', 'guitar_focus', 'amplified_sound']
            },
            'electronic_age': {
                'tempo_range': (100, 160),
                'complexity_focus': 'sound_design',
                'harmony_style': 'ambient',
                'key_features': ['synthesized_sounds', 'loop_based', 'digital_effects']
            }
        }

    def analyze_influencers(self, midi_path: str) -> Dict[str, Any]:
        """
        Analyze musical influencers and stylistic influences.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Analysis of musical influences and trends
        """
        features_data = self.feature_extractor.extract_features(midi_path)

        if 'error' in features_data:
            return {'error': features_data['error']}

        # Analyze features against different eras
        era_scores = self._score_eras(features_data)

        # Find primary influences
        primary_influences = sorted(era_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            'primary_influence': primary_influences[0][0],
            'influence_scores': dict(primary_influences),
            'trend_analysis': self._analyze_trends(features_data),
            'stylistic_characteristics': self._extract_stylistic_characteristics(features_data)
        }

    def _score_eras(self, features_data: Dict) -> Dict[str, float]:
        """Score how well the music fits different musical eras."""
        scores = {}

        temporal = features_data.get('temporal', {})
        harmonic = features_data.get('harmonic', {})
        textural = features_data.get('textural', {})

        tempo = temporal.get('tempo_bpm', 120)
        harmony_complexity = harmonic.get('harmonic_complexity', 0.5)
        polyphony = textural.get('polyphony_mean', 2)

        for era, characteristics in self.eras.items():
            score = 0.0

            # Tempo match
            tempo_min, tempo_max = characteristics['tempo_range']
            if tempo_min <= tempo <= tempo_max:
                score += 0.3
            elif abs(tempo - tempo_min) <= 20 or abs(tempo - tempo_max) <= 20:
                score += 0.1

            # Harmony complexity match
            if characteristics['harmony_style'] == 'functional' and harmony_complexity < 0.4:
                score += 0.25
            elif characteristics['harmony_style'] == 'tonal' and 0.3 <= harmony_complexity <= 0.7:
                score += 0.25
            elif characteristics['harmony_style'] == 'chromatic' and harmony_complexity > 0.6:
                score += 0.25
            elif characteristics['harmony_style'] == 'extended' and harmony_complexity > 0.5:
                score += 0.25

            # Texture match
            if characteristics['complexity_focus'] == 'counterpoint' and polyphony > 3:
                score += 0.2
            elif characteristics['complexity_focus'] == 'form' and 2 <= polyphony <= 4:
                score += 0.2
            elif characteristics['complexity_focus'] == 'energy' and polyphony <= 2:
                score += 0.2

            scores[era] = min(1.0, score)

        return scores

    def _analyze_trends(self, features_data: Dict) -> Dict[str, Any]:
        """Analyze musical trends and contemporary influences."""
        trends = {}

        temporal = features_data.get('temporal', {})
        tempo = temporal.get('tempo_bpm', 120)
        rhythm_regularity = temporal.get('rhythm_regularity', 0.7)

        # Trend analysis
        if tempo > 140 and rhythm_regularity > 0.8:
            trends['electronic_dance'] = 0.8
        elif tempo > 120 and rhythm_regularity < 0.6:
            trends['indie_lofi'] = 0.7
        elif 100 <= tempo <= 130 and rhythm_regularity > 0.7:
            trends['modern_pop'] = 0.75

        return trends

    def _extract_stylistic_characteristics(self, features_data: Dict) -> List[str]:
        """Extract key stylistic characteristics."""
        characteristics = []

        temporal = features_data.get('temporal', {})
        harmonic = features_data.get('harmonic', {})
        textural = features_data.get('textural', {})

        # Analyze key characteristics
        if temporal.get('tempo_bpm', 120) > 150:
            characteristics.append('fast_tempo')
        if harmonic.get('harmonic_complexity', 0.5) > 0.7:
            characteristics.append('complex_harmony')
        if textural.get('polyphony_mean', 2) > 3:
            characteristics.append('polyphonic_texture')
        if temporal.get('rhythm_regularity', 0.7) > 0.85:
            characteristics.append('strict_rhythm')
        if harmonic.get('dissonance_index', 0.3) > 0.6:
            characteristics.append('dissonant_harmony')

        return characteristics