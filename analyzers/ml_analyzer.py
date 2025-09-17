"""ML Analyzer Module for MIDI Pattern Analysis.

This module provides machine learning capabilities for analyzing MIDI patterns,
including similarity clustering, genre classification, quality assessment, and
predictive completion. Integrates with ReferencePatternLibrary for data loading.
Uses scikit-learn for ML algorithms.
"""

import os
import numpy as np
from typing import List, Optional, Dict, Tuple, Any, Callable
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict, Counter
import warnings

# Local imports
try:
    from analyzers.reference_pattern_library import ReferencePatternLibrary, PatternMetadata, MidiPatternData
    from music_theory import MusicTheory
    from midi_pattern_extractor import NoteEvent
except ImportError as e:
    raise ImportError(f"Required modules not found: {e}. Ensure project dependencies are installed.")


class MLAnalyzer:
    """
    MLAnalyzer class for ML-based analysis of MIDI patterns.

    Provides:
    - Pattern similarity clustering using unsupervised learning (KMeans, DBSCAN)
    - Automatic genre classification of patterns using supervised learning
    - Quality assessment algorithms scoring groove, harmony, consistency
    - Predictive pattern completion using Markov chains

    Integrates with ReferencePatternLibrary for loading reference data.
    """

    def __init__(self, library: Optional[ReferencePatternLibrary] = None):
        """
        Initialize MLAnalyzer.

        Args:
            library: Optional ReferencePatternLibrary instance. If None, creates a new one.

        Raises:
            ValueError: If library loading fails.
        """
        self.library = library or ReferencePatternLibrary()
        if not library:
            try:
                self.library.load_from_directory("reference_midis")
            except Exception as e:
                warnings.warn(f"Failed to load library from reference_midis: {e}")
        self.scaler = StandardScaler()
        self.genre_classifier: Optional[RandomForestClassifier] = None
        self.cluster_model = None
        self.feature_cache: Dict[MidiPatternData, np.ndarray] = {}
        self.pattern_features: Dict[MidiPatternData, np.ndarray] = {}
        self.labels: Dict[MidiPatternData, str] = {}
        self._load_data()

    def _load_data(self):
        """Load patterns and extract features for training."""
        try:
            all_patterns = self.library.get_patterns()
            self.pattern_features = {}
            self.labels = {}
            for pattern in all_patterns:
                features = self.extract_features(pattern)
                self.pattern_features[pattern] = features
                meta = self.library.get_metadata(pattern)
                self.labels[pattern] = meta.genre
            if not self.pattern_features:
                warnings.warn("No patterns loaded for ML analysis.")
        except Exception as e:
            raise ValueError(f"Failed to load data: {e}")

    def extract_features(self, pattern: MidiPatternData) -> np.ndarray:
        """
        Extract feature vector from a MIDI pattern for ML.

        Features include pitch, duration, velocity, density, syncopation, etc.
        Returns a 15-dimensional vector.

        Args:
            pattern: MidiPatternData to extract features from.

        Returns:
            np.ndarray: Feature vector.
        """
        if pattern in self.feature_cache:
            return self.feature_cache[pattern]

        notes = [n for n in pattern.notes if hasattr(n, 'note') and n.note is not None]
        if not notes:
            features = np.zeros(15)
            self.feature_cache[pattern] = features
            return features

        pitches = np.array([n.note for n in notes])
        durations = np.array([n.duration for n in notes if n.duration > 0])
        velocities = np.array([n.velocity for n in notes])
        start_times = np.array([n.start_time for n in notes])

        # Basic features
        avg_pitch = np.mean(pitches)
        pitch_range = np.max(pitches) - np.min(pitches)
        avg_dur = np.mean(durations) if len(durations) > 0 else 1.0
        avg_vel = np.mean(velocities)
        ticks_per_beat = pattern.ticks_per_beat if hasattr(pattern, 'ticks_per_beat') else 480
        length_beats = pattern.length_ticks / ticks_per_beat if hasattr(pattern, 'length_ticks') else 16
        density = len(notes) / length_beats if length_beats > 0 else 0

        # Rhythmic features
        beat_positions = start_times % ticks_per_beat
        syncopation = np.std(beat_positions) / ticks_per_beat if ticks_per_beat > 0 else 0
        groove = 1 - (np.std(durations) / avg_dur if avg_dur > 0 and len(durations) > 1 else 0)

        # Melodic features
        interval_var = 0.0
        avg_interval = 0.0
        if len(pitches) > 1:
            intervals = np.diff(pitches)
            interval_var = np.var(intervals)
            avg_interval = np.mean(np.abs(intervals)) / 12
        velocity_var = np.var(velocities)

        # Harmonic tension (simple)
        tension = 0
        time_to_pitches = defaultdict(list)
        for n in notes:
            time_to_pitches[n.start_time].append(n.note)
        chord_counts = sum(1 for group in time_to_pitches.values() if len(group) > 1)
        if len(notes) > 0:
            tension = chord_counts / len(notes) * interval_var / 12  # Normalized

        # Tempo
        bpm = pattern.tempos[0].bpm if pattern.tempos else 120.0
        tempo_norm = bpm / 200.0

        # Complexity from metadata
        meta = self.library.get_metadata(pattern)
        complexity = meta.complexity

        # Contour variety (simple direction changes)
        contour_var = 0
        if len(pitches) > 2:
            directions = np.sign(np.diff(pitches))
            contour_var = np.sum(np.diff(directions) != 0) / (len(directions) - 1) if len(directions) > 1 else 0

        # Features vector (15 dims)
        features = np.array([
            avg_pitch / 128,  # 0
            pitch_range / 127,  # 1
            avg_dur / ticks_per_beat,  # 2
            avg_vel / 127,  # 3
            density / 4,  # 4 (notes per beat max 4)
            syncopation,  # 5
            groove,  # 6
            interval_var / 12,  # 7 (octave)
            velocity_var / 127,  # 8
            tension,  # 9
            tempo_norm,  # 10
            complexity,  # 11
            contour_var,  # 12
            len(set(pitches)) / len(pitches) if len(pitches) > 0 else 0,  # 13 pitch diversity
            avg_interval  # 14 avg interval
        ], dtype=np.float64)

        self.feature_cache[pattern] = features
        return features

    def cluster_patterns(self, method: str = 'kmeans', **kwargs) -> Tuple[Dict[int, List[MidiPatternData]], np.ndarray]:
        """
        Perform clustering on loaded patterns.

        Args:
            method: 'kmeans' or 'dbscan'
            **kwargs: Parameters for clusterer (e.g., n_clusters=5, eps=0.5)

        Returns:
            Tuple of (clusters dict, labels array)

        Raises:
            ValueError: If insufficient data or invalid method.
        """
        if not self.pattern_features:
            raise ValueError("No patterns loaded. Call _load_data or provide library.")

        X = np.array(list(self.pattern_features.values()))
        if len(X) < 2:
            warnings.warn("Insufficient patterns for clustering.")
            return {}, np.zeros(len(X))

        X_scaled = self.scaler.fit_transform(X)

        try:
            if method.lower() == 'kmeans':
                n_clusters = kwargs.get('n_clusters', min(5, len(X) // 2))
                self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = self.cluster_model.fit_predict(X_scaled)
            elif method.lower() == 'dbscan':
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 2)
                self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = self.cluster_model.fit_predict(X_scaled)
            else:
                raise ValueError("Method must be 'kmeans' or 'dbscan'.")
        except Exception as e:
            raise ValueError(f"Clustering failed: {e}")

        clusters = defaultdict(list)
        patterns_list = list(self.pattern_features.keys())
        for i, label in enumerate(labels):
            clusters[label].append(patterns_list[i])

        return dict(clusters), labels

    def fit_genre_classifier(self):
        """Train genre classifier on loaded patterns."""
        if not self.pattern_features:
            raise ValueError("No patterns loaded.")

        genres = list(self.labels.values())
        if len(set(genres)) < 2:
            warnings.warn("Insufficient genre variety for classification.")
            return

        X = np.array(list(self.pattern_features.values()))
        y = np.array(list(self.labels.values()))

        try:
            X_scaled = self.scaler.fit_transform(X)
            self.genre_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.genre_classifier.fit(X_scaled, y)
        except Exception as e:
            warnings.warn(f"Failed to train genre classifier: {e}")
            self.genre_classifier = None

    def classify_genre(self, pattern: MidiPatternData) -> Tuple[str, float]:
        """
        Classify genre of a pattern.

        Args:
            pattern: MidiPatternData to classify.

        Returns:
            Tuple of (predicted_genre, confidence)

        Raises:
            ValueError: If classifier not fitted.
        """
        if self.genre_classifier is None:
            self.fit_genre_classifier()

        if self.genre_classifier is None:
            return 'unknown', 0.0

        feat = self.extract_features(pattern).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        pred = self.genre_classifier.predict(feat_scaled)[0]
        probs = self.genre_classifier.predict_proba(feat_scaled)[0]
        confidence = np.max(probs)

        return pred, confidence

    def find_similar_patterns(self, pattern: MidiPatternData, n: int = 5, threshold: float = 0.7) -> List[Tuple[MidiPatternData, float]]:
        """
        Find similar patterns using cosine similarity on features.

        Args:
            pattern: Query MidiPatternData.
            n: Max number to return.
            threshold: Min similarity score.

        Returns:
            List of (similar_pattern, similarity_score) tuples.
        """
        feat = self.extract_features(pattern).reshape(1, -1)
        similarities = []
        for p, f in self.pattern_features.items():
            if p is pattern:
                continue
            f_arr = f.reshape(1, -1)
            sim = 1 - cosine_distances(feat, f_arr)[0, 0]
            if sim >= threshold:
                similarities.append((p, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    def assess_quality(self, pattern: MidiPatternData) -> Dict[str, Any]:
        """
        Assess quality of a pattern using multiple metrics.

        Metrics: groove (rhythmic), harmony (tension), consistency (velocity/dur var).

        Args:
            pattern: MidiPatternData to assess.

        Returns:
            Dict of metric scores (0-1) and feedback.
        """
        feat = self.extract_features(pattern)
        notes = [n for n in pattern.notes if n.note is not None]

        # Groove: density, syncopation, groove
        groove = 0.3 * feat[4] + 0.4 * feat[5] + 0.3 * feat[6]

        # Harmony: low tension good
        tension = feat[9]
        harmony = 1 - min(tension, 1.0)

        # Consistency: low variance good
        consistency = 1 - min(feat[8], 1.0)  # velocity var

        # Overall weighted
        overall = 0.4 * groove + 0.3 * harmony + 0.3 * consistency

        # Feedback
        feedback = []
        if groove < 0.5:
            feedback.append("Improve rhythmic variety.")
        if harmony < 0.5:
            feedback.append("Reduce harmonic tension.")
        if consistency < 0.5:
            feedback.append("Stabilize dynamics.")

        return {
            'groove': float(groove),
            'harmony': float(harmony),
            'consistency': float(consistency),
            'overall': float(overall),
            'feedback': feedback
        }

    def complete_pattern(self, pattern: MidiPatternData, num_notes: int = 8) -> List[NoteEvent]:
        """
        Predictively complete a partial pattern using Markov chain on pitches.

        Args:
            pattern: Partial MidiPatternData.
            num_notes: Number of notes to generate.

        Returns:
            List of new NoteEvent objects.
        """
        notes = [n for n in pattern.notes if hasattr(n, 'note') and n.note is not None]
        if len(notes) < 1:
            warnings.warn("Cannot complete empty pattern.")
            return []

        # Build pitch transitions
        transitions = defaultdict(list)
        for i in range(len(notes) - 1):
            curr = notes[i].note
            next_p = notes[i + 1].note
            transitions[curr].append(next_p)

        # Generate
        new_notes = []
        last_note = notes[-1]
        last_pitch = last_note.note
        current_time = last_note.start_time + last_note.duration

        avg_dur = float(np.mean([n.duration for n in notes if n.duration > 0]) if len(notes) > 0 else 1.0)
        avg_vel = float(np.mean([n.velocity for n in notes]))
        channel = last_note.channel
        instrument = getattr(last_note, 'instrument', None)

        for _ in range(num_notes):
            if last_pitch in transitions and transitions[last_pitch]:
                next_pitch = np.random.choice(transitions[last_pitch])
            else:
                # Fallback: repeat or random in range
                next_pitch = last_pitch + np.random.choice([-2, -1, 0, 1, 2])

            # Clamp to MIDI range
            next_pitch = np.clip(next_pitch, 21, 108)

            dur = max(float(avg_dur + np.random.normal(0, avg_dur * 0.1)), 0.1)
            vel = np.clip(float(avg_vel + np.random.normal(0, 10)), 1, 127)

            new_note = NoteEvent(
                note=int(next_pitch),
                start_time=current_time,
                duration=dur,
                velocity=int(vel),
                channel=channel,
                instrument=instrument
            )
            new_notes.append(new_note)

            current_time += dur
            last_pitch = next_pitch

        return new_notes


if __name__ == "__main__":
    """Example usage of MLAnalyzer."""
    # Load library
    lib = ReferencePatternLibrary()
    try:
        lib.load_from_directory("reference_midis")
        print(f"Loaded {lib.get_stats()['total_patterns']} patterns.")
    except Exception as e:
        print(f"Example: Could not load real data: {e}. Using empty library.")

    # Initialize analyzer
    analyzer = MLAnalyzer(lib)

    # Get a sample pattern (if available)
    all_patterns = list(analyzer.pattern_features.keys())
    if all_patterns:
        sample_pattern = all_patterns[0]
        print(f"\nAnalyzing sample pattern: {getattr(sample_pattern, 'file_path', 'unknown')}")

        # Cluster
        clusters, labels = analyzer.cluster_patterns('kmeans', n_clusters=3)
        print(f"Clustering: {len(clusters)} clusters found.")

        # Genre classification
        genre, confidence = analyzer.classify_genre(sample_pattern)
        print(f"Genre: {genre} (confidence: {confidence:.2f})")

        # Similar patterns
        similars = analyzer.find_similar_patterns(sample_pattern, n=3)
        print(f"Similar patterns: {len(similars)} found.")

        # Quality assessment
        quality = analyzer.assess_quality(sample_pattern)
        print(f"Quality assessment: {quality['overall']:.2f}")
        if quality['feedback']:
            print("Feedback:", quality['feedback'])

        # Pattern completion
        completed = analyzer.complete_pattern(sample_pattern, num_notes=4)
        print(f"Pattern completion: Generated {len(completed)} new notes.")
    else:
        print("No patterns available for example analysis.")