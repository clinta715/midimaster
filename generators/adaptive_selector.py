import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from analyzers.reference_pattern_library import ReferencePatternLibrary, MidiPatternData, PatternMetadata


class AdaptivePatternSelector:
    """
    Adaptive pattern selection system that intelligently chooses patterns based on context.

    This class integrates with the ReferencePatternLibrary to select, score, and recommend
    patterns based on mood, genre, complexity, BPM range, and instrument type. It supports
    both exact matching and fuzzy matching using similarity metrics.

    Key Features:
    - Mood mapping: Correlates pattern characteristics with emotional qualities.
    - Genre-specific clustering: Uses pattern similarity for genre-aware selection.
    - BPM-aware selection: Filters and scores patterns within a tempo range.
    - Pattern scoring: Ranks patterns based on multiple weighted criteria.
    - Best matching retrieval: Returns top patterns for given parameters.
    - Pattern blending: Recommends complementary patterns for blending.

    Usage Example:
        library = ReferencePatternLibrary()
        library.load_from_directory("reference_midis")
        selector = AdaptivePatternSelector(library)
        patterns = selector.get_best_patterns(
            mood="energetic",
            genre="drum-and-bass",
            complexity=0.5,
            bpm_min=140,
            bpm_max=160,
            instrument="drums",
            n=3
        )
        for pat, score in patterns:
            print(f"Pattern score: {score}")
    """

    def __init__(self, pattern_library: ReferencePatternLibrary):
        """
        Initialize the AdaptivePatternSelector.

        Args:
            pattern_library: The ReferencePatternLibrary instance to use for pattern storage and retrieval.
        """
        self.library = pattern_library
        self.mood_mappings = self._initialize_mood_mappings()
        self.genre_clusters = self._precompute_genre_clusters()

    def _initialize_mood_mappings(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize mood-to-feature mappings for emotional correlation.

        Mappings define preferred characteristics for each mood:
        - energetic: High density, fast notes, high velocity.
        - calm: Low density, sustained notes, low velocity.
        - happy: Major contours, moderate density, upbeat rhythms.
        - sad: Minor contours, slow notes, low velocity.

        Returns:
            Dict mapping moods to feature preferences (density, avg_velocity, note_speed, contour_bias).
        """
        return {
            "energetic": {
                "density": 0.8,      # High note density
                "avg_velocity": 0.9, # High dynamics
                "note_speed": 0.7,   # Faster note durations (shorter)
                "contour_bias": 0.5  # Neutral to ascending
            },
            "calm": {
                "density": 0.3,      # Low density
                "avg_velocity": 0.4, # Soft dynamics
                "note_speed": 0.3,   # Sustained notes (longer)
                "contour_bias": 0.2  # Slight descending for relaxation
            },
            "happy": {
                "density": 0.6,      # Moderate density
                "avg_velocity": 0.7, # Bright dynamics
                "note_speed": 0.5,   # Balanced rhythm
                "contour_bias": 0.8  # Ascending/major bias
            },
            "sad": {
                "density": 0.4,      # Sparse
                "avg_velocity": 0.3, # Gentle dynamics
                "note_speed": 0.4,   # Slower rhythm
                "contour_bias": 0.1  # Descending/minor bias
            }
        }

    def _precompute_genre_clusters(self) -> Dict[str, List[MidiPatternData]]:
        """
        Precompute genre-specific pattern clusters using similarity metrics.

        Groups patterns within each genre into clusters based on feature similarity.
        Uses k-means like grouping (simplified: top similar patterns per genre).

        Returns:
            Dict of genre to list of clustered patterns (all patterns per genre for simplicity).
        """
        clusters = {}
        for (genre, instr), patterns in self.library.patterns.items():
            if genre not in clusters:
                clusters[genre] = []
            # Simple clustering: all patterns for now; could implement k-means on features
            for pattern in patterns:
                clusters[genre].append(pattern)
        return clusters

    def _extract_pattern_features(self, pattern: MidiPatternData) -> Dict[str, float]:
        """
        Extract key features from a pattern for scoring and similarity.

        Features include density, average velocity, note speed (inverse duration),
        contour (ascending bias), complexity, bpm.

        Args:
            pattern: The MidiPatternData to extract features from.

        Returns:
            Dict of extracted features.
        """
        if not pattern.notes:
            return {"density": 0, "avg_velocity": 0, "note_speed": 0, "contour_bias": 0.5, "complexity": 0, "bpm": 120}

        notes = pattern.notes
        durations = [n.duration for n in notes]
        velocities = [n.velocity for n in notes]
        pitches = [n.note for n in notes]

        length_beats = pattern.length_ticks / pattern.ticks_per_beat
        density = len(notes) / length_beats if length_beats > 0 else 0
        avg_velocity = float(np.mean(velocities) / 127) if velocities else 0.0
        avg_duration = float(np.mean(durations) / pattern.ticks_per_beat) if durations else 1.0
        note_speed = float(1 / (avg_duration + 0.01))  # Inverse duration, normalized roughly

        # Contour bias: fraction of ascending intervals
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)] if len(pitches) > 1 else [0]
        contour_bias = sum(1 for i in intervals if i > 0) / len(intervals) if intervals else 0.5

        meta = self.library.metadata.get(pattern.file_path, PatternMetadata(120, (4,4), 0, "", ""))
        complexity = meta.complexity
        bpm = meta.bpm

        return {
            "density": min(density / 4, 1.0),  # Normalize to 0-1 (assume max 4 notes/beat)
            "avg_velocity": avg_velocity,
            "note_speed": min(note_speed, 1.0),
            "contour_bias": contour_bias,
            "complexity": min(complexity, 1.0),
            "bpm": bpm
        }

    def _mood_mapping_score(self, features: Dict[str, float], mood: str) -> float:
        """
        Compute mood matching score using mood mappings.

        Args:
            features: Extracted pattern features.
            mood: The target mood.

        Returns:
            Score (0-1) indicating how well the pattern matches the mood.
        """
        if mood not in self.mood_mappings:
            return 0.5

        mapping = self.mood_mappings[mood]
        score = 0
        total_weight = 0

        for key, target in mapping.items():
            actual = features.get(key, 0.5)
            # Simple absolute difference, normalized
            diff = abs(actual - target)
            match_score = 1 - diff
            weight = 1.0  # Equal weights
            score += match_score * weight
            total_weight += weight

        return score / total_weight if total_weight > 0 else 0

    def _genre_match_score(self, pattern: MidiPatternData, target_genre: str) -> float:
        """
        Compute genre matching score.

        For exact match: 1.0
        For similar genres (placeholder): partial score.

        Args:
            pattern: The pattern.
            target_genre: Target genre.

        Returns:
            Genre match score (0-1).
        """
        meta = self.library.metadata.get(pattern.file_path, PatternMetadata(120, (4,4), 0, "", ""))
        if meta.genre == target_genre:
            return 1.0
        # Placeholder for genre similarity (e.g., electronic ~ drum-and-bass)
        genre_similarities = {
            "drum-and-bass": ["electronic"],
            "electronic": ["drum-and-bass", "pop"],
            # Add more as needed
        }
        if target_genre in genre_similarities and meta.genre in genre_similarities[target_genre]:
            return 0.7
        return 0.3

    def _bpm_match_score(self, bpm: float, bpm_min: float, bpm_max: float) -> float:
        """
        Compute BPM matching score within range.

        Args:
            bpm: Pattern BPM.
            bpm_min: Minimum BPM.
            bpm_max: Maximum BPM.

        Returns:
            Score (0-1) based on proximity to range.
        """
        if bpm_min <= bpm <= bpm_max:
            return 1.0
        elif bpm < bpm_min:
            return max(0, 1 - (bpm_min - bpm) / bpm_min)
        else:
            return max(0, 1 - (bpm - bpm_max) / bpm_max)

    def _complexity_match_score(self, complexity: float, target_complexity: float) -> float:
        """
        Compute complexity matching score.

        Args:
            complexity: Pattern complexity.
            target_complexity: Target complexity level (0-1).

        Returns:
            Score (0-1) based on closeness.
        """
        diff = abs(complexity - target_complexity)
        return max(0, 1 - diff)

    def _instrument_match_score(self, pattern: MidiPatternData, target_instrument: str) -> float:
        """
        Compute instrument type matching score.

        Args:
            pattern: The pattern.
            target_instrument: Target instrument type.

        Returns:
            Exact match: 1.0, else 0.5 (partial compatibility).
        """
        meta = self.library.metadata.get(pattern.file_path, PatternMetadata(120, (4,4), 0, "", ""))
        if meta.instrument_type == target_instrument:
            return 1.0
        # Placeholder for instrument compatibility (e.g., synth ~ melody)
        return 0.5

    def score_pattern(
        self,
        pattern: MidiPatternData,
        mood: str,
        genre: str,
        complexity: float,
        bpm_range: Tuple[float, float],
        instrument: str
    ) -> float:
        """
        Compute a comprehensive score for a pattern based on all criteria.

        Weights: mood=0.3, genre=0.2, bpm=0.2, complexity=0.15, instrument=0.15

        Args:
            pattern: The pattern to score.
            mood: Target mood.
            genre: Target genre.
            complexity: Target complexity (0-1).
            bpm_range: (min_bpm, max_bpm) tuple.
            instrument: Target instrument type.

        Returns:
            Overall score (0-1).
        """
        features = self._extract_pattern_features(pattern)
        scores = {
            "mood": self._mood_mapping_score(features, mood),
            "genre": self._genre_match_score(pattern, genre),
            "bpm": self._bpm_match_score(features["bpm"], bpm_range[0], bpm_range[1]),
            "complexity": self._complexity_match_score(features["complexity"], complexity),
            "instrument": self._instrument_match_score(pattern, instrument)
        }

        weights = {
            "mood": 0.3,
            "genre": 0.2,
            "bpm": 0.2,
            "complexity": 0.15,
            "instrument": 0.15
        }

        total_score = sum(scores[k] * weights[k] for k in scores)
        return total_score

    def get_best_patterns(
        self,
        mood: str,
        genre: str,
        complexity: float,
        bpm_min: float,
        bpm_max: float,
        instrument: str,
        n: int = 5,
        threshold: float = 0.5,
        fuzzy: bool = True
    ) -> List[Tuple[MidiPatternData, float]]:
        """
        Retrieve the best matching patterns for given parameters.

        If fuzzy=True, uses similarity for broader matching; else exact filters.

        Args:
            mood: Target mood.
            genre: Target genre.
            complexity: Target complexity (0-1).
            bpm_min: Minimum BPM.
            bpm_max: Maximum BPM.
            instrument: Target instrument type.
            n: Number of top patterns to return.
            threshold: Minimum score threshold.
            fuzzy: Use fuzzy matching (similarity-based).

        Returns:
            List of (pattern, score) tuples, sorted by score descending.
        """
        bpm_range = (bpm_min, bpm_max)
        candidates = []

        # Get candidate patterns
        if fuzzy:
            # Broader search: all patterns in genre or similar
            all_patterns = self.library.get_patterns(instrument=instrument, genre=genre)
            # Add similar genres if needed, but for now use get_patterns
        else:
            # Exact filter
            all_patterns = self.library.get_patterns(instrument=instrument, genre=genre)

        for pattern in all_patterns:
            score = self.score_pattern(pattern, mood, genre, complexity, bpm_range, instrument)
            if score >= threshold:
                candidates.append((pattern, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def get_exact_matches(
        self,
        genre: str,
        instrument: str,
        bpm_range: Tuple[float, float],
        complexity_range: Tuple[float, float]
    ) -> List[MidiPatternData]:
        """
        Retrieve exact matching patterns without scoring.

        Filters strictly on genre, instrument, BPM, and complexity range.

        Args:
            genre: Exact genre.
            instrument: Exact instrument type.
            bpm_range: (min, max) BPM.
            complexity_range: (min, max) complexity.

        Returns:
            List of matching patterns.
        """
        def exact_filter(pat: MidiPatternData, meta: PatternMetadata) -> bool:
            bpm_match = bpm_range[0] <= meta.bpm <= bpm_range[1]
            comp_match = complexity_range[0] <= meta.complexity <= complexity_range[1]
            return bpm_match and comp_match

        return self.library.get_patterns(
            instrument=instrument,
            genre=genre,
            min_complexity=complexity_range[0],
            max_complexity=complexity_range[1],
            filter_func=exact_filter
        )

    def recommend_blends(
        self,
        base_pattern: MidiPatternData,
        mood: str,
        genre: str,
        instrument: str,
        max_recommendations: int = 3,
        complementarity_threshold: float = 0.6
    ) -> List[Tuple[MidiPatternData, float]]:
        """
        Recommend patterns for blending with a base pattern.

        Finds complementary patterns: similar in structure (high similarity) but
        different in details (e.g., variation in contour or density) for blending.

        Args:
            base_pattern: Base pattern to blend with.
            mood: Target mood for recommendations.
            genre: Target genre.
            instrument: Target instrument (same as base for cohesion).
            max_recommendations: Max number of recommendations.
            complementarity_threshold: Min similarity for blend suitability.

        Returns:
            List of (recommended_pattern, blend_score) tuples.
        """
        base_features = self._extract_pattern_features(base_pattern)
        base_meta = self.library.metadata.get(base_pattern.file_path, PatternMetadata(120, (4,4), 0, genre, instrument))
        base_genre = base_meta.genre
        base_instrument = base_meta.instrument_type

        recommendations = []
        candidates = self.library.get_patterns(instrument=base_instrument, genre=base_genre)

        for candidate in candidates:
            if candidate == base_pattern:
                continue

            sim = self.library.compute_similarity(base_pattern, candidate)
            if sim < complementarity_threshold:
                continue

            # Blend score: similarity * mood match * variation (diff in some features)
            cand_features = self._extract_pattern_features(candidate)
            variation = abs(cand_features["contour_bias"] - base_features["contour_bias"]) + \
                        abs(cand_features["density"] - base_features["density"])
            variation = min(variation / 2, 1.0)  # Normalize 0-1

            mood_score = self._mood_mapping_score(cand_features, mood)
            blend_score = sim * 0.4 + mood_score * 0.3 + variation * 0.3

            recommendations.append((candidate, blend_score))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:max_recommendations]

    def get_pattern_clusters(self, genre: str, instrument: str, cluster_size: int = 5) -> Dict[str, List[MidiPatternData]]:
        """
        Retrieve genre-specific pattern clusters.

        Args:
            genre: Genre to cluster.
            instrument: Instrument type.
            cluster_size: Approximate size per cluster (simplified grouping).

        Returns:
            Dict of cluster_id to list of patterns.
        """
        if genre in self.genre_clusters:
            patterns = [p for p in self.genre_clusters[genre] if self.library.get_metadata(p).instrument_type == instrument]
            # Simplified: one big cluster; could implement actual clustering
            return {"primary_cluster": patterns[:cluster_size * 2]}  # Dummy split
        return {}


# Example usage (for testing)
if __name__ == "__main__":
    # Assuming library is loaded
    library = ReferencePatternLibrary()
    # library.load_from_directory("reference_midis")  # Uncomment to load
    selector = AdaptivePatternSelector(library)

    # Example: Get best energetic drum patterns for drum-and-bass
    # best = selector.get_best_patterns("energetic", "drum-and-bass", 0.6, 140, 160, "drums", n=2)
    # print(f"Best patterns: {[score for _, score in best]}")

    # Example blending
    # if library.patterns:
    #     base = next(iter(library.patterns.values()))[0]
    #     blends = selector.recommend_blends(base, "energetic", "drum-and-bass", "drums")
    #     print(f"Blend scores: {[score for _, score in blends]}")
    pass