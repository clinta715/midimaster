#!/usr/bin/env python3
"""
Reference Library Integration System

Purpose:
- Integrate with reference MIDI libraries for style matching
- Implement trend analysis across reference collections
- Provide style matching algorithms for genre classification
- Enable comparison between generated music and reference tracks

Features:
- Reference track database management
- Style similarity scoring
- Trend analysis across time periods
- Genre-specific reference matching
"""

import argparse
import json
import os
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import mido


@dataclass
class ReferenceTrack:
    """Represents a reference track in the library."""
    file_path: str
    genre: str
    tempo_bpm: float
    key_signature: str
    time_signature: str
    duration_seconds: float
    note_count: int
    average_velocity: float
    pitch_range: Tuple[int, int]
    complexity_score: float
    style_features: Dict[str, float]


@dataclass
class StyleMatch:
    """Result of style matching between query and reference."""
    reference_track: ReferenceTrack
    similarity_score: float
    matched_features: Dict[str, float]
    genre_consistency: float
    recommendations: List[str]


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    time_period: str
    genre_trends: Dict[str, float]
    tempo_trends: Dict[str, float]
    complexity_trends: Dict[str, float]
    key_trends: Dict[str, float]
    style_evolution: List[Dict[str, Any]]


class ReferenceLibrary:
    """Manages reference track library and analysis."""

    def __init__(self, library_path: str = "reference_midis"):
        self.library_path = library_path
        self.reference_tracks: List[ReferenceTrack] = []
        self._load_library()

    def _load_library(self):
        """Load reference tracks from library directory."""
        if not os.path.exists(self.library_path):
            print(f"Reference library path {self.library_path} does not exist")
            return

        for root, dirs, files in os.walk(self.library_path):
            for file in files:
                if file.endswith('.mid'):
                    file_path = os.path.join(root, file)
                    track = self._analyze_reference_track(file_path)
                    if track:
                        self.reference_tracks.append(track)

        print(f"Loaded {len(self.reference_tracks)} reference tracks")

    def _analyze_reference_track(self, file_path: str) -> Optional[ReferenceTrack]:
        """Analyze a single reference track."""
        try:
            midi_file = mido.MidiFile(file_path)

            # Extract basic information
            tempo = self._extract_tempo(midi_file)
            key_sig = self._detect_key_signature(midi_file)
            time_sig = self._detect_time_signature(midi_file)
            duration = self._calculate_duration(midi_file)
            notes = self._extract_notes(midi_file)

            if not notes:
                return None

            # Calculate statistics
            note_count = len(notes)
            avg_velocity = statistics.mean([n['velocity'] for n in notes])
            pitch_range = self._calculate_pitch_range(notes)
            complexity = self._calculate_complexity_score(notes, tempo)

            # Genre detection from path
            genre = self._detect_genre_from_path(file_path)

            # Extract style features
            style_features = self._extract_style_features(notes, tempo, midi_file)

            return ReferenceTrack(
                file_path=file_path,
                genre=genre,
                tempo_bpm=tempo,
                key_signature=key_sig,
                time_signature=time_sig,
                duration_seconds=duration,
                note_count=note_count,
                average_velocity=avg_velocity,
                pitch_range=pitch_range,
                complexity_score=complexity,
                style_features=style_features
            )

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def _extract_tempo(self, midi_file: mido.MidiFile) -> float:
        """Extract tempo from MIDI file."""
        for track in midi_file.tracks:
            for msg in track:
                if hasattr(msg, 'type') and msg.type == 'set_tempo':
                    return 60000000 / msg.tempo
        return 120.0

    def _detect_key_signature(self, midi_file: mido.MidiFile) -> str:
        """Detect key signature from MIDI content."""
        # Simple key detection - in practice this would be more sophisticated
        return "C major"

    def _detect_time_signature(self, midi_file: mido.MidiFile) -> str:
        """Detect time signature from MIDI content."""
        return "4/4"

    def _calculate_duration(self, midi_file: mido.MidiFile) -> float:
        """Calculate total duration in seconds."""
        total_ticks = 0
        tempo = self._extract_tempo(midi_file)

        for track in midi_file.tracks:
            track_ticks = 0
            for msg in track:
                track_ticks += msg.time
            total_ticks = max(total_ticks, track_ticks)

        # Convert ticks to seconds (approximate)
        return total_ticks / (midi_file.ticks_per_beat * tempo / 60)

    def _extract_notes(self, midi_file: mido.MidiFile) -> List[Dict]:
        """Extract notes from MIDI file."""
        notes = []
        active_notes = {}

        for track_idx, track in enumerate(midi_file.tracks):
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if hasattr(msg, 'type'):
                    if msg.type == 'note_on' and msg.velocity > 0:
                        key = (msg.note, msg.channel)
                        active_notes[key] = {
                            'pitch': msg.note,
                            'velocity': msg.velocity,
                            'start_time': abs_time,
                            'channel': msg.channel,
                            'track': track_idx
                        }
                    elif ((msg.type == 'note_off') or
                          (msg.type == 'note_on' and msg.velocity == 0)):
                        key = (msg.note, msg.channel)
                        if key in active_notes:
                            note = active_notes[key]
                            note['end_time'] = abs_time
                            note['duration'] = abs_time - note['start_time']
                            notes.append(note)
                            del active_notes[key]

        return notes

    def _calculate_pitch_range(self, notes: List[Dict]) -> Tuple[int, int]:
        """Calculate pitch range."""
        if not notes:
            return (60, 60)
        pitches = [n['pitch'] for n in notes]
        return (min(pitches), max(pitches))

    def _calculate_complexity_score(self, notes: List[Dict], tempo: float) -> float:
        """Calculate musical complexity score."""
        if not notes:
            return 0.0

        # Factors: note density, velocity variation, rhythm complexity
        density = len(notes) / 30.0  # Notes per 30 seconds
        velocity_variance = statistics.variance([n['velocity'] for n in notes]) if len(notes) > 1 else 0

        # Rhythm complexity based on note timing
        times = [n['start_time'] for n in notes]
        rhythm_complexity = 0.5
        if len(times) > 1:
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            if intervals:
                avg_interval = statistics.mean(intervals)
                variance = statistics.variance(intervals) if len(intervals) > 1 else 0
                rhythm_complexity = min(1.0, variance / (avg_interval * avg_interval))

        return (density * 0.4 + velocity_variance / 1000.0 * 0.3 + rhythm_complexity * 0.3)

    def _detect_genre_from_path(self, file_path: str) -> str:
        """Detect genre from file path."""
        path_lower = file_path.lower()

        genre_hints = {
            'rock': 'rock',
            'jazz': 'jazz',
            'pop': 'pop',
            'electronic': 'electronic',
            'hip-hop': 'hip-hop',
            'hiphop': 'hip-hop',
            'classical': 'classical',
            'dnb': 'electronic',
            'drum-and-bass': 'electronic',
            'techno': 'electronic',
            'house': 'electronic',
            'ambient': 'electronic'
        }

        for hint, genre in genre_hints.items():
            if hint in path_lower:
                return genre

        return 'unknown'

    def _extract_style_features(self, notes: List[Dict], tempo: float,
                               midi_file: mido.MidiFile) -> Dict[str, float]:
        """Extract style-specific features."""
        features = {}

        if not notes:
            return features

        # Syncopation level
        features['syncopation'] = self._calculate_syncopation(notes, tempo)

        # Swing factor
        features['swing'] = self._calculate_swing(notes, tempo)

        # Dynamic range
        velocities = [n['velocity'] for n in notes]
        features['dynamic_range'] = (max(velocities) - min(velocities)) / 127.0 if velocities else 0

        # Harmonic density
        features['harmonic_density'] = self._calculate_harmonic_density(notes)

        # Rhythmic complexity
        features['rhythmic_complexity'] = self._calculate_rhythmic_complexity(notes)

        return features

    def _calculate_syncopation(self, notes: List[Dict], tempo: float) -> float:
        """Calculate syncopation level."""
        return 0.5  # Placeholder

    def _calculate_swing(self, notes: List[Dict], tempo: float) -> float:
        """Calculate swing factor."""
        return 0.5  # Placeholder

    def _calculate_harmonic_density(self, notes: List[Dict]) -> float:
        """Calculate harmonic density."""
        if not notes:
            return 0.0

        # Group notes by time windows
        time_windows = defaultdict(list)
        for note in notes:
            window = note['start_time'] // 100
            time_windows[window].append(note)

        # Calculate average simultaneous notes
        avg_simultaneous = statistics.mean([len(window) for window in time_windows.values()])
        return min(1.0, avg_simultaneous / 4.0)

    def _calculate_rhythmic_complexity(self, notes: List[Dict]) -> float:
        """Calculate rhythmic complexity."""
        if len(notes) < 2:
            return 0.0

        # Based on interval variance
        times = [n['start_time'] for n in notes]
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        if intervals:
            variance = statistics.variance(intervals)
            avg_interval = statistics.mean(intervals)
            return min(1.0, variance / (avg_interval * avg_interval))
        return 0.0


class StyleMatcher:
    """Matches query tracks against reference library."""

    def __init__(self, reference_library: ReferenceLibrary):
        self.library = reference_library

    def find_style_matches(self, query_track_path: str, target_genre: Optional[str] = None,
                          top_n: int = 5) -> List[StyleMatch]:
        """Find best style matches for a query track."""
        try:
            # Analyze query track
            query_track = self.library._analyze_reference_track(query_track_path)
            if not query_track:
                return []

            matches = []

            # Compare against all reference tracks
            for ref_track in self.library.reference_tracks:
                if target_genre and ref_track.genre != target_genre:
                    continue

                similarity = self._calculate_similarity(query_track, ref_track)
                genre_consistency = 1.0 if query_track.genre == ref_track.genre else 0.5

                matched_features = self._identify_matched_features(query_track, ref_track)
                recommendations = self._generate_recommendations(query_track, ref_track)

                match = StyleMatch(
                    reference_track=ref_track,
                    similarity_score=similarity,
                    matched_features=matched_features,
                    genre_consistency=genre_consistency,
                    recommendations=recommendations
                )

                matches.append(match)

            # Sort by similarity and return top matches
            matches.sort(key=lambda m: m.similarity_score, reverse=True)
            return matches[:top_n]

        except Exception as e:
            print(f"Error finding style matches: {e}")
            return []

    def _calculate_similarity(self, query: ReferenceTrack, reference: ReferenceTrack) -> float:
        """Calculate similarity score between two tracks."""
        scores = []

        # Tempo similarity
        tempo_diff = abs(query.tempo_bpm - reference.tempo_bpm)
        tempo_score = max(0.0, 1.0 - tempo_diff / 30.0)
        scores.append(tempo_score)

        # Complexity similarity
        complexity_diff = abs(query.complexity_score - reference.complexity_score)
        complexity_score = max(0.0, 1.0 - complexity_diff)
        scores.append(complexity_score)

        # Pitch range similarity
        range_overlap = self._calculate_range_overlap(query.pitch_range, reference.pitch_range)
        scores.append(range_overlap)

        # Style feature similarity
        style_similarity = self._calculate_style_similarity(query.style_features, reference.style_features)
        scores.append(style_similarity)

        return statistics.mean(scores)

    def _calculate_range_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
        """Calculate overlap between two pitch ranges."""
        start1, end1 = range1
        start2, end2 = range2

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_size = overlap_end - overlap_start
        union_size = max(end1, end2) - min(start1, start2)

        return overlap_size / union_size if union_size > 0 else 0.0

    def _calculate_style_similarity(self, features1: Dict[str, float],
                                  features2: Dict[str, float]) -> float:
        """Calculate similarity between style feature sets."""
        if not features1 or not features2:
            return 0.5

        similarities = []
        for key in set(features1.keys()) | set(features2.keys()):
            val1 = features1.get(key, 0.5)
            val2 = features2.get(key, 0.5)
            diff = abs(val1 - val2)
            similarities.append(1.0 - diff)

        return statistics.mean(similarities) if similarities else 0.5

    def _identify_matched_features(self, query: ReferenceTrack,
                                  reference: ReferenceTrack) -> Dict[str, float]:
        """Identify which features match well between tracks."""
        matched = {}

        # Tempo match
        tempo_diff = abs(query.tempo_bpm - reference.tempo_bpm)
        matched['tempo'] = max(0.0, 1.0 - tempo_diff / 20.0)

        # Complexity match
        complexity_diff = abs(query.complexity_score - reference.complexity_score)
        matched['complexity'] = max(0.0, 1.0 - complexity_diff)

        # Style features
        for key in set(query.style_features.keys()) | set(reference.style_features.keys()):
            val1 = query.style_features.get(key, 0.5)
            val2 = reference.style_features.get(key, 0.5)
            diff = abs(val1 - val2)
            matched[f'style_{key}'] = 1.0 - diff

        return matched

    def _generate_recommendations(self, query: ReferenceTrack,
                                reference: ReferenceTrack) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []

        # Tempo recommendations
        tempo_diff = abs(query.tempo_bpm - reference.tempo_bpm)
        if tempo_diff > 10:
            if query.tempo_bpm < reference.tempo_bpm:
                recommendations.append("Consider increasing tempo for better genre fit")
            else:
                recommendations.append("Consider decreasing tempo for better genre fit")

        # Complexity recommendations
        complexity_diff = abs(query.complexity_score - reference.complexity_score)
        if complexity_diff > 0.2:
            if query.complexity_score < reference.complexity_score:
                recommendations.append("Add more rhythmic variation and dynamics")
            else:
                recommendations.append("Simplify rhythm and dynamics for better fit")

        return recommendations


class TrendAnalyzer:
    """Analyzes trends across reference library."""

    def __init__(self, reference_library: ReferenceLibrary):
        self.library = reference_library

    def analyze_genre_trends(self, time_periods: Optional[List[str]] = None) -> List[TrendAnalysis]:
        """Analyze trends in music characteristics over time."""
        if time_periods is None:
            time_periods = ["1980s", "1990s", "2000s", "2010s", "2020s"]

        trends = []

        for period in time_periods:
            period_tracks = [t for t in self.library.reference_tracks
                           if self._track_in_period(t, period)]

            if not period_tracks:
                continue

            genre_trends = self._analyze_genre_distribution(period_tracks)
            tempo_trends = self._analyze_tempo_trends(period_tracks)
            complexity_trends = self._analyze_complexity_trends(period_tracks)
            key_trends = self._analyze_key_trends(period_tracks)
            style_evolution = self._analyze_style_evolution(period_tracks)

            trend = TrendAnalysis(
                time_period=period,
                genre_trends=genre_trends,
                tempo_trends=tempo_trends,
                complexity_trends=complexity_trends,
                key_trends=key_trends,
                style_evolution=style_evolution
            )

            trends.append(trend)

        return trends
    def _track_in_period(self, track: ReferenceTrack, period: str) -> bool:
        """Check if track belongs to given time period."""
        # This would need actual metadata - for now, use filename hints
        period_hints = {
            "1980s": ["80", "eighties"],
            "1990s": ["90", "nineties"],
            "2000s": ["00", "2000"],
            "2010s": ["10", "2010"],
            "2020s": ["20", "2020"]
        }

        if period in period_hints:
            hints = period_hints[period]
            filename = track.file_path.lower()
            return any(hint in filename for hint in hints)

        return False

    def _analyze_genre_distribution(self, tracks: List[ReferenceTrack]) -> Dict[str, float]:
        """Analyze genre distribution in tracks."""
        genres = [t.genre for t in tracks]
        total = len(genres)

        distribution = {}
        for genre, count in Counter(genres).items():
            distribution[genre] = count / total

        return distribution

    def _analyze_tempo_trends(self, tracks: List[ReferenceTrack]) -> Dict[str, float]:
        """Analyze tempo trends."""
        tempos = [t.tempo_bpm for t in tracks]

        return {
            "average": statistics.mean(tempos) if tempos else 120.0,
            "median": statistics.median(tempos) if tempos else 120.0,
            "range_width": (max(tempos) - min(tempos)) if tempos else 0.0
        }

    def _analyze_complexity_trends(self, tracks: List[ReferenceTrack]) -> Dict[str, float]:
        """Analyze complexity trends."""
        complexities = [t.complexity_score for t in tracks]

        return {
            "average": statistics.mean(complexities) if complexities else 0.5,
            "median": statistics.median(complexities) if complexities else 0.5,
            "variance": statistics.variance(complexities) if len(complexities) > 1 else 0.0
        }

    def _analyze_key_trends(self, tracks: List[ReferenceTrack]) -> Dict[str, float]:
        """Analyze key distribution trends."""
        keys = [t.key_signature for t in tracks]
        total = len(keys)

        distribution = {}
        for key, count in Counter(keys).items():
            distribution[key] = count / total

        return distribution

    def _analyze_style_evolution(self, tracks: List[ReferenceTrack]) -> List[Dict]:
        """Analyze how style features evolved."""
        # Group by tempo ranges
        tempo_ranges = [(80, 100), (100, 120), (120, 140), (140, 180)]

        evolution = []
        for min_tempo, max_tempo in tempo_ranges:
            range_tracks = [t for t in tracks if min_tempo <= t.tempo_bpm < max_tempo]

            if range_tracks:
                avg_features = {}
                for track in range_tracks:
                    for key, value in track.style_features.items():
                        if key not in avg_features:
                            avg_features[key] = []
                        avg_features[key].append(value)

                # Calculate averages
                for key in avg_features:
                    avg_features[key] = statistics.mean(avg_features[key])

                evolution.append({
                    "tempo_range": f"{min_tempo}-{max_tempo} BPM",
                    "track_count": len(range_tracks),
                    "average_features": avg_features
                })

        return evolution


def main():
    parser = argparse.ArgumentParser(description="Reference library integration demo.")
    parser.add_argument("--action", choices=["match", "trends"], default="match",
                       help="Action to perform.")
    parser.add_argument("--input", help="Input MIDI file for style matching.")
    parser.add_argument("--genre", default="pop", help="Target genre.")
    parser.add_argument("--output", default="test_outputs",
                       help="Output directory for results.")

    args = parser.parse_args()

    # Initialize library
    library = ReferenceLibrary()
    matcher = StyleMatcher(library)
    trend_analyzer = TrendAnalyzer(library)

    if args.action == "match" and args.input:
        print(f"Finding style matches for {args.input}...")

        matches = matcher.find_style_matches(args.input, args.genre)

        for i, match in enumerate(matches[:3], 1):
            print(f"\nMatch {i}: {os.path.basename(match.reference_track.file_path)}")
            print(f"  Similarity: {match.similarity_score:.2f}")
            print(f"  Genre consistency: {match.genre_consistency:.2f}")
            print("  Top matched features:")
            for feature, score in list(match.matched_features.items())[:3]:
                print(f"    {feature}: {score:.2f}")
            if match.recommendations:
                print("  Recommendations:")
                for rec in match.recommendations:
                    print(f"    - {rec}")

        # Save results
        os.makedirs(args.output, exist_ok=True)
        results = {
            "query_file": args.input,
            "target_genre": args.genre,
            "matches": [
                {
                    "reference_file": match.reference_track.file_path,
                    "similarity_score": match.similarity_score,
                    "genre_consistency": match.genre_consistency,
                    "matched_features": match.matched_features,
                    "recommendations": match.recommendations
                } for match in matches
            ]
        }

        with open(os.path.join(args.output, "style_matches.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {args.output}/style_matches.json")

    elif args.action == "trends":
        print("Analyzing trends in reference library...")

        trends = trend_analyzer.analyze_genre_trends()

        for trend in trends:
            print(f"\n{trend.time_period}:")
            print("  Genre distribution:")
            for genre, percentage in trend.genre_trends.items():
                print(f"    {genre}: {percentage*100:.1f}%")
            print(f"  Average tempo: {trend.tempo_trends['average']:.1f} BPM")
            print(f"  Average complexity: {trend.complexity_trends['average']:.2f}")

        # Save trends
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "trend_analysis.json"), "w") as f:
            json.dump([
                {
                    "period": t.time_period,
                    "genre_trends": t.genre_trends,
                    "tempo_trends": t.tempo_trends,
                    "complexity_trends": t.complexity_trends,
                    "key_trends": t.key_trends,
                    "style_evolution": t.style_evolution
                } for t in trends
            ], f, indent=2)

        print(f"\nTrend analysis saved to {args.output}/trend_analysis.json")


if __name__ == "__main__":
    main()