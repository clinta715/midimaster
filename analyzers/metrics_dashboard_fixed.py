#!/usr/bin/env python3
"""
Advanced Metrics Dashboard with Genre Consistency Scoring

Purpose:
- Comprehensive metrics dashboard combining rhythm, harmony, melody, and production quality analysis
- Genre consistency scoring to evaluate how well generated music matches target genre characteristics
- Production quality metrics for dynamics, arrangement, and mixing evaluation

Outputs:
- test_outputs/metrics_dashboard_report.json
- test_outputs/metrics_dashboard_report.html
"""

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set

import mido


# ------------------------
# Data structures
# ------------------------

@dataclass
class GenreConsistencyScore:
    """Genre consistency evaluation for generated music."""
    overall_score: float  # 0.0 to 1.0
    rhythm_consistency: float
    harmony_consistency: float
    melody_consistency: float
    instrumentation_consistency: float
    tempo_consistency: float
    dynamics_consistency: float


@dataclass
class ProductionQualityMetrics:
    """Production quality evaluation metrics."""
    overall_quality_score: float  # 0.0 to 1.0
    velocity_range: float
    velocity_variance: float
    note_density_score: float
    polyphony_balance: float
    texture_variety: float
    section_contrast: float
    channel_balance: float
    frequency_distribution: float
    rhythmic_clarity: float
    mix_readiness_score: float


@dataclass
class ComprehensiveMetrics:
    """Complete analysis result combining all metrics."""
    file_path: str
    target_genre: str
    detected_genre: str
    genre_consistency: GenreConsistencyScore
    production_quality: ProductionQualityMetrics
    overall_score: float
    recommendations: List[str]
    strengths: List[str]
    weaknesses: List[str]


# ------------------------
# Genre Consistency Analyzer
# ------------------------


class GenreConsistencyAnalyzer:
    """Analyzes how well generated music matches target genre characteristics."""

    def __init__(self):
        self.genre_profiles = self._load_genre_profiles()

    def _load_genre_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive genre characteristic profiles."""
        return {
            "pop": {
                "tempo_range": (90, 140),
                "swing_factor": 0.55,
                "syncopation_level": 0.3,
                "scale_types": ["major", "minor"],
                "chord_progressions": ["I-V-vi-IV", "I-IV-V-I", "vi-IV-I-V"],
                "typical_structure": ["verse", "chorus", "verse", "chorus", "bridge", "chorus"],
                "instrumentation": ["piano", "guitar", "bass", "drums", "vocals"],
                "velocity_range": (60, 110),
                "polyphony_level": 0.6,
                "harmonic_complexity": 0.4
            },
            "rock": {
                "tempo_range": (100, 160),
                "swing_factor": 0.5,
                "syncopation_level": 0.2,
                "scale_types": ["major", "minor", "blues"],
                "chord_progressions": ["I-IV-V", "I-V-vi-IV", "I-bVII-IV"],
                "typical_structure": ["verse", "chorus", "verse", "chorus", "solo", "chorus"],
                "instrumentation": ["electric_guitar", "bass", "drums", "vocals"],
                "velocity_range": (70, 120),
                "polyphony_level": 0.7,
                "harmonic_complexity": 0.3
            },
            "jazz": {
                "tempo_range": (120, 200),
                "swing_factor": 0.66,
                "syncopation_level": 0.6,
                "scale_types": ["major", "minor", "dorian", "mixolydian"],
                "chord_progressions": ["ii-V-I", "I-vi-ii-V", "iii-vi-ii-V"],
                "typical_structure": ["head", "solo", "solo", "solo", "head"],
                "instrumentation": ["piano", "saxophone", "trumpet", "bass", "drums"],
                "velocity_range": (40, 100),
                "polyphony_level": 0.8,
                "harmonic_complexity": 0.8
            }
        }

    def analyze_genre_consistency(self, target_genre: str, midi_file: mido.MidiFile) -> GenreConsistencyScore:
        """Calculate comprehensive genre consistency score."""
        if target_genre not in self.genre_profiles:
            target_genre = "pop"

        profile = self.genre_profiles[target_genre]

        # Extract basic MIDI information
        tempo = self._extract_tempo(midi_file)
        notes = self._extract_notes(midi_file)

        # Calculate individual consistency scores
        tempo_score = self._analyze_tempo_consistency(profile, tempo)
        rhythm_score = self._analyze_rhythm_consistency(profile, midi_file, notes)
        harmony_score = self._analyze_harmony_consistency(profile, notes)
        melody_score = self._analyze_melody_consistency(profile, notes)
        dynamics_score = self._analyze_dynamics_consistency(profile, notes)

        # Calculate overall score
        weights = {
            'tempo': 0.2,
            'rhythm': 0.25,
            'harmony': 0.20,
            'melody': 0.15,
            'dynamics': 0.20
        }

        overall_score = (
            tempo_score * weights['tempo'] +
            rhythm_score * weights['rhythm'] +
            harmony_score * weights['harmony'] +
            melody_score * weights['melody'] +
            dynamics_score * weights['dynamics']
        )

        return GenreConsistencyScore(
            overall_score=overall_score,
            rhythm_consistency=rhythm_score,
            harmony_consistency=harmony_score,
            melody_consistency=melody_score,
            instrumentation_consistency=0.7,  # Placeholder
            tempo_consistency=tempo_score,
            dynamics_consistency=dynamics_score
        )

    def _extract_tempo(self, midi_file: mido.MidiFile) -> float:
        """Extract average tempo from MIDI file."""
        tempos = []
        for track in midi_file.tracks:
            for msg in track:
                if hasattr(msg, 'type') and msg.type == 'set_tempo':
                    tempo_uspb = msg.tempo
                    tempo_bpm = 60000000 / tempo_uspb
                    tempos.append(tempo_bpm)

        return statistics.mean(tempos) if tempos else 120.0

    def _extract_notes(self, midi_file: mido.MidiFile) -> List[Dict[str, Any]]:
        """Extract note information from MIDI file."""
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

    def _analyze_tempo_consistency(self, profile: Dict[str, Any], tempo: float) -> float:
        """Analyze tempo consistency against genre profile."""
        min_tempo, max_tempo = profile['tempo_range']

        if min_tempo <= tempo <= max_tempo:
            return 1.0
        else:
            distance = min(abs(tempo - min_tempo), abs(tempo - max_tempo))
            return max(0.0, 1.0 - distance / 20.0)

    def _analyze_rhythm_consistency(self, profile: Dict[str, Any],
                                   midi_file: mido.MidiFile, notes: List[Dict[str, Any]]) -> float:
        """Analyze rhythm consistency."""
        # Simple rhythm analysis based on note timing
        if not notes:
            return 0.5

        times = [note['start_time'] for note in notes]
        if len(times) < 2:
            return 0.5

        # Calculate inter-onset intervals
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        avg_interval = statistics.mean(intervals)
        variance = statistics.variance(intervals) if len(intervals) > 1 else 0

        # Score based on regularity
        regularity_score = max(0.0, 1.0 - variance / (avg_interval * 2))

        return regularity_score

    def _analyze_harmony_consistency(self, profile: Dict[str, Any], notes: List[Dict[str, Any]]) -> float:
        """Analyze harmony consistency."""
        # Basic harmony analysis based on pitch distribution
        if not notes:
            return 0.5

        pitches = [note['pitch'] for note in notes]
        pitch_classes = [pitch % 12 for pitch in pitches]

        # Count pitch class distribution
        pc_counts = Counter(pitch_classes)
        total_notes = len(pitch_classes)

        # Calculate distribution evenness
        if pc_counts:
            avg_count = total_notes / 12
            variance = sum((count - avg_count) ** 2 for count in pc_counts.values()) / 12
            evenness_score = max(0.0, 1.0 - variance / (avg_count * 4))
        else:
            evenness_score = 0.5

        return evenness_score

    def _analyze_melody_consistency(self, profile: Dict[str, Any], notes: List[Dict[str, Any]]) -> float:
        """Analyze melody consistency."""
        if not notes:
            return 0.5

        pitches = [note['pitch'] for note in notes]
        velocities = [note['velocity'] for note in notes]

        # Analyze pitch range
        if pitches:
            pitch_range = max(pitches) - min(pitches)
            range_score = min(pitch_range / 24.0, 1.0)  # 2 octaves ideal
        else:
            range_score = 0.5

        # Analyze velocity variation
        if len(velocities) > 1:
            velocity_variance = statistics.variance(velocities)
            velocity_score = min(velocity_variance / 100.0, 1.0)
        else:
            velocity_score = 0.5

        return (range_score + velocity_score) / 2.0

    def _analyze_dynamics_consistency(self, profile: Dict[str, Any], notes: List[Dict[str, Any]]) -> float:
        """Analyze dynamics consistency."""
        if not notes:
            return 0.5

        velocities = [note['velocity'] for note in notes]

        if velocities:
            min_vel, max_vel = min(velocities), max(velocities)
            profile_min, profile_max = profile['velocity_range']

            # Check if velocity range matches profile
            range_overlap = max(0, min(max_vel, profile_max) - max(min_vel, profile_min))
            expected_range = profile_max - profile_min
            overlap_score = range_overlap / expected_range if expected_range > 0 else 0.5

            return overlap_score
        else:
            return 0.5


# ------------------------
# Production Quality Analyzer
# ------------------------

class ProductionQualityAnalyzer:
    """Analyzes production quality aspects of generated music."""

    def analyze_production_quality(self, midi_file: mido.MidiFile,
                                  notes: List[Dict[str, Any]]) -> ProductionQualityMetrics:
        """Analyze production quality metrics."""

        # Dynamics analysis
        velocity_range, velocity_variance = self._analyze_dynamics(notes)
        note_density_score = self._analyze_note_density(notes)

        # Arrangement analysis
        polyphony_balance = self._analyze_polyphony(notes)
        texture_variety = self._analyze_texture_variety(notes)
        section_contrast = self._analyze_section_contrast(notes)

        # Mixing analysis
        channel_balance = self._analyze_channel_balance(notes)
        frequency_distribution = self._analyze_frequency_distribution(notes)
        rhythmic_clarity = self._analyze_rhythmic_clarity(midi_file, notes)

        # Calculate individual scores
        dynamics_score = (velocity_range + velocity_variance + note_density_score) / 3.0
        arrangement_score = (polyphony_balance + texture_variety + section_contrast) / 3.0
        mixing_score = (channel_balance + frequency_distribution + rhythmic_clarity) / 3.0

        overall_quality = (
            dynamics_score * 0.3 +
            arrangement_score * 0.4 +
            mixing_score * 0.3
        )

        mix_readiness = overall_quality * 0.8 + rhythmic_clarity * 0.2

        return ProductionQualityMetrics(
            overall_quality_score=overall_quality,
            velocity_range=velocity_range,
            velocity_variance=velocity_variance,
            note_density_score=note_density_score,
            polyphony_balance=polyphony_balance,
            texture_variety=texture_variety,
            section_contrast=section_contrast,
            channel_balance=channel_balance,
            frequency_distribution=frequency_distribution,
            rhythmic_clarity=rhythmic_clarity,
            mix_readiness_score=mix_readiness
        )

    def _analyze_dynamics(self, notes: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Analyze dynamics range and variance."""
        if not notes:
            return 0.7, 0.6

        velocities = [note['velocity'] for note in notes]

        if len(velocities) > 1:
            velocity_range = (max(velocities) - min(velocities)) / 127.0
            velocity_variance = statistics.variance(velocities) / 100.0
            return min(velocity_range, 1.0), min(velocity_variance, 1.0)
        else:
            return 0.5, 0.5

    def _analyze_note_density(self, notes: List[Dict[str, Any]]) -> float:
        """Analyze note density appropriateness."""
        if not notes:
            return 0.5

        total_duration = max((note['end_time'] for note in notes), default=1)
        note_count = len(notes)

        density = note_count / (total_duration / 1000)  # notes per second

        # Ideal density depends on genre, but generally 2-10 notes/sec is good
        if 2 <= density <= 10:
            return 1.0
        elif 1 <= density <= 15:
            return 0.8
        else:
            return max(0.3, 1.0 - abs(density - 6) / 10.0)

    def _analyze_polyphony(self, notes: List[Dict[str, Any]]) -> float:
        """Analyze polyphonic balance."""
        if not notes:
            return 0.5

        # Group notes by time windows
        time_windows = {}
        for note in notes:
            start_time = note['start_time']
            window = start_time // 100  # 100-tick windows
            if window not in time_windows:
                time_windows[window] = []
            time_windows[window].append(note)

        # Calculate average polyphony
        avg_polyphony = statistics.mean(len(window) for window in time_windows.values())

        # Score based on polyphony level (2-4 voices ideal)
        if 1.5 <= avg_polyphony <= 4.5:
            return 1.0
        elif 1 <= avg_polyphony <= 6:
            return 0.8
        else:
            return max(0.3, 1.0 - abs(avg_polyphony - 3) / 5.0)

    def _analyze_texture_variety(self, notes: List[Dict[str, Any]]) -> float:
        """Analyze texture variety."""
        if not notes:
            return 0.5

        # Simple texture analysis based on channel distribution
        channels = [note['channel'] for note in notes]
        channel_counts = Counter(channels)

        # More channels generally indicate better texture variety
        channel_count = len(channel_counts)
        return min(channel_count / 4.0, 1.0)

    def _analyze_section_contrast(self, notes: List[Dict[str, Any]]) -> float:
        """Analyze contrast between sections."""
        # Placeholder - would need more sophisticated section detection
        return 0.6

    def _analyze_channel_balance(self, notes: List[Dict[str, Any]]) -> float:
        """Analyze balance between different instrument channels."""
        if not notes:
            return 0.5

        channels = [note['channel'] for note in notes]
        channel_counts = Counter(channels)

        if channel_counts:
            # Check if any channel dominates too much
            total_notes = sum(channel_counts.values())
            max_channel_ratio = max(channel_counts.values()) / total_notes

            # Ideal: no channel has more than 50% of notes
            if max_channel_ratio <= 0.5:
                return 1.0
            else:
                return max(0.5, 1.0 - (max_channel_ratio - 0.5) * 2)

        return 0.5

    def _analyze_frequency_distribution(self, notes: List[Dict[str, Any]]) -> float:
        """Analyze frequency distribution of notes."""
        if not notes:
            return 0.5

        pitches = [note['pitch'] for note in notes]

        if pitches:
            pitch_range = max(pitches) - min(pitches)
            # Good frequency distribution covers reasonable range
            return min(pitch_range / 48.0, 1.0)  # 4 octaves ideal

        return 0.5

    def _analyze_rhythmic_clarity(self, midi_file: mido.MidiFile, notes: List[Dict[str, Any]]) -> float:
        """Analyze rhythmic clarity."""
        if not notes:
            return 0.5

        # Simple rhythmic clarity based on note timing consistency
        times = [note['start_time'] for note in notes]

        if len(times) > 1:
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            avg_interval = statistics.mean(intervals)

            if len(intervals) > 1:
                variance = statistics.variance(intervals)
                clarity_score = max(0.0, 1.0 - variance / (avg_interval * avg_interval))
                return clarity_score

        return 0.5


# ------------------------
# Main Metrics Dashboard Analyzer
# ------------------------

class MetricsDashboardAnalyzer:
    """Main analyzer that combines all metrics and provides comprehensive analysis."""

    def __init__(self):
        self.genre_analyzer = GenreConsistencyAnalyzer()
        self.quality_analyzer = ProductionQualityAnalyzer()

    def analyze_file(self, file_path: str, target_genre: str = "pop",
                    verbose: bool = False) -> Optional[ComprehensiveMetrics]:
        """
        Perform comprehensive analysis of a MIDI file.

        Args:
            file_path: Path to MIDI file to analyze
            target_genre: Target genre to evaluate against
            verbose: Whether to print verbose output

        Returns:
            ComprehensiveMetrics with all analysis results
        """
        if verbose:
            print(f"Analyzing {file_path}...")

        try:
            midi_file = mido.MidiFile(file_path)
            notes = self.genre_analyzer._extract_notes(midi_file)
        except Exception as e:
            if verbose:
                print(f"Error parsing {file_path}: {e}")
            return None

        # Analyze genre consistency
        genre_consistency = self.genre_analyzer.analyze_genre_consistency(
            target_genre, midi_file
        )

        # Analyze production quality
        production_quality = self.quality_analyzer.analyze_production_quality(
            midi_file, notes
        )

        # Calculate overall score
        overall_score = (
            genre_consistency.overall_score * 0.6 +
            production_quality.overall_quality_score * 0.4
        )

        # Generate recommendations based on scores
        recommendations = self._generate_recommendations(
            genre_consistency, production_quality, target_genre
        )

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(
            genre_consistency, production_quality
        )

        return ComprehensiveMetrics(
            file_path=file_path,
            target_genre=target_genre,
            detected_genre=self._detect_genre_from_midi(midi_file),
            genre_consistency=genre_consistency,
            production_quality=production_quality,
            overall_score=overall_score,
            recommendations=recommendations,
            strengths=strengths,
            weaknesses=weaknesses
        )

    def _detect_genre_from_midi(self, midi_file: mido.MidiFile) -> str:
        """Detect genre from MIDI file characteristics."""
        tempo = self.genre_analyzer._extract_tempo(midi_file)

        if 160 <= tempo <= 180:
            return "electronic"
        elif 100 <= tempo <= 140:
            return "pop"
        elif 120 <= tempo <= 160:
            return "rock"
        elif 110 <= tempo <= 180:
            return "jazz"
        else:
            return "classical"

    def _generate_recommendations(self, genre_consistency: GenreConsistencyScore,
                                production_quality: ProductionQualityMetrics,
                                target_genre: str) -> List[str]:
        """Generate specific recommendations based on analysis results."""
        recommendations = []

        if genre_consistency.tempo_consistency < 0.7:
            recommendations.append(f"Adjust tempo to better match {target_genre} characteristics")

        if genre_consistency.rhythm_consistency < 0.6:
            recommendations.append("Improve rhythmic consistency and timing")

        if genre_consistency.harmony_consistency < 0.6:
            recommendations.append("Enhance harmonic structure and chord progressions")

        if production_quality.velocity_range < 0.6:
            recommendations.append("Add more dynamic variation in note velocities")

        if production_quality.note_density_score < 0.5:
            recommendations.append("Adjust note density for better musical balance")

        return recommendations if recommendations else ["Analysis complete - no major issues found"]

    def _identify_strengths_weaknesses(self, genre_consistency: GenreConsistencyScore,
                                    production_quality: ProductionQualityMetrics) -> Tuple[List[str], List[str]]:
        """Identify key strengths and weaknesses."""
        strengths = []
        weaknesses = []

        if genre_consistency.overall_score > 0.8:
            strengths.append("Strong genre consistency")
        elif genre_consistency.overall_score < 0.5:
            weaknesses.append("Genre consistency needs improvement")

        if production_quality.overall_quality_score > 0.8:
            strengths.append("High production quality")
        elif production_quality.overall_quality_score < 0.5:
            weaknesses.append("Production quality needs attention")

        if production_quality.mix_readiness_score > 0.8:
            strengths.append("Ready for mixing/mastering")
        elif production_quality.mix_readiness_score < 0.5:
            weaknesses.append("Additional mixing preparation needed")

        return strengths, weaknesses


# ------------------------
# Reporting and CLI
# ------------------------

def ensure_out_dirs():
    os.makedirs("test_outputs", exist_ok=True)


def glob_inputs(inputs: List[str]) -> List[str]:
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            files.extend(glob.glob(os.path.join(inp, "*.mid")))
        else:
            matched = glob.glob(inp)
            if matched:
                files.extend(matched)
            else:
                files.append(inp)
    # Deduplicate
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    return unique_files


def write_json_report(results: List[ComprehensiveMetrics], path: str):
    """Write comprehensive JSON report."""
    data = {
        "summary": {
            "total_files": len(results),
            "average_overall_score": statistics.mean([r.overall_score for r in results]) if results else 0,
            "average_genre_consistency": statistics.mean([r.genre_consistency.overall_score for r in results]) if results else 0,
            "average_production_quality": statistics.mean([r.production_quality.overall_quality_score for r in results]) if results else 0,
        },
        "files": [asdict(result) for result in results]
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def write_html_report(results: List[ComprehensiveMetrics], path: str):
    """Write comprehensive HTML report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MIDI Master Metrics Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .score-high {{ color: #28a745; font-weight: bold; }}
        .score-medium {{ color: #ffc107; font-weight: bold; }}
        .score-low {{ color: #dc3545; font-weight: bold; }}
        .section {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>MIDI Master Advanced Metrics Dashboard</h1>

    <div class="section">
        <h2>Summary</h2>
        <p>Analyzed {len(results)} files</p>
        <p>Average Overall Score: <span class="score-{'high' if statistics.mean([r.overall_score for r in results]) > 0.7 else 'medium' if statistics.mean([r.overall_score for r in results]) > 0.5 else 'low'}">{statistics.mean([r.overall_score for r in results]):.2f}</span></p>
    </div>

    {"".join([f'''
    <div class="metric-card">
        <h3>{os.path.basename(result.file_path)}</h3>
        <p><strong>Target Genre:</strong> {result.target_genre}</p>
        <p><strong>Detected Genre:</strong> {result.detected_genre}</p>
        <p><strong>Overall Score:</strong> <span class="score-{'high' if result.overall_score > 0.7 else 'medium' if result.overall_score > 0.5 else 'low'}">{result.overall_score:.2f}</span></p>

        <h4>Genre Consistency</h4>
        <ul>
            <li>Rhythm: {result.genre_consistency.rhythm_consistency:.2f}</li>
            <li>Harmony: {result.genre_consistency.harmony_consistency:.2f}</li>
            <li>Melody: {result.genre_consistency.melody_consistency:.2f}</li>
            <li>Tempo: {result.genre_consistency.tempo_consistency:.2f}</li>
            <li>Dynamics: {result.genre_consistency.dynamics_consistency:.2f}</li>
        </ul>

        <h4>Production Quality</h4>
        <ul>
            <li>Velocity Range: {result.production_quality.velocity_range:.2f}</li>
            <li>Note Density: {result.production_quality.note_density_score:.2f}</li>
            <li>Polyphony Balance: {result.production_quality.polyphony_balance:.2f}</li>
            <li>Mix Readiness: {result.production_quality.mix_readiness_score:.2f}</li>
        </ul>

        {"<h4>Recommendations</h4><ul>" + "".join([f"<li>{rec}</li>" for rec in result.recommendations]) + "</ul>" if result.recommendations else ""}

        {"<h4>Strengths</h4><ul>" + "".join([f"<li>{strength}</li>" for strength in result.strengths]) + "</ul>" if result.strengths else ""}

        {"<h4>Weaknesses</h4><ul>" + "".join([f"<li>{weakness}</li>" for weakness in result.weaknesses]) + "</ul>" if result.weaknesses else ""}
    </div>
    ''' for result in results])}
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Advanced metrics dashboard for MIDI analysis.")
    parser.add_argument("--inputs", nargs="+", required=True,
                       help="Input MIDI paths, directories, or glob patterns.")
    parser.add_argument("--genre", default="pop",
                       help="Target genre for consistency analysis (default: pop).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ensure_out_dirs()

    paths = glob_inputs(args.inputs)
    if args.verbose:
        print(f"Discovered {len(paths)} input(s).")

    analyzer = MetricsDashboardAnalyzer()
    results = []

    for path in paths:
        result = analyzer.analyze_file(path, target_genre=args.genre, verbose=args.verbose)
        if result:
            results.append(result)

    # Write reports
    json_path = os.path.join("test_outputs", "metrics_dashboard_report.json")
    html_path = os.path.join("test_outputs", "metrics_dashboard_report.html")

    write_json_report(results, json_path)
    write_html_report(results, html_path)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {html_path}")

    if results:
        avg_score = statistics.mean([r.overall_score for r in results])
        print(f"Average score: {avg_score:.2f}")

if __name__ == "__main__":
    main()