#!/usr/bin/env python3
"""
Advanced Metrics Dashboard with Genre Consistency Scoring

Purpose:
- Comprehensive metrics dashboard combining rhythm, harmony, melody, and production quality analysis
- Genre consistency scoring to evaluate how well generated music matches target genre characteristics
- Production quality metrics for dynamics, arrangement, and mixing evaluation
- Real-time analysis capabilities for live feedback during music generation

Outputs:
- test_outputs/metrics_dashboard_report.json
- test_outputs/metrics_dashboard_report.html
- test_outputs/genre_consistency_scores.json

Usage:
  python analyzers/metrics_dashboard.py --inputs path1.mid path2.mid dir_or_glob ... [--genre target_genre] [--verbose]
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

# Import existing analyzers
# Import individual analysis functions instead of classes
# Beat audit functionality will be called directly
from beat_audit import analyze_file as beat_analyze_file
# from beat_audit import analyze_file as beat_analyze_file
from structure_insights import StructureInsightsAnalyzer
from music_theory import MusicTheory


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

    # Detailed breakdowns
    beat_strength_score: float
    grid_alignment_score: float
    scale_adherence_score: float
    chord_progression_score: float
    key_stability_score: float
    motif_development_score: float


@dataclass
class ProductionQualityMetrics:
    """Production quality evaluation metrics."""
    overall_quality_score: float  # 0.0 to 1.0

    # Dynamics and expression
    velocity_range: float
    velocity_variance: float
    note_density_score: float

    # Arrangement quality
    polyphony_balance: float
    texture_variety: float
    section_contrast: float

    # Mixing quality
    channel_balance: float
    frequency_distribution: float
    rhythmic_clarity: float

    # Overall production readiness
    mix_readiness_score: float


@dataclass
class ComprehensiveMetrics:
    """Complete analysis result combining all metrics."""
    file_path: str
    target_genre: str
    detected_genre: str

    # Genre consistency
    genre_consistency: GenreConsistencyScore

    # Production quality
    production_quality: ProductionQualityMetrics

    # Individual analyzer results
    beat_audit_data: Dict[str, Any]
    structure_insights_data: Dict[str, Any]

    # Overall assessment
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
        # Genre characteristic definitions
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
            },
            "electronic": {
                "tempo_range": (120, 140),
                "swing_factor": 0.5,
                "syncopation_level": 0.4,
                "scale_types": ["minor", "major", "phrygian"],
                "chord_progressions": ["i-VI-III-VII", "i-iv-VII-VI", "i-VI-IV-V"],
                "typical_structure": ["build", "drop", "break", "drop", "outro"],
                "instrumentation": ["synthesizer", "drums", "bass", "effects"],
                "velocity_range": (50, 127),
                "polyphony_level": 0.5,
                "harmonic_complexity": 0.6
            },
            "hip-hop": {
                "tempo_range": (80, 110),
                "swing_factor": 0.6,
                "syncopation_level": 0.7,
                "scale_types": ["minor", "major", "blues"],
                "chord_progressions": ["i-VII-VI-V", "i-iv-VII-VI", "I-IV-V-I"],
                "typical_structure": ["verse", "hook", "verse", "hook", "bridge", "hook"],
                "instrumentation": ["drums", "bass", "vocals", "samples"],
                "velocity_range": (60, 110),
                "polyphony_level": 0.4,
                "harmonic_complexity": 0.3
            },
            "classical": {
                "tempo_range": (60, 160),
                "swing_factor": 0.5,
                "syncopation_level": 0.1,
                "scale_types": ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian"],
                "chord_progressions": ["I-IV-V-I", "i-VII-VI-V", "I-V-vi-IV"],
                "typical_structure": ["exposition", "development", "recapitulation"],
                "instrumentation": ["orchestra", "piano", "strings", "woodwinds"],
                "velocity_range": (30, 100),
                "polyphony_level": 0.9,
                "harmonic_complexity": 0.7
            },
            "dnb": {
                "tempo_range": (160, 180),
                "swing_factor": 0.5,
                "syncopation_level": 0.8,
                "scale_types": ["minor", "major"],
                "chord_progressions": ["i-VII-VI-V", "i-iv-VII-VI"],
                "typical_structure": ["intro", "build", "drop", "break", "drop", "outro"],
                "instrumentation": ["drums", "bass", "synth", "effects"],
                "velocity_range": (80, 127),
                "polyphony_level": 0.6,
                "harmonic_complexity": 0.5
            }
        }

    def analyze_genre_consistency(self, target_genre: str, beat_data: Dict[str, Any],
                                  structure_data: Dict[str, Any]) -> GenreConsistencyScore:
        """
        Calculate comprehensive genre consistency score.

        Args:
            target_genre: Target genre to evaluate against
            beat_data: Results from beat audit analysis
            structure_data: Results from structure insights analysis

        Returns:
            GenreConsistencyScore with detailed breakdown
        """
        if target_genre not in self.genre_profiles:
            # Default to pop if genre not found
            target_genre = "pop"

        profile = self.genre_profiles[target_genre]

        # Rhythm consistency analysis
        rhythm_score = self._analyze_rhythm_consistency(profile, beat_data, target_genre)

        # Harmony consistency analysis
        harmony_score = self._analyze_harmony_consistency(profile, structure_data)

        # Melody consistency analysis
        melody_score = self._analyze_melody_consistency(profile, structure_data, target_genre)

        # Instrumentation consistency analysis
        instrumentation_score = self._analyze_instrumentation_consistency(profile, beat_data, structure_data)

        # Tempo consistency analysis
        tempo_score = self._analyze_tempo_consistency(profile, beat_data)

        # Dynamics consistency analysis
        dynamics_score = self._analyze_dynamics_consistency(profile, beat_data, structure_data)

        # Calculate overall score as weighted average
        weights = {
            'rhythm': 0.25,
            'harmony': 0.20,
            'melody': 0.15,
            'instrumentation': 0.15,
            'tempo': 0.15,
            'dynamics': 0.10
        }

        overall_score = (
            rhythm_score * weights['rhythm'] +
            harmony_score * weights['harmony'] +
            melody_score * weights['melody'] +
            instrumentation_score * weights['instrumentation'] +
            tempo_score * weights['tempo'] +
            dynamics_score * weights['dynamics']
        )

        # Extract detailed scores from individual analyses
        beat_strength = self._extract_beat_strength_score(beat_data)
        grid_alignment = self._extract_grid_alignment_score(beat_data)
        scale_adherence = self._extract_scale_adherence_score(structure_data)
        chord_progression = self._extract_chord_progression_score(structure_data, profile)
        key_stability = self._extract_key_stability_score(structure_data)
        motif_development = self._extract_motif_development_score(structure_data)

        return GenreConsistencyScore(
            overall_score=overall_score,
            rhythm_consistency=rhythm_score,
            harmony_consistency=harmony_score,
            melody_consistency=melody_score,
            instrumentation_consistency=instrumentation_score,
            tempo_consistency=tempo_score,
            dynamics_consistency=dynamics_score,
            beat_strength_score=beat_strength,
            grid_alignment_score=grid_alignment,
            scale_adherence_score=scale_adherence,
            chord_progression_score=chord_progression,
            key_stability_score=key_stability,
            motif_development_score=motif_development
        )

    def _analyze_rhythm_consistency(self, profile: Dict[str, Any], beat_data: Dict[str, Any], target_genre: str) -> float:
        """Analyze rhythm consistency against genre profile."""
        score = 0.0
        factors = 0

        # Tempo analysis
        if 'tempo_bpm_avg' in beat_data:
            tempo = beat_data['tempo_bpm_avg']
            min_tempo, max_tempo = profile['tempo_range']
            if min_tempo <= tempo <= max_tempo:
                score += 1.0
            else:
                # Partial credit for being close
                distance = min(abs(tempo - min_tempo), abs(tempo - max_tempo))
                score += max(0.0, 1.0 - distance / 50.0)
            factors += 1

        # Grid alignment
        if 'grid_alignment_percent' in beat_data:
            alignment = beat_data['grid_alignment_percent']
            if target_genre == "dnb":
                # DnB needs tighter alignment
                score += min(alignment / 90.0, 1.0)
            else:
                score += min(alignment / 80.0, 1.0)
            factors += 1

        # Cross-part alignment
        if 'cross_alignment_percent' in beat_data:
            cross_align = beat_data['cross_alignment_percent']
            score += min(cross_align / 85.0, 1.0)
            factors += 1

        return score / max(factors, 1)

    def _analyze_harmony_consistency(self, profile: Dict[str, Any], structure_data: Dict[str, Any]) -> float:
        """Analyze harmony consistency against genre profile."""
        score = 0.0
        factors = 0

        # Scale type analysis
        if 'key_detections' in structure_data and structure_data['key_detections']:
            detected_key = structure_data['key_detections'][0]
            detected_scale = detected_key.get('scale_type', '').lower()

            if detected_scale in [s.lower() for s in profile['scale_types']]:
                score += 1.0
            else:
                score += 0.5  # Partial credit for common scales
            factors += 1

        # Harmonic complexity
        if 'harmonic_complexity_score' in structure_data:
            complexity = structure_data['harmonic_complexity_score']
            expected_complexity = profile['harmonic_complexity']

            # Score based on how close complexity is to expected
            distance = abs(complexity - expected_complexity)
            score += max(0.0, 1.0 - distance)
            factors += 1

        return score / max(factors, 1)

    def _analyze_melody_consistency(self, profile: Dict[str, Any], structure_data: Dict[str, Any], target_genre: str) -> float:
        """Analyze melody consistency against genre profile."""
        score = 0.0
        factors = 0

        # Pitch range analysis
        if 'pitch_range' in structure_data:
            min_pitch, max_pitch = structure_data['pitch_range']
            range_size = max_pitch - min_pitch

            # Different genres have different expected ranges
            if target_genre in ['classical', 'jazz']:
                expected_range = 36  # 3 octaves
            elif target_genre in ['rock', 'pop']:
                expected_range = 24  # 2 octaves
            else:
                expected_range = 24

            if abs(range_size - expected_range) <= 12:  # Within 1 octave
                score += 1.0
            else:
                score += max(0.0, 1.0 - abs(range_size - expected_range) / 24.0)
            factors += 1

        # Motif development
        if 'motifs' in structure_data:
            motif_count = len(structure_data['motifs'])
            # More motifs generally indicate better development
            motif_score = min(motif_count / 5.0, 1.0)
            score += motif_score
            factors += 1

        return score / max(factors, 1)

    def _analyze_instrumentation_consistency(self, profile: Dict[str, Any],
                                           beat_data: Dict[str, Any],
                                           structure_data: Dict[str, Any]) -> float:
        """Analyze instrumentation consistency against genre profile."""
        # For now, return a neutral score - would need MIDI program analysis
        return 0.7

    def _analyze_tempo_consistency(self, profile: Dict[str, Any], beat_data: Dict[str, Any]) -> float:
        """Analyze tempo consistency against genre profile."""
        if 'tempo_bpm_avg' not in beat_data:
            return 0.5

        tempo = beat_data['tempo_bpm_avg']
        min_tempo, max_tempo = profile['tempo_range']

        if min_tempo <= tempo <= max_tempo:
            return 1.0
        else:
            # Calculate how far outside the range
            if tempo < min_tempo:
                distance = min_tempo - tempo
            else:
                distance = tempo - max_tempo

            # Return score based on distance (20 BPM tolerance)
            return max(0.0, 1.0 - distance / 20.0)

    def _analyze_dynamics_consistency(self, profile: Dict[str, Any],
                                     beat_data: Dict[str, Any],
                                     structure_data: Dict[str, Any]) -> float:
        """Analyze dynamics consistency against genre profile."""
        # For now, return a neutral score - would need velocity analysis
        return 0.7

    # Helper methods for extracting detailed scores
    def _extract_beat_strength_score(self, beat_data: Dict[str, Any]) -> float:
        """Extract beat strength score from beat data."""
        if 'rhythm_note_count' in beat_data and 'rhythm_channels' in beat_data:
            note_count = beat_data['rhythm_note_count']
            channel_count = len(beat_data['rhythm_channels'])
            # Higher note count and more channels generally indicate stronger beat
            return min((note_count / 100.0 + channel_count / 4.0) / 2.0, 1.0)
        return 0.5

    def _extract_grid_alignment_score(self, beat_data: Dict[str, Any]) -> float:
        """Extract grid alignment score from beat data."""
        if 'grid_alignment_percent' in beat_data:
            return beat_data['grid_alignment_percent'] / 100.0
        return 0.5

    def _extract_scale_adherence_score(self, structure_data: Dict[str, Any]) -> float:
        """Extract scale adherence score from structure data."""
        if 'key_detections' in structure_data and structure_data['key_detections']:
            # Use confidence of top key detection
            confidence = structure_data['key_detections'][0].get('confidence', 0.5)
            return confidence
        return 0.5

    def _extract_chord_progression_score(self, structure_data: Dict[str, Any],
                                        profile: Dict[str, Any]) -> float:
        """Extract chord progression score."""
        if 'detected_chords' in structure_data:
            chord_count = len(structure_data['detected_chords'])
            # More chords generally indicate better harmonic development
            return min(chord_count / 20.0, 1.0)
        return 0.5

    def _extract_key_stability_score(self, structure_data: Dict[str, Any]) -> float:
        """Extract key stability score."""
        if 'key_detections' in structure_data:
            key_count = len(structure_data['key_detections'])
            # Fewer keys indicate better stability
            if key_count == 1:
                return 1.0
            elif key_count == 2:
                return 0.8
            else:
                return max(0.3, 1.0 - (key_count - 2) * 0.2)
        return 0.5

    def _extract_motif_development_score(self, structure_data: Dict[str, Any]) -> float:
        """Extract motif development score."""
        if 'motifs' in structure_data:
            motif_count = len(structure_data['motifs'])
            return min(motif_count / 8.0, 1.0)  # Expect 8 motifs for full score
        return 0.3


# ------------------------
# Production Quality Analyzer
# ------------------------

class ProductionQualityAnalyzer:
    """Analyzes production quality aspects of generated music."""

    def analyze_production_quality(self, beat_data: Dict[str, Any],
                                  structure_data: Dict[str, Any]) -> ProductionQualityMetrics:
        """
        Analyze production quality metrics.

        Args:
            beat_data: Results from beat audit analysis
            structure_data: Results from structure insights analysis

        Returns:
            ProductionQualityMetrics with detailed breakdown
        """

        # Dynamics analysis
        velocity_range, velocity_variance = self._analyze_dynamics(beat_data, structure_data)
        note_density_score = self._analyze_note_density(beat_data, structure_data)

        # Arrangement analysis
        polyphony_balance = self._analyze_polyphony(structure_data)
        texture_variety = self._analyze_texture_variety(structure_data)
        section_contrast = self._analyze_section_contrast(structure_data)

        # Mixing analysis
        channel_balance = self._analyze_channel_balance(beat_data, structure_data)
        frequency_distribution = self._analyze_frequency_distribution(structure_data)
        rhythmic_clarity = self._analyze_rhythmic_clarity(beat_data)

        # Calculate individual scores
        dynamics_score = (velocity_range + velocity_variance + note_density_score) / 3.0
        arrangement_score = (polyphony_balance + texture_variety + section_contrast) / 3.0
        mixing_score = (channel_balance + frequency_distribution + rhythmic_clarity) / 3.0

        # Overall quality score as weighted average
        overall_quality = (
            dynamics_score * 0.3 +
            arrangement_score * 0.4 +
            mixing_score * 0.3
        )

        # Mix readiness is a composite of all factors
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

    def _analyze_dynamics(self, beat_data: Dict[str, Any],
                         structure_data: Dict[str, Any]) -> Tuple[float, float]:
        """Analyze dynamics range and variance."""
        # Placeholder - would need actual velocity data from MIDI analysis
        return 0.7, 0.6

    def _analyze_note_density(self, beat_data: Dict[str, Any],
                             structure_data: Dict[str, Any]) -> float:
        """Analyze note density appropriateness."""
        if 'rhythm_note_count' in beat_data:
            count = beat_data['rhythm_note_count']
            # Ideal density varies by genre and tempo
            # This is a simplified scoring
            if 50 <= count <= 200:
                return 1.0
            elif 20 <= count <= 300:
                return 0.8
            else:
                return 0.4
        return 0.5

    def _analyze_polyphony(self, structure_data: Dict[str, Any]) -> float:
        """Analyze polyphonic balance."""
        # Placeholder - would analyze simultaneous note counts
        return 0.7

    def _analyze_texture_variety(self, structure_data: Dict[str, Any]) -> float:
        """Analyze texture variety throughout the piece."""
        if 'sections' in structure_data:
            section_count = len(structure_data['sections'])
            # More sections generally indicate better variety
            return min(section_count / 6.0, 1.0)
        return 0.5

    def _analyze_section_contrast(self, structure_data: Dict[str, Any]) -> float:
        """Analyze contrast between sections."""
        # Placeholder - would compare characteristics across sections
        return 0.6

    def _analyze_channel_balance(self, beat_data: Dict[str, Any],
                                structure_data: Dict[str, Any]) -> float:
        """Analyze balance between different instrument channels."""
        # Placeholder - would analyze relative volumes and presence
        return 0.7

    def _analyze_frequency_distribution(self, structure_data: Dict[str, Any]) -> float:
        """Analyze frequency distribution of notes."""
        if 'pitch_range' in structure_data:
            min_pitch, max_pitch = structure_data['pitch_range']
            range_size = max_pitch - min_pitch
            # Good frequency distribution covers reasonable range
            return min(range_size / 48.0, 1.0)  # 4 octaves ideal
        return 0.5

    def _analyze_rhythmic_clarity(self, beat_data: Dict[str, Any]) -> float:
        """Analyze rhythmic clarity and definition."""
        if 'grid_alignment_percent' in beat_data:
            alignment = beat_data['grid_alignment_percent']
            return alignment / 100.0
        return 0.5


# ------------------------
# Main Metrics Dashboard Analyzer
# ------------------------

class MetricsDashboardAnalyzer:
    """Main analyzer that combines all metrics and provides comprehensive analysis."""

    def __init__(self):
        
        self.structure_analyzer = StructureInsightsAnalyzer()
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

        # Run individual analyzers
        diversity_signature_sets: Dict[str, Set[str]] = {}
        beat_result = beat_analyze_file(file_path, window_bars=2, diversity_signature_sets=diversity_signature_sets, verbose=verbose)
        structure_result = self.structure_analyzer.analyze_file(file_path, verbose=verbose)

        if not beat_result or not structure_result:
            if verbose:
                print(f"Failed to analyze {file_path}")
            return None

        # Convert results to dictionaries for processing
        beat_data = asdict(beat_result)
        structure_data = asdict(structure_result)

        # Analyze genre consistency
        genre_consistency = self.genre_analyzer.analyze_genre_consistency(
            target_genre, beat_data, structure_data
        )

        # Analyze production quality
        production_quality = self.quality_analyzer.analyze_production_quality(
            beat_data, structure_data
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
            detected_genre=self._detect_genre_from_data(beat_data),
            genre_consistency=genre_consistency,
            production_quality=production_quality,
            beat_audit_data=beat_data,
            structure_insights_data=structure_data,
            overall_score=overall_score,
            recommendations=recommendations,
            strengths=strengths,
            weaknesses=weaknesses
        )

    def _detect_genre_from_data(self, beat_data: Dict[str, Any]) -> str:
        """Detect genre from beat data characteristics."""
        tempo = beat_data.get('tempo_bpm_avg', 120)

        if 160 <= tempo <= 180:
            return "dnb"
        elif 100 <= tempo <= 140:
            return "electronic"
        elif 80 <= tempo <= 100:
            return "hip-hop"
        elif 90 <= tempo <= 130:
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

        # Genre consistency recommendations
        if genre_consistency.tempo_consistency < 0.7:
            recommendations.append(f"Adjust tempo to better match {target_genre} characteristics")

        if genre_consistency.grid_alignment_score < 0.8:
            recommendations.append("Improve rhythmic quantization and grid alignment")

        if genre_consistency.scale_adherence_score < 0.6:
            recommendations.append("Ensure melody follows appropriate scale for the genre")

        # Production quality recommendations
        if production_quality.velocity_range < 0.6:
            recommendations.append("Add more dynamic variation in note velocities")

        if production_quality.note_density_score < 0.5:
            recommendations.append("Adjust note density for better musical balance")

        if production_quality.polyphony_balance < 0.7:
            recommendations.append("Improve balance between melodic and harmonic elements")

        return recommendations if recommendations else ["Analysis complete - no major issues found"]

    def _identify_strengths_weaknesses(self, genre_consistency: GenreConsistencyScore,
                                    production_quality: ProductionQualityMetrics) -> Tuple[List[str], List[str]]:
        """Identify key strengths and weaknesses."""
        strengths = []
        weaknesses = []

        # Genre consistency strengths/weaknesses
        if genre_consistency.rhythm_consistency > 0.8:
            strengths.append("Strong rhythmic consistency")
        elif genre_consistency.rhythm_consistency < 0.5:
            weaknesses.append("Rhythmic consistency needs improvement")

        if genre_consistency.harmony_consistency > 0.8:
            strengths.append("Good harmonic structure")
        elif genre_consistency.harmony_consistency < 0.5:
            weaknesses.append("Harmonic structure could be improved")

        if genre_consistency.melody_consistency > 0.8:
            strengths.append("Strong melodic development")
        elif genre_consistency.melody_consistency < 0.5:
            weaknesses.append("Melodic development needs work")

        # Production quality strengths/weaknesses
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
        .chart {{ background: #f8f9fa; padding: 10px; border-radius: 3px; }}
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
        </ul>

        <h4>Production Quality</h4>
        <ul>
            <li>Dynamics: {(result.production_quality.velocity_range + result.production_quality.velocity_variance + result.production_quality.note_density_score) / 3:.2f}</li>
            <li>Arrangement: {(result.production_quality.polyphony_balance + result.production_quality.texture_variety + result.production_quality.section_contrast) / 3:.2f}</li>
            <li>Mixing: {(result.production_quality.channel_balance + result.production_quality.frequency_distribution + result.production_quality.rhythmic_clarity) / 3:.2f}</li>
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

    # Summary
    if results:
        avg_score = statistics.mean([r.overall_score for r in results])
        print(f"\nAnalysis Summary:")
        print(f"  Files analyzed: {len(results)}")
        print(f"  Average score: {avg_score:.2f}")
    else:
        print("No files analyzed.")
if __name__ == "__main__":
    main()