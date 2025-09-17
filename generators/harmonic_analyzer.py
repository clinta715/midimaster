"""
Harmonic Analyzer for comprehensive harmonic analysis and chord progression features.

This module provides advanced harmonic analysis capabilities including:
- Chord progression analysis and validation
- Harmonic tension and resolution tracking
- Cadential analysis (authentic, plagal, deceptive)
- Functional harmony analysis
- Harmonic rhythm analysis
- Key center detection and modulation tracking
- Chord voicing optimization
- Harmonic complexity metrics
"""

import math
from typing import List, Dict, Tuple, Optional, Set, Union, NamedTuple
from enum import Enum
from collections import defaultdict, Counter
from dataclasses import dataclass

from music_theory import MusicTheory, Note, ScaleType


class HarmonicFunction(Enum):
    """Harmonic functions in tonal music."""
    TONIC = "I"           # Tonic function (I, iii, vi)
    DOMINANT = "V"        # Dominant function (V, vii°)
    SUBDOMINANT = "IV"    # Subdominant function (IV, ii, vi)
    TONIC_SUBSTITUTE = "III"  # Tonic substitute (iii, vi)
    DOMINANT_SUBSTITUTE = "VII"  # Dominant substitute (vii°)
    SUBDOMINANT_SUBSTITUTE = "II"  # Subdominant substitute (ii)


class CadenceType(Enum):
    """Types of musical cadences."""
    AUTHENTIC = "authentic"      # V-I or V7-I (perfect or imperfect)
    PLAGAL = "plagal"           # IV-I or ii-I
    DECEPTIVE = "deceptive"      # V-vi or V-IV
    HALF = "half"               # I-V or iv-V (no resolution)
    PHRYGIAN = "phrygian"       # iv-V-I
    IMPERFECT = "imperfect"     # I-V with inverted chords


class TensionLevel(Enum):
    """Levels of harmonic tension."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ChordAnalysis:
    """Analysis result for a single chord."""
    chord_name: str
    root: int
    pitches: List[int]
    function: HarmonicFunction
    tension_level: TensionLevel
    stability_score: float  # 0-1, higher = more stable
    dissonance_score: float  # 0-1, higher = more dissonant
    roman_numeral: str
    scale_degree: int


@dataclass
class ProgressionAnalysis:
    """Analysis result for a chord progression."""
    chords: List[ChordAnalysis]
    key_center: str
    overall_tension_curve: List[float]
    cadences: List[Tuple[int, CadenceType]]  # (chord_index, cadence_type)
    functional_balance: Dict[HarmonicFunction, float]  # Percentage of each function
    harmonic_rhythm: List[float]  # Durations of each chord
    modulation_points: List[int]  # Indices where modulation occurs
    quality_score: float  # 0-1 overall quality metric


class HarmonicAnalyzer:
    """Main harmonic analysis engine."""

    def __init__(self):
        """Initialize the harmonic analyzer."""
        self.music_theory = MusicTheory()
        self.key_context = "C major"  # Default key

    def analyze_chord_progression(self, chord_names: List[str],
                                key_context: Optional[str] = None) -> ProgressionAnalysis:
        """
        Analyze a complete chord progression.

        Args:
            chord_names: List of chord names (e.g., ['C', 'Dm', 'G', 'C'])
            key_context: Key context (e.g., 'C major'), auto-detected if None

        Returns:
            Comprehensive progression analysis
        """
        if key_context:
            self.key_context = key_context

        # Analyze individual chords
        chord_analyses = []
        for chord_name in chord_names:
            analysis = self.analyze_single_chord(chord_name)
            chord_analyses.append(analysis)

        # Detect key if not provided
        if not key_context:
            self.key_context = self._detect_key_center(chord_analyses)

        # Analyze progression-level features
        tension_curve = self._calculate_tension_curve(chord_analyses)
        cadences = self._identify_cadences(chord_analyses)
        functional_balance = self._analyze_functional_balance(chord_analyses)
        modulations = self._detect_modulations(chord_analyses)

        # Calculate harmonic rhythm (assuming equal duration for simplicity)
        harmonic_rhythm = [1.0] * len(chord_analyses)

        quality_score = self._calculate_progression_quality(
            chord_analyses, tension_curve, cadences, functional_balance
        )

        return ProgressionAnalysis(
            chords=chord_analyses,
            key_center=self.key_context,
            overall_tension_curve=tension_curve,
            cadences=cadences,
            functional_balance=functional_balance,
            harmonic_rhythm=harmonic_rhythm,
            modulation_points=modulations,
            quality_score=quality_score
        )

    def analyze_single_chord(self, chord_name: str) -> ChordAnalysis:
        """
        Analyze a single chord within the current key context.

        Args:
            chord_name: Chord name (e.g., 'C', 'Dm7', 'G7')

        Returns:
            Detailed chord analysis
        """
        try:
            # Parse chord to get pitches
            pitches = self._chord_name_to_pitches(chord_name)
            if not pitches:
                return self._create_empty_analysis(chord_name)

            root = pitches[0] % 12
            roman_numeral, scale_degree = self._identify_roman_numeral(root)
            function = self._determine_harmonic_function(scale_degree, chord_name)
            tension_level = self._assess_tension_level(pitches, chord_name)
            stability = self._calculate_stability_score(scale_degree, function)
            dissonance = self._calculate_dissonance_score(pitches)

            return ChordAnalysis(
                chord_name=chord_name,
                root=root,
                pitches=pitches,
                function=function,
                tension_level=tension_level,
                stability_score=stability,
                dissonance_score=dissonance,
                roman_numeral=roman_numeral,
                scale_degree=scale_degree
            )

        except Exception as e:
            print(f"Error analyzing chord {chord_name}: {e}")
            return self._create_empty_analysis(chord_name)

    def find_optimal_voicing(self, chord_name: str, register: str = "mid") -> List[int]:
        """
        Find optimal voicing for a chord.

        Args:
            chord_name: Chord name
            register: Preferred register ('low', 'mid', 'high')

        Returns:
            Optimally voiced chord pitches
        """
        try:
            base_pitches = self._chord_name_to_pitches(chord_name)
            if not base_pitches:
                return []

            # Apply voicing optimizations based on register
            if register == "low":
                return self._voice_for_bass_register(base_pitches)
            elif register == "mid":
                return self._voice_for_mid_register(base_pitches)
            elif register == "high":
                return self._voice_for_high_register(base_pitches)
            else:
                return base_pitches

        except Exception as e:
            print(f"Error optimizing voicing for {chord_name}: {e}")
            return self._chord_name_to_pitches(chord_name) or []

    def analyze_harmonic_rhythm(self, chord_durations: List[float]) -> Dict[str, float]:
        """
        Analyze the rhythmic aspect of harmony.

        Args:
            chord_durations: Duration of each chord in beats

        Returns:
            Analysis metrics for harmonic rhythm
        """
        if not chord_durations:
            return {}

        analysis = {
            'average_duration': sum(chord_durations) / len(chord_durations),
            'duration_variety': self._calculate_variety(chord_durations),
            'harmonic_density': len(chord_durations) / sum(chord_durations),
            'rhythmic_stability': self._calculate_rhythmic_stability(chord_durations)
        }

        return analysis

    def suggest_progression_improvements(self, progression: ProgressionAnalysis) -> List[str]:
        """
        Suggest improvements for a chord progression.

        Args:
            progression: Analyzed progression

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Check functional balance
        tonic_pct = progression.functional_balance.get(HarmonicFunction.TONIC, 0)
        dominant_pct = progression.functional_balance.get(HarmonicFunction.DOMINANT, 0)

        if tonic_pct < 0.3:
            suggestions.append("Add more tonic function chords for better resolution")
        if dominant_pct < 0.15:
            suggestions.append("Include more dominant function chords for stronger tension-resolution")

        # Check for cadences
        if not progression.cadences:
            suggestions.append("Consider adding cadences for better structural definition")

        # Check tension curve
        if len(progression.overall_tension_curve) > 2:
            tension_variance = self._calculate_variety(progression.overall_tension_curve)
            if tension_variance < 0.1:
                suggestions.append("Add more harmonic variety to the tension curve")

        # Check quality score
        if progression.quality_score < 0.6:
            suggestions.append("Consider revising chord choices for better harmonic flow")

        return suggestions

    def _chord_name_to_pitches(self, chord_name: str) -> Optional[List[int]]:
        """Convert chord name to MIDI pitches (simplified implementation)."""
        if not chord_name:
            return None

        # Basic chord parsing (can be enhanced)
        chord_name = chord_name.strip()

        # Handle simple major/minor chords
        if chord_name.endswith('m'):
            root_name = chord_name[:-1]
            root = self._note_name_to_midi(root_name)
            return [root, root + 3, root + 7] if root is not None else None
        else:
            root = self._note_name_to_midi(chord_name)
            return [root, root + 4, root + 7] if root is not None else None

    def _note_name_to_midi(self, note_name: str) -> Optional[int]:
        """Convert note name to MIDI pitch (simplified)."""
        note_map = {
            'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
            'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68,
            'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
        }
        return note_map.get(note_name.upper())

    def _identify_roman_numeral(self, root: int) -> Tuple[str, int]:
        """Identify Roman numeral and scale degree for a root pitch."""
        # Simplified - assumes C major context
        scale_degrees = [0, 2, 4, 5, 7, 9, 11]  # C major scale degrees
        roman_numerals = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']

        for i, degree in enumerate(scale_degrees):
            if root % 12 == degree:
                return roman_numerals[i], i + 1

        return 'I', 1  # Default fallback

    def _determine_harmonic_function(self, scale_degree: int, chord_name: str) -> HarmonicFunction:
        """Determine the harmonic function of a chord."""
        if scale_degree in [1, 3, 6]:  # I, iii, vi
            return HarmonicFunction.TONIC
        elif scale_degree in [5, 7]:  # V, vii°
            return HarmonicFunction.DOMINANT
        elif scale_degree in [2, 4]:  # ii, IV
            return HarmonicFunction.SUBDOMINANT
        else:
            return HarmonicFunction.TONIC  # Default

    def _assess_tension_level(self, pitches: List[int], chord_name: str) -> TensionLevel:
        """Assess the tension level of a chord."""
        num_notes = len(pitches)

        if num_notes <= 3:
            return TensionLevel.LOW
        elif num_notes == 4:
            return TensionLevel.MEDIUM
        elif chord_name and ('7' in chord_name or '9' in chord_name or '11' in chord_name):
            return TensionLevel.HIGH
        else:
            return TensionLevel.EXTREME

    def _calculate_stability_score(self, scale_degree: int, function: HarmonicFunction) -> float:
        """Calculate stability score for a chord."""
        base_scores = {
            HarmonicFunction.TONIC: 1.0,
            HarmonicFunction.DOMINANT: 0.7,
            HarmonicFunction.SUBDOMINANT: 0.8,
            HarmonicFunction.TONIC_SUBSTITUTE: 0.9,
            HarmonicFunction.DOMINANT_SUBSTITUTE: 0.6,
            HarmonicFunction.SUBDOMINANT_SUBSTITUTE: 0.7
        }
        return base_scores.get(function, 0.5)

    def _calculate_dissonance_score(self, pitches: List[int]) -> float:
        """Calculate dissonance score for chord pitches."""
        if len(pitches) < 2:
            return 0.0

        dissonance = 0.0
        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                interval = abs(pitches[i] - pitches[j]) % 12
                # Minor seconds and tritones are most dissonant
                if interval in [1, 6, 11]:
                    dissonance += 1.0
                elif interval in [2, 10]:
                    dissonance += 0.5

        return min(dissonance / len(pitches), 1.0)

    def _detect_key_center(self, chord_analyses: List[ChordAnalysis]) -> str:
        """Detect the most likely key center from chord analysis."""
        if not chord_analyses:
            return "C major"

        # Simple heuristic: most common root with major/minor bias
        roots = [analysis.root for analysis in chord_analyses]
        most_common_root = Counter(roots).most_common(1)[0][0]

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        root_name = note_names[most_common_root % 12]

        # Simple major/minor detection based on chord functions
        dominant_count = sum(1 for ca in chord_analyses if ca.function == HarmonicFunction.DOMINANT)
        if dominant_count > len(chord_analyses) * 0.3:
            return f"{root_name} major"
        else:
            return f"{root_name} minor"

    def _calculate_tension_curve(self, chord_analyses: List[ChordAnalysis]) -> List[float]:
        """Calculate the harmonic tension curve for a progression."""
        tension_curve = []

        for analysis in chord_analyses:
            # Combine dissonance and tension level
            base_tension = (analysis.dissonance_score + analysis.stability_score) / 2

            # Adjust based on tension level
            level_multiplier = {
                TensionLevel.LOW: 0.3,
                TensionLevel.MEDIUM: 0.6,
                TensionLevel.HIGH: 0.8,
                TensionLevel.EXTREME: 1.0
            }

            tension = base_tension * level_multiplier.get(analysis.tension_level, 0.5)
            tension_curve.append(min(max(tension, 0.0), 1.0))

        return tension_curve

    def _identify_cadences(self, chord_analyses: List[ChordAnalysis]) -> List[Tuple[int, CadenceType]]:
        """Identify cadences in the progression."""
        cadences = []

        for i in range(len(chord_analyses) - 1):
            current = chord_analyses[i]
            next_chord = chord_analyses[i + 1]

            # Check for common cadences
            if (current.function == HarmonicFunction.DOMINANT and
                next_chord.function == HarmonicFunction.TONIC):
                cadences.append((i, CadenceType.AUTHENTIC))

            elif (current.function == HarmonicFunction.SUBDOMINANT and
                  next_chord.function == HarmonicFunction.TONIC):
                cadences.append((i, CadenceType.PLAGAL))

            elif (current.function == HarmonicFunction.DOMINANT and
                  next_chord.scale_degree == 6):  # vi after V
                cadences.append((i, CadenceType.DECEPTIVE))

        return cadences

    def _analyze_functional_balance(self, chord_analyses: List[ChordAnalysis]) -> Dict[HarmonicFunction, float]:
        """Analyze the functional balance of the progression."""
        if not chord_analyses:
            return {}

        function_counts = Counter(ca.function for ca in chord_analyses)
        total_chords = len(chord_analyses)

        return {func: count / total_chords for func, count in function_counts.items()}

    def _detect_modulations(self, chord_analyses: List[ChordAnalysis]) -> List[int]:
        """Detect points where modulation occurs."""
        modulations = []

        for i in range(1, len(chord_analyses)):
            prev_key = chord_analyses[i-1].roman_numeral
            curr_key = chord_analyses[i].roman_numeral

            # Simple modulation detection: significant change in scale degrees
            if abs(chord_analyses[i].scale_degree - chord_analyses[i-1].scale_degree) > 3:
                modulations.append(i)

        return modulations

    def _calculate_progression_quality(self, chord_analyses: List[ChordAnalysis],
                                     tension_curve: List[float], cadences: List[Tuple[int, CadenceType]],
                                     functional_balance: Dict[HarmonicFunction, float]) -> float:
        """Calculate overall quality score for the progression."""
        if not chord_analyses:
            return 0.0

        score = 0.5  # Base score

        # Reward functional balance
        tonic_balance = functional_balance.get(HarmonicFunction.TONIC, 0)
        dominant_balance = functional_balance.get(HarmonicFunction.DOMINANT, 0)
        if 0.2 <= tonic_balance <= 0.5 and dominant_balance >= 0.1:
            score += 0.2

        # Reward cadences
        if cadences:
            score += 0.2

        # Reward tension variety
        if len(tension_curve) > 1:
            tension_variance = self._calculate_variety(tension_curve)
            score += tension_variance * 0.2

        # Penalize excessive dissonance
        avg_dissonance = sum(ca.dissonance_score for ca in chord_analyses) / len(chord_analyses)
        if avg_dissonance > 0.7:
            score -= 0.2

        return max(0.0, min(score, 1.0))

    def _calculate_variety(self, values: List[float]) -> float:
        """Calculate variety metric for a list of values."""
        if len(values) < 2:
            return 0.0

        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return min(math.sqrt(variance), 1.0)  # Normalize

    def _calculate_rhythmic_stability(self, durations: List[float]) -> float:
        """Calculate rhythmic stability of chord durations."""
        if len(durations) < 2:
            return 1.0

        # Measure how consistent durations are
        mean_duration = sum(durations) / len(durations)
        consistency = sum(abs(d - mean_duration) for d in durations) / len(durations)
        return max(0.0, 1.0 - consistency / mean_duration)

    def _create_empty_analysis(self, chord_name: str) -> ChordAnalysis:
        """Create empty analysis for failed chord parsing."""
        return ChordAnalysis(
            chord_name=chord_name,
            root=0,
            pitches=[],
            function=HarmonicFunction.TONIC,
            tension_level=TensionLevel.LOW,
            stability_score=0.5,
            dissonance_score=0.0,
            roman_numeral="I",
            scale_degree=1
        )

    def _voice_for_bass_register(self, pitches: List[int]) -> List[int]:
        """Voice chord for bass register (lowest possible)."""
        if not pitches:
            return pitches

        # Invert to put root in bass if needed
        root = pitches[0]
        # Simple voicing: root, fifth, third (or octave equivalents)
        return [root, (root + 7) % 12 + 48, (root + 4) % 12 + 52]

    def _voice_for_mid_register(self, pitches: List[int]) -> List[int]:
        """Voice chord for mid register."""
        if not pitches:
            return pitches

        root = pitches[0]
        # Mid-range voicing
        return [(root + 12) % 12 + 48, (root + 4 + 12) % 12 + 52, (root + 7 + 12) % 12 + 55]

    def _voice_for_high_register(self, pitches: List[int]) -> List[int]:
        """Voice chord for high register."""
        if not pitches:
            return pitches

        root = pitches[0]
        # High-range voicing
        return [(root + 24) % 12 + 60, (root + 4 + 24) % 12 + 64, (root + 7 + 24) % 12 + 67]