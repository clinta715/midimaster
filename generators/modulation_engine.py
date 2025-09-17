"""
Advanced Modulation Engine for sophisticated key changes and modulations.

This module provides comprehensive key modulation capabilities including:
- Smooth key transitions with voice leading
- Pivot chords and common tone modulation
- Enharmonic modulations
- Chromatic mediant relationships
- Real-time modulation analysis
- Modulation planning and execution
"""

import math
from typing import List, Dict, Tuple, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

from music_theory import MusicTheory, Note, ScaleType


class ModulationType(Enum):
    """Types of key modulations available."""
    DIRECT = "direct"           # Immediate key change
    PIVOT_CHORD = "pivot_chord" # Using pivot chord
    COMMON_TONE = "common_tone" # Common tone modulation
    ENHARMONIC = "enharmonic"   # Enharmonic equivalent
    CHROMATIC_MEDIANT = "chromatic_mediant"  # Third relationships
    SMOOTH_VOICE_LEADING = "smooth_voice_leading"  # Gradual transition


class ModulationDirection(Enum):
    """Direction of key modulation."""
    UP = "up"
    DOWN = "down"
    CHROMATIC_UP = "chromatic_up"
    CHROMATIC_DOWN = "chromatic_down"


@dataclass
class KeyChange:
    """Represents a key change event."""
    start_time: float
    end_time: float
    from_key: str
    to_key: str
    from_scale: str
    to_scale: str
    modulation_type: ModulationType
    pivot_chords: Optional[List[str]] = None

    def __post_init__(self):
        if self.pivot_chords is None:
            self.pivot_chords = []


class VoiceLeadingResult:
    """Result of voice leading analysis for key changes."""

    def __init__(self, original_pitches: List[int], target_pitches: List[int],
                 voice_movements: List[int], smoothness_score: float):
        """
        Initialize voice leading result.

        Args:
            original_pitches: Original chord pitches
            target_pitches: Target chord pitches
            voice_movements: Movement in semitones for each voice
            smoothness_score: Score from 0-1 indicating smoothness
        """
        self.original_pitches = original_pitches
        self.target_pitches = target_pitches
        self.voice_movements = voice_movements
        self.smoothness_score = smoothness_score

    def is_smooth(self, threshold: float = 0.7) -> bool:
        """Check if the voice leading is considered smooth."""
        return self.smoothness_score >= threshold


class ModulationAnalyzer:
    """Analyzes modulation possibilities between keys."""

    def __init__(self):
        """Initialize the modulation analyzer."""
        self.music_theory = MusicTheory()

    def find_pivot_chords(self, from_key: str, to_key: str) -> List[str]:
        """
        Find pivot chords that work in both keys.

        Args:
            from_key: Source key (e.g., 'C major')
            to_key: Target key (e.g., 'G major')

        Returns:
            List of chord names that function in both keys
        """
        try:
            # Parse keys
            from_root, from_scale = self.music_theory.parse_scale_string(from_key)
            to_root, to_scale = self.music_theory.parse_scale_string(to_key)

            # Get scale pitches
            from_pitches = self.music_theory.build_scale(from_root, from_scale)
            to_pitches = self.music_theory.build_scale(to_root, to_scale)

            # Find common pitches
            common_pitches = set(from_pitches) & set(to_pitches)

            # Generate possible pivot chords from common pitches
            pivot_chords = []

            # Try triads and seventh chords built on common pitches
            for pitch in common_pitches:
                # Try major triad
                triad = [pitch, pitch + 4, pitch + 7]
                if all(p in from_pitches for p in triad) and all(p in to_pitches for p in triad):
                    chord_name = self._midi_pitches_to_chord_name(triad, from_key)
                    if chord_name:
                        pivot_chords.append(chord_name)

                # Try minor triad
                min_triad = [pitch, pitch + 3, pitch + 7]
                if all(p in from_pitches for p in min_triad) and all(p in to_pitches for p in min_triad):
                    chord_name = self._midi_pitches_to_chord_name(min_triad, from_key)
                    if chord_name:
                        pivot_chords.append(chord_name)

            return pivot_chords

        except Exception as e:
            print(f"Error finding pivot chords: {e}")
            return []

    def calculate_modulation_distance(self, from_key: str, to_key: str) -> float:
        """
        Calculate the harmonic distance of a modulation.

        Args:
            from_key: Source key
            to_key: Target key

        Returns:
            Distance score (0-1, higher = more distant modulation)
        """
        try:
            from_root, from_scale = self.music_theory.parse_scale_string(from_key)
            to_root, to_scale = self.music_theory.parse_scale_string(to_key)

            # Calculate root movement in semitones
            root_distance = abs(to_root.value - from_root.value) % 12

            # Normalize to 0-1 scale
            # Perfect fifth/circle of fifths = 0.2, tritone = 0.5, etc.
            normalized_distance = min(root_distance / 6.0, 1.0)

            # Apply scale compatibility factor
            scale_compatibility = 1.0 if from_scale == to_scale else 0.8

            return normalized_distance * scale_compatibility

        except Exception as e:
            print(f"Error calculating modulation distance: {e}")
            return 1.0

    def find_enharmonic_equivalents(self, key: str) -> List[str]:
        """
        Find enharmonic equivalents of a key.

        Args:
            key: Key to find equivalents for

        Returns:
            List of enharmonically equivalent keys
        """
        equivalents = []
        try:
            root, scale = self.music_theory.parse_scale_string(key)

            # Common enharmonic equivalents
            enharmonic_map = {
                'C#': ['Db'],
                'D#': ['Eb'],
                'F#': ['Gb'],
                'G#': ['Ab'],
                'A#': ['Bb'],
                'Db': ['C#'],
                'Eb': ['D#'],
                'Gb': ['F#'],
                'Ab': ['G#'],
                'Bb': ['A#']
            }

            root_name = root.name if hasattr(root, 'name') else str(root)
            if root_name in enharmonic_map:
                for equiv_root in enharmonic_map[root_name]:
                    equivalents.append(f"{equiv_root} {scale}")

        except Exception as e:
            print(f"Error finding enharmonic equivalents: {e}")

        return equivalents

    def find_chromatic_mediants(self, key: str) -> List[str]:
        """
        Find chromatic mediant relationships.

        Args:
            key: Source key

        Returns:
            List of chromatic mediant keys
        """
        mediants = []
        try:
            root, scale = self.music_theory.parse_scale_string(key)

            # Chromatic mediants are up/down 3 semitones
            mediant_up = (root.value + 3) % 12
            mediant_down = (root.value - 3) % 12

            # Convert back to note names
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

            if mediant_up < len(note_names):
                mediants.append(f"{note_names[mediant_up]} {scale}")
            if mediant_down < len(note_names):
                mediants.append(f"{note_names[mediant_down]} {scale}")

        except Exception as e:
            print(f"Error finding chromatic mediants: {e}")

        return mediants

    def _midi_pitches_to_chord_name(self, pitches: List[int], key: str) -> Optional[str]:
        """Convert MIDI pitches to chord name (simplified implementation)."""
        if len(pitches) < 3:
            return None

        try:
            root = pitches[0] % 12
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

            if root < len(note_names):
                root_name = note_names[root]

                # Check interval pattern for chord type
                intervals = [(p - pitches[0]) % 12 for p in pitches[1:]]

                if intervals == [4, 7]:  # Major triad
                    return f"{root_name}"
                elif intervals == [3, 7]:  # Minor triad
                    return f"{root_name}m"
                elif intervals == [3, 6]:  # Diminished triad
                    return f"{root_name}dim"
                elif intervals == [4, 8]:  # Augmented triad
                    return f"{root_name}aug"

        except Exception:
            pass

        return None


class VoiceLeadingEngine:
    """Engine for calculating optimal voice leading between chords."""

    def __init__(self):
        """Initialize the voice leading engine."""
        self.music_theory = MusicTheory()

    def calculate_voice_leading(self, from_chord: List[int], to_chord: List[int],
                              max_voices: int = 4) -> VoiceLeadingResult:
        """
        Calculate optimal voice leading between two chords.

        Args:
            from_chord: Source chord pitches
            to_chord: Target chord pitches
            max_voices: Maximum number of voices to consider

        Returns:
            VoiceLeadingResult with analysis
        """
        # Extend chords to match voice count
        from_extended = from_chord + [from_chord[0] + 12] * (max_voices - len(from_chord))
        to_extended = to_chord + [to_chord[0] + 12] * (max_voices - len(to_chord))

        # Calculate all possible voice assignments
        min_movement = float('inf')
        best_movements = []

        # Simple assignment: sort and match by index
        movements = []
        for i in range(min(len(from_extended), len(to_extended))):
            movement = to_extended[i] - from_extended[i]
            movements.append(movement)
            min_movement = min(min_movement, abs(movement))

        # Calculate smoothness score (inverse of average movement)
        avg_movement = sum(abs(m) for m in movements) / len(movements)
        smoothness_score = max(0, 1 - avg_movement / 12)  # 12 semitones = octave

        return VoiceLeadingResult(from_extended, to_extended, movements, smoothness_score)

    def find_optimal_pivot_chord(self, from_key: str, to_key: str) -> Optional[str]:
        """
        Find the optimal pivot chord for modulation.

        Args:
            from_key: Source key
            to_key: Target key

        Returns:
            Best pivot chord name or None
        """
        analyzer = ModulationAnalyzer()
        pivot_chords = analyzer.find_pivot_chords(from_key, to_key)

        if not pivot_chords:
            return None

        # For now, return the first pivot chord
        # Could be enhanced to choose based on harmonic function
        return pivot_chords[0] if pivot_chords else None


class ModulationEngine:
    """Main modulation engine coordinating all modulation features."""

    def __init__(self):
        """Initialize the modulation engine."""
        self.analyzer = ModulationAnalyzer()
        self.voice_leading = VoiceLeadingEngine()
        self.current_key = "C major"
        self.modulation_history: List[KeyChange] = []

    def plan_modulation(self, from_key: str, to_key: str,
                       modulation_type: Optional[ModulationType] = None) -> Optional[KeyChange]:
        """
        Plan a key modulation with optimal parameters.

        Args:
            from_key: Source key
            to_key: Target key
            modulation_type: Preferred modulation type (auto-selected if None)

        Returns:
            Planned KeyChange or None if impossible
        """
        if modulation_type is None:
            modulation_type = self._choose_optimal_modulation_type(from_key, to_key)

        try:
            # Validate keys
            self.music_theory.validate_key_mode(from_key.split()[0], from_key.split()[1])
            self.music_theory.validate_key_mode(to_key.split()[0], to_key.split()[1])

            # Create modulation plan
            if modulation_type == ModulationType.PIVOT_CHORD:
                pivot_chords = self.analyzer.find_pivot_chords(from_key, to_key)
                if not pivot_chords:
                    # Fallback to direct modulation
                    modulation_type = ModulationType.DIRECT
                    pivot_chords = []

            key_change = KeyChange(
                start_time=0.0,  # Will be set when executed
                end_time=0.0,    # Will be set when executed
                from_key=from_key,
                to_key=to_key,
                from_scale=from_key.split()[1],
                to_scale=to_key.split()[1],
                modulation_type=modulation_type,
                pivot_chords=pivot_chords if modulation_type == ModulationType.PIVOT_CHORD else []
            )

            return key_change

        except ValueError as e:
            print(f"Invalid modulation: {e}")
            return None

    def execute_modulation(self, key_change: KeyChange) -> bool:
        """
        Execute a planned key modulation.

        Args:
            key_change: The key change to execute

        Returns:
            True if successful
        """
        try:
            # Update current key
            self.current_key = key_change.to_key
            self.modulation_history.append(key_change)

            print(f"Modulated from {key_change.from_key} to {key_change.to_key} "
                  f"using {key_change.modulation_type.value}")

            return True

        except Exception as e:
            print(f"Failed to execute modulation: {e}")
            return False

    def get_modulation_suggestions(self, current_key: str,
                                 max_distance: float = 0.5) -> List[Tuple[str, float]]:
        """
        Get modulation suggestions within a distance threshold.

        Args:
            current_key: Current key
            max_distance: Maximum modulation distance (0-1)

        Returns:
            List of (suggested_key, distance) tuples
        """
        suggestions = []
        common_keys = [
            "C major", "G major", "D major", "A major", "E major", "F major",
            "A minor", "E minor", "D minor", "G minor", "C minor"
        ]

        for key in common_keys:
            if key != current_key:
                distance = self.analyzer.calculate_modulation_distance(current_key, key)
                if distance <= max_distance:
                    suggestions.append((key, distance))

        # Sort by distance (closest first)
        suggestions.sort(key=lambda x: x[1])
        return suggestions

    def analyze_modulation_sequence(self, key_sequence: List[str]) -> Dict:
        """
        Analyze a sequence of key changes.

        Args:
            key_sequence: List of keys in sequence

        Returns:
            Analysis dictionary with various metrics
        """
        analysis = {
            'total_distance': 0.0,
            'smoothness_score': 0.0,
            'pivot_chord_usage': 0,
            'modulation_types': defaultdict(int)
        }

        for i in range(len(key_sequence) - 1):
            from_key = key_sequence[i]
            to_key = key_sequence[i + 1]

            # Calculate distance
            distance = self.analyzer.calculate_modulation_distance(from_key, to_key)
            analysis['total_distance'] += distance

            # Analyze modulation type
            plan = self.plan_modulation(from_key, to_key)
            if plan:
                analysis['modulation_types'][plan.modulation_type.value] += 1
                if plan.pivot_chords:
                    analysis['pivot_chord_usage'] += 1

        # Calculate smoothness (inverse of average distance)
        if len(key_sequence) > 1:
            avg_distance = analysis['total_distance'] / (len(key_sequence) - 1)
            analysis['smoothness_score'] = max(0, 1 - avg_distance)

        return dict(analysis)

    def _choose_optimal_modulation_type(self, from_key: str, to_key: str) -> ModulationType:
        """Choose the optimal modulation type based on key relationship."""
        distance = self.analyzer.calculate_modulation_distance(from_key, to_key)

        if distance < 0.2:  # Close relationship (circle of fifths)
            return ModulationType.SMOOTH_VOICE_LEADING
        elif distance < 0.4:  # Medium distance
            # Check for pivot chords
            pivot_chords = self.analyzer.find_pivot_chords(from_key, to_key)
            if pivot_chords:
                return ModulationType.PIVOT_CHORD
            else:
                return ModulationType.SMOOTH_VOICE_LEADING
        else:  # Distant relationship
            # Check if enharmonic equivalent
            equivalents = self.analyzer.find_enharmonic_equivalents(from_key)
            if to_key in equivalents:
                return ModulationType.ENHARMONIC
            else:
                return ModulationType.DIRECT

    @property
    def music_theory(self):
        """Access to music theory utilities."""
        return self.analyzer.music_theory