"""
Density Manager for MIDI Master music generation program.

This module provides utilities for managing note and beat density in musical patterns.
It allows fine-grained control over the overall density of generated music.
"""

import random
from typing import List, Dict, Any


class DensityManager:
    """Manages density parameters for musical pattern generation.

    This class provides methods to control the density of notes and beats
    across different musical elements (melody, harmony, rhythm, bass).
    """

    def __init__(self, note_density: float = 0.5, rhythm_density: float = 0.5,
                 chord_density: float = 0.5, bass_density: float = 0.5):
        """
        Initialize the DensityManager.

        Args:
            note_density: Overall note density (0.0-1.0)
            rhythm_density: Rhythm complexity density (0.0-1.0)
            chord_density: Chord voicing density (0.0-1.0)
            bass_density: Bass line density (0.0-1.0)
        """
        self.note_density = self._clamp_density(note_density)
        self.rhythm_density = self._clamp_density(rhythm_density)
        self.chord_density = self._clamp_density(chord_density)
        self.bass_density = self._clamp_density(bass_density)

    def _clamp_density(self, density: float) -> float:
        """Clamp density value to valid range (0.0-1.0)."""
        return max(0.0, min(1.0, density))

    def calculate_note_probability(self, base_probability: float = 1.0) -> float:
        """
        Calculate the probability of placing a note based on density.

        Args:
            base_probability: Base probability before density adjustment

        Returns:
            Adjusted probability based on note density
        """
        # Higher density = higher probability of placing notes
        return base_probability * (0.3 + 0.7 * self.note_density)

    def get_available_durations(self, density: float) -> List[float]:
        """
        Get available note durations based on density level.

        Args:
            density: Density level (0.0-1.0)

        Returns:
            List of available note durations in beats
        """
        # Duration options from longest to shortest
        all_durations = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]

        if density <= 0.2:
            # Very sparse - only long notes
            return [4.0, 2.0, 1.0]
        elif density <= 0.4:
            # Sparse - add quarter notes
            return [4.0, 2.0, 1.0, 0.5]
        elif density <= 0.6:
            # Medium - add eighth notes
            return [2.0, 1.0, 0.5, 0.25]
        elif density <= 0.8:
            # Dense - add sixteenth notes
            return [1.0, 0.5, 0.25, 0.125]
        else:
            # Very dense - all durations available
            return all_durations

    def get_chord_voicing_size(self, max_notes: int = 4) -> int:
        """
        Determine how many notes to include in chord voicing.

        Args:
            max_notes: Maximum number of notes in full voicing

        Returns:
            Number of notes to include in chord voicing
        """
        if self.chord_density <= 0.3:
            # Minimal voicing - just root and fifth
            return min(2, max_notes)
        elif self.chord_density <= 0.6:
            # Standard voicing - root, third, fifth
            return min(3, max_notes)
        else:
            # Extended voicing - full chord
            return max_notes

    def get_rhythm_pattern_complexity(self) -> int:
        """
        Determine rhythm pattern complexity level.

        Returns:
            Complexity level (0-3, where higher = more complex)
        """
        if self.rhythm_density <= 0.25:
            return 0  # Very simple
        elif self.rhythm_density <= 0.5:
            return 1  # Simple
        elif self.rhythm_density <= 0.75:
            return 2  # Medium
        else:
            return 3  # Complex

    def get_bass_note_count(self, bar_length: int = 4) -> int:
        """
        Determine how many bass notes to place per bar.

        Args:
            bar_length: Length of bar in beats

        Returns:
            Number of bass notes to place
        """
        if self.bass_density <= 0.2:
            # Very sparse - one note per 2 bars
            return max(1, bar_length // 8)
        elif self.bass_density <= 0.4:
            # Sparse - one note per bar
            return max(1, bar_length // 4)
        elif self.bass_density <= 0.6:
            # Medium - one note per beat
            return bar_length
        elif self.bass_density <= 0.8:
            # Dense - multiple notes per beat
            return bar_length * 2
        else:
            # Very dense - walking bass
            return bar_length * 4

    def should_place_note(self, position: float, total_length: float) -> bool:
        """
        Determine if a note should be placed at a given position.

        Args:
            position: Position in beats from start
            total_length: Total length in beats

        Returns:
            True if a note should be placed at this position
        """
        # Use density to determine probability of note placement
        probability = self.calculate_note_probability()

        # Adjust probability based on position (slight bias toward strong beats)
        beat_position = position % 4.0
        if beat_position in [0.0, 2.0]:  # Downbeats
            probability *= 1.2
        elif beat_position in [1.0, 3.0]:  # Off-beats
            probability *= 0.8

        return random.random() < probability

    def get_melody_note_count(self, bar_count: int) -> int:
        """
        Determine how many melody notes to generate.

        Args:
            bar_count: Number of bars to generate for

        Returns:
            Number of melody notes to generate
        """
        base_notes_per_bar = 4  # Quarter notes per bar

        if self.note_density <= 0.2:
            # Very sparse melody
            return max(1, int(bar_count * base_notes_per_bar * 0.3))
        elif self.note_density <= 0.4:
            # Sparse melody
            return max(1, int(bar_count * base_notes_per_bar * 0.5))
        elif self.note_density <= 0.6:
            # Medium melody
            return int(bar_count * base_notes_per_bar * 0.7)
        elif self.note_density <= 0.8:
            # Dense melody
            return int(bar_count * base_notes_per_bar * 0.9)
        else:
            # Very dense melody
            return int(bar_count * base_notes_per_bar * 1.2)

    def get_density_settings(self) -> Dict[str, float]:
        """
        Get current density settings.

        Returns:
            Dictionary of current density parameters
        """
        return {
            'note_density': self.note_density,
            'rhythm_density': self.rhythm_density,
            'chord_density': self.chord_density,
            'bass_density': self.bass_density
        }

    def set_density_settings(self, **kwargs):
        """
        Update density settings.

        Args:
            **kwargs: Density parameters to update
        """
        if 'note_density' in kwargs:
            self.note_density = self._clamp_density(kwargs['note_density'])
        if 'rhythm_density' in kwargs:
            self.rhythm_density = self._clamp_density(kwargs['rhythm_density'])
        if 'chord_density' in kwargs:
            self.chord_density = self._clamp_density(kwargs['chord_density'])
        if 'bass_density' in kwargs:
            self.bass_density = self._clamp_density(kwargs['bass_density'])


# Preset density configurations
DENSITY_PRESETS = {
    'minimal': {
        'note_density': 0.1,
        'rhythm_density': 0.2,
        'chord_density': 0.1,
        'bass_density': 0.1
    },
    'sparse': {
        'note_density': 0.3,
        'rhythm_density': 0.4,
        'chord_density': 0.3,
        'bass_density': 0.3
    },
    'balanced': {
        'note_density': 0.5,
        'rhythm_density': 0.5,
        'chord_density': 0.5,
        'bass_density': 0.5
    },
    'dense': {
        'note_density': 0.7,
        'rhythm_density': 0.6,
        'chord_density': 0.8,
        'bass_density': 0.7
    },
    'complex': {
        'note_density': 0.9,
        'rhythm_density': 0.8,
        'chord_density': 0.9,
        'bass_density': 0.9
    }
}


def create_density_manager_from_preset(preset_name: str) -> DensityManager:
    """
    Create a DensityManager from a preset configuration.

    Args:
        preset_name: Name of the preset ('minimal', 'sparse', 'balanced', 'dense', 'complex')

    Returns:
        DensityManager configured with preset values

    Raises:
        ValueError: If preset_name is not recognized
    """
    if preset_name not in DENSITY_PRESETS:
        available = list(DENSITY_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

    settings = DENSITY_PRESETS[preset_name]
    return DensityManager(**settings)