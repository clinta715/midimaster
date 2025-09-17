"""
Rhythm Generator Module

This module contains the RhythmGenerator class responsible for generating
rhythmic patterns based on genre-specific timing and patterns.
"""

import random
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from structures.data_structures import Note, Pattern, PatternType
from generators.generator_utils import get_velocity_for_mood, initialize_key_and_scale

if TYPE_CHECKING:
    from generators.generator_context import GeneratorContext


class RhythmGenerator:
    """Generates rhythm patterns with genre-specific timing.

    The RhythmGenerator creates rhythmic foundation using genre-specific rhythm patterns.
    These patterns can be used for percussion or rhythmic accompaniment.
    The rhythm is influenced by both the genre and the selected mood.
    """

    def __init__(self, context: 'GeneratorContext'):
        """
        Initialize the RhythmGenerator.

        Args:
            context: Shared GeneratorContext containing music theory and configuration
        """
        self.context = context

    def generate(self, num_bars: int, beat_complexity: float = 0.5) -> Pattern:
        """Generate a rhythm pattern.

        Creates a rhythmic foundation using genre-specific rhythm patterns.
        These patterns can be used for percussion or rhythmic accompaniment.
        The rhythm is influenced by both the genre and the selected mood.

        Args:
            num_bars: Number of bars to generate
            beat_complexity: Complexity of the beat (0.0-1.0, default 0.5)

        Returns:
            Pattern object containing the rhythm notes
        """
        # Validate beat complexity parameter
        if not 0.0 <= beat_complexity <= 1.0:
            raise ValueError("beat_complexity must be between 0.0 and 1.0")

        # Ensure key and scale are established
        if not self.context.scale_pitches:
            initialize_key_and_scale(self.context)

        notes = []
        chords = []
        start_time = 0.0

        # Select subgenre-aware rhythm patterns (fallback to base patterns)
        subgenre = getattr(self.context, 'subgenre', None)
        rhythm_patterns = self.context.genre_rules.get_rhythm_patterns_for_subgenre(subgenre)
        if not rhythm_patterns:
            rhythm_patterns = self.context.genre_rules.get_rhythm_patterns()
        # Fallback to straight eighths if still missing
        if not rhythm_patterns:
            rhythm_patterns = [{'name': 'straight_eight_fallback', 'pattern': [0.5, 0.5, 0.5, 0.5]}]
        selected = random.choice(rhythm_patterns)
        selected_rhythm = selected['pattern']
        # Swing parameters
        swing_factor = self.context.beat_characteristics.get('swing_factor', 0.5)
        apply_swing = abs(swing_factor - 0.5) > 1e-6
        swing_toggle = False  # alternate long-short for eighths when applicable

        # Generate rhythm notes
        # 4 beats per bar is the default time signature
        for i in range(num_bars * 4):
            # Use the rhythm pattern cyclically
            base_duration = selected_rhythm[i % len(selected_rhythm)]

            # Apply simple eighth-note swing by warping pairs of 0.5 durations
            duration = base_duration
            if apply_swing and abs(base_duration - 0.5) < 1e-6:
                duration = swing_factor if not swing_toggle else (1.0 - swing_factor)
                swing_toggle = not swing_toggle

            # Percussion-like notes (could be mapped to drum sounds in MIDI)
            # Use pitches that are more likely to be in scale for melodic percussion
            if self.context.scale_pitches and i % 3 == 0:  # Every few hits, use a scale tone
                pitch = random.choice([p for p in self.context.scale_pitches if p > 60])
            else:
                # Simple mapping to percussion sounds (MIDI pitches 35-4 are common percussion)
                pitch = 35 + (i % 10)

            # Get rhythm pattern complexity from density
            complexity = self.context.density_manager.get_rhythm_pattern_complexity()

            # Adjust velocity based on mood and metric position
            base_velocity = get_velocity_for_mood(self.context.mood)
            
            # Emphasize notes on strong beats based on subgenre-aware beat characteristics
            beat_in_bar = int(start_time % 4) + 1 # 1-indexed beat in bar
            emphasis_patterns = self.context.beat_characteristics.get('emphasis_patterns', [])

            if beat_in_bar in emphasis_patterns:
                velocity = min(127, base_velocity + 25) # Strong emphasis
            elif beat_in_bar % 2 == 1: # Other odd beats (1, 3)
                velocity = min(127, base_velocity + 10) # Medium emphasis
            else: # Even beats (2, 4)
                velocity = base_velocity

            # Add a small random variation for humanization
            velocity = max(0, min(127, velocity + random.randint(-7, 7)))

            # Only place note based on rhythm density probability
            if self.context.density_manager.calculate_note_probability() > random.random():
                note = Note(pitch, duration, velocity, start_time)
                notes.append(note)

            # Advance the start time
            start_time += duration

        # Return the rhythm pattern
        return Pattern(PatternType.RHYTHM, notes, chords)