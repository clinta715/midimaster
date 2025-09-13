"""
Generator Context Module

This module contains the GeneratorContext class which holds shared state
used by all individual generators during the pattern generation process.
"""

from typing import Dict, Any, Optional
from music_theory import MusicTheory
from generators.density_manager import DensityManager
from genres.genre_rules import GenreRules


class GeneratorContext:
    """Holds shared state for all music generators.

    The GeneratorContext encapsulates the common state and configuration
    that is shared across all generators (melody, harmony, bass, rhythm).
    This promotes better separation of concerns and easier testing by
    removing tight coupling between generators and the shared state.
    """

    def __init__(self, genre_rules: GenreRules, mood: str,
                 note_density: float = 0.5, rhythm_density: float = 0.5,
                 chord_density: float = 0.5, bass_density: float = 0.5):
        """
        Initialize the GeneratorContext.

        Args:
            genre_rules: Dictionary of genre-specific rules including scales,
                         chord progressions, rhythm patterns, etc.
            mood: Mood for the music generation ('happy', 'sad', 'energetic', 'calm')
            note_density: Density of notes in melody patterns (0.0-1.0)
            rhythm_density: Density of rhythm patterns (0.0-1.0)
            chord_density: Density of harmonic content (0.0-1.0)
            bass_density: Density of bass patterns (0.0-1.0)
        """
        self.genre_rules = genre_rules
        self.mood = mood
        self.base_pitch = 60  # Middle C as default reference point
        self.music_theory = MusicTheory()
        self.current_key = 'C'  # Default key (will be updated based on genre rules)
        self.current_scale = 'major'  # Default scale (will be updated based on genre rules)
        self.scale_pitches = []  # Will be populated with valid scale pitches for the selected key/scale

        # User-specified key/mode overrides
        self.user_key: Optional[str] = None  # User-specified root note (e.g., 'A')
        self.user_mode: Optional[str] = None  # User-specified mode (e.g., 'dorian')

        # Create DensityManager with provided density settings
        self.density_manager = DensityManager(
            note_density=note_density,
            rhythm_density=rhythm_density,
            chord_density=chord_density,
            bass_density=bass_density
        )

        # Get beat characteristics from genre rules for rhythm realism
        self.beat_characteristics = genre_rules.get_beat_characteristics()

    def set_user_key_mode(self, key: str, mode: str) -> None:
        """
        Set user-specified key and mode with validation.

        Args:
            key: Root note (e.g., 'A')
            mode: Scale type (e.g., 'dorian')

        Raises:
            ValueError: If the key/mode combination is invalid
        """
        if not self.music_theory.validate_key_mode(key, mode):
            raise ValueError(f"Invalid key/mode combination: {key} {mode}")
        self.user_key = key
        self.user_mode = mode