"""
Generator Context Module

This module contains the GeneratorContext class which holds shared state
used by all individual generators during the pattern generation process.
"""

from typing import Dict, Any, Optional
from music_theory import MusicTheory
from generators.density_manager import DensityManager
from genres.genre_rules import GenreRules


class PerformanceProfile:
    """
    Performance realism and expression shaping parameters applied at MIDI rendering time.
    These control microtiming, articulation, velocity shaping, and swing mode.
    """
    def __init__(
        self,
        swing_mode: str = "eighth",              # {"eighth","sixteenth","triplet"}
        micro_timing_range_ms: float = 6.0,      # ± ms humanization range per note
        grid_bias_ms: float = 0.0,               # constant push/pull relative to grid
        note_length_variance: float = 0.10,      # ± percentage applied randomly to note length
        staccato_prob: float = 0.10,
        staccato_scale: float = 0.60,            # multiply duration when staccato triggers
        tenuto_prob: float = 0.20,
        tenuto_scale: float = 1.00,              # multiply duration when tenuto triggers
        marcato_prob: float = 0.10,
        marcato_velocity_boost: int = 12,        # add to velocity when marcato triggers
        velocity_profile: dict | None = None     # e.g., {"shape":"arch","intensity":0.3,"phrase_length_beats":4.0}
    ):
        self.swing_mode = swing_mode
        self.micro_timing_range_ms = micro_timing_range_ms
        self.grid_bias_ms = grid_bias_ms
        self.note_length_variance = note_length_variance
        self.staccato_prob = staccato_prob
        self.staccato_scale = staccato_scale
        self.tenuto_prob = tenuto_prob
        self.tenuto_scale = tenuto_scale
        self.marcato_prob = marcato_prob
        self.marcato_velocity_boost = marcato_velocity_boost
        self.velocity_profile = velocity_profile or {
            "shape": "arch",
            "intensity": 0.3,
            "phrase_length_beats": 4.0,
        }


class GeneratorContext:
    """Holds shared state for all music generators.

    The GeneratorContext encapsulates the common state and configuration
    that is shared across all generators (melody, harmony, bass, rhythm).
    This promotes better separation of concerns and easier testing by
    removing tight coupling between generators and the shared state.
    """

    def __init__(self, genre_rules: GenreRules, mood: str,
                     note_density: float = 0.5, rhythm_density: float = 0.5,
                     chord_density: float = 0.5, bass_density: float = 0.5,
                     subgenre: Optional[str] = None):
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
        self.subgenre: Optional[str] = subgenre
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

        # Performance expression profile used by MIDI renderer
        self.performance = PerformanceProfile()

        # Get beat characteristics from genre rules for rhythm realism (subgenre-aware)
        try:
            self.beat_characteristics = genre_rules.get_beat_characteristics(self.subgenre)  # type: ignore[attr-defined]
        except Exception:
            # Fallback for dict-based or minimal genre_rules used in tests
            self.beat_characteristics = {
                'swing_factor': 0.5,
                'syncopation_level': 0.0,
                'emphasis_patterns': [],
                'tempo_range': (60, 180)
            }
        
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

    def set_performance(self, **kwargs) -> None:
        """
        Update performance profile parameters.
        Allowed keys mirror PerformanceProfile fields (e.g., swing_mode, micro_timing_range_ms, grid_bias_ms, etc.)
        """
        if not hasattr(self, "performance") or self.performance is None:
            self.performance = PerformanceProfile()
        for k, v in kwargs.items():
            if hasattr(self.performance, k):
                setattr(self.performance, k, v)

    def get_performance(self) -> PerformanceProfile:
        """Return the current performance profile object."""
        if not hasattr(self, "performance") or self.performance is None:
            self.performance = PerformanceProfile()
        return self.performance
