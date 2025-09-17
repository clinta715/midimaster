"""
Pattern Orchestrator Module

This module contains the PatternOrchestrator class which coordinates
the generation of musical patterns using individual generator components.
It maintains the same public API as the original PatternGenerator while
using the refactored modular generator architecture.
"""

from typing import List, Optional, TYPE_CHECKING

from structures.data_structures import Pattern, PatternType
from structures.song_skeleton import SongSkeleton
from generators.generator_context import GeneratorContext
from generators.melody_generator import MelodyGenerator
from generators.harmony_generator import HarmonyGenerator
from generators.bass_generator import BassGenerator
from generators.rhythm_generator import RhythmGenerator
from generators.generator_utils import initialize_key_and_scale
from genres.genre_rules import GenreRules

if TYPE_CHECKING:
    from typing import Dict, Any


class PatternOrchestrator:
    """Orchestrates musical pattern generation using individual generators.

    The PatternOrchestrator coordinates calls to specialized generators:
    MelodyGenerator, HarmonyGenerator, BassGenerator, and RhythmGenerator.
    It maintains the same public interface as the original PatternGenerator
    while using dependency injection for the shared GeneratorContext.

    The orchestrator:
    1. Creates a shared GeneratorContext with all necessary state
    2. Initializes individual generators with the context
    3. Coordinates pattern generation across all generator types
    4. Returns the same pattern structure as the original implementation
    """

    def __init__(self, genre_rules: GenreRules, mood: str,
                 note_density: float = 0.5, rhythm_density: float = 0.5,
                 chord_density: float = 0.5, bass_density: float = 0.5,
                 harmonic_variance: str = 'medium',
                 subgenre: Optional[str] = None,
                 context: Optional[GeneratorContext] = None):
        """
        Initialize the PatternOrchestrator.

        Args:
            genre_rules: Dictionary of genre-specific rules including scales,
                          chord progressions, rhythm patterns, etc.
            mood: Mood for the music generation ('happy', 'sad', 'energetic', 'calm')
            note_density: Density of notes in melody patterns (0.0-1.0)
            rhythm_density: Density of rhythm patterns (0.0-1.0)
            chord_density: Density of harmonic content (0.0-1.0)
            bass_density: Density of bass patterns (0.0-1.0)
            harmonic_variance: Level of harmonic movement between chords ('close', 'medium', 'distant')
            subgenre: Optional subgenre/style within the genre (e.g., 'deep_house', 'drill', 'jungle')
            context: Optional pre-configured GeneratorContext (for user key/mode settings)
        """
        # Use provided context or create new one
        if context is not None:
            self.context = context
            # Update context with provided parameters
            self.context.genre_rules = genre_rules
            self.context.mood = mood
            self.context.subgenre = subgenre
            self.context.density_manager.note_density = note_density
            self.context.density_manager.rhythm_density = rhythm_density
            self.context.density_manager.chord_density = chord_density
            self.context.density_manager.bass_density = bass_density
            # Refresh beat characteristics with subgenre-awareness
            self.context.beat_characteristics = self.context.genre_rules.get_beat_characteristics(self.context.subgenre)
        else:
            # Create shared context
            self.context = GeneratorContext(
                genre_rules=genre_rules,
                mood=mood,
                note_density=note_density,
                rhythm_density=rhythm_density,
                chord_density=chord_density,
                bass_density=bass_density,
                subgenre=subgenre
            )

        # Store harmonic variance setting
        self.harmonic_variance = harmonic_variance

        # Initialize individual generators with shared context
        self.melody_generator = MelodyGenerator(self.context)
        self.harmony_generator = HarmonyGenerator(self.context)
        self.bass_generator = BassGenerator(self.context)
        self.rhythm_generator = RhythmGenerator(self.context)

    @property
    def current_key(self):
        """Access to the current key from context (for backward compatibility)."""
        return self.context.current_key

    @property
    def current_scale(self):
        """Access to the current scale from context (for backward compatibility)."""
        return self.context.current_scale

    @property
    def scale_pitches(self):
        """Access to scale pitches from context (for backward compatibility)."""
        return self.context.scale_pitches

    @property
    def density_manager(self):
        """Expose DensityManager for backward compatibility with tests."""
        return self.context.density_manager

    def generate_beats_only(self, song_skeleton: SongSkeleton, num_bars: int, beat_complexity: float = 0.5) -> Pattern:
        """Generate only a rhythm pattern."""
        initialize_key_and_scale(self.context)
        return self.rhythm_generator.generate(num_bars, beat_complexity)

    def generate_chords_only(self, song_skeleton: SongSkeleton, num_bars: int, chord_complexity: str = 'medium') -> Pattern:
        """Generate only a harmony (chords) pattern."""
        initialize_key_and_scale(self.context)
        return self.harmony_generator.generate(num_bars, chord_complexity, self.harmonic_variance)

    def generate_selective_patterns(self, song_skeleton: SongSkeleton, num_bars: int,
                                    pattern_types: List[str],
                                    beat_complexity: float = 0.5,
                                    chord_complexity: str = 'medium') -> List[Pattern]:
        """Generate a selection of patterns."""
        initialize_key_and_scale(self.context)
        patterns = []
        for p_type in pattern_types:
            if p_type == 'melody':
                patterns.append(self.melody_generator.generate(num_bars))
            elif p_type == 'harmony':
                patterns.append(self.harmony_generator.generate(num_bars, chord_complexity, self.harmonic_variance))
            elif p_type == 'bass':
                patterns.append(self.bass_generator.generate(num_bars))
            elif p_type == 'rhythm':
                patterns.append(self.rhythm_generator.generate(num_bars, beat_complexity))
        return patterns

    def generate_patterns(self, song_skeleton: SongSkeleton, num_bars: int,
                          pattern_length: Optional[int] = None, average_notes_per_chord: Optional[int] = None,
                          beat_complexity: float = 0.5, chord_complexity: str = 'medium') -> List[Pattern]:
        """
        Generate musical patterns for a song (enhanced with backward compatibility).

        This method orchestrates the generation of all musical patterns for a composition:
        melody, harmony, bass, and rhythm. Each pattern type is generated using
        genre-specific rules and the selected mood.

        Enhanced version with additional parameters while maintaining backward compatibility.

        Args:
            song_skeleton: The song structure to generate patterns for
            num_bars: Number of bars to generate
            pattern_length: (Optional) Pattern length in bars (for future use)
            average_notes_per_chord: (Optional) Average notes per chord metric (for future use)
            beat_complexity: Complexity of the beat (0.0-1.0, default 0.5)
            chord_complexity: Complexity level of chords ('simple', 'medium', 'complex')

        Returns:
            List of generated patterns (melody, harmony, bass, rhythm)
        """
        # Validate new parameters
        if beat_complexity is not None and not 0.0 <= beat_complexity <= 1.0:
            raise ValueError("beat_complexity must be between 0.0 and 1.0")

        if chord_complexity is not None:
            valid_complexities = ['simple', 'medium', 'complex']
            if chord_complexity not in valid_complexities:
                raise ValueError(f"chord_complexity must be one of {valid_complexities}")

        patterns = []

        # Establish key and scale from genre rules
        # This determines the musical foundation for all generated patterns
        initialize_key_and_scale(self.context)

        # Generate melody pattern using scale-based pitch selection
        melody_pattern = self.melody_generator.generate(num_bars)
        patterns.append(melody_pattern)

        # Generate harmony pattern using Roman numeral chord progressions
        harmony_pattern = self.harmony_generator.generate(num_bars, chord_complexity, self.harmonic_variance)
        patterns.append(harmony_pattern)

        # Generate bass pattern using chord roots from the progression
        bass_pattern = self.bass_generator.generate(num_bars)
        patterns.append(bass_pattern)

        # Generate rhythm pattern with genre-specific timing
        rhythm_pattern = self.rhythm_generator.generate(num_bars, beat_complexity)
        patterns.append(rhythm_pattern)

        return patterns

    def _generate_fill_pattern(self, num_bars: int = 1, fill_type: str = 'rhythmic') -> Pattern:
        """
        Generates a short transitional fill pattern.

        Args:
            num_bars: Number of bars for the fill (default: 1)
            fill_type: Type of fill to generate ('rhythmic' or 'melodic')

        Returns:
            A Pattern object representing the fill.
        """
        initialize_key_and_scale(self.context)
        if fill_type == 'rhythmic':
            return self.rhythm_generator.generate(num_bars, beat_complexity=0.8) # More complex rhythm for fills
        elif fill_type == 'melodic':
            return self.melody_generator.generate(num_bars) # Simple melody for fills
        else:
            raise ValueError("Invalid fill_type. Must be 'rhythmic' or 'melodic'.")