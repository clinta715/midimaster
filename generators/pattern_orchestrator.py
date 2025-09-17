"""
Pattern Orchestrator Module
    
This module contains the PatternOrchestrator class which coordinates
the generation of musical patterns using individual generator components.
It maintains the same public API as the original PatternGenerator while
using the refactored modular generator architecture.
"""

from typing import List, Optional, TYPE_CHECKING, Union, Dict

from structures.data_structures import Pattern, PatternType
from structures.song_skeleton import SongSkeleton
from generators.generator_context import GeneratorContext
from generators.melody_generator import MelodyGenerator
from generators.harmony_generator import HarmonyGenerator
from generators.bass_generator import BassGenerator
from generators.rhythm_generator import RhythmGenerator
from generators.generator_utils import initialize_key_and_scale
from generators.loop_manager import LoopManager
from generators.melody_refinement_loop import MelodyRefinementLoop
from generators.harmony_refinement_loop import HarmonyRefinementLoop
from generators.timing_engine import AdvancedTimingEngine
from generators.rhythm_refinement_loop import RhythmRefinementLoop
from genres.genre_rules import GenreRules
from instruments.advanced_arrangement_engine import AdvancedArrangementEngine
from instruments.instrumentation_manager import InstrumentationManager

if TYPE_CHECKING:
    from typing import Dict, Any
    from data_store.pattern_repository import PatternRepository


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
                      pattern_strength: float = 1.0,
                      subgenre: Optional[str] = None,
                      context: Optional[GeneratorContext] = None,
                      pattern_repository: Optional['PatternRepository'] = None):
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
            pattern_strength: Velocity scaling strength for patterns (0.0-1.0)
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
            self.context.pattern_strength = pattern_strength
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
                pattern_strength=pattern_strength,
                subgenre=subgenre
            )
    
        # Store harmonic variance setting
        self.harmonic_variance = harmonic_variance
    
        # Setup optional pattern repository (non-fatal on failure)
        self.pattern_repository = pattern_repository
        if self.pattern_repository is None:
            try:
                from data_store.pattern_repository import PatternRepository as _PR  # type: ignore
                try:
                    self.pattern_repository = _PR()
                except Exception:
                    # On DB open or init failure, continue without repository
                    self.pattern_repository = None
            except Exception:
                # Import failed; continue without repository
                self.pattern_repository = None

        # Initialize individual generators with shared context
        self.melody_generator = MelodyGenerator(self.context)
        self.harmony_generator = HarmonyGenerator(self.context)
        self.bass_generator = BassGenerator(self.context)
        self.rhythm_generator = RhythmGenerator(
            self.context,
            pattern_strength=self.context.pattern_strength,
            pattern_repository=self.pattern_repository
        )
    
        # Initialize loop-based generation system
        self.loop_manager = None
        self.use_loop_based_generation = False
    
        # Initialize advanced timing engine
        self.timing_engine = AdvancedTimingEngine(base_tempo=120.0)  # Default tempo, can be configured
        self.use_advanced_timing = False

    def enable_advanced_timing(self, base_tempo: float = 120.0):
        """Enable advanced timing and microtiming features.
    
        Args:
            base_tempo: Base tempo in BPM for the timing engine
        """
        self.timing_engine = AdvancedTimingEngine(base_tempo=base_tempo)
        self.use_advanced_timing = True
    
    def disable_advanced_timing(self):
        """Disable advanced timing and return to basic timing."""
        self.use_advanced_timing = False
        self.use_loop_based_generation = False
    
    def enable_loop_based_generation(self, max_iterations: int = 5, convergence_threshold: float = 0.05):
        """Enable loop-based generation with refinement loops.
    
        Args:
            max_iterations: Maximum iterations for each refinement loop
            convergence_threshold: Quality improvement threshold for convergence
        """
        self.loop_manager = LoopManager(max_iterations, convergence_threshold)
    
        # Add refinement loops for each pattern type
        self.loop_manager.add_generation_loop(MelodyRefinementLoop(max_iterations, convergence_threshold))
        self.loop_manager.add_generation_loop(HarmonyRefinementLoop(max_iterations, convergence_threshold))
        self.loop_manager.add_generation_loop(RhythmRefinementLoop(max_iterations, convergence_threshold))
    
        self.use_loop_based_generation = True
    
    def disable_loop_based_generation(self):
        """Disable loop-based generation and return to standard generation."""
        self.loop_manager = None
        self.use_loop_based_generation = False
    
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
    
        # Establish key and scale from genre rules
        # This determines the musical foundation for all generated patterns
        initialize_key_and_scale(self.context)
    
        # Use loop-based generation if enabled
        if self.use_loop_based_generation and self.loop_manager is not None:
            return self._generate_with_loops(num_bars, beat_complexity, chord_complexity)
        else:
            return self._generate_standard(num_bars, beat_complexity, chord_complexity)
    
    def _generate_with_loops(self, num_bars: int, beat_complexity: float, chord_complexity: str) -> List[Pattern]:
        """Generate patterns using loop-based refinement."""
        # Generate initial patterns
        initial_patterns = self._generate_standard(num_bars, beat_complexity, chord_complexity)
    
        # Apply refinement loops
        refined_patterns = []
    
        # Create a mapping of pattern types to patterns
        pattern_map = {pattern.pattern_type.value: pattern for pattern in initial_patterns}
    
        # Execute each refinement loop
        if 'melody' in pattern_map and self.loop_manager is not None:
            melody_loop = next((loop for loop in self.loop_manager.generation_loops
                              if isinstance(loop, MelodyRefinementLoop)), None)
            if melody_loop:
                refined_melody = melody_loop.execute(self.context, pattern_map['melody'])
                refined_patterns.append(refined_melody)
            else:
                refined_patterns.append(pattern_map['melody'])
    
        if 'harmony' in pattern_map and self.loop_manager is not None:
            harmony_loop = next((loop for loop in self.loop_manager.generation_loops
                               if isinstance(loop, HarmonyRefinementLoop)), None)
            if harmony_loop:
                refined_harmony = harmony_loop.execute(self.context, pattern_map['harmony'])
                refined_patterns.append(refined_harmony)
            else:
                refined_patterns.append(pattern_map['harmony'])
    
        # For bass, use harmony as input for now (simplified)
        if 'bass' in pattern_map:
            refined_patterns.append(pattern_map['bass'])
    
        if 'rhythm' in pattern_map and self.loop_manager is not None:
            rhythm_loop = next((loop for loop in self.loop_manager.generation_loops
                              if isinstance(loop, RhythmRefinementLoop)), None)
            if rhythm_loop:
                refined_rhythm = rhythm_loop.execute(self.context, pattern_map['rhythm'])
                refined_patterns.append(refined_rhythm)
            else:
                refined_patterns.append(pattern_map['rhythm'])
    
        return refined_patterns
    
    def _generate_standard(self, num_bars: int, beat_complexity: float, chord_complexity: str) -> List[Pattern]:
        """Generate patterns using standard generation (no loops)."""
        patterns = []
    
        # Generate melody pattern using scale-based pitch selection
        melody_pattern = self.melody_generator.generate(num_bars)
        patterns.append(melody_pattern)
    
        # Generate harmony pattern using Roman numeral chord progressions
        genre = self.context.genre_rules.get_genre_name()
        mood = self.context.mood
        harmony_pattern = self.harmony_generator.generate(num_bars, chord_complexity, self.harmonic_variance)
        patterns.append(harmony_pattern)
    
        # Generate bass pattern using chord roots from the progression
        bass_pattern = self.bass_generator.generate(num_bars)
        patterns.append(bass_pattern)
    
        # Generate rhythm pattern with genre-specific timing
        rhythm_pattern = self.rhythm_generator.generate(num_bars, beat_complexity)
        patterns.append(rhythm_pattern)
    
        # Apply advanced timing variations if enabled
        if self.use_advanced_timing and self.timing_engine is not None:
            # Determine genre and mood for timing profiles
            genre = getattr(self.context, 'subgenre', None) or self.context.genre_rules.get_genre_name()
            mood = self.context.mood
    
            # Apply timing variations to each pattern
            for i, pattern in enumerate(patterns):
                # Determine section intensity (simplified - could be enhanced)
                section_intensity = 'medium'  # Default
                if mood == 'energetic':
                    section_intensity = 'high'
                elif mood in ['calm', 'sad']:
                    section_intensity = 'low'
    
                patterns[i] = self.timing_engine.apply_timing_variations(
                    pattern, genre, mood, section_intensity
                )
    
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
            return self.rhythm_generator.generate(num_bars, 0.8) # More complex rhythm for fills
        elif fill_type == 'melodic':
            return self.melody_generator.generate(num_bars) # Simple melody for fills
        else:
            raise ValueError("Invalid fill_type. Must be 'rhythmic' or 'melodic'")
    
    def get_loop_quality_scores(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get quality scores from the most recent loop-based generation.
    
        Returns:
            Dictionary of quality scores by pattern type, or None if no loop generation was performed
        """
        if not self.use_loop_based_generation or self.loop_manager is None:
            return None
    
        # This would need to be implemented to track patterns from the last generation
        # For now, return None as we don't store the last generated patterns
        return None