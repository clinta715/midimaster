"""
Rhythm Generator Module
 
This module contains the RhythmGenerator class responsible for generating
rhythmic patterns based on genre-specific timing and patterns.
"""
import random
import math
from typing import TYPE_CHECKING, Optional, List
import logging

from structures.data_structures import Note, Pattern, PatternType
from generators.generator_utils import get_velocity_for_mood, initialize_key_and_scale, _convert_repo_pattern_to_durations

from rhythm_generator_variations import RhythmVariationEngine

if TYPE_CHECKING:
    from generators.generator_context import GeneratorContext
    from analyzers.reference_pattern_library import ReferencePatternLibrary
    from data_store.pattern_repository import PatternRepository


class RhythmGenerator:
    """Generates rhythm patterns with genre-specific timing.
 
    The RhythmGenerator creates rhythmic foundation using genre-specific rhythm patterns.
    These patterns can be used for percussion or rhythmic accompaniment.
    The rhythm is influenced by both the genre and the selected mood.
    Supports integration with ReferencePatternLibrary for template-based generation via
    generate_with_templates method.
    """
 
    def __init__(self, context: 'GeneratorContext', pattern_strength: float = 1.0, pattern_library: Optional['ReferencePatternLibrary'] = None, pattern_repository: Optional['PatternRepository'] = None):
 
        """
        Initialize the RhythmGenerator.
 
        Args:
            context: Shared GeneratorContext containing music theory and configuration
            pattern_library: Optional ReferencePatternLibrary for template-based generation
        """
        self.context = context
        self.pattern_strength = pattern_strength
        self.pattern_library = pattern_library
        self.pattern_repository = pattern_repository
 
 
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
        # Validate num_bars parameter
        if not isinstance(num_bars, int) or num_bars <= 0:
            raise ValueError(f"num_bars must be a positive integer, got {num_bars} of type {type(num_bars)}")
        # Validate beat complexity parameter
        if not 0.0 <= beat_complexity <= 1.0:
            raise ValueError("beat_complexity must be between 0.0 and 1.0")
 
        # Ensure key and scale are established
        if not self.context.scale_pitches:
            initialize_key_and_scale(self.context)
 
        notes = []
        chords = []
        # Genre validation
        if not hasattr(self.context, 'genre_rules') or self.context.genre_rules is None:
            raise ValueError("No genre rules provided for rhythm generation. Genre must be specified.")
 
        # Determine genre and characteristics
        genre_rules_class = type(self.context.genre_rules).__name__
        genre = genre_rules_class.lower().rstrip('rules')
        char = self.context.genre_rules.get_genre_characteristics(genre)

        # Try repository-based pattern selection first (non-fatal on failure)
        selected_rhythm = None
        pr = getattr(self, "pattern_repository", None)
        if pr is not None:
            try:
                time_signature = "4/4"
                target_bpm = getattr(self.context, 'tempo', char.get('tempo_min', 120.0))
                target_length_beats = 4.0  # prefer single-bar base patterns
                sync_center = char.get('syncopation_level', char.get('syncopation', 0.3))
                sync_range = (max(0.0, sync_center - 0.2), min(1.0, sync_center + 0.2))
                dens_target = getattr(self.context.density_manager, 'rhythm_density', None)
                dens_range = None
                if isinstance(dens_target, (int, float)):
                    dens_range = (max(0.0, float(dens_target) - 0.2), min(1.0, float(dens_target) + 0.2))

                candidates = pr.find_rhythm_patterns(
                    instrument="drums",
                    genre=genre,
                    time_signature=time_signature,
                    mood=getattr(self.context, 'mood', None),
                    subdivision=None,
                    target_length_beats=target_length_beats,
                    target_bpm=target_bpm,
                    syncopation_range=sync_range,
                    density_range=dens_range,
                    min_quality_score=0.0,
                    limit=25,
                )
                if candidates:
                    c = candidates[0]  # repository already orders by quality_score DESC
                    pattern_obj = c.get("pattern")
                    length_beats = float(c.get("length_beats") or target_length_beats)
                    subdivision_val = int(c.get("subdivision") or 16)
                    converted = _convert_repo_pattern_to_durations(pattern_obj, length_beats, subdivision_val)
                    if converted:
                        selected_rhythm = converted
                        print(f"Selected repository rhythm pattern for {genre}")
            except Exception as e:
                logging.debug(f"PatternRepository fetch failed: {e}")

        # If no repository candidate, fall back to genre rules/hardcoded catalogs
        if selected_rhythm is None:
            # Get rhythm patterns from genre rules
            # Fallback to straight eighths if no patterns specified
            rhythm_patterns = self.context.genre_rules.get_rhythm_patterns()
            if not rhythm_patterns:
                # Fallback mechanism for missing rhythm patterns
                print("Warning: No rhythm patterns found for genre. Using straight eighths fallback.")
                rhythm_patterns = [{'pattern': [0.25] * 16}]  # 4 bars of eighth notes

            # Pattern compatibility checking
            compatible_patterns = [p for p in rhythm_patterns if self._is_compatible_pattern(p['pattern'], char)]
            if compatible_patterns:
                selected_rhythm = random.choice(compatible_patterns)['pattern']
                print(f"Selected compatible rhythm pattern for {genre}")
            else:
                # Fallback to any available pattern if no compatible ones found
                selected_rhythm = random.choice(rhythm_patterns)['pattern']
                print(f"Warning: No compatible rhythm pattern found for {genre}. Using fallback pattern.")
 
        # Generate rhythm notes
        # 4 beats per bar is the default time signature
        current_time = 0.0
        total_beats = num_bars * 4.0
        pattern_length = len(selected_rhythm)
        note_index = 0
        base_velocity = get_velocity_for_mood(self.context.mood)
        base_velocity = max(40, min(127, base_velocity * self.pattern_strength))
        emphasis_patterns = self.context.genre_rules.get_beat_characteristics().get('emphasis_patterns', [])
        probability = self.context.density_manager.calculate_note_probability()
 
        while current_time < total_beats:
            duration = selected_rhythm[note_index % pattern_length]
 
            # Percussion-like notes (could be mapped to drum sounds in MIDI)
            # Use pitches that are more likely to be in scale for melodic percussion
            if self.context.scale_pitches and note_index % 3 == 0:  # Every few hits, use a scale tone
                pitch = random.choice([p for p in self.context.scale_pitches if p > 60])
            else:
                # Simple mapping to percussion sounds (MIDI pitches 35- are common percussion)
                pitch = 35 + (note_index % 10)
 
            # Adjust velocity based on mood and metric position
            beat_position = current_time % 4
            beat_in_bar = math.floor(beat_position) + 1
 
            if beat_in_bar in emphasis_patterns:
                velocity = min(127, base_velocity + 25) # Strong emphasis
            elif beat_in_bar % 2 == 1: # Other odd beats (1, 3)
                velocity = min(127, base_velocity + 10) # Medium emphasis
            else: # Even beats (2, 4)
                velocity = base_velocity
 
            # Add a small random variation for humanization
            velocity = max(0, min(127, velocity + random.randint(-7, 7)))
 
            is_emphasis_beat = math.isclose(current_time % 1, 0, abs_tol=1e-6) and (beat_in_bar in emphasis_patterns)
 
            # Only place note based on rhythm density probability
            if is_emphasis_beat or probability > random.random():
                note = Note(pitch, duration, int(velocity), current_time)
                notes.append(note)
 
            # Advance the current time
            current_time += duration
            note_index += 1
 
        # Set genre-specific properties
        char = self.context.genre_rules.get_genre_characteristics(genre)
        tempo = getattr(self.context, 'tempo', char.get('tempo_min', 120.0))
 
        # Fine-tune syncopation for pop and rock to match expected levels
        target_sync = char.get('syncopation_level', char.get('syncopation', 0.0))
        if genre in ['pop', 'rock']:
            notes = self._adjust_syncopation(notes, target_sync, genre)
 
        return Pattern(PatternType.RHYTHM, notes, chords, tempo=tempo, swing_factor=char.get('swing_factor', char.get('swing', 0.0)), syncopation_level=char.get('syncopation_level', char.get('syncopation', 0.0)))
 
    def _adjust_syncopation(self, notes: List[Note], target_sync: float, genre: str = 'pop') -> List[Note]:
        """Adjust the syncopation level by selectively removing off-beat notes for low-sync genres."""
        if not notes:
            return notes
 
        off_notes = [n for n in notes if n.start_time % 1 != 0]
        on_notes = [n for n in notes if n.start_time % 1 == 0]
        current_ratio = len(off_notes) / len(notes)
 
        # If already within tolerance, no adjustment needed
        if current_ratio <= target_sync + 0.2:
            return notes
 
        # Calculate number of off-beat notes to remove to reach target
        desired_off = int(len(notes) * target_sync)
        to_remove = len(off_notes) - desired_off
 
        if to_remove <= 0:
            return notes
 
        # Randomly select off-beat notes to remove
        remove_indices = random.sample(range(len(off_notes)), min(to_remove, len(off_notes)))
        kept_off = [off_notes[i] for i in range(len(off_notes)) if i not in set(remove_indices)]
 
        adjusted_notes = on_notes + kept_off
        adjusted_notes.sort(key=lambda n: n.start_time)
        return adjusted_notes
 
    def _is_compatible_pattern(self, pattern: List[float], genre_char: dict) -> bool:
        """Check if the rhythm pattern is compatible with genre characteristics based on syncopation."""
        if not pattern:
            return False
        # Count off-beat positions (non-integer beat positions)
        off_beats = sum(1 for t in pattern if not math.isclose(t % 1, 0, abs_tol=0.01))
        sync_ratio = off_beats / len(pattern)
        target_sync = genre_char.get('syncopation_level', genre_char.get('syncopation', 0.3))
        tolerance = 0.2
        is_compatible = abs(sync_ratio - target_sync) < tolerance
        genre_name = self.context.genre_rules.__class__.__name__.lower().rstrip('rules')
        if not is_compatible:
            logging.warning(f"Rejected {genre_name} pattern: syncopation ratio {sync_ratio:.2f} too far from target {target_sync:.2f} (tolerance {tolerance})")
        return is_compatible
    def generate_with_templates(self, num_bars: int, beat_complexity: float = 0.5, variation_level: float = 0.2) -> Pattern:
        """Generate a rhythm pattern using reference templates from the pattern library.
 
        Selects professional drum patterns as templates based on genre and complexity.
        Applies mood-based velocity scaling and variation to create unique but authentic rhythms.
        Falls back to standard generation if no suitable templates are available.
 
        Args:
            num_bars: Number of bars to generate
            beat_complexity: Target complexity level (0.0-1.0, influences pattern selection)
            variation_level: Amount of variation to apply to timing and velocity (0.0-1.0)
 
        Returns:
            Pattern object with template-based rhythm notes
 
        Raises:
            ValueError: If parameters are out of valid range
        """
        # Validate parameters
        if not 0.0 <= beat_complexity <= 1.0:
            raise ValueError("beat_complexity must be between 0.0 and 1.0")
        if not 0.0 <= variation_level <= 1.0:
            raise ValueError("variation_level must be between 0.0 and 1.0")
 
        # Fallback if no pattern library provided
        if self.pattern_library is None:
            return self.generate(num_bars, beat_complexity)
 
        # Determine genre from genre_rules class name
        genre_rules_class = type(self.context.genre_rules).__name__
        genre = genre_rules_class.lower().rstrip('rules')
        # Get genre characteristics for pattern parameters
        char = self.context.genre_rules.get_genre_characteristics(genre)
        tempo = getattr(self.context, 'tempo', char.get('tempo_min', 120.0))
 
        # Map beat_complexity to expected notes per beat (0.5 -> ~1.0 npb, 1.0 -> ~3.0 npb)
        target_complexity = beat_complexity * 2.5 + 0.5
        min_complexity = max(0.0, target_complexity - 0.5)
        max_complexity = target_complexity + 0.5
 
        # Retrieve suitable drum patterns
        patterns = self.pattern_library.get_patterns(
            genre=genre,
            instrument='drums',
            min_complexity=min_complexity,
            max_complexity=max_complexity
        )
 
        # Fallback if no matching patterns found
        if not patterns:
            return self.generate(num_bars, beat_complexity)
 
        # Select a random template
        selected = random.choice(patterns)
 
        # Extract metadata (fallback if not available)
        try:
            meta = self.pattern_library.get_metadata(selected)
        except (KeyError, AttributeError):
            meta = self.pattern_library.extract_metadata(selected)
 
        # Convert ticks to beats
        ticks_per_beat = getattr(selected, 'ticks_per_beat', 480)  # Default PPQ
        pattern_length_beats = getattr(selected, 'length_ticks', ticks_per_beat * 4) / ticks_per_beat
 
        # Mood-based velocity scaling
        mood_velocity = get_velocity_for_mood(self.context.mood)
        original_velocities = [event.velocity for event in getattr(selected, 'notes', [])]
        avg_original_vel = sum(original_velocities) / len(original_velocities) if original_velocities else 64.0
        velocity_scale = mood_velocity / avg_original_vel
 
        # Prepare notes list
        notes_list = []
        beats_per_bar = 4  # Assuming 4/4 time
        total_beats = num_bars * beats_per_bar
        num_repeats = math.ceil(total_beats / pattern_length_beats)
 
        for repeat in range(num_repeats):
            repeat_offset = repeat * pattern_length_beats
            for event in getattr(selected, 'notes', []):
                # Assume NoteEvent has time, duration, note, velocity in ticks
                start_beats = repeat_offset + (getattr(event, 'time', 0) / ticks_per_beat)
                duration_beats = getattr(event, 'duration', ticks_per_beat / 4) / ticks_per_beat  # Default quarter note
 
                if start_beats + duration_beats > total_beats:
                    continue
 
                pitch = getattr(event, 'note', 60)  # Default pitch
                vel = max(1, min(127, int(getattr(event, 'velocity', 64) * velocity_scale)))
 
                # Apply pattern_strength scaling to template velocities
                vel = max(1, min(127, int(vel * self.pattern_strength)))
                print(f"RhythmGenerator: Template velocity scaled - original: {getattr(event, 'velocity', 64)}, after mood_scale: {int(getattr(event, 'velocity', 64) * velocity_scale)}, after pattern_strength {self.pattern_strength}: {vel}")
 
                # Apply variation for uniqueness
                if variation_level > 0.0:
                    time_offset = random.uniform(-variation_level * 0.125, variation_level * 0.125) * duration_beats
                    start_beats += time_offset
                    vel_offset = random.randint(-int(variation_level * 20), int(variation_level * 20))
                    vel = max(1, min(127, vel + vel_offset))
 
                # Clamp start time
                start_beats = max(0.0, start_beats)
 
                note = Note(pitch, duration_beats, vel, start_beats)
                notes_list.append(note)
 
        # Sort notes by start time
        notes_list.sort(key=lambda n: n.start_time)
 
        # Fine-tune syncopation for pop and rock to match expected levels
        target_sync = char.get('syncopation_level', char.get('syncopation', 0.0))
        if genre in ['pop', 'rock']:
            notes_list = self._adjust_syncopation(notes_list, target_sync, genre)
 
        # Return the pattern
        return Pattern(PatternType.RHYTHM, notes_list, [], tempo=tempo, swing_factor=char.get('swing_factor', char.get('swing', 0.0)), syncopation_level=char.get('syncopation_level', char.get('syncopation', 0.0)))