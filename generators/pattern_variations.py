"""
Pattern variation algorithms for generating unique but authentic patterns.

This module provides the PatternVariationEngine class, which implements various
algorithms to create variations of musical patterns while preserving their
essential musical characteristics. The engine works with the existing Pattern
data structure and can apply transformations like time stretching, velocity
randomization, pattern recombination, complexity scaling, pitch transposition,
rhythmic density adjustment, and ornamentation addition.
"""

from typing import List, Optional, Tuple
from enum import Enum
import random
import copy
import math

from structures.data_structures import Note, Pattern, PatternType, Chord


class VariationType(Enum):
    """Types of pattern variations supported by the engine."""
    TIME_STRETCH = "time_stretch"
    VELOCITY_RANDOMIZATION = "velocity_randomization"
    RECOMBINATION = "recombination"
    COMPLEXITY_SCALING = "complexity_scaling"
    PITCH_TRANSPOSITION = "pitch_transposition"
    RHYTHMIC_DENSITY = "rhythmic_density"
    ORNAMENTATION = "ornamentation"


class PatternVariationEngine:
    """
    Engine for applying variations to musical patterns.

    This class provides methods to generate unique variations of Pattern objects
    while maintaining their musical authenticity. Each variation method accepts
    an intensity parameter (0.0 to 1.0) to control the strength of the transformation.

    Usage:
        engine = PatternVariationEngine()
        varied_pattern = engine.time_stretch(original_pattern, stretch_factor=1.2, intensity=0.8)
    """

    def __init__(self):
        """
        Initialize the PatternVariationEngine.

        No parameters required; uses random seed for reproducibility if needed.
        """
        random.seed(42)  # For reproducible variations; can be overridden

    def _copy_pattern(self, pattern: Pattern) -> Pattern:
        """
        Create a deep copy of a Pattern.

        Args:
            pattern: The original Pattern to copy.

        Returns:
            A deep copy of the pattern.
        """
        return copy.deepcopy(pattern)

    def _get_all_notes(self, pattern: Pattern) -> List[Note]:
        """
        Extract all notes from a pattern, including those in chords.

        Args:
            pattern: The Pattern to extract notes from.

        Returns:
            List of all Note objects in the pattern.
        """
        all_notes = pattern.notes.copy()
        for chord in pattern.chords:
            all_notes.extend(chord.notes)
        return all_notes

    def _update_pattern_notes(self, pattern: Pattern, notes: List[Note]):
        """
        Update the notes in a pattern from a list, distributing to notes and chords.

        Args:
            pattern: The Pattern to update.
            notes: The new list of notes.
        """
        # Simple distribution: assign to pattern.notes, clear chords for simplicity
        # In a full impl, would need to reconstruct chords based on simultaneous notes
        pattern.notes = notes
        pattern.chords = []  # Flatten to notes for variations; chords can be rebuilt if needed

    def time_stretch(self, pattern: Pattern, factor: float, intensity: float = 1.0) -> Pattern:
        """
        Apply time stretching to maintain groove while changing tempo.

        Scales start times and durations by a factor, adjusted by intensity.
        Preserves the relative rhythmic structure.

        Args:
            pattern: The original Pattern.
            factor: Stretch factor (>1 slows down, <1 speeds up).
            intensity: Strength of the variation (0.0-1.0).

        Returns:
            A new stretched Pattern.
        """
        if not 0 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")

        new_pattern = self._copy_pattern(pattern)
        scale = 1.0 + (factor - 1.0) * intensity

        all_notes = self._get_all_notes(new_pattern)
        for note in all_notes:
            note.start_time *= scale
            note.duration *= scale

        # Update chords' start times
        for chord in new_pattern.chords:
            chord.start_time *= scale

        return new_pattern

    def velocity_randomization(self, pattern: Pattern, std_dev: float = 20, intensity: float = 1.0) -> Pattern:
        """
        Randomize velocities using a professional reference curve (Gaussian noise).

        Applies Gaussian noise to velocities, clamped to 0-127, simulating human variation.
        Higher velocities get more variation to mimic dynamic playing.

        Args:
            pattern: The original Pattern.
            std_dev: Standard deviation of the noise (default 20).
            intensity: Strength of randomization (0.0-1.0).

        Returns:
            A new Pattern with varied velocities.
        """
        if not 0 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")

        new_pattern = self._copy_pattern(pattern)
        all_notes = self._get_all_notes(new_pattern)
        adjusted_std = std_dev * intensity

        for note in all_notes:
            if note.velocity > 0:  # Only vary existing notes
                noise = random.gauss(0, adjusted_std)
                new_velocity = note.velocity + noise
                # Apply curve: more variation for louder notes
                if note.velocity > 64:
                    new_velocity += random.uniform(-5, 5) * intensity
                note.velocity = max(0, min(127, int(new_velocity)))

        return new_pattern

    def recombine_patterns(self, base_pattern: Pattern, templates: List[Pattern], intensity: float = 1.0) -> Pattern:
        """
        Recombine elements from multiple template patterns into the base.

        Randomly selects and interleaves notes from templates into the base pattern,
        weighted by intensity.

        Args:
            base_pattern: The base Pattern to modify.
            templates: List of template Patterns to recombine from.
            intensity: Proportion of base notes to replace (0.0-1.0).

        Returns:
            A recombined Pattern.
        """
        if not 0 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")

        new_pattern = self._copy_pattern(base_pattern)
        all_base_notes = self._get_all_notes(new_pattern)

        if not all_base_notes:
            return new_pattern

        num_to_replace = int(len(all_base_notes) * intensity)
        replace_indices = random.sample(range(len(all_base_notes)), min(num_to_replace, len(all_base_notes)))

        # Collect all template notes
        template_notes = []
        for temp in templates:
            template_notes.extend(self._get_all_notes(temp))

        if not template_notes:
            return new_pattern

        for idx in replace_indices:
            # Replace with a random template note, adjust start_time to match
            replacement = random.choice(template_notes)
            all_base_notes[idx].pitch = replacement.pitch
            all_base_notes[idx].duration = replacement.duration
            all_base_notes[idx].velocity = replacement.velocity
            all_base_notes[idx].start_time = all_base_notes[idx].start_time  # Preserve timing

        self._update_pattern_notes(new_pattern, all_base_notes)
        return new_pattern

    def complexity_scale(self, pattern: Pattern, target_density: float, intensity: float = 1.0) -> Pattern:
        """
        Scale the complexity of a pattern (simplify or add notes).

        For density <1, remove notes; for >1, subdivide durations and add notes.
        Density is notes per beat.

        Args:
            pattern: The original Pattern.
            target_density: Target notes per beat (current density calculated internally).
            intensity: How aggressively to apply changes (0.0-1.0).

        Returns:
            A Pattern with scaled complexity.
        """
        if not 0 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")

        new_pattern = self._copy_pattern(pattern)
        all_notes = self._get_all_notes(new_pattern)

        if not all_notes:
            return new_pattern

        # Calculate current total duration and density
        total_duration = max(note.start_time + note.duration for note in all_notes)
        current_density = len(all_notes) / total_duration if total_duration > 0 else 1.0
        density_ratio = target_density / current_density

        adjusted_ratio = 1.0 + (density_ratio - 1.0) * intensity

        if adjusted_ratio < 1.0:
            # Simplify: remove notes randomly
            num_to_remove = int(len(all_notes) * (1 - adjusted_ratio))
            remove_indices = random.sample(range(len(all_notes)), num_to_remove)
            for idx in sorted(remove_indices, reverse=True):
                del all_notes[idx]
        else:
            # Add complexity: subdivide some notes
            num_to_add = int(len(all_notes) * (adjusted_ratio - 1))
            for _ in range(num_to_add):
                # Pick a random note to subdivide
                base_note = random.choice(all_notes)
                if base_note.duration > 0.125:  # Only subdivide if long enough
                    sub_duration = base_note.duration / 2
                    sub_note = Note(
                        pitch=random.choice([base_note.pitch, base_note.pitch + random.choice([-1, 1])]),
                        duration=sub_duration,
                        velocity=int(base_note.velocity * random.uniform(0.5, 0.8)),
                        start_time=base_note.start_time + random.uniform(0, base_note.duration - sub_duration),
                        channel=base_note.channel
                    )
                    all_notes.append(sub_note)

        self._update_pattern_notes(new_pattern, all_notes)
        return new_pattern

    def pitch_transpose(self, pattern: Pattern, semitones: int, intensity: float = 1.0) -> Pattern:
        """
        Transpose pitches while maintaining harmonic relationships.

        Shifts all pitches by semitones, preserving intervals in chords/melodies.

        Args:
            pattern: The original Pattern.
            semitones: Number of semitones to transpose (+ up, - down).
            intensity: Partial transposition (e.g., 0.5 transposes half the notes).

        Returns:
            A transposed Pattern.
        """
        if not 0 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")

        new_pattern = self._copy_pattern(pattern)
        all_notes = self._get_all_notes(new_pattern)

        num_to_transpose = int(len(all_notes) * intensity)
        transpose_indices = random.sample(range(len(all_notes)), num_to_transpose) if intensity < 1.0 else range(len(all_notes))

        for idx in transpose_indices:
            note = all_notes[idx]
            new_pitch = note.pitch + semitones
            new_pitch = max(0, min(127, new_pitch))  # Clamp to MIDI range
            all_notes[idx].pitch = new_pitch

        self._update_pattern_notes(new_pattern, all_notes)
        return new_pattern

    def rhythmic_density_adjust(self, pattern: Pattern, density_factor: float, intensity: float = 1.0) -> Pattern:
        """
        Adjust rhythmic density while preserving characteristic accents.

        Increases/decreases note density around strong beats (high velocity), preserving accents.

        Args:
            pattern: The original Pattern.
            density_factor: Factor for density change (>1 increase, <1 decrease).
            intensity: Strength of adjustment (0.0-1.0).

        Returns:
            A Pattern with adjusted density.
        """
        if not 0 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")

        new_pattern = self._copy_pattern(pattern)
        all_notes = self._get_all_notes(new_pattern)

        if not all_notes:
            return new_pattern

        # Identify accent notes (high velocity)
        accents = [n for n in all_notes if n.velocity > 80]
        non_accents = [n for n in all_notes if n.velocity <= 80]

        adjusted_factor = 1.0 + (density_factor - 1.0) * intensity

        if adjusted_factor < 1.0:
            # Reduce density, but preserve more accents
            keep_ratio = adjusted_factor
            num_keep_non = int(len(non_accents) * keep_ratio * 0.7)  # Reduce non-accents more
            num_keep_acc = int(len(accents) * keep_ratio * 0.9)     # Preserve accents more
            keep_non_indices = random.sample(range(len(non_accents)), num_keep_non)
            keep_acc_indices = random.sample(range(len(accents)), num_keep_acc)

            kept_notes = [accents[i] for i in keep_acc_indices] + [non_accents[i] for i in keep_non_indices]
        else:
            # Increase density, add around non-accents but duplicate accents carefully
            kept_notes = all_notes.copy()
            num_to_add = int(len(all_notes) * (adjusted_factor - 1))
            for _ in range(num_to_add):
                if random.random() < 0.3 and accents:  # 30% chance to duplicate accent
                    base = random.choice(accents)
                else:
                    base = random.choice(non_accents) if non_accents else random.choice(all_notes)
                # Add subdivided note
                if base.duration > 0.25:
                    add_note = Note(
                        pitch=base.pitch,
                        duration=base.duration / random.randint(2, 4),
                        velocity=int(base.velocity * random.uniform(0.6, 0.9)),
                        start_time=base.start_time + random.uniform(0, base.duration * 0.8),
                        channel=base.channel
                    )
                    kept_notes.append(add_note)

        # Sort by start_time
        kept_notes.sort(key=lambda n: n.start_time)
        self._update_pattern_notes(new_pattern, kept_notes)
        return new_pattern

    def add_ornamentation(self, pattern: Pattern, ornament_type: str = "ghost", intensity: float = 1.0) -> Pattern:
        """
        Add ornamentation like ghost notes or grace notes.

        Ghost notes: low-velocity short notes between main notes.
        Grace notes: quick notes before main notes.

        Args:
            pattern: The original Pattern.
            ornament_type: "ghost" or "grace".
            intensity: Number/probability of ornaments (0.0-1.0).

        Returns:
            A Pattern with added ornaments.
        """
        if not 0 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0.0 and 1.0")

        new_pattern = self._copy_pattern(pattern)
        all_notes = self._get_all_notes(new_pattern)

        num_ornaments = int(len(all_notes) * intensity * 2)  # Up to 2 per note

        for _ in range(num_ornaments):
            if not all_notes:
                break
            main_note = random.choice(all_notes)
            if ornament_type == "ghost":
                # Ghost note: low vel, short, between notes
                orn_duration = random.uniform(0.0625, 0.125)  # 1/16 to 1/8 beat
                orn_time = random.uniform(main_note.start_time, main_note.start_time + main_note.duration - orn_duration)
                orn_pitch = main_note.pitch + random.choice([-2, -1, 0, 1, 2])
                orn_vel = random.randint(20, 40)
            elif ornament_type == "grace":
                # Grace note: very short before main
                orn_duration = random.uniform(0.03125, 0.0625)  # 1/32 to 1/16
                orn_time = main_note.start_time - orn_duration
                orn_pitch = main_note.pitch + random.choice([-1, 1, 2, -2])
                orn_vel = random.randint(60, 90)
            else:
                continue

            orn_pitch = max(0, min(127, orn_pitch))
            ornament = Note(pitch=orn_pitch, duration=orn_duration, velocity=orn_vel,
                            start_time=orn_time, channel=main_note.channel)
            all_notes.append(ornament)

        all_notes.sort(key=lambda n: n.start_time)
        self._update_pattern_notes(new_pattern, all_notes)
        return new_pattern

    def apply_variation(self, pattern: Pattern, variation_type: VariationType,
                        params: dict, intensity: float = 1.0) -> Pattern:
        """
        Apply a specific variation type using parameters.

        Args:
            pattern: The original Pattern.
            variation_type: Type of variation to apply.
            params: Dictionary of parameters for the variation.
            intensity: Strength (0.0-1.0).

        Returns:
            The varied Pattern.
        """
        method_map = {
            VariationType.TIME_STRETCH: self.time_stretch,
            VariationType.VELOCITY_RANDOMIZATION: self.velocity_randomization,
            VariationType.RECOMBINATION: self.recombine_patterns,
            VariationType.COMPLEXITY_SCALING: self.complexity_scale,
            VariationType.PITCH_TRANSPOSITION: self.pitch_transpose,
            VariationType.RHYTHMIC_DENSITY: self.rhythmic_density_adjust,
            VariationType.ORNAMENTATION: self.add_ornamentation,
        }

        if variation_type not in method_map:
            raise ValueError(f"Unsupported variation type: {variation_type}")

        method = method_map[variation_type]
        # Handle varying arg counts
        if variation_type == VariationType.RECOMBINATION:
            templates = params.get('templates', [])
            return method(pattern, templates, intensity)
        elif variation_type == VariationType.ORNAMENTATION:
            orn_type = params.get('ornament_type', 'ghost')
            return method(pattern, orn_type, intensity)
        else:
            # Assume first param is the main one
            main_param = list(params.values())[0] if params else 1.0
            return method(pattern, main_param, intensity)


# Example usage
if __name__ == "__main__":
    # Create a simple example pattern
    notes = [
        Note(pitch=60, duration=1.0, velocity=80, start_time=0.0),
        Note(pitch=62, duration=0.5, velocity=70, start_time=1.0),
        Note(pitch=64, duration=0.5, velocity=90, start_time=1.5),
    ]
    example_pattern = Pattern(PatternType.MELODY, notes, [])

    engine = PatternVariationEngine()

    # Example: Time stretch
    stretched = engine.time_stretch(example_pattern, factor=1.2, intensity=0.8)
    print("Time Stretched Pattern:", stretched)

    # Example: Velocity randomization
    randomized = engine.velocity_randomization(example_pattern, std_dev=15, intensity=0.6)
    print("Velocity Randomized Pattern:", randomized)

    # Example: Pitch transposition
    transposed = engine.pitch_transpose(example_pattern, semitones=2, intensity=1.0)
    print("Transposed Pattern:", transposed)

    # For recombination, need multiple patterns
    # template1 = ... (create another)
    # recombined = engine.recombine_patterns(example_pattern, [template1], 0.5)

    # For ornamentation
    ornamented = engine.add_ornamentation(example_pattern, "ghost", 0.7)
    print("Ornamented Pattern:", len(ornamented.notes), "notes")