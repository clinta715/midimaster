"""
Melody Refinement Loop Module

This module contains the MelodyRefinementLoop class which implements iterative
refinement algorithms for melody generation and improvement.
"""

from typing import List, Optional, Any
import random
import copy

from structures.data_structures import Pattern, PatternType, Note
from generators.loop_manager import GenerationLoop
from music_theory import MusicTheory


class MelodyRefinementLoop(GenerationLoop):
    """Refines melody through iterative generation and evaluation.

    This loop implements sophisticated melody refinement algorithms including:
    - Contour analysis and improvement
    - Harmonic consistency checking
    - Motivic development
    - Phrasing optimization
    """

    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.05):
        """Initialize the melody refinement loop.

        Args:
            max_iterations: Maximum number of refinement iterations
            convergence_threshold: Quality improvement threshold for convergence
        """
        super().__init__(max_iterations, convergence_threshold)
        self.music_theory = MusicTheory()

    def execute(self, context: Any, input_pattern: Optional[Pattern]) -> Pattern:
        """Generate and refine melody pattern.

        Args:
            context: Generator context containing shared state
            input_pattern: Input melody pattern to refine (None for first generation)

        Returns:
            Pattern: Refined melody pattern
        """
        # Generate initial melody if none provided
        if input_pattern is None:
            current_pattern = self._generate_initial_melody(context)
        else:
            current_pattern = copy.deepcopy(input_pattern)

        best_pattern = current_pattern
        best_score = self.evaluate_quality(current_pattern)

        # Iterative refinement
        for iteration in range(self.max_iterations):
            # Generate melody variations
            candidates = self._generate_melody_variations(context, best_pattern)

            # Find best candidate
            for candidate in candidates:
                score = self.evaluate_quality(candidate)
                if score > best_score:
                    best_pattern = candidate
                    best_score = score

            # Check convergence
            if not self.should_continue(iteration, best_score, best_score):
                break

        return best_pattern

    def evaluate_quality(self, pattern: Pattern) -> float:
        """Evaluate the quality of a melody pattern.

        Args:
            pattern: Melody pattern to evaluate

        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if pattern.pattern_type != PatternType.MELODY:
            return 0.0

        scores = []

        # Contour smoothness (weight: 0.3)
        contour_score = self._evaluate_contour_smoothness(pattern)
        scores.append(contour_score * 0.3)

        # Harmonic consistency (weight: 0.3)
        harmony_score = self._evaluate_harmonic_consistency(pattern)
        scores.append(harmony_score * 0.3)

        # Rhythmic variety (weight: 0.2)
        rhythm_score = self._evaluate_rhythmic_variety(pattern)
        scores.append(rhythm_score * 0.2)

        # Motivic development (weight: 0.2)
        motive_score = self._evaluate_motivic_development(pattern)
        scores.append(motive_score * 0.2)

        return sum(scores)

    def _generate_initial_melody(self, context: Any) -> Pattern:
        """Generate an initial melody pattern.

        Args:
            context: Generator context

        Returns:
            Pattern: Initial melody pattern
        """
        # Use existing melody generator to create initial pattern
        try:
            from generators.melody_generator import MelodyGenerator
            generator = MelodyGenerator(context)
            return generator.generate(num_bars=4)  # Generate 4 bars as default
        except ImportError:
            # Fallback: create a simple melody pattern
            return self._create_simple_melody(context)

    def _create_simple_melody(self, context: Any) -> Pattern:
        """Create a simple fallback melody pattern.

        Args:
            context: Generator context

        Returns:
            Pattern: Simple melody pattern
        """
        notes = []
        current_time = 0.0
        base_pitch = 60  # Middle C
        scale = context.scale_pitches if hasattr(context, 'scale_pitches') else [60, 62, 64, 65, 67, 69, 71]

        # Generate 16 notes (4 bars of 4 notes each)
        for i in range(16):
            # Select pitch from scale
            pitch_idx = random.randint(0, len(scale) - 1)
            pitch = scale[pitch_idx]

            # Vary duration slightly
            duration = random.choice([0.5, 1.0, 1.5, 2.0])

            # Vary velocity
            velocity = random.randint(60, 100)

            note = Note(pitch, duration, velocity, current_time)
            notes.append(note)
            current_time += duration

        return Pattern(PatternType.MELODY, notes, [])

    def _generate_melody_variations(self, context: Any, base_pattern: Pattern) -> List[Pattern]:
        """Generate variations of a melody pattern.

        Args:
            context: Generator context
            base_pattern: Base melody pattern to vary

        Returns:
            List[Pattern]: List of melody variations
        """
        variations = []

        # Create multiple variations
        for _ in range(3):
            variation = copy.deepcopy(base_pattern)

            # Apply random refinements
            refinement_type = random.choice(['contour', 'rhythm', 'register', 'ornament'])

            if refinement_type == 'contour':
                variation = self._refine_contour(variation)
            elif refinement_type == 'rhythm':
                variation = self._refine_rhythm(variation)
            elif refinement_type == 'register':
                variation = self._refine_register(variation)
            elif refinement_type == 'ornament':
                variation = self._refine_ornamentation(variation)

            variations.append(variation)

        return variations

    def _refine_contour(self, pattern: Pattern) -> Pattern:
        """Refine melody contour for better smoothness.

        Args:
            pattern: Pattern to refine

        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)

        for i in range(1, len(refined_pattern.notes) - 1):
            current_note = refined_pattern.notes[i]
            prev_note = refined_pattern.notes[i-1]
            next_note = refined_pattern.notes[i+1]

            # Calculate pitch differences
            prev_diff = abs(current_note.pitch - prev_note.pitch)
            next_diff = abs(next_note.pitch - current_note.pitch)

            # If both differences are large, smooth the transition
            if prev_diff > 4 and next_diff > 4:
                # Choose a pitch that's intermediate between prev and next
                avg_pitch = (prev_note.pitch + next_note.pitch) // 2
                current_note.pitch = avg_pitch

        return refined_pattern

    def _refine_rhythm(self, pattern: Pattern) -> Pattern:
        """Refine rhythmic variety in the melody.

        Args:
            pattern: Pattern to refine

        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)

        # Add some rhythmic variation
        for i, note in enumerate(refined_pattern.notes):
            if random.random() < 0.3:  # 30% chance to modify
                # Slightly vary duration
                variation = random.uniform(0.8, 1.2)
                note.duration *= variation

                # Adjust start time to maintain continuity
                if i > 0:
                    note.start_time = refined_pattern.notes[i-1].start_time + refined_pattern.notes[i-1].duration

        return refined_pattern

    def _refine_register(self, pattern: Pattern) -> Pattern:
        """Refine pitch register for better range.

        Args:
            pattern: Pattern to refine

        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)

        pitches = [note.pitch for note in refined_pattern.notes]
        current_range = max(pitches) - min(pitches)

        # If range is too narrow, expand it
        if current_range < 12:  # Less than an octave
            # Shift some notes up or down by an octave
            for note in refined_pattern.notes:
                if random.random() < 0.4:  # 40% chance
                    if random.random() < 0.5:
                        note.pitch += 12  # Up an octave
                    else:
                        note.pitch -= 12  # Down an octave

                    # Keep within MIDI range
                    note.pitch = max(21, min(108, note.pitch))

        return refined_pattern

    def _refine_ornamentation(self, pattern: Pattern) -> Pattern:
        """Add subtle ornamentation to the melody.

        Args:
            pattern: Pattern to refine

        Returns:
            Pattern: Refined pattern with ornamentation
        """
        refined_pattern = copy.deepcopy(pattern)
        new_notes = []

        for note in refined_pattern.notes:
            new_notes.append(note)

            # Occasionally add grace notes
            if random.random() < 0.2 and note.duration > 1.0:  # 20% chance for longer notes
                grace_duration = min(0.1, note.duration * 0.1)

                # Create grace note a step above
                grace_pitch = note.pitch + 1
                grace_note = Note(grace_pitch, grace_duration, note.velocity - 10, note.start_time)

                new_notes.append(grace_note)

                # Adjust main note start time
                note.start_time += grace_duration

        refined_pattern.notes = new_notes
        return refined_pattern

    def _evaluate_contour_smoothness(self, pattern: Pattern) -> float:
        """Evaluate melody contour smoothness.

        Args:
            pattern: Pattern to evaluate

        Returns:
            float: Smoothness score (0.0-1.0)
        """
        if len(pattern.notes) < 2:
            return 0.5

        total_transitions = 0
        smooth_transitions = 0

        for i in range(1, len(pattern.notes)):
            pitch_diff = abs(pattern.notes[i].pitch - pattern.notes[i-1].pitch)
            total_transitions += 1

            # Consider transitions of 3 semitones or less as smooth
            if pitch_diff <= 3:
                smooth_transitions += 1

        return smooth_transitions / total_transitions if total_transitions > 0 else 0.5

    def _evaluate_harmonic_consistency(self, pattern: Pattern) -> float:
        """Evaluate how well the melody fits with harmony.

        Args:
            pattern: Pattern to evaluate

        Returns:
            float: Consistency score (0.0-1.0)
        """
        # For now, return a moderate score
        # In a full implementation, this would check against chord progressions
        return 0.7

    def _evaluate_rhythmic_variety(self, pattern: Pattern) -> float:
        """Evaluate rhythmic variety in the melody.

        Args:
            pattern: Pattern to evaluate

        Returns:
            float: Variety score (0.0-1.0)
        """
        if len(pattern.notes) < 2:
            return 0.5

        durations = [note.duration for note in pattern.notes]
        unique_durations = len(set(durations))

        # More unique durations = more variety, but cap at reasonable maximum
        variety_score = min(unique_durations / 6.0, 1.0)

        return variety_score

    def _evaluate_motivic_development(self, pattern: Pattern) -> float:
        """Evaluate motivic development and repetition.

        Args:
            pattern: Pattern to evaluate

        Returns:
            float: Development score (0.0-1.0)
        """
        # Simple implementation - check for some repetition patterns
        if len(pattern.notes) < 8:
            return 0.5

        # Look for repeated pitch patterns
        pitch_sequence = [note.pitch for note in pattern.notes]
        has_repetition = False

        for i in range(len(pitch_sequence) - 4):
            motif = pitch_sequence[i:i+4]
            for j in range(i + 4, len(pitch_sequence) - 4):
                if pitch_sequence[j:j+4] == motif:
                    has_repetition = True
                    break
            if has_repetition:
                break

        return 0.8 if has_repetition else 0.4