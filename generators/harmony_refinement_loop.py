"""
Harmony Refinement Loop Module

This module contains the HarmonyRefinementLoop class which implements iterative
refinement algorithms for harmony generation and improvement.
"""

from typing import List, Optional, Any
import random
import copy

from structures.data_structures import Pattern, PatternType, Note, Chord
from generators.loop_manager import GenerationLoop
from music_theory import MusicTheory


class HarmonyRefinementLoop(GenerationLoop):
    """Refines harmony through iterative generation and evaluation.

    This loop implements sophisticated harmony refinement algorithms including:
    - Voice leading optimization
    - Tension/release analysis
    - Chord progression smoothing
    - Harmonic rhythm refinement
    """

    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.05):
        """Initialize the harmony refinement loop.

        Args:
            max_iterations: Maximum number of refinement iterations
            convergence_threshold: Quality improvement threshold for convergence
        """
        super().__init__(max_iterations, convergence_threshold)
        self.music_theory = MusicTheory()

    def execute(self, context: Any, input_pattern: Optional[Pattern]) -> Pattern:
        """Generate and refine harmony pattern.

        Args:
            context: Generator context containing shared state
            input_pattern: Input harmony pattern to refine (None for first generation)

        Returns:
            Pattern: Refined harmony pattern
        """
        # Generate initial harmony if none provided
        if input_pattern is None:
            current_pattern = self._generate_initial_harmony(context)
        else:
            current_pattern = copy.deepcopy(input_pattern)

        best_pattern = current_pattern
        best_score = self.evaluate_quality(current_pattern)

        # Iterative refinement
        for iteration in range(self.max_iterations):
            # Generate harmony variations
            candidates = self._generate_harmony_variations(context, best_pattern)

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
        """Evaluate the quality of a harmony pattern.

        Args:
            pattern: Harmony pattern to evaluate

        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if pattern.pattern_type != PatternType.HARMONY:
            return 0.0

        scores = []

        # Voice leading quality (weight: 0.4)
        voice_leading_score = self._evaluate_voice_leading(pattern)
        scores.append(voice_leading_score * 0.4)

        # Tension/release balance (weight: 0.3)
        tension_release_score = self._evaluate_tension_release(pattern)
        scores.append(tension_release_score * 0.3)

        # Chord progression coherence (weight: 0.3)
        progression_score = self._evaluate_progression_coherence(pattern)
        scores.append(progression_score * 0.3)

        return sum(scores)

    def _generate_initial_harmony(self, context: Any) -> Pattern:
        """Generate an initial harmony pattern.

        Args:
            context: Generator context

        Returns:
            Pattern: Initial harmony pattern
        """
        # Use existing harmony generator to create initial pattern
        try:
            from generators.harmony_generator import HarmonyGenerator
            generator = HarmonyGenerator(context)
            return generator.generate(num_bars=4, chord_complexity='medium', harmonic_variance='medium')
        except ImportError:
            # Fallback: create a simple harmony pattern
            return self._create_simple_harmony(context)

    def _create_simple_harmony(self, context: Any) -> Pattern:
        """Create a simple fallback harmony pattern.

        Args:
            context: Generator context

        Returns:
            Pattern: Simple harmony pattern
        """
        chords = []
        current_time = 0.0

        # Common chord progression: I - IV - V - I
        progression = ['I', 'IV', 'V', 'I']

        for chord_symbol in progression:
            # Convert roman numeral to actual chord notes
            chord_notes = self._roman_to_chord_notes(chord_symbol, context)

            # Create chord with 2 beats duration
            chord = Chord(chord_notes, current_time)
            chords.append(chord)
            current_time += 2.0

        return Pattern(PatternType.HARMONY, [], chords)

    def _roman_to_chord_notes(self, roman: str, context: Any) -> List[Note]:
        """Convert roman numeral to chord notes.

        Args:
            roman: Roman numeral chord symbol
            context: Generator context

        Returns:
            List[Note]: Chord notes
        """
        # Simple mapping - in full implementation would use music theory
        base_pitch = 60  # C4

        if roman == 'I':
            # Major chord: root, major third, fifth
            return [
                Note(base_pitch, 2.0, 80, 0.0),
                Note(base_pitch + 4, 2.0, 80, 0.0),
                Note(base_pitch + 7, 2.0, 80, 0.0)
            ]
        elif roman == 'IV':
            # Major chord up a fourth
            return [
                Note(base_pitch + 5, 2.0, 80, 0.0),
                Note(base_pitch + 9, 2.0, 80, 0.0),
                Note(base_pitch + 12, 2.0, 80, 0.0)
            ]
        elif roman == 'V':
            # Major chord up a fifth
            return [
                Note(base_pitch + 7, 2.0, 80, 0.0),
                Note(base_pitch + 11, 2.0, 80, 0.0),
                Note(base_pitch + 14, 2.0, 80, 0.0)
            ]
        else:
            # Default to I chord
            return [
                Note(base_pitch, 2.0, 80, 0.0),
                Note(base_pitch + 4, 2.0, 80, 0.0),
                Note(base_pitch + 7, 2.0, 80, 0.0)
            ]

    def _generate_harmony_variations(self, context: Any, base_pattern: Pattern) -> List[Pattern]:
        """Generate variations of a harmony pattern.

        Args:
            context: Generator context
            base_pattern: Base harmony pattern to vary

        Returns:
            List[Pattern]: List of harmony variations
        """
        variations = []

        # Create multiple variations
        for _ in range(3):
            variation = copy.deepcopy(base_pattern)

            # Apply random refinements
            refinement_type = random.choice(['voice_leading', 'tension_release', 'progression', 'voicing'])

            if refinement_type == 'voice_leading':
                variation = self._refine_voice_leading(variation)
            elif refinement_type == 'tension_release':
                variation = self._refine_tension_release(variation)
            elif refinement_type == 'progression':
                variation = self._refine_progression(variation)
            elif refinement_type == 'voicing':
                variation = self._refine_voicing(variation)

            variations.append(variation)

        return variations

    def _refine_voice_leading(self, pattern: Pattern) -> Pattern:
        """Refine voice leading between chords.

        Args:
            pattern: Pattern to refine

        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)

        if len(refined_pattern.chords) < 2:
            return refined_pattern

        for i in range(len(refined_pattern.chords) - 1):
            current_chord = refined_pattern.chords[i]
            next_chord = refined_pattern.chords[i + 1]

            if len(current_chord.notes) >= 2 and len(next_chord.notes) >= 2:
                # Smooth the transition between chord notes
                for j in range(min(len(current_chord.notes), len(next_chord.notes))):
                    current_pitch = current_chord.notes[j].pitch
                    next_pitch = next_chord.notes[j].pitch

                    # If large leap, try to find a better voice leading
                    if abs(next_pitch - current_pitch) > 7:  # More than a fifth
                        # Try to find a note in the next chord closer to current
                        closest_pitch = min(next_chord.notes, key=lambda n: abs(n.pitch - current_pitch))
                        current_chord.notes[j].pitch = closest_pitch.pitch

        return refined_pattern

    def _refine_tension_release(self, pattern: Pattern) -> Pattern:
        """Refine tension and release in the harmonic progression.

        Args:
            pattern: Pattern to refine

        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)

        # Adjust chord voicings to create better tension/release arcs
        for i, chord in enumerate(refined_pattern.chords):
            if i % 4 == 0:  # First chord of phrase - more consonant
                self._make_more_consonant(chord)
            elif i % 4 == 2:  # Third chord - build tension
                self._add_tension(chord)
            elif i % 4 == 3:  # Fourth chord - prepare resolution
                self._prepare_resolution(chord)

        return refined_pattern

    def _refine_progression(self, pattern: Pattern) -> Pattern:
        """Refine the chord progression for better flow.

        Args:
            pattern: Pattern to refine

        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)

        if len(refined_pattern.chords) < 3:
            return refined_pattern

        # Look for common progression patterns and improve them
        for i in range(len(refined_pattern.chords) - 2):
            chord1 = refined_pattern.chords[i]
            chord2 = refined_pattern.chords[i + 1]
            chord3 = refined_pattern.chords[i + 2]

            # If we have I - I - I, try to add some movement
            if self._chords_similar(chord1, chord2) and self._chords_similar(chord2, chord3):
                # Change the middle chord to create movement
                self._modify_chord_for_movement(chord2, chord1, chord3)

        return refined_pattern

    def _refine_voicing(self, pattern: Pattern) -> Pattern:
        """Refine chord voicings for better sound.

        Args:
            pattern: Pattern to refine

        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)

        for chord in refined_pattern.chords:
            if len(chord.notes) >= 3:
                # Sort notes to create better spacing
                chord.notes.sort(key=lambda n: n.pitch)

                # Invert if spacing is too wide at bottom
                if len(chord.notes) >= 2:
                    bottom_interval = chord.notes[1].pitch - chord.notes[0].pitch
                    if bottom_interval > 12:  # More than an octave
                        # Drop the lowest note an octave
                        chord.notes[0].pitch -= 12

        return refined_pattern

    def _make_more_consonant(self, chord: Chord):
        """Make a chord more consonant."""
        if len(chord.notes) >= 3:
            # Ensure notes are in a major or minor triad arrangement
            pitches = sorted([n.pitch for n in chord.notes])
            # Adjust to create more consonant intervals
            pass  # Implementation would adjust specific intervals

    def _add_tension(self, chord: Chord):
        """Add tension to a chord."""
        if random.random() < 0.3:  # 30% chance
            # Add a seventh or ninth
            if len(chord.notes) >= 3:
                seventh_pitch = chord.notes[0].pitch + 10  # Minor seventh
                seventh_note = Note(seventh_pitch, chord.notes[0].duration,
                                  chord.notes[0].velocity - 10, chord.start_time)
                chord.notes.append(seventh_note)

    def _prepare_resolution(self, chord: Chord):
        """Prepare a chord for resolution."""
        # Could add leading tone or adjust voicing for resolution
        pass

    def _chords_similar(self, chord1: Chord, chord2: Chord) -> bool:
        """Check if two chords are similar."""
        if len(chord1.notes) != len(chord2.notes):
            return False

        # Check if root notes are the same (simplified)
        if chord1.notes and chord2.notes:
            return chord1.notes[0].pitch % 12 == chord2.notes[0].pitch % 12

        return False

    def _modify_chord_for_movement(self, chord: Chord, prev_chord: Chord, next_chord: Chord):
        """Modify a chord to create better harmonic movement."""
        # Simple implementation - shift the chord up or down
        if random.random() < 0.5:
            shift = 7  # Up a fifth
        else:
            shift = 5  # Up a fourth

        for note in chord.notes:
            note.pitch += shift

    def _evaluate_voice_leading(self, pattern: Pattern) -> float:
        """Evaluate voice leading quality.

        Args:
            pattern: Pattern to evaluate

        Returns:
            float: Voice leading score (0.0-1.0)
        """
        if len(pattern.chords) < 2:
            return 0.5

        total_transitions = 0
        smooth_transitions = 0

        for i in range(len(pattern.chords) - 1):
            chord1 = pattern.chords[i]
            chord2 = pattern.chords[i + 1]

            if len(chord1.notes) >= 2 and len(chord2.notes) >= 2:
                # Check voice leading between corresponding notes
                for j in range(min(len(chord1.notes), len(chord2.notes))):
                    interval = abs(chord2.notes[j].pitch - chord1.notes[j].pitch)
                    total_transitions += 1

                    # Smooth transitions are small intervals
                    if interval <= 5:  # Within a fourth
                        smooth_transitions += 1

        return smooth_transitions / total_transitions if total_transitions > 0 else 0.5

    def _evaluate_tension_release(self, pattern: Pattern) -> float:
        """Evaluate tension/release balance.

        Args:
            pattern: Pattern to evaluate

        Returns:
            float: Tension/release score (0.0-1.0)
        """
        # Simplified evaluation - check for variety in chord complexity
        if len(pattern.chords) < 2:
            return 0.5

        complexities = []
        for chord in pattern.chords:
            complexity = len(chord.notes)  # More notes = more complex/tension
            complexities.append(complexity)

        # Look for tension/release pattern
        has_tension_release = False
        for i in range(len(complexities) - 1):
            if complexities[i] < complexities[i + 1]:  # Building tension
                has_tension_release = True
                break

        return 0.8 if has_tension_release else 0.4

    def _evaluate_progression_coherence(self, pattern: Pattern) -> float:
        """Evaluate chord progression coherence.

        Args:
            pattern: Pattern to evaluate

        Returns:
            float: Coherence score (0.0-1.0)
        """
        if len(pattern.chords) < 2:
            return 0.5

        # Check for reasonable harmonic movement
        total_movement = 0
        coherent_movement = 0

        for i in range(len(pattern.chords) - 1):
            if pattern.chords[i].notes and pattern.chords[i + 1].notes:
                root1 = pattern.chords[i].notes[0].pitch % 12
                root2 = pattern.chords[i + 1].notes[0].pitch % 12

                interval = abs(root2 - root1)
                total_movement += 1

                # Common progressions move by fourths/fifths
                if interval in [5, 7]:  # Fourth or fifth
                    coherent_movement += 1

        return coherent_movement / total_movement if total_movement > 0 else 0.5