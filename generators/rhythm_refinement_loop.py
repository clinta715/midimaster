"""
Rhythm Refinement Loop Module
 
This module contains the RhythmRefinementLoop class which implements iterative
refinement algorithms for rhythm generation and improvement.
"""

from typing import List, Optional, Any
import random
import copy

from structures.data_structures import Pattern, PatternType, Note
from generators.loop_manager import GenerationLoop

# Import advanced rhythm engines for enhanced refinement
try:
    from generators.advanced_rhythm_engine import AdvancedRhythmEngine, PolyrhythmEngine, SyncopationEngine
    ADVANCED_ENGINES_AVAILABLE = True
except ImportError:
    AdvancedRhythmEngine = None
    PolyrhythmEngine = None
    SyncopationEngine = None
    ADVANCED_ENGINES_AVAILABLE = False


class RhythmRefinementLoop(GenerationLoop):
    """Refines rhythm through iterative generation and evaluation.
 
    This loop implements sophisticated rhythm refinement algorithms including:
    - Groove consistency analysis
    - Rhythmic complexity optimization
    - Syncopation enhancement
    - Pattern variation and development
    """
 
    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.05):
        """Initialize the rhythm refinement loop.
 
        Args:
            max_iterations: Maximum number of refinement iterations
            convergence_threshold: Quality improvement threshold for convergence
        """
        super().__init__(max_iterations, convergence_threshold)
 
    def execute(self, context: Any, input_pattern: Optional[Pattern] = None, pattern_strength: float = 1.0) -> Pattern:
        """Generate and refine rhythm pattern.
 
        Args:
            context: Generator context containing shared state
            input_pattern: Input rhythm pattern to refine (None for first generation)
            pattern_strength: Strength for velocity scaling in pattern generation (0.0-1.0)
 
        Returns:
            Pattern: Refined rhythm pattern
        """
        # Generate initial rhythm if none provided
        if input_pattern is None:
            current_pattern = self._generate_initial_rhythm(context, pattern_strength)
        else:
            current_pattern = copy.deepcopy(input_pattern)
 
        best_pattern = current_pattern
        best_score = self.evaluate_quality(current_pattern)
 
        # Iterative refinement
        for iteration in range(self.max_iterations):
            # Generate rhythm variations
            candidates = self._generate_rhythm_variations(context, best_pattern, pattern_strength)
 
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
        """Evaluate the quality of a rhythm pattern.
 
        Args:
            pattern: Rhythm pattern to evaluate
 
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if pattern.pattern_type != PatternType.RHYTHM:
            return 0.0
 
        scores = []
 
        # Groove consistency (weight: 0.4)
        groove_score = self._evaluate_groove_consistency(pattern)
        scores.append(groove_score * 0.4)
 
        # Complexity balance (weight: 0.3)
        complexity_score = self._evaluate_complexity_balance(pattern)
        scores.append(complexity_score * 0.3)
 
        # Syncopation quality (weight: 0.3)
        syncopation_score = self._evaluate_syncopation_quality(pattern)
        scores.append(syncopation_score * 0.3)
 
        return sum(scores)
 
    def _generate_initial_rhythm(self, context: Any, pattern_strength: float = 1.0) -> Pattern:
        """Generate an initial rhythm pattern.
 
        Args:
            context: Generator context
            pattern_strength: Strength for velocity scaling (0.0-1.0)
 
        Returns:
            Pattern: Initial rhythm pattern
        """
        # Use existing rhythm generator to create initial pattern
        try:
            from generators.rhythm_generator import RhythmGenerator
            generator = RhythmGenerator(context, pattern_strength=pattern_strength)
            return generator.generate(num_bars=4, beat_complexity=0.5)
        except ImportError:
            # Fallback: create a simple rhythm pattern
            return self._create_simple_rhythm(context, pattern_strength)
 
    def _create_simple_rhythm(self, context: Any, pattern_strength: float = 1.0) -> Pattern:
        """Create a simple fallback rhythm pattern.
 
        Args:
            context: Generator context
            pattern_strength: Strength for velocity scaling (0.0-1.0)
 
        Returns:
            Pattern: Simple rhythm pattern
        """
        notes = []
        current_time = 0.0
        tempo = 120  # BPM
 
        # Create a basic 4/4 pattern with kick and snare
        pattern_16ths = [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0]  # Basic rock pattern
 
        for bar in range(4):
            for i, hit in enumerate(pattern_16ths):
                if hit:
                    # Determine drum type based on position within bar
                    if i % 8 == 0:  # Downbeat
                        pitch = 36  # Kick
                        velocity = max(1, min(127, int(100 * pattern_strength)))
                    elif i % 4 == 2:  # Snare position
                        pitch = 38  # Snare
                        velocity = max(1, min(127, int(90 * pattern_strength)))
                    else:
                        pitch = 42  # Closed hat
                        velocity = max(1, min(127, int(70 * pattern_strength)))
 
                    note = Note(pitch, 0.25, velocity, current_time)  # 16th note
                    notes.append(note)
 
                current_time += 0.25  # Move to next 16th note
 
        return Pattern(PatternType.RHYTHM, notes, [])
 
    def _generate_rhythm_variations(self, context: Any, base_pattern: Pattern, pattern_strength: float = 1.0) -> List[Pattern]:
        """Generate variations of a rhythm pattern.
 
        Args:
            context: Generator context
            base_pattern: Base rhythm pattern to vary
            pattern_strength: Strength for velocity scaling (0.0-1.0)
 
        Returns:
            List[Pattern]: List of rhythm variations
        """
        variations = []
 
        # Create multiple variations
        for _ in range(3):
            variation = copy.deepcopy(base_pattern)
 
            # Apply random refinements
            refinement_type = random.choice(['groove', 'complexity', 'syncopation', 'variation'])
 
            if refinement_type == 'groove':
                variation = self._refine_groove(variation)
            elif refinement_type == 'complexity':
                variation = self._refine_complexity(variation, pattern_strength)
            elif refinement_type == 'syncopation':
                variation = self._refine_syncopation(variation)
            elif refinement_type == 'variation':
                variation = self._refine_variation(variation, pattern_strength)
 
            variations.append(variation)
 
        return variations
 
    def _refine_groove(self, pattern: Pattern) -> Pattern:
        """Refine groove consistency in the rhythm.
 
        Args:
            pattern: Pattern to refine
 
        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)
 
        # Add subtle timing variations for groove
        for note in refined_pattern.notes:
            # Add small random timing variations (±5% of 16th note duration)
            variation = random.uniform(-0.0125, 0.0125)  # ±5% of 0.25 beat
            note.start_time += variation
 
            # Ensure timing stays reasonable
            note.start_time = max(0, note.start_time)
 
        # Sort notes by start time after timing adjustments
        refined_pattern.notes.sort(key=lambda n: n.start_time)
        return refined_pattern
 
    def _refine_complexity(self, pattern: Pattern, pattern_strength: float = 1.0) -> Pattern:
        """Refine rhythmic complexity balance.
 
        Args:
            pattern: Pattern to refine
            pattern_strength: Strength for velocity scaling (0.0-1.0)
 
        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)
 
        # Analyze current complexity
        note_count = len(refined_pattern.notes)
        total_duration = sum(note.duration for note in refined_pattern.notes)
 
        # If too simple, add some ghost notes or fills
        if note_count < 8:  # Less than 8 notes per 4 bars
            self._add_ghost_notes(refined_pattern, pattern_strength)
        # If too complex, simplify some areas
        elif note_count > 20:  # More than 20 notes per 4 bars
            self._simplify_complex_areas(refined_pattern)
 
        # Sort notes after modifications
        refined_pattern.notes.sort(key=lambda n: n.start_time)
        return refined_pattern
 
    def _refine_syncopation(self, pattern: Pattern) -> Pattern:
        """Refine syncopation in the rhythm.
 
        Args:
            pattern: Pattern to refine
 
        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)
 
        # Find notes on strong beats and consider moving some off-beat for syncopation
        strong_beats = [0, 4, 8, 12]  # Downbeats in 4 bars (4/4)
 
        for note in refined_pattern.notes:
            # Check if note is on a strong beat (within tolerance)
            note_beat = round(note.start_time)
 
            if note_beat in strong_beats and random.random() < 0.3:  # 30% chance
                # Move to syncopated position
                offset = random.choice([0.25, 0.5, 0.75, -0.25])  # Off-beat positions
                note.start_time += offset
 
                # Ensure the note stays within pattern bounds (16 beats for 4 bars)
                note.start_time = max(0, min(note.start_time, 15.75))
 
        # Sort notes by start time after modifications
        refined_pattern.notes.sort(key=lambda n: n.start_time)
        return refined_pattern
 
    def _refine_variation(self, pattern: Pattern, pattern_strength: float = 1.0) -> Pattern:
        """Add pattern variation and development.
 
        Args:
            pattern: Pattern to refine
            pattern_strength: Strength for velocity scaling (0.0-1.0)
 
        Returns:
            Pattern: Refined pattern
        """
        refined_pattern = copy.deepcopy(pattern)
 
        # Add some dynamic variation
        for note in refined_pattern.notes:
            # Slightly vary velocities for more natural feel
            velocity_variation = random.randint(-10, 10)
            note.velocity = max(1, min(127, note.velocity + velocity_variation))
 
        # Occasionally add or remove notes for variation
        if random.random() < 0.2:  # 20% chance to add a fill note
            self._add_fill_note(refined_pattern, pattern_strength)
 
        # Sort notes after modifications
        refined_pattern.notes.sort(key=lambda n: n.start_time)
        return refined_pattern
 
    def _add_ghost_notes(self, pattern: Pattern, pattern_strength: float = 1.0):
        """Add subtle ghost notes to increase complexity."""
        # Add some soft notes between main hits
        new_notes = []
 
        for i in range(len(pattern.notes) - 1):
            current_note = pattern.notes[i]
            next_note = pattern.notes[i + 1]
 
            # If there's space between notes, add a ghost note
            gap = next_note.start_time - (current_note.start_time + current_note.duration)
            if gap > 0.25:  # At least a 16th note gap
                ghost_time = current_note.start_time + current_note.duration + gap / 2
                ghost_velocity = max(1, min(127, int(40 * pattern_strength)))
                ghost_note = Note(42, 0.125, ghost_velocity, ghost_time)  # Soft closed hat
                print(f"RhythmRefinementLoop: Added ghost note with scaled velocity {ghost_velocity} (pattern_strength: {pattern_strength})")
                new_notes.append(ghost_note)
 
        pattern.notes.extend(new_notes)
 
    def _simplify_complex_areas(self, pattern: Pattern):
        """Simplify overly complex rhythmic areas."""
        # Remove some notes that are too close together
        notes_to_remove = []
 
        for i in range(len(pattern.notes) - 1):
            current_note = pattern.notes[i]
            next_note = pattern.notes[i + 1]
 
            # If notes are very close, consider removing one
            if next_note.start_time - current_note.start_time < 0.125:  # Less than 32nd note apart
                if random.random() < 0.3:  # 30% chance to remove
                    notes_to_remove.append(pattern.notes[i])
 
        for note in notes_to_remove:
            if note in pattern.notes:
                pattern.notes.remove(note)
 
    def _add_fill_note(self, pattern: Pattern, pattern_strength: float = 1.0):
        """Add a fill note for variation."""
        if pattern.notes:
            # Add a note near the end of the pattern
            last_note = pattern.notes[-1]
            fill_time = min(last_note.start_time + 1.0, 15.5)  # Within last beat of 4 bars
            fill_velocity = max(1, min(127, int(60 * pattern_strength)))
            fill_note = Note(46, 0.25, fill_velocity, fill_time)  # Open hat
            print(f"RhythmRefinementLoop: Added fill note with scaled velocity {fill_velocity} (pattern_strength: {pattern_strength})")
            pattern.notes.append(fill_note)
 
    def _evaluate_groove_consistency(self, pattern: Pattern) -> float:
        """Evaluate groove consistency.
 
        Args:
            pattern: Pattern to evaluate
 
        Returns:
            float: Groove consistency score (0.0-1.0)
        """
        if len(pattern.notes) < 4:
            return 0.5
 
        # Check timing regularity
        start_times = [note.start_time for note in pattern.notes]
        start_times.sort()
 
        # Calculate intervals between consecutive notes
        intervals = []
        for i in range(1, len(start_times)):
            intervals.append(start_times[i] - start_times[i-1])
 
        if not intervals:
            return 0.5
 
        # Check how consistent the intervals are
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(intervals)
        std_dev = variance ** 0.5
 
        # Lower standard deviation = more consistent groove
        consistency_score = max(0, 1 - (std_dev / avg_interval))
 
        return consistency_score
 
    def _evaluate_complexity_balance(self, pattern: Pattern) -> float:
        """Evaluate rhythmic complexity balance.
 
        Args:
            pattern: Pattern to evaluate
 
        Returns:
            float: Complexity balance score (0.0-1.0)
        """
        if len(pattern.notes) < 2:
            return 0.5
 
        note_count = len(pattern.notes)
 
        # Ideal range: 8-16 notes per 4 bars (2-4 notes per bar)
        if 8 <= note_count <= 16:
            return 1.0
        elif 4 <= note_count <= 24:
            # Linear interpolation for scores outside ideal range
            if note_count < 8:
                return 0.5 + (note_count - 4) * 0.5 / 4
            else:
                return 1.0 - (note_count - 16) * 0.5 / 8
        else:
            return 0.3  # Too simple or complex
 
    def _evaluate_syncopation_quality(self, pattern: Pattern) -> float:
        """Evaluate syncopation quality.
 
        Args:
            pattern: Pattern to evaluate
 
        Returns:
            float: Syncopation quality score (0.0-1.0)
        """
        if len(pattern.notes) < 2:
            return 0.5
 
        # Count notes that fall off the main beats
        off_beat_count = 0
        total_notes = len(pattern.notes)
 
        for note in pattern.notes:
            fractional_part = note.start_time % 1
            if 0.1 < fractional_part < 0.9:
                off_beat_count += 1
 
        # Some syncopation is good, but not too much
        syncopation_ratio = off_beat_count / total_notes if total_notes > 0 else 0
 
        if 0.1 <= syncopation_ratio <= 0.4:  # 10-40% syncopation is ideal
            return 1.0
        elif 0 <= syncopation_ratio <= 0.6:  # Up to 60% is acceptable
            return 0.8
        else:
            return 0.4  # Too much or too little syncopation