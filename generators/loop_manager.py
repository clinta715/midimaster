"""
Loop Manager Module

This module contains the LoopManager class and related components for implementing
iterative generation and refinement loops in the MIDI Master music generation system.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from structures.data_structures import Pattern


class GenerationLoop(ABC):
    """Base class for generation loops.

    Generation loops implement iterative refinement algorithms that can improve
    musical patterns through multiple generation-evaluation cycles.
    """

    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.1):
        """Initialize the generation loop.

        Args:
            max_iterations: Maximum number of iterations to perform
            convergence_threshold: Quality improvement threshold for convergence detection
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    @abstractmethod
    def execute(self, context: Any, input_pattern: Optional[Pattern]) -> Pattern:
        """Execute one iteration of the generation loop.

        Args:
            context: Generator context containing shared state
            input_pattern: Input pattern to refine (None for first iteration)

        Returns:
            Pattern: Refined pattern after this iteration
        """
        pass

    @abstractmethod
    def evaluate_quality(self, pattern: Pattern) -> float:
        """Evaluate the quality of a generated pattern.

        Args:
            pattern: Pattern to evaluate

        Returns:
            float: Quality score between 0.0 and 1.0
        """
        pass

    def should_continue(self, iteration: int, current_quality: float,
                       previous_quality: Optional[float] = None) -> bool:
        """Determine if loop should continue iterating.

        Args:
            iteration: Current iteration number (0-based)
            current_quality: Current pattern quality score
            previous_quality: Previous iteration's quality score

        Returns:
            bool: True if should continue, False if should stop
        """
        # Stop if we've reached max iterations
        if iteration >= self.max_iterations:
            return False

        # Stop if quality improvement is below threshold
        if previous_quality is not None:
            improvement = current_quality - previous_quality
            if improvement < self.convergence_threshold:
                return False

        return True


class QualityEvaluator:
    """Evaluates the quality of generated musical patterns.

    Provides comprehensive quality metrics for melody, harmony, rhythm, and bass patterns.
    """

    def __init__(self):
        """Initialize the quality evaluator."""
        self.weights = {
            'melody': {
                'contour_smoothness': 0.3,
                'harmonic_consistency': 0.3,
                'rhythmic_variety': 0.2,
                'pitch_range': 0.2
            },
            'harmony': {
                'voice_leading': 0.4,
                'tension_release': 0.3,
                'chord_progression': 0.3
            },
            'rhythm': {
                'groove_consistency': 0.4,
                'complexity_balance': 0.3,
                'syncopation_quality': 0.3
            },
            'bass': {
                'harmonic_support': 0.4,
                'rhythmic_drive': 0.3,
                'register_appropriateness': 0.3
            }
        }

    def evaluate_pattern(self, pattern: Pattern) -> Dict[str, float]:
        """Evaluate a pattern across multiple quality dimensions.

        Args:
            pattern: Pattern to evaluate

        Returns:
            Dict containing quality scores for different aspects
        """
        pattern_type = pattern.pattern_type.value

        if pattern_type == 'melody':
            return self._evaluate_melody_quality(pattern)
        elif pattern_type == 'harmony':
            return self._evaluate_harmony_quality(pattern)
        elif pattern_type == 'rhythm':
            return self._evaluate_rhythm_quality(pattern)
        elif pattern_type == 'bass':
            return self._evaluate_bass_quality(pattern)
        else:
            return {'overall_quality': 0.5}

    def _evaluate_melody_quality(self, pattern: Pattern) -> Dict[str, float]:
        """Evaluate melody-specific quality metrics."""
        scores = {}

        # Contour smoothness - measure pitch movement consistency
        scores['contour_smoothness'] = self._calculate_contour_smoothness(pattern)

        # Harmonic consistency - check if notes fit the underlying harmony
        scores['harmonic_consistency'] = self._calculate_harmonic_consistency(pattern)

        # Rhythmic variety - assess note duration diversity
        scores['rhythmic_variety'] = self._calculate_rhythmic_variety(pattern)

        # Pitch range - evaluate appropriate register usage
        scores['pitch_range'] = self._calculate_pitch_range(pattern)

        # Overall quality as weighted sum
        weights = self.weights['melody']
        scores['overall_quality'] = sum(
            scores[metric] * weights[metric] for metric in weights.keys()
        )

        return scores

    def _evaluate_harmony_quality(self, pattern: Pattern) -> Dict[str, float]:
        """Evaluate harmony-specific quality metrics."""
        scores = {}

        # Voice leading - smoothness of chord transitions
        scores['voice_leading'] = self._calculate_voice_leading(pattern)

        # Tension/release - balance of dissonance and consonance
        scores['tension_release'] = self._calculate_tension_release(pattern)

        # Chord progression - quality of harmonic movement
        scores['chord_progression'] = self._calculate_chord_progression_quality(pattern)

        # Overall quality as weighted sum
        weights = self.weights['harmony']
        scores['overall_quality'] = sum(
            scores[metric] * weights[metric] for metric in weights.keys()
        )

        return scores

    def _evaluate_rhythm_quality(self, pattern: Pattern) -> Dict[str, float]:
        """Evaluate rhythm-specific quality metrics."""
        scores = {}

        # Groove consistency - rhythmic pattern regularity
        scores['groove_consistency'] = self._calculate_groove_consistency(pattern)

        # Complexity balance - appropriate complexity for the style
        scores['complexity_balance'] = self._calculate_complexity_balance(pattern)

        # Syncopation quality - effectiveness of off-beat accents
        scores['syncopation_quality'] = self._calculate_syncopation_quality(pattern)

        # Overall quality as weighted sum
        weights = self.weights['rhythm']
        scores['overall_quality'] = sum(
            scores[metric] * weights[metric] for metric in weights.keys()
        )

        return scores

    def _evaluate_bass_quality(self, pattern: Pattern) -> Dict[str, float]:
        """Evaluate bass-specific quality metrics."""
        scores = {}

        # Harmonic support - how well bass supports the harmony
        scores['harmonic_support'] = self._calculate_harmonic_support(pattern)

        # Rhythmic drive - contribution to rhythmic momentum
        scores['rhythmic_drive'] = self._calculate_rhythmic_drive(pattern)

        # Register appropriateness - suitability of pitch range
        scores['register_appropriateness'] = self._calculate_register_appropriateness(pattern)

        # Overall quality as weighted sum
        weights = self.weights['bass']
        scores['overall_quality'] = sum(
            scores[metric] * weights[metric] for metric in weights.keys()
        )

        return scores

    def _calculate_contour_smoothness(self, pattern: Pattern) -> float:
        """Calculate melody contour smoothness (0.0-1.0)."""
        if len(pattern.notes) < 2:
            return 0.5

        total_movement = 0
        smooth_transitions = 0

        for i in range(1, len(pattern.notes)):
            pitch_diff = abs(pattern.notes[i].pitch - pattern.notes[i-1].pitch)

            # Small intervals (1-3 semitones) are considered smooth
            if pitch_diff <= 3:
                smooth_transitions += 1

            total_movement += 1

        return smooth_transitions / total_movement if total_movement > 0 else 0.5

    def _calculate_harmonic_consistency(self, pattern: Pattern) -> float:
        """Calculate harmonic consistency (0.0-1.0)."""
        # Placeholder - would need chord context to implement properly
        return 0.7

    def _calculate_rhythmic_variety(self, pattern: Pattern) -> float:
        """Calculate rhythmic variety (0.0-1.0)."""
        if len(pattern.notes) < 2:
            return 0.5

        durations = [note.duration for note in pattern.notes]
        unique_durations = len(set(durations))

        # More unique durations = more rhythmic variety
        variety_score = min(unique_durations / 8.0, 1.0)  # Cap at 8 different durations

        return variety_score

    def _calculate_pitch_range(self, pattern: Pattern) -> float:
        """Calculate pitch range appropriateness (0.0-1.0)."""
        if len(pattern.notes) < 2:
            return 0.5

        pitches = [note.pitch for note in pattern.notes]
        pitch_range = max(pitches) - min(pitches)

        # Ideal melody range is about 12-24 semitones (octave to two octaves)
        if 12 <= pitch_range <= 24:
            return 1.0
        elif 8 <= pitch_range <= 36:
            return 0.8
        else:
            return 0.5

    def _calculate_voice_leading(self, pattern: Pattern) -> float:
        """Calculate voice leading quality (0.0-1.0)."""
        # Placeholder - would analyze chord transitions
        return 0.7

    def _calculate_tension_release(self, pattern: Pattern) -> float:
        """Calculate tension/release balance (0.0-1.0)."""
        # Placeholder - would analyze dissonance/consonance patterns
        return 0.7

    def _calculate_chord_progression_quality(self, pattern: Pattern) -> float:
        """Calculate chord progression quality (0.0-1.0)."""
        # Placeholder - would evaluate harmonic movement
        return 0.7

    def _calculate_groove_consistency(self, pattern: Pattern) -> float:
        """Calculate groove consistency (0.0-1.0)."""
        # Placeholder - would analyze rhythmic regularity
        return 0.7

    def _calculate_complexity_balance(self, pattern: Pattern) -> float:
        """Calculate complexity balance (0.0-1.0)."""
        # Placeholder - would assess appropriate complexity
        return 0.7

    def _calculate_syncopation_quality(self, pattern: Pattern) -> float:
        """Calculate syncopation quality (0.0-1.0)."""
        # Placeholder - would evaluate off-beat effectiveness
        return 0.7

    def _calculate_harmonic_support(self, pattern: Pattern) -> float:
        """Calculate harmonic support (0.0-1.0)."""
        # Placeholder - would check bass-chord alignment
        return 0.7

    def _calculate_rhythmic_drive(self, pattern: Pattern) -> float:
        """Calculate rhythmic drive (0.0-1.0)."""
        # Placeholder - would assess rhythmic momentum
        return 0.7

    def _calculate_register_appropriateness(self, pattern: Pattern) -> float:
        """Calculate register appropriateness (0.0-1.0)."""
        if len(pattern.notes) < 2:
            return 0.5

        pitches = [note.pitch for note in pattern.notes]
        avg_pitch = sum(pitches) / len(pitches)

        # Bass should typically be in lower register (MIDI 24-48)
        if 24 <= avg_pitch <= 48:
            return 1.0
        elif 12 <= avg_pitch <= 60:
            return 0.8
        else:
            return 0.5


class LoopManager:
    """Manages iterative generation and refinement loops.

    The LoopManager coordinates multiple generation loops that can iteratively
    improve musical patterns through generation-evaluation cycles.
    """

    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.1):
        """Initialize the loop manager.

        Args:
            max_iterations: Maximum iterations for convergence detection
            convergence_threshold: Quality improvement threshold for stopping
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.generation_loops: List[GenerationLoop] = []
        self.quality_evaluator = QualityEvaluator()

    def add_generation_loop(self, loop: GenerationLoop):
        """Add a generation loop to the manager.

        Args:
            loop: GenerationLoop instance to add
        """
        self.generation_loops.append(loop)

    def execute_all_loops(self, context: Any) -> List[Pattern]:
        """Execute all registered loops and return refined patterns.

        Args:
            context: Generator context containing shared state

        Returns:
            List[Pattern]: List of refined patterns, one per loop
        """
        refined_patterns = []

        for loop in self.generation_loops:
            # Execute the loop with no initial pattern
            refined_pattern = loop.execute(context, None)
            refined_patterns.append(refined_pattern)

        return refined_patterns

    def get_quality_scores(self, patterns: List[Pattern]) -> Dict[str, Dict[str, float]]:
        """Get quality scores for a list of patterns.

        Args:
            patterns: List of patterns to evaluate

        Returns:
            Dict mapping pattern types to their quality scores
        """
        scores = {}

        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            scores[pattern_type] = self.quality_evaluator.evaluate_pattern(pattern)

        return scores

    def has_converged(self, current_scores: Dict[str, float],
                     previous_scores: Optional[Dict[str, float]] = None) -> bool:
        """Check if the generation process has converged.

        Args:
            current_scores: Current iteration quality scores
            previous_scores: Previous iteration quality scores

        Returns:
            bool: True if converged (no significant improvement)
        """
        if previous_scores is None:
            return False

        total_improvement = 0.0
        for pattern_type in current_scores:
            if pattern_type in previous_scores:
                improvement = current_scores[pattern_type] - previous_scores[pattern_type]
                total_improvement += improvement

        avg_improvement = total_improvement / len(current_scores) if current_scores else 0.0

        return avg_improvement < self.convergence_threshold

    def get_overall_quality(self, pattern_scores: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall quality across all patterns.

        Args:
            pattern_scores: Quality scores for each pattern type

        Returns:
            float: Weighted overall quality score
        """
        if not pattern_scores:
            return 0.0

        total_quality = 0.0
        pattern_weights = {
            'melody': 0.3,
            'harmony': 0.3,
            'rhythm': 0.25,
            'bass': 0.15
        }

        for pattern_type, scores in pattern_scores.items():
            if 'overall_quality' in scores:
                weight = pattern_weights.get(pattern_type, 0.25)
                total_quality += scores['overall_quality'] * weight

        return total_quality