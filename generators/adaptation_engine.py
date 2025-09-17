"""
Adaptation Engine for real-time musical adaptation and performance adjustments.

This module provides intelligent adaptation capabilities including:
- Real-time tempo adaptation based on performance context
- Dynamic key adaptation for optimal playability
- Performance-based adjustment algorithms
- Musical context awareness
- Player skill level adaptation
- Genre-specific adaptation rules
- Feedback-driven parameter adjustment
"""

import math
import time
from typing import List, Dict, Tuple, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import statistics

from generators.tempo_curve_manager import TempoManager, TempoEvent, CurveType
from generators.modulation_engine import ModulationEngine, KeyChange, ModulationType
from generators.harmonic_analyzer import HarmonicAnalyzer, HarmonicFunction, TensionLevel


class AdaptationTrigger(Enum):
    """Types of events that can trigger adaptation."""
    PERFORMANCE_FEEDBACK = "performance_feedback"
    PLAYER_SKILL_CHANGE = "player_skill_change"
    GENRE_SWITCH = "genre_switch"
    MOOD_CHANGE = "mood_change"
    TEMPO_DRIFT = "tempo_drift"
    HARMONIC_COMPLEXITY = "harmonic_complexity"
    RHYTHMIC_DIFFICULTY = "rhythmic_difficulty"
    MANUAL_OVERRIDE = "manual_override"


class PlayerSkill(Enum):
    """Player skill levels for adaptation."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    PROFESSIONAL = "professional"


@dataclass
class AdaptationContext:
    """Context information for adaptation decisions."""
    current_tempo: float = 120.0
    current_key: str = "C major"
    current_mood: str = "energetic"
    current_genre: str = "pop"
    player_skill: PlayerSkill = PlayerSkill.INTERMEDIATE
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    adaptation_history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_adaptation_time: float = 0.0
    adaptation_cooldown: float = 5.0  # Seconds between adaptations


@dataclass
class AdaptationSuggestion:
    """A suggested adaptation with reasoning."""
    trigger: AdaptationTrigger
    parameter: str  # What to adapt ('tempo', 'key', 'complexity', etc.)
    current_value: Union[float, str]
    suggested_value: Union[float, str]
    confidence: float  # 0-1 confidence level
    reasoning: str
    priority: int  # 1-10 priority level
    timestamp: float = field(default_factory=time.time)


class PerformanceMetrics:
    """Metrics for evaluating musical performance."""

    def __init__(self, window_size: int = 50):
        """Initialize performance metrics tracking."""
        self.window_size = window_size
        self.tempo_stability = deque(maxlen=window_size)
        self.rhythm_accuracy = deque(maxlen=window_size)
        self.harmonic_complexity = deque(maxlen=window_size)
        self.player_engagement = deque(maxlen=window_size)
        self.adaptation_effectiveness = deque(maxlen=window_size)

    def add_measurement(self, metric_type: str, value: float):
        """Add a performance measurement."""
        if metric_type == "tempo_stability":
            self.tempo_stability.append(value)
        elif metric_type == "rhythm_accuracy":
            self.rhythm_accuracy.append(value)
        elif metric_type == "harmonic_complexity":
            self.harmonic_complexity.append(value)
        elif metric_type == "player_engagement":
            self.player_engagement.append(value)
        elif metric_type == "adaptation_effectiveness":
            self.adaptation_effectiveness.append(value)

    def get_average(self, metric_type: str) -> float:
        """Get average value for a metric."""
        if metric_type == "tempo_stability":
            return statistics.mean(self.tempo_stability) if self.tempo_stability else 0.5
        elif metric_type == "rhythm_accuracy":
            return statistics.mean(self.rhythm_accuracy) if self.rhythm_accuracy else 0.5
        elif metric_type == "harmonic_complexity":
            return statistics.mean(self.harmonic_complexity) if self.harmonic_complexity else 0.5
        elif metric_type == "player_engagement":
            return statistics.mean(self.player_engagement) if self.player_engagement else 0.5
        elif metric_type == "adaptation_effectiveness":
            return statistics.mean(self.adaptation_effectiveness) if self.adaptation_effectiveness else 0.5
        return 0.5


class AdaptationEngine:
    """Main engine for real-time musical adaptation."""

    def __init__(self):
        """Initialize the adaptation engine."""
        self.tempo_manager = TempoManager()
        self.modulation_engine = ModulationEngine()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.context = AdaptationContext()
        self.metrics = PerformanceMetrics()
        self.adaptation_strategies = self._initialize_strategies()
        self.feedback_processors = self._initialize_feedback_processors()

    def adapt_to_performance_feedback(self, feedback_type: str, feedback_value: float) -> List[AdaptationSuggestion]:
        """
        Process performance feedback and generate adaptation suggestions.

        Args:
            feedback_type: Type of feedback ('tempo', 'accuracy', 'engagement', etc.)
            feedback_value: Value of the feedback (0-1 scale)

        Returns:
            List of adaptation suggestions
        """
        suggestions = []

        # Update performance metrics
        self.metrics.add_measurement(feedback_type, feedback_value)

        # Process feedback through registered processors
        for processor in self.feedback_processors:
            processor_suggestions = processor(feedback_type, feedback_value, self.context)
            suggestions.extend(processor_suggestions)

        # Filter and prioritize suggestions
        suggestions = self._filter_suggestions(suggestions)

        return suggestions

    def apply_adaptation(self, suggestion: AdaptationSuggestion) -> bool:
        """
        Apply an adaptation suggestion.

        Args:
            suggestion: The adaptation suggestion to apply

        Returns:
            True if adaptation was successful
        """
        # Check cooldown period
        current_time = time.time()
        if current_time - self.context.last_adaptation_time < self.context.adaptation_cooldown:
            return False

        try:
            if suggestion.parameter == "tempo":
                self._adapt_tempo(suggestion)
            elif suggestion.parameter == "key":
                self._adapt_key(suggestion)
            elif suggestion.parameter == "complexity":
                self._adapt_complexity(suggestion)
            elif suggestion.parameter == "mood":
                self._adapt_mood(suggestion)

            # Update context
            self.context.last_adaptation_time = current_time
            self.context.adaptation_history.append(suggestion)

            # Measure effectiveness (will be updated later with performance feedback)
            self.metrics.add_measurement("adaptation_effectiveness", 0.5)

            return True

        except Exception as e:
            print(f"Failed to apply adaptation: {e}")
            return False

    def get_context_aware_suggestions(self) -> List[AdaptationSuggestion]:
        """
        Generate context-aware adaptation suggestions based on current state.

        Returns:
            List of context-aware suggestions
        """
        suggestions = []

        # Tempo adaptation based on player skill and genre
        tempo_suggestion = self._analyze_tempo_adaptation()
        if tempo_suggestion:
            suggestions.append(tempo_suggestion)

        # Key adaptation for optimal playability
        key_suggestion = self._analyze_key_adaptation()
        if key_suggestion:
            suggestions.append(key_suggestion)

        # Complexity adaptation based on performance
        complexity_suggestion = self._analyze_complexity_adaptation()
        if complexity_suggestion:
            suggestions.append(complexity_suggestion)

        return self._filter_suggestions(suggestions)

    def update_player_skill(self, skill_level: PlayerSkill):
        """Update the assessed player skill level."""
        self.context.player_skill = skill_level

        # Trigger skill-based adaptations
        if skill_level == PlayerSkill.BEGINNER:
            self._simplify_for_beginners()
        elif skill_level == PlayerSkill.ADVANCED:
            self._increase_complexity_for_advanced()

    def analyze_performance_trends(self) -> Dict[str, float]:
        """
        Analyze performance trends to inform adaptation decisions.

        Returns:
            Dictionary of trend metrics
        """
        trends = {}

        # Calculate trends for key metrics
        metrics_to_analyze = [
            "tempo_stability", "rhythm_accuracy",
            "harmonic_complexity", "player_engagement"
        ]

        for metric in metrics_to_analyze:
            trend = self._calculate_trend(metric)
            trends[f"{metric}_trend"] = trend

        # Overall performance trend
        overall_trend = sum(trends.values()) / len(trends) if trends else 0.0
        trends["overall_performance_trend"] = overall_trend

        return trends

    def _tempo_adaptation_strategy(self):
        """Strategy for tempo adaptations."""
        pass

    def _key_adaptation_strategy(self):
        """Strategy for key adaptations."""
        pass

    def _complexity_adaptation_strategy(self):
        """Strategy for complexity adaptations."""
        pass

    def _mood_adaptation_strategy(self):
        """Strategy for mood adaptations."""
        pass
    def _initialize_strategies(self) -> Dict[str, Callable]:
        """Initialize adaptation strategies."""
        return {
            "tempo_adaptation": self._tempo_adaptation_strategy,
            "key_adaptation": self._key_adaptation_strategy,
            "complexity_adaptation": self._complexity_adaptation_strategy,
            "mood_adaptation": self._mood_adaptation_strategy
        }

    def _initialize_feedback_processors(self) -> List[Callable]:
        """Initialize feedback processors."""
        return [
            self._process_tempo_feedback,
            self._process_accuracy_feedback,
            self._process_engagement_feedback,
            self._process_complexity_feedback
        ]

    def _process_tempo_feedback(self, feedback_type: str, value: float,
                               context: AdaptationContext) -> List[AdaptationSuggestion]:
        """Process tempo-related feedback."""
        suggestions = []

        if feedback_type == "tempo_stability":
            if value < 0.3:  # Very unstable
                suggestions.append(AdaptationSuggestion(
                    trigger=AdaptationTrigger.TEMPO_DRIFT,
                    parameter="tempo",
                    current_value=context.current_tempo,
                    suggested_value=context.current_tempo * 0.9,  # Slow down
                    confidence=0.8,
                    reasoning="Tempo instability detected - suggesting slower tempo for better control",
                    priority=8
                ))
            elif value > 0.9:  # Very stable
                # Could suggest tempo increase for more challenge
                pass

        return suggestions

    def _process_accuracy_feedback(self, feedback_type: str, value: float,
                                  context: AdaptationContext) -> List[AdaptationSuggestion]:
        """Process accuracy-related feedback."""
        suggestions = []

        if feedback_type == "rhythm_accuracy":
            if value < 0.4:  # Poor accuracy
                suggestions.append(AdaptationSuggestion(
                    trigger=AdaptationTrigger.RHYTHMIC_DIFFICULTY,
                    parameter="complexity",
                    current_value="current",
                    suggested_value="simplified",
                    confidence=0.9,
                    reasoning="Low rhythmic accuracy - suggesting simplified rhythms",
                    priority=9
                ))

        return suggestions

    def _process_engagement_feedback(self, feedback_type: str, value: float,
                                    context: AdaptationContext) -> List[AdaptationSuggestion]:
        """Process engagement-related feedback."""
        suggestions = []

        if feedback_type == "player_engagement":
            if value < 0.3:  # Low engagement
                # Try changing mood or key to increase engagement
                new_mood = self._suggest_mood_change(context.current_mood)
                suggestions.append(AdaptationSuggestion(
                    trigger=AdaptationTrigger.MOOD_CHANGE,
                    parameter="mood",
                    current_value=context.current_mood,
                    suggested_value=new_mood,
                    confidence=0.7,
                    reasoning="Low engagement detected - suggesting mood change",
                    priority=6
                ))

        return suggestions

    def _process_complexity_feedback(self, feedback_type: str, value: float,
                                    context: AdaptationContext) -> List[AdaptationSuggestion]:
        """Process complexity-related feedback."""
        suggestions = []

        if feedback_type == "harmonic_complexity":
            if value > 0.8:  # Too complex
                suggestions.append(AdaptationSuggestion(
                    trigger=AdaptationTrigger.HARMONIC_COMPLEXITY,
                    parameter="complexity",
                    current_value="complex",
                    suggested_value="balanced",
                    confidence=0.8,
                    reasoning="High harmonic complexity - suggesting simpler harmonies",
                    priority=7
                ))

        return suggestions

    def _analyze_tempo_adaptation(self) -> Optional[AdaptationSuggestion]:
        """Analyze if tempo adaptation is needed."""
        skill = self.context.player_skill
        current_tempo = self.context.current_tempo
        tempo_stability = self.metrics.get_average("tempo_stability")

        # Adjust tempo based on skill level
        optimal_tempo_ranges = {
            PlayerSkill.BEGINNER: (60, 90),
            PlayerSkill.INTERMEDIATE: (90, 130),
            PlayerSkill.ADVANCED: (110, 160),
            PlayerSkill.EXPERT: (120, 180),
            PlayerSkill.PROFESSIONAL: (100, 200)
        }

        optimal_min, optimal_max = optimal_tempo_ranges[skill]

        if current_tempo < optimal_min:
            return AdaptationSuggestion(
                trigger=AdaptationTrigger.PLAYER_SKILL_CHANGE,
                parameter="tempo",
                current_value=current_tempo,
                suggested_value=optimal_min + (optimal_max - optimal_min) * 0.3,
                confidence=0.6,
                reasoning=f"Tempo too slow for {skill.value} level - suggesting increase",
                priority=5
            )
        elif current_tempo > optimal_max:
            return AdaptationSuggestion(
                trigger=AdaptationTrigger.PLAYER_SKILL_CHANGE,
                parameter="tempo",
                current_value=current_tempo,
                suggested_value=optimal_max * 0.9,
                confidence=0.6,
                reasoning=f"Tempo too fast for {skill.value} level - suggesting decrease",
                priority=7
            )

        return None

    def _analyze_key_adaptation(self) -> Optional[AdaptationSuggestion]:
        """Analyze if key adaptation is needed."""
        current_key = self.context.current_key

        # Suggest easier keys for beginners
        if self.context.player_skill == PlayerSkill.BEGINNER:
            easy_keys = ["C major", "G major", "F major", "A minor", "E minor"]
            if current_key not in easy_keys:
                return AdaptationSuggestion(
                    trigger=AdaptationTrigger.PLAYER_SKILL_CHANGE,
                    parameter="key",
                    current_value=current_key,
                    suggested_value="C major",
                    confidence=0.5,
                    reasoning="Suggesting easier key for beginner level",
                    priority=4
                )

        return None

    def _analyze_complexity_adaptation(self) -> Optional[AdaptationSuggestion]:
        """Analyze if complexity adaptation is needed."""
        accuracy = self.metrics.get_average("rhythm_accuracy")
        engagement = self.metrics.get_average("player_engagement")

        if accuracy < 0.4 and engagement < 0.4:
            return AdaptationSuggestion(
                trigger=AdaptationTrigger.PERFORMANCE_FEEDBACK,
                parameter="complexity",
                current_value="current",
                suggested_value="simplified",
                confidence=0.9,
                reasoning="Poor performance - suggesting complexity reduction",
                priority=10
            )

        return None

    def _filter_suggestions(self, suggestions: List[AdaptationSuggestion]) -> List[AdaptationSuggestion]:
        """Filter and prioritize adaptation suggestions."""
        if not suggestions:
            return []

        # Remove duplicates
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            key = (suggestion.parameter, suggestion.trigger)
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)

        # Sort by priority (highest first)
        unique_suggestions.sort(key=lambda s: s.priority, reverse=True)

        # Limit to top suggestions
        return unique_suggestions[:5]

    def _adapt_tempo(self, suggestion: AdaptationSuggestion):
        """Apply tempo adaptation."""
        new_tempo = float(suggestion.suggested_value)
        self.tempo_manager.add_tempo_change(
            self.tempo_manager.current_time,
            self.tempo_manager.current_time + 4.0,  # 4 beat transition
            self.context.current_tempo,
            new_tempo,
            CurveType.EASE_IN_OUT
        )
        self.context.current_tempo = new_tempo

    def _adapt_key(self, suggestion: AdaptationSuggestion):
        """Apply key adaptation."""
        new_key = str(suggestion.suggested_value)
        key_change = self.modulation_engine.plan_modulation(
            self.context.current_key,
            new_key,
            ModulationType.SMOOTH_VOICE_LEADING
        )
        if key_change:
            self.modulation_engine.execute_modulation(key_change)
            self.context.current_key = new_key

    def _adapt_complexity(self, suggestion: AdaptationSuggestion):
        """Apply complexity adaptation."""
        # This would interface with the generation system to adjust complexity
        complexity_level = str(suggestion.suggested_value)
        print(f"Adapting complexity to: {complexity_level}")

    def _adapt_mood(self, suggestion: AdaptationSuggestion):
        """Apply mood adaptation."""
        new_mood = str(suggestion.suggested_value)
        self.context.current_mood = new_mood

    def _suggest_mood_change(self, current_mood: str) -> str:
        """Suggest a mood change for increased engagement."""
        mood_transitions = {
            "happy": "energetic",
            "sad": "calm",
            "energetic": "happy",
            "calm": "energetic"
        }
        return mood_transitions.get(current_mood, "energetic")

    def _simplify_for_beginners(self):
        """Apply beginner-friendly simplifications."""
        # Reduce tempo
        if self.context.current_tempo > 100:
            self._adapt_tempo(AdaptationSuggestion(
                trigger=AdaptationTrigger.PLAYER_SKILL_CHANGE,
                parameter="tempo",
                current_value=self.context.current_tempo,
                suggested_value=90.0,
                confidence=0.8,
                reasoning="Simplifying tempo for beginners",
                priority=8
            ))

        # Change to simpler key
        if self.context.current_key not in ["C major", "G major", "F major"]:
            self._adapt_key(AdaptationSuggestion(
                trigger=AdaptationTrigger.PLAYER_SKILL_CHANGE,
                parameter="key",
                current_value=self.context.current_key,
                suggested_value="C major",
                confidence=0.7,
                reasoning="Simplifying key for beginners",
                priority=6
            ))

    def _increase_complexity_for_advanced(self):
        """Apply advanced-level complexity increases."""
        # Increase tempo slightly
        if self.context.current_tempo < 140:
            self._adapt_tempo(AdaptationSuggestion(
                trigger=AdaptationTrigger.PLAYER_SKILL_CHANGE,
                parameter="tempo",
                current_value=self.context.current_tempo,
                suggested_value=min(self.context.current_tempo + 20, 160),
                confidence=0.6,
                reasoning="Increasing complexity for advanced players",
                priority=5
            ))

    def _calculate_trend(self, metric_type: str) -> float:
        """Calculate trend for a metric (-1 to 1, negative = declining)."""
        metric_values = getattr(self.metrics, metric_type)

        if len(metric_values) < 5:
            return 0.0

        # Simple linear trend calculation
        n = len(metric_values)
        x = list(range(n))
        y = list(metric_values)

        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, slope * 10))  # Scale factor for sensitivity

    def get_adaptation_report(self) -> Dict:
        """
        Generate a comprehensive adaptation report.

        Returns:
            Dictionary containing adaptation statistics and recommendations
        """
        report = {
            "current_context": {
                "tempo": self.context.current_tempo,
                "key": self.context.current_key,
                "mood": self.context.current_mood,
                "player_skill": self.context.player_skill.value
            },
            "performance_metrics": {
                "tempo_stability": self.metrics.get_average("tempo_stability"),
                "rhythm_accuracy": self.metrics.get_average("rhythm_accuracy"),
                "harmonic_complexity": self.metrics.get_average("harmonic_complexity"),
                "player_engagement": self.metrics.get_average("player_engagement"),
                "adaptation_effectiveness": self.metrics.get_average("adaptation_effectiveness")
            },
            "trends": self.analyze_performance_trends(),
            "adaptation_history": len(self.context.adaptation_history),
            "last_adaptation": self.context.last_adaptation_time
        }

        return report