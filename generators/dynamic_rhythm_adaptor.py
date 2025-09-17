"""
Dynamic Rhythm Adaptor Module

This module contains the DynamicRhythmAdaptor class for adapting rhythm patterns
dynamically based on ambient characteristics, tempo, mood, and existing content analysis.
It integrates with AmbientRhythmEngine and AtmosphereGenerator by modifying base patterns
to enhance atmospheric feel through sparsity, sustain, and velocity adjustments.
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass

from structures.data_structures import Note
from music_theory import MusicTheory
from generators.generator_context import GeneratorContext
from generators.generator_utils import get_velocity_for_mood


@dataclass
class AdaptationMetrics:
    """Metrics for rhythm adaptation."""
    tempo: float = 120.0
    mood: str = "calm"
    sparsity: float = 0.5
    complexity: float = 0.5
    density: float = 0.5


class DynamicRhythmAdaptor:
    """
    Adaptor for dynamic rhythm modifications based on ambient style and metrics.

    This class takes base patterns (List[Note]) and adapts them according to:
    - Tempo-based complexity: slower tempos increase sustain/duration
    - Mood-based sparsity: calm moods increase sparsity
    - Velocity modifications for atmospheric feel
    - Sparse placement algorithms (probabilistic removal/addition)
    - Content analysis: adjust based on existing pattern density, repetition, etc.
    - Integration: Can be called post AmbientRhythmEngine generation.

    Uses GeneratorContext for shared state (mood, key, scale).
    """

    def __init__(self, context: Optional[GeneratorContext] = None):
        """
        Initialize the DynamicRhythmAdaptor.

        Args:
            context: Optional GeneratorContext for mood, scale, etc.
        """
        self.context = context
        self.music_theory = MusicTheory()
        if context:
            self.mood = context.mood
            self.scale_pitches = context.scale_pitches or []
        else:
            self.mood = "calm"
            self.scale_pitches = self.music_theory.get_scale_pitches_from_string("C major", octave_range=3)

    def adapt_rhythm_to_ambient_style(
        self, base_pattern: List[Note], 
        ambient_metrics: Dict[str, float],
        adaptation_mode: str = "full"
    ) -> List[Note]:
        """
        Main adaptation function for ambient style.

        Adjusts note durations, velocities, placement based on metrics.
        Modes: "full" (all adaptations), "duration", "velocity", "sparsity", "complexity".

        Args:
            base_pattern: List[Note] from e.g., AmbientRhythmEngine
            ambient_metrics: Dict with 'tempo', 'mood_sparsity', 'complexity', etc.
            adaptation_mode: Mode of adaptation

        Returns:
            Adapted List[Note]
        """
        if not base_pattern:
            return []

        adapted = base_pattern.copy()
        metrics = AdaptationMetrics(
            tempo=ambient_metrics.get('tempo', 120.0),
            
            sparsity=ambient_metrics.get('sparsity', 0.5),
            complexity=ambient_metrics.get('complexity', 0.5),
            density=self._analyze_pattern_density(base_pattern)
        )

        if adaptation_mode in ["full", "duration"]:
            adapted = self._adapt_note_durations(adapted, metrics)
        if adaptation_mode in ["full", "velocity"]:
            adapted = self._modify_velocities_for_atmosphere(adapted, metrics)
        if adaptation_mode in ["full", "sparsity"]:
            adapted = self._apply_sparse_placement(adapted, metrics.sparsity)
        if adaptation_mode in ["full", "complexity"]:
            adapted = self._adjust_complexity_by_tempo(adapted, metrics)
        if adaptation_mode in ["full", "mood"]:
            adapted = self._scale_sparsity_by_mood(adapted, metrics.mood, metrics.sparsity)
        if adaptation_mode == "full":
            adapted = self._modify_based_on_content_analysis(adapted, metrics.density)

        # Ensure pitches stay in scale
        adapted = [self._ensure_scale_pitch(note) for note in adapted]
        adapted.sort(key=lambda n: n.start_time)
        return adapted

    def _adapt_note_durations(self, notes: List[Note], metrics: AdaptationMetrics) -> List[Note]:
        """Adapt durations based on ambient characteristics (e.g., sustain for calm)."""
        for note in notes:
            # Increase sustain for slower tempos or calm moods
            sustain_factor = 1.0 + (120.0 / metrics.tempo) * 0.5  # Slower = longer
            if metrics.mood == "calm":
                sustain_factor *= 1.3
            new_duration = note.duration * sustain_factor
            # Cap to avoid overlap issues
            new_duration = min(new_duration, 8.0)
            note.duration = new_duration
        return notes

    def _modify_velocities_for_atmosphere(self, notes: List[Note], metrics: AdaptationMetrics) -> List[Note]:
        """Modify velocities for atmospheric feel (softer for ambient)."""
        base_vel = get_velocity_for_mood(metrics.mood)
        for note in notes:
            # Softer for calm, atmospheric
            if metrics.mood in ["calm", "sad"]:
                vel_factor = 0.6 + random.uniform(-0.1, 0.1)
            else:
                vel_factor = 0.8 + random.uniform(-0.1, 0.1)
            # Fade based on duration (longer notes softer)
            fade_factor = 1.0 - (note.duration / 8.0) * 0.4
            new_vel = int(base_vel * vel_factor * fade_factor)
            note.velocity = max(20, min(127, new_vel))
        return notes

    def _apply_sparse_placement(self, notes: List[Note], sparsity: float) -> List[Note]:
        """Apply sparse placement: probabilistically remove/add notes."""
        sparse_notes = []
        for note in notes:
            if random.random() > sparsity:  # Higher sparsity removes more
                sparse_notes.append(note)
            else:
                # Occasionally add a sparse accent
                if random.random() < sparsity * 0.2:
                    accent = Note(
                        pitch=note.pitch,
                        duration=note.duration * 0.5,
                        velocity=int(note.velocity * 1.2),
                        start_time=note.start_time + note.duration * 0.5,
                        channel=note.channel
                    )
                    sparse_notes.append(accent)
        return sparse_notes

    def _adjust_complexity_by_tempo(self, notes: List[Note], metrics: AdaptationMetrics) -> List[Note]:
        """Adjust complexity: slower tempo = more sustained, less notes."""
        if metrics.tempo < 80:
            # More sustained: merge short notes
            merged = []
            current_note = None
            for note in sorted(notes, key=lambda n: n.start_time):
                if current_note is None or abs(note.start_time - (current_note.start_time + current_note.duration)) > 0.1:
                    if current_note:
                        merged.append(current_note)
                    current_note = note
                else:
                    # Merge: extend duration, average velocity
                    current_note.duration += note.duration
                    current_note.velocity = int((current_note.velocity + note.velocity) / 2)
            if current_note:
                merged.append(current_note)
            return merged
        return notes

    def _scale_sparsity_by_mood(self, notes: List[Note], mood: str, base_sparsity: float) -> List[Note]:
        """Scale sparsity with mood: calm = sparser."""
        mood_sparsity = base_sparsity
        if mood == "calm":
            mood_sparsity *= 1.5  # Sparser
        elif mood == "energetic":
            mood_sparsity *= 0.7  # Denser
        mood_sparsity = min(1.0, max(0.0, mood_sparsity))
        return self._apply_sparse_placement(notes, mood_sparsity)

    def _modify_based_on_content_analysis(self, notes: List[Note], density: float) -> List[Note]:
        """Modify patterns based on existing content: balance density."""
        if density > 0.7:  # High density: sparsify
            return self._apply_sparse_placement(notes, 0.6)
        elif density < 0.3:  # Low density: add notes
            augmented = notes.copy()
            for note in notes:
                if random.random() < 0.3:
                    new_note = Note(
                        pitch=random.choice(self.scale_pitches),
                        duration=note.duration * 0.7,
                        velocity=int(note.velocity * 0.8),
                        start_time=note.start_time + random.uniform(0, note.duration),
                        channel=note.channel
                    )
                    augmented.append(new_note)
            return augmented
        return notes

    def _analyze_pattern_density(self, notes: List[Note]) -> float:
        """Simple density analysis: notes per unit time."""
        if not notes:
            return 0.0
        total_duration = max((n.start_time + n.duration for n in notes), default=1.0)
        num_notes = len(notes)
        return num_notes / total_duration

    def _ensure_scale_pitch(self, note: Note) -> Note:
        """Ensure pitch is in scale, transpose if needed."""
        if not self.scale_pitches or note.pitch in self.scale_pitches:
            return note
        # Find closest in scale
        closest = min(self.scale_pitches, key=lambda p: abs(p - note.pitch))
        return Note(pitch=closest, duration=note.duration, velocity=note.velocity,
                    start_time=note.start_time, channel=note.channel)