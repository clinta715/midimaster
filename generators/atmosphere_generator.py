"""
Atmosphere Generator Module

This module contains the AtmosphereGenerator class for creating layered atmospheric
textures with sustained notes, subtle rhythmic variations, and dynamic evolution.
It builds upon the AmbientRhythmEngine for base texture generation and adds
specialized features for ambient-specific effects like reverb simulation and fades.
"""

import random
from typing import List

from structures.data_structures import Note
from music_theory import MusicTheory
from generators.ambient_rhythm_engine import AmbientRhythmEngine
from generators.generator_context import GeneratorContext
from generators.generator_utils import get_velocity_for_mood

from .dynamic_rhythm_adaptor import DynamicRhythmAdaptor

from .ambient_patterns import AmbientPatternTemplates
class AtmosphereGenerator:
    """
    Generator for layered atmospheric textures with dynamic evolution.

    This class creates evolving atmospheric layers using the AmbientRhythmEngine
    as a foundation, then applies micro-timing variations, texture evolution curves,
    and ambient effects like simulated reverb tails (via overlapping sustains) and
    gradual fades (via velocity gradients across overlapping notes).

    Integrates with GeneratorContext for shared state (key, scale, mood, density)
    and works in conjunction with AmbientRhythmEngine for base rhythm/texture.
    """

    def __init__(self, context: GeneratorContext):
        """
        Initialize the AtmosphereGenerator.

        Args:
            context: Shared generator context containing key, scale, mood, and density settings.
        """
        self.context = context
        self.music_theory = MusicTheory()
        # Initialize ambient engine with current key/scale from context
        self.ambient_engine = AmbientRhythmEngine(
            base_key=context.current_key,
            base_scale=context.current_scale
        )
        # Get scale pitches from context (fallback to music_theory if needed)
        if not self.context.scale_pitches:
            scale_str = f"{context.current_key} {context.current_scale}"
            self.context.scale_pitches = self.music_theory.get_scale_pitches_from_string(
                scale_str, octave_range=3
            )
        self.scale_pitches = self.context.scale_pitches

        # Define texture-specific pitch ranges (expanded from ambient engine)
        self.texture_ranges = {
            "drone": [p for p in self.scale_pitches if 36 <= p <= 60],  # Low sustains
            "pad": [p for p in self.scale_pitches if 48 <= p <= 72],     # Mid textures
            "ethereal": [p for p in self.scale_pitches if 60 <= p <= 84], # High sparse
            "ambient": [p for p in self.scale_pitches if 48 <= p <= 72]   # General mid
        }

        # Micro-timing parameters for subtle variations (in beats, ~human feel)
        self.micro_timing_range = 0.05  # ±5% of beat for subtle offsets
        self.performance = context.get_performance()  # Access performance profile
        self.pattern_templates = AmbientPatternTemplates(self.context)

    def create_atmospheric_layer(
        self, duration_beats: float, complexity: float, texture_type: str
    ) -> List[Note]:
        """
        Generate layered atmospheric notes with evolution and effects.

        This method uses the AmbientRhythmEngine for base texture, then enhances
        with:
        - Layered notes for depth
        - Subtle micro-timing variations
        - Dynamic texture evolution (velocity/density curves over time)
        - Ambient effects: reverb tails (overlapping long sustains), gradual fades
          (velocity decay via overlapping notes)

        Args:
            duration_beats: Total duration of the layer in beats
            complexity: Complexity level (0.0=simple, 1.0=intricate textures)
            texture_type: Type of texture ("drone", "pad", "ethereal", "ambient")

        Returns:
            List[Note]: Generated atmospheric notes with applied variations
        """
        if not 0.0 <= complexity <= 1.0:
            raise ValueError("complexity must be between 0.0 and 1.0")
        if texture_type not in self.texture_ranges:
            texture_type = "ambient"  # Fallback

        # Base generation using AmbientRhythmEngine
        sparsity_level = max(0.2, 1.0 - complexity * 0.8)  # Less sparse for complex
        sustain_probability = min(0.9, 0.5 + complexity * 0.4)  # More sustains for complex
        base_notes = self.ambient_engine.generate_ambient_rhythm(
            duration_beats, sparsity_level, sustain_probability
        )
        # Integrate dynamic rhythm adaptation
        adaptor = DynamicRhythmAdaptor(self.context)
        ambient_metrics = {
            'tempo': getattr(self.context, 'tempo', 120.0),
            'sparsity': sparsity_level,
            'complexity': complexity
        }
        base_notes = adaptor.adapt_rhythm_to_ambient_style(base_notes, ambient_metrics)

        # Select texture-specific range
        pitch_range = self.texture_ranges[texture_type]
        base_velocity = get_velocity_for_mood(self.context.mood)

        # Apply subtle rhythmic variations (micro-timing)
        varied_notes = self._apply_micro_timing(base_notes, duration_beats)

        # Implement dynamic texture evolution: divide into segments and vary density/velocity
        evolved_notes = self._apply_texture_evolution(
            varied_notes, duration_beats, complexity, base_velocity
        )

        # Add ambient-specific effects: reverb tails and gradual fades
        final_notes = self._add_ambient_effects(
            evolved_notes, duration_beats, complexity, texture_type, base_velocity
        )

        # Sort by start_time for compatibility
        final_notes.sort(key=lambda n: n.start_time)
        return final_notes

    def _apply_micro_timing(self, notes: List[Note], total_duration: float) -> List[Note]:
        """
        Apply subtle micro-timing variations to notes.

        Adds small random offsets to start_time for human-like feel, using
        performance profile settings.

        Args:
            notes: Base notes to vary
            total_duration: Total duration for scaling offsets

        Returns:
            List[Note]: Notes with micro-timing applied
        """
        varied = []
        for note in notes:
            # Base offset from performance profile
            offset_range = self.micro_timing_range * self.performance.micro_timing_range_ms / 1000.0
            # Scale by note duration for proportional feel
            note_offset = random.uniform(-offset_range * note.duration, offset_range * note.duration)
            # Apply swing if enabled
            if self.performance.swing_mode != "none":
                swing_factor = self._get_swing_offset(note.start_time, self.performance.swing_mode)
                note_offset += swing_factor * note.duration

            new_start = max(0.0, note.start_time + note_offset)
            # Adjust duration if offset pushes beyond end
            adjusted_duration = note.duration - (new_start - note.start_time)
            adjusted_duration = min(adjusted_duration, total_duration - new_start)

            varied_note = Note(
                pitch=note.pitch,
                duration=adjusted_duration,
                velocity=note.velocity,
                start_time=new_start,
                channel=note.channel
            )
            varied.append(varied_note)
        return varied

    def _get_swing_offset(self, beat_position: float, swing_mode: str) -> float:
        """
        Calculate swing offset based on mode.

        Args:
            beat_position: Current beat position
            swing_mode: Swing type ("eighth", "sixteenth", "none")

        Returns:
            float: Swing offset factor (-1 to 1)
        """
        if swing_mode == "none":
            return 0.0
        # Simple eighth-note swing: delay off-beats
        beat_frac = beat_position % 1.0
        if 0.5 <= beat_frac < 1.0:  # Off-beat
            return random.uniform(0.05, 0.15)  # Subtle delay
        return random.uniform(-0.05, 0.0)  # Slight anticipation on-beat

    def _apply_texture_evolution(
        self, notes: List[Note], duration_beats: float, complexity: float, base_velocity: int
    ) -> List[Note]:
        """
        Apply dynamic texture evolution over time.

        Divides duration into 3-5 segments (based on complexity) and applies
        evolving curves: increasing density/velocity for build-up, then fade.

        Args:
            notes: Notes to evolve
            duration_beats: Total duration
            complexity: Guides number of evolution segments
            base_velocity: Base velocity for scaling

        Returns:
            List[Note]: Evolved notes with varying velocity/density
        """
        # Number of evolution segments (more for complex textures)
        num_segments = max(3, int(3 + complexity * 2))
        segment_length = duration_beats / num_segments

        evolved = []
        for note in notes:
            # Find segment for this note
            segment_idx = int(note.start_time / segment_length)
            segment_progress = (note.start_time % segment_length) / segment_length

            # Evolution curve: build (0-0.5), peak (0.5), decay (0.5-1.0) per segment
            global_progress = segment_idx / num_segments + segment_progress / num_segments
            if global_progress < 0.5:
                # Build-up: increase velocity/density
                vel_factor = 0.7 + (global_progress * 0.6 * complexity)
                # Optionally add more notes for density (simulate by duplicating short ones)
                if random.random() < complexity * 0.3 and note.duration < 1.0:
                    extra_note = Note(
                        pitch=note.pitch,
                        duration=note.duration * 0.5,
                        velocity=int(note.velocity * 0.7),
                        start_time=note.start_time + note.duration * 0.3,
                        channel=note.channel + 1 if note.channel < 16 else 1
                    )
                    evolved.append(extra_note)
            else:
                # Decay: decrease velocity
                decay_progress = (global_progress - 0.5) * 2
                vel_factor = 1.0 - (decay_progress * 0.5)

            new_velocity = int(base_velocity * vel_factor + random.uniform(-5, 5))
            new_velocity = max(20, min(127, new_velocity))

            evolved_note = Note(
                pitch=note.pitch,
                duration=note.duration,
                velocity=new_velocity,
                start_time=note.start_time,
                channel=note.channel
            )
            evolved.append(evolved_note)

        return evolved

    def _add_ambient_effects(
        self, notes: List[Note], duration_beats: float, complexity: float,
        texture_type: str, base_velocity: int
    ) -> List[Note]:
        """
        Add ambient-specific effects: reverb tails and gradual fades.

        Simulates reverb with overlapping sustained tails (longer notes fading out).
        Gradual fades via layered decreasing-velocity notes.

        Args:
            notes: Base evolved notes
            duration_beats: Total duration
            complexity: Influences effect intensity
            texture_type: Guides effect style
            base_velocity: For scaling fades

        Returns:
            List[Note]: Notes with ambient effects applied
        """
        enhanced = notes.copy()
        effect_intensity = complexity * 0.5  # Scale effects by complexity

        for note in notes:
            # Reverb tails: add 1-3 overlapping sustains if long enough
            if note.duration > 1.0 and random.random() < effect_intensity:
                num_tails = min(3, int(1 + complexity * 2))
                tail_decay = 0.8  # Velocity decay per tail
                current_vel = note.velocity
                tail_start = note.start_time + note.duration * 0.2  # Overlap start
                tail_duration = note.duration * random.uniform(1.2, 2.0)
                tail_duration = min(tail_duration, duration_beats - tail_start)

                for _ in range(num_tails):
                    if tail_start < duration_beats:
                        tail_note = Note(
                            pitch=note.pitch,
                            duration=tail_duration,
                            velocity=int(current_vel * random.uniform(0.3, 0.6)),
                            start_time=tail_start,
                            channel=(note.channel % 16) + 1  # Alternate channels for layering
                        )
                        enhanced.append(tail_note)
                        current_vel *= tail_decay
                        tail_start += note.duration * 0.1  # Stagger tails

            # Gradual fades: for shorter notes, add 2-4 fading overlaps
            if note.duration < 2.0 and random.random() < effect_intensity * 0.7:
                num_fades = min(4, int(2 + complexity * 2))
                fade_vel = note.velocity
                fade_offset = note.duration / (num_fades + 1)
                for i in range(num_fades):
                    fade_start = note.start_time + i * fade_offset
                    if fade_start + fade_offset > duration_beats:
                        break
                    fade_note = Note(
                        pitch=note.pitch,
                        duration=fade_offset * random.uniform(0.8, 1.2),
                        velocity=int(fade_vel * (1.0 - i * 0.2)),  # Linear decay
                        start_time=fade_start,
                        channel=note.channel
                    )
                    enhanced.append(fade_note)
                    fade_vel *= 0.7  # Exponential decay

        return enhanced