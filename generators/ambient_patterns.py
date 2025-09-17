"""
Ambient Pattern Templates Module

This module provides rhythm pattern templates specifically designed for
ambient/electronic compositions. These templates create sustained drones,
sparse percussion, texture-based evolutions, and micro-rhythm variations
that integrate with AmbientRhythmEngine and AtmosphereGenerator.

Templates support subgenre variations (dark, ethereal, cinematic) and
leverage DynamicRhythmAdaptor for tempo/mood-based complexity adjustments.
"""

import random
from typing import List, Dict, Optional

from structures.data_structures import Note
from generators.generator_context import GeneratorContext
from generators.dynamic_rhythm_adaptor import DynamicRhythmAdaptor
from music_theory import MusicTheory
from generators.generator_utils import get_velocity_for_mood


class AmbientPatternTemplates:
    """
    Provides specialized rhythm pattern templates for ambient/electronic music.

    These templates generate Note lists compatible with the existing pattern
    system. They can be used directly in AmbientRhythmEngine or AtmosphereGenerator
    and are adapted via DynamicRhythmAdaptor for dynamic complexity based on
    tempo (slower = simpler/sustained) and mood (calm = sparser).

    Subgenres:
    - 'dark': Ominous, low-register, sparse
    - 'ethereal': Airy, high-register, sustained
    - 'cinematic': Evolving, mid-register, dramatic
    """

    def __init__(self, context: Optional[GeneratorContext] = None):
        """
        Initialize with optional GeneratorContext for shared state.

        Args:
            context: GeneratorContext for mood, scale, etc. (default: creates basic)
        """
        if context:
            self.context = context
            self.music_theory = context.music_theory
            self.scale_pitches = context.scale_pitches or []
            self.mood = context.mood
        else:
            self.context = None
            self.music_theory = MusicTheory()
            self.scale_pitches = self.music_theory.get_scale_pitches_from_string("C dorian", octave_range=3)
            self.mood = "calm"
        
        self.adaptor = DynamicRhythmAdaptor(context)
        self.subgenre_params = self._get_subgenre_params()

    def _get_subgenre_params(self) -> Dict[str, Dict[str, float]]:
        """Subgenre-specific parameters for variation."""
        return {
            'dark': {
                'sparsity': 0.8, 'sustain_factor': 1.2, 'velocity_factor': 0.6,
                'pitch_offset': -12, 'complexity': 0.3  # Low, ominous
            },
            'ethereal': {
                'sparsity': 0.6, 'sustain_factor': 1.5, 'velocity_factor': 0.7,
                'pitch_offset': 12, 'complexity': 0.5  # Airy, sustained
            },
            'cinematic': {
                'sparsity': 0.5, 'sustain_factor': 1.3, 'velocity_factor': 0.8,
                'pitch_offset': 0, 'complexity': 0.7  # Evolving, dramatic
            }
        }

    def sustained_drone_pattern(
        self, duration_beats: float, subgenre: str = 'ethereal',
        num_layers: int = 3
    ) -> List[Note]:
        """
        Generate sustained drone patterns with subtle variations.

        Creates long, overlapping notes in low-mid register with gradual
        pitch/velocity shifts for evolution. Adapted for tempo: slower = longer drones.

        Args:
            duration_beats: Total duration in beats
            subgenre: 'dark', 'ethereal', 'cinematic'
            num_layers: Number of overlapping drone layers

        Returns:
            List[Note]: Sustained drone notes
        """
        params = self.subgenre_params.get(subgenre, self.subgenre_params['ethereal'])
        notes = []
        base_velocity = get_velocity_for_mood(self.mood) * params['velocity_factor']
        
        for layer in range(num_layers):
            # Layer-specific pitch range (low for drones)
            pitch_range = [p + params['pitch_offset'] for p in self.scale_pitches if 36 <= p <= 60]
            if not pitch_range:
                pitch_range = [48, 52, 55]  # Fallback low pitches
            
            start_time = random.uniform(0, duration_beats * 0.3)  # Stagger starts
            # Long sustains, adjusted by subgenre and tempo via adaptor
            base_duration = duration_beats * random.uniform(0.6, 0.9) * params['sustain_factor']
            duration = min(base_duration, duration_beats - start_time)
            
            pitch = random.choice(pitch_range)
            velocity = int(base_velocity + random.uniform(-10, 10))
            velocity = max(20, min(80, velocity))  # Soft for drones
            
            note = Note(
                pitch=int(pitch), duration=duration, velocity=velocity,
                start_time=start_time, channel=layer + 1  # Layer on channels
            )
            notes.append(note)
        
        # Adapt for dynamic complexity (e.g., slower tempo = fewer layers)
        metrics = {
            'tempo': getattr(self.context, 'tempo', 60.0) if self.context else 60.0,
            'sparsity': params['sparsity'], 'complexity': params['complexity']
        }
        adapted_notes = self.adaptor.adapt_rhythm_to_ambient_style(notes, metrics)
        return adapted_notes

    def sparse_percussion_layer(
        self, duration_beats: float, subgenre: str = 'ethereal',
        hit_density: float = 0.2
    ) -> List[Note]:
        """
        Generate sparse percussion layers with occasional atmospheric hits.

        Places infrequent, soft hits (e.g., distant kicks, chimes) for subtle pulse.
        Density adapts to mood: calm = sparser.

        Args:
            duration_beats: Total duration in beats
            subgenre: 'dark', 'ethereal', 'cinematic'
            hit_density: Base probability of hits per beat (0.0-1.0)

        Returns:
            List[Note]: Sparse percussion notes
        """
        params = self.subgenre_params.get(subgenre, self.subgenre_params['ethereal'])
        notes = []
        adjusted_density = hit_density * (1 - params['sparsity'])  # Sparser for dark
        
        # Percussion pitches: low for kicks, high for chimes
        perc_pitches = {
            'dark': [35, 42],  # Kick, low tom
            'ethereal': [76, 80],  # High chimes
            'cinematic': [42, 52]  # Mid toms/bells
        }.get(subgenre, [42, 52])
        
        base_velocity = int(get_velocity_for_mood(self.mood) * 0.4)  # Soft hits
        
        # Place hits probabilistically
        time_step = 0.5  # Half-beat steps for sparsity
        current_time = 0.0
        while current_time < duration_beats:
            if random.random() < adjusted_density:
                pitch = random.choice(perc_pitches)
                duration = random.uniform(0.25, 0.5)  # Short hits
                velocity = int(base_velocity + random.uniform(-15, 5))
                velocity = max(15, min(60, velocity))  # Very soft
                
                note = Note(
                    pitch=int(pitch), duration=duration, velocity=velocity,
                    start_time=current_time, channel=10  # Perc channel
                )
                notes.append(note)
            
            current_time += time_step
        
        metrics = {
            'tempo': getattr(self.context, 'tempo', 60.0) if self.context else 60.0,
            'sparsity': params['sparsity'], 'complexity': params['complexity']
        }
        adapted_notes = self.adaptor.adapt_rhythm_to_ambient_style(notes, metrics)
        return adapted_notes

    def texture_based_rhythm(
        self, duration_beats: float, subgenre: str = 'cinematic',
        evolution_stages: int = 4
    ) -> List[Note]:
        """
        Generate texture-based rhythms for soundscape evolution.

        Creates evolving layers that build/dissipate over time, simulating
        atmospheric development. Stages divide duration for gradual changes.

        Args:
            duration_beats: Total duration in beats
            subgenre: 'dark', 'ethereal', 'cinematic'
            evolution_stages: Number of evolution phases

        Returns:
            List[Note]: Evolving texture notes
        """
        params = self.subgenre_params.get(subgenre, self.subgenre_params['cinematic'])
        notes = []
        stage_length = duration_beats / evolution_stages
        base_velocity = get_velocity_for_mood(self.mood) * params['velocity_factor']
        
        for stage in range(evolution_stages):
            stage_start = stage * stage_length
            stage_progress = stage / evolution_stages
            # Evolve density/velocity: build to peak, then fade
            stage_density = min(1.0, params['complexity'] * (0.5 + stage_progress * 1.5))
            stage_velocity_factor = 0.5 + stage_progress * 0.8 if stage_progress < 0.7 else 1.0 - (stage_progress - 0.7) * 2
            
            num_notes_stage = int(stage_length * stage_density * 0.5)  # Sparse base
            pitch_range = [p + params['pitch_offset'] for p in self.scale_pitches if 48 <= p <= 72]
            
            for _ in range(num_notes_stage):
                start_time = stage_start + random.uniform(0, stage_length)
                duration = random.uniform(0.5, 2.0) * params['sustain_factor']
                duration = min(duration, duration_beats - start_time)
                
                pitch = random.choice(pitch_range)
                velocity = int(base_velocity * stage_velocity_factor + random.uniform(-5, 5))
                
                note = Note(
                    pitch=int(pitch), duration=duration, velocity=velocity,
                    start_time=start_time, channel=random.randint(1, 8)
                )
                notes.append(note)
        
        metrics = {
            'tempo': getattr(self.context, 'tempo', 60.0) if self.context else 60.0,
            'sparsity': params['sparsity'], 'complexity': params['complexity']
        }
        adapted_notes = self.adaptor.adapt_rhythm_to_ambient_style(notes, metrics)
        return adapted_notes

    def micro_rhythm_variations(
        self, base_notes: List[Note], subgenre: str = 'ethereal',
        variation_intensity: float = 0.05
    ) -> List[Note]:
        """
        Apply micro-rhythm variations to base notes.

        Adds subtle timing offsets (± variation_intensity of beat) and
        slight duration/velocity jitter for organic feel. Integrates with
        performance profile if available.

        Args:
            base_notes: Existing notes to vary
            subgenre: Influences variation style
            variation_intensity: Timing variation as fraction of beat (0.0-0.1)

        Returns:
            List[Note]: Varied notes
        """
        if not base_notes:
            return []
        
        params = self.subgenre_params.get(subgenre, self.subgenre_params['ethereal'])
        varied_notes = []
        performance = getattr(self.context, 'performance', None) if self.context else None
        
        for note in base_notes:
            # Micro-timing: subtle offset
            timing_offset = random.uniform(-variation_intensity, variation_intensity) * note.duration
            new_start = max(0.0, note.start_time + timing_offset)
            
            # Adjust duration if offset
            adjusted_duration = note.duration - timing_offset
            adjusted_duration = min(adjusted_duration, 16.0 - new_start)  # Cap
            
            # Subtle duration variation
            dur_jitter = random.uniform(0.95, 1.05) * params['sustain_factor']
            adjusted_duration *= dur_jitter
            
            # Velocity jitter for expression
            vel_jitter = random.uniform(0.95, 1.05)
            new_velocity = int(note.velocity * vel_jitter)
            
            # Swing if from performance
            if performance and performance.swing_mode != "none":
                swing_offset = 0.02 if random.random() > 0.5 else -0.01  # Subtle
                new_start += swing_offset * note.duration
            
            varied_note = Note(
                pitch=note.pitch, duration=adjusted_duration, velocity=new_velocity,
                start_time=new_start, channel=note.channel
            )
            varied_notes.append(varied_note)
        
        # Final adaptation
        metrics = {
            'tempo': getattr(self.context, 'tempo', 60.0) if self.context else 60.0,
            'sparsity': params['sparsity'], 'complexity': params['complexity']
        }
        return self.adaptor.adapt_rhythm_to_ambient_style(varied_notes, metrics)

    def generate_full_ambient_pattern(
        self, duration_beats: float, subgenre: str, pattern_type: str = 'all'
    ) -> List[Note]:
        """
        Generate a complete ambient pattern combining templates.

        Args:
            duration_beats: Total duration
            subgenre: Subgenre
            pattern_type: 'drone', 'perc', 'texture', 'micro', or 'all'

        Returns:
            Combined List[Note]
        """
        if pattern_type == 'all':
            drones = self.sustained_drone_pattern(duration_beats * 0.6, subgenre)
            perc = self.sparse_percussion_layer(duration_beats, subgenre)
            texture = self.texture_based_rhythm(duration_beats, subgenre)
            # Apply micro to combined
            combined = drones + perc + texture
            return self.micro_rhythm_variations(combined, subgenre)
        elif pattern_type == 'drone':
            return self.sustained_drone_pattern(duration_beats, subgenre)
        elif pattern_type == 'perc':
            return self.sparse_percussion_layer(duration_beats, subgenre)
        elif pattern_type == 'texture':
            return self.texture_based_rhythm(duration_beats, subgenre)
        elif pattern_type == 'micro':
            base = self.sustained_drone_pattern(duration_beats * 0.5, subgenre)
            return self.micro_rhythm_variations(base, subgenre)
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")