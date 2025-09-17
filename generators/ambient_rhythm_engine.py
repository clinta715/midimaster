"""
Ambient Rhythm Engine Module

This module contains the AmbientRhythmEngine class for generating sustained note patterns
and sparse element placement suitable for ambient/electronic compositions. It focuses on
texture-based rhythm creation rather than traditional beat-based patterns, emphasizing
atmospheric pads, drones, and probabilistic element distribution.
"""

import random
from typing import List

from structures.data_structures import Note
from music_theory import MusicTheory, ScaleType

from .ambient_patterns import AmbientPatternTemplates

class AmbientRhythmEngine:
    """
    Engine for generating ambient rhythms with sustained notes and sparse placement.
    
    This class creates texture-based rhythmic patterns suitable for atmospheric pads,
    drones, and sparse electronic elements. It uses probabilistic algorithms to control
    note density and sustain characteristics, integrating with the existing rhythm
    generation system by producing lists of Note objects that can be incorporated
    into Patterns or used directly in composition pipelines.
    """
    
    def __init__(self, base_key: str = "C", base_scale: str = "major"):
        """
        Initialize the AmbientRhythmEngine.
        
        Args:
            base_key: Base key for scale generation (default: "C")
            base_scale: Base scale type (default: "major")
        """
        # Establish base scale for ambient textures
        scale_str = f"{base_key} {base_scale}"
        self.scale_pitches = MusicTheory.get_scale_pitches_from_string(scale_str, octave_range=3)
        if not self.scale_pitches:
            # Fallback to C major if scale generation fails
            self.scale_pitches = MusicTheory.get_scale_pitches_from_string("C major", octave_range=3)
        
        # Ambient-friendly pitch ranges: lower for drones, mid for pads
        self.drone_range = [p for p in self.scale_pitches if 36 <= p <= 60]
        self.pad_range = [p for p in self.scale_pitches if 48 <= p <= 72]
        self.sparse_range = [p for p in self.scale_pitches if 60 <= p <= 84]
    
    def generate_ambient_rhythm(self, duration_beats: float, sparsity_level: float, 
                               sustain_probability: float) -> List[Note]:
        """
        Generate sustained atmospheric notes with sparse placement.
        
        This method creates texture-based ambient rhythms using probabilistic placement
        and variable sustain lengths. Notes are layered to form atmospheric textures
        rather than rigid beat patterns. The output is a list of Note objects compatible
        with the existing rhythm generation system (e.g., can be wrapped in a Pattern).
        
        Args:
            duration_beats: Total duration of the pattern in beats
            sparsity_level: Sparsity control (0.0 = dense, 1.0 = very sparse)
            sustain_probability: Probability of generating sustained notes (0.0-1.0)
            
        Returns:
            List[Note]: List of generated ambient notes
        """
        if not 0.0 <= sparsity_level <= 1.0:
            raise ValueError("sparsity_level must be between 0.0 and 1.0")
        if not 0.0 <= sustain_probability <= 1.0:
            raise ValueError("sustain_probability must be between 0.0 and 1.0")
        
        notes = []
        
        # Calculate expected number of notes based on sparsity (texture density)
        base_density = 0.8  # Base notes per beat for ambient textures
        expected_notes = int(duration_beats * base_density * (1 - sparsity_level))
        
        # Generate layers: drones (long sustain), pads (medium), sparse elements (short)
        layers = [
            ("drone", self.drone_range, 0.3, sustain_probability * 0.8),  # Rare long sustains
            ("pad", self.pad_range, 0.5, sustain_probability * 0.6),      # Medium textures
            ("sparse", self.sparse_range, 0.2, sustain_probability * 0.3) # Accents/elements
        ]
        
        for layer_name, pitch_range, layer_weight, layer_sustain_prob in layers:
            layer_notes = int(expected_notes * layer_weight)
            for _ in range(layer_notes):
                # Random start time within duration (allows overlap for texture)
                start_time = random.uniform(0, duration_beats)
                
                # Duration: sustained or short based on probability and layer
                if random.random() < layer_sustain_prob:
                    # Sustained notes (drones/pads)
                    min_dur, max_dur = (3.0, duration_beats * 0.75) if layer_name == "drone" else (1.0, 3.0)
                    duration = random.uniform(min_dur, min(max_dur, duration_beats - start_time))
                else:
                    # Sparse elements (short attacks)
                    duration = random.uniform(0.25, 1.0)
                
                # Select pitch from layer-appropriate range
                pitch = random.choice(pitch_range)
                
                # Soft velocities for ambient feel, varied by layer
                base_vel = 40 if layer_name == "drone" else 60 if layer_name == "pad" else 70
                velocity = int(base_vel + random.uniform(-10, 20))
                velocity = max(20, min(127, velocity))  # Ensure audible but soft
                
                # Channel variation for layering (1-4 for ambient stems)
                channel = random.randint(1, 4)
                
                note = Note(pitch=pitch, duration=duration, velocity=velocity, 
                           start_time=start_time, channel=channel)
                notes.append(note)
        
        # Sort notes by start_time for sequential processing compatibility
        notes.sort(key=lambda n: n.start_time)
        
        return notes
    
    def _generate_texture_layer(self, duration_beats: float, pitch_range: List[int], 
                               num_notes: int, sustain_prob: float, layer_type: str) -> List[Note]:
        """
        Internal helper to generate a single texture layer.
        
        Args:
            duration_beats: Duration for this layer
            pitch_range: Available pitches for the layer
            num_notes: Number of notes in this layer
            sustain_prob: Sustain probability for this layer
            layer_type: Type of layer ("drone", "pad", "sparse")
            
        Returns:
            List[Note]: Notes for this texture layer
        """
        layer_notes = []
        for _ in range(num_notes):
            start_time = random.uniform(0, duration_beats)
            
            if random.random() < sustain_prob:
                if layer_type == "drone":
                    duration = random.uniform(4.0, 8.0)
                else:
                    duration = random.uniform(1.5, 4.0)
            else:
                duration = random.uniform(0.125, 0.5)
            
            duration = min(duration, duration_beats - start_time)  # Don't exceed end
            
            pitch = random.choice(pitch_range)
            velocity = random.randint(30, 70)
            
            layer_notes.append(Note(pitch, duration, velocity, start_time))
        
        return layer_notes