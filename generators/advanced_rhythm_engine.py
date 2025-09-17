"""
Advanced Rhythm Engine Module

This module contains the AdvancedRhythmEngine class and its supporting engines
(PolyrhythmEngine and SyncopationEngine) for sophisticated rhythmic pattern generation.
"""

import random
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from structures.data_structures import Note, Pattern, PatternType


@dataclass
class RhythmPattern:
    """Represents a complex rhythmic pattern with multiple layers."""
    notes: List[Note]
    tempo: float
    time_signature: Tuple[int, int]
    complexity: float
    groove_intensity: float


class PolyrhythmEngine:
    """Handles polyrhythmic pattern generation and cross-rhythm creation."""

    def __init__(self):
        self.polyrhythm_templates = self._load_polyrhythm_templates()
        self.cross_rhythm_patterns = self._load_cross_rhythm_patterns()

    def _load_polyrhythm_templates(self) -> Dict[str, Dict]:
        """Load predefined polyrhythm templates."""
        return {
            '3_over_4': {
                'primary_pulse': 4,
                'secondary_pulse': 3,
                'description': '3 notes over 4 beats (common in jazz)',
                'intensity_levels': ['subtle', 'moderate', 'intense']
            },
            '5_over_4': {
                'primary_pulse': 4,
                'secondary_pulse': 5,
                'description': '5 notes over 4 beats (West African influence)',
                'intensity_levels': ['subtle', 'moderate', 'intense']
            },
            '7_over_8': {
                'primary_pulse': 8,
                'secondary_pulse': 7,
                'description': '7 notes over 8 beats (Brazilian rhythms)',
                'intensity_levels': ['subtle', 'moderate', 'intense']
            },
            '2_over_3': {
                'primary_pulse': 3,
                'secondary_pulse': 2,
                'description': '2 notes over 3 beats (hemiola effect)',
                'intensity_levels': ['subtle', 'moderate', 'intense']
            }
        }

    def _load_cross_rhythm_patterns(self) -> Dict[str, List[float]]:
        """Load cross-rhythm pattern templates."""
        return {
            '3_over_4_basic': [0, 1.33, 2.67],  # 3 evenly spaced hits over 4 beats
            '3_over_4_syncopated': [0.25, 1.58, 2.92],  # Off-beat version
            '5_over_4_fibonacci': [0, 0.8, 1.6, 2.4, 3.2],  # Fibonacci spacing
            '7_over_8_triplet_feel': [0, 0.57, 1.14, 1.71, 2.29, 2.86, 3.43],  # 7 over 8
            'hemiola_2_over_3': [0, 1.5, 3.0]  # 2 over 3 pattern
        }

    def apply_polyrhythm(self, base_pattern: RhythmPattern, genre: str = 'jazz',
                        complexity: str = 'complex') -> RhythmPattern:
        """Apply polyrhythmic elements to a base rhythm pattern.

        Args:
            base_pattern: The base rhythmic pattern to modify
            genre: Target genre for polyrhythm selection
            complexity: Complexity level ('simple', 'complex', 'dense')

        Returns:
            RhythmPattern: Enhanced pattern with polyrhythmic elements
        """
        if complexity in ['simple', 'basic']:
            return base_pattern

        # Select appropriate polyrhythm based on genre and complexity
        polyrhythm_type = self._select_polyrhythm_type(genre, complexity)

        if polyrhythm_type:
            enhanced_pattern = self._apply_specific_polyrhythm(
                base_pattern, polyrhythm_type, genre
            )
            return enhanced_pattern

        return base_pattern

    def _select_polyrhythm_type(self, genre: str, complexity: str) -> Optional[str]:
        """Select appropriate polyrhythm type based on genre and complexity."""
        genre_polyrhythms = {
            'jazz': ['3_over_4', '7_over_8', '2_over_3'],
            'latin': ['3_over_4', '5_over_4', '7_over_8'],
            'african': ['3_over_4', '5_over_4', '7_over_8'],
            'electronic': ['3_over_4', '7_over_8'],
            'classical': ['2_over_3', '3_over_4'],
            'folk': ['2_over_3', '3_over_4']
        }

        available_types = genre_polyrhythms.get(genre, ['3_over_4'])

        if complexity == 'dense':
            # For dense complexity, use more complex polyrhythms
            complex_types = [t for t in available_types if t in ['5_over_4', '7_over_8', '2_over_3']]
            if complex_types:
                return random.choice(complex_types)

        return random.choice(available_types)

    def _apply_specific_polyrhythm(self, base_pattern: RhythmPattern,
                                 polyrhythm_type: str, genre: str) -> RhythmPattern:
        """Apply a specific polyrhythm type to the pattern."""
        template = self.polyrhythm_templates.get(polyrhythm_type)
        if not template:
            return base_pattern

        # Create polyrhythmic layer
        polyrhythmic_notes = []

        # Calculate timing for polyrhythmic layer
        primary_pulse = template['primary_pulse']
        secondary_pulse = template['secondary_pulse']

        # Determine which voice/instrument gets the polyrhythm
        polyrhythm_voice = self._select_polyrhythm_voice(base_pattern, genre)

        # Generate polyrhythmic timing
        pattern_duration = 4.0  # 4 beats (one bar in 4/4)
        polyrhythm_interval = pattern_duration / secondary_pulse

        for i in range(secondary_pulse):
            timing = i * polyrhythm_interval
            if timing < pattern_duration:
                # Create note with appropriate velocity and pitch
                velocity = self._calculate_polyrhythm_velocity(i, secondary_pulse, genre)
                pitch = self._select_polyrhythm_pitch(polyrhythm_voice, genre)

                note = Note(pitch, 0.1, velocity, timing)  # Short duration for polyrhythm
                polyrhythmic_notes.append(note)

        # Blend polyrhythmic notes with base pattern
        enhanced_notes = base_pattern.notes.copy() + polyrhythmic_notes

        return RhythmPattern(
            notes=enhanced_notes,
            tempo=base_pattern.tempo,
            time_signature=base_pattern.time_signature,
            complexity=min(1.0, base_pattern.complexity + 0.2),
            groove_intensity=base_pattern.groove_intensity
        )

    def _select_polyrhythm_voice(self, base_pattern: RhythmPattern, genre: str) -> str:
        """Select which voice/instrument should carry the polyrhythm."""
        # Analyze existing notes to determine voice distribution
        pitches = [note.pitch for note in base_pattern.notes]

        # Group by typical drum voices
        voice_groups = {
            'kick': [35, 36],  # Bass drum
            'snare': [37, 38, 39, 40],  # Snare variations
            'hihat': [41, 42, 43, 44, 45, 46],  # Hi-hat variations
            'percussion': [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]  # Other percussion
        }

        # Choose voice based on genre preferences
        if genre in ['latin', 'african']:
            return 'percussion'
        elif genre in ['jazz', 'funk']:
            return 'hihat' if random.random() < 0.7 else 'snare'
        else:
            return random.choice(['hihat', 'percussion'])

    def _calculate_polyrhythm_velocity(self, position: int, total_positions: int, genre: str) -> int:
        """Calculate velocity for polyrhythmic notes."""
        base_velocity = 60  # Moderate velocity

        # Add variation based on position
        position_factor = abs(position - total_positions // 2) / (total_positions // 2)
        velocity_variation = int(20 * (1 - position_factor))  # Stronger in middle

        # Genre-specific adjustments
        if genre in ['latin', 'african']:
            base_velocity += 15  # Louder for rhythmic emphasis
        elif genre == 'jazz':
            base_velocity -= 10  # Softer for subtlety

        return max(1, min(127, base_velocity + velocity_variation))

    def _select_polyrhythm_pitch(self, voice: str, genre: str) -> int:
        """Select appropriate pitch for polyrhythmic voice."""
        voice_pitches = {
            'kick': [35, 36],
            'snare': [37, 38, 39, 40],
            'hihat': [42, 44, 46],  # Closed, pedal, open hi-hat
            'percussion': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]  # Various percussion
        }

        pitches = voice_pitches.get(voice, [42])  # Default to hi-hat
        return random.choice(pitches)

    def generate_cross_rhythm(self, primary_pulse: int, secondary_pulse: int,
                            pattern_duration: float = 4.0) -> List[float]:
        """Generate cross-rhythmic timing patterns.

        Args:
            primary_pulse: Primary rhythmic pulse (e.g., 4 for quarter notes)
            secondary_pulse: Secondary rhythmic pulse (e.g., 3 for triplets)
            pattern_duration: Duration in beats

        Returns:
            List[float]: List of timing positions for cross-rhythm
        """
        # Calculate timing intervals
        primary_interval = pattern_duration / primary_pulse
        secondary_interval = pattern_duration / secondary_pulse

        # Generate cross-rhythmic positions
        positions = []
        for i in range(secondary_pulse):
            position = i * secondary_interval
            if position < pattern_duration:
                positions.append(position)

        return positions


class SyncopationEngine:
    """Handles sophisticated syncopation and groove generation."""

    def __init__(self):
        self.syncopation_patterns = self._load_syncopation_patterns()
        self.swing_profiles = self._load_swing_profiles()

    def _load_syncopation_patterns(self) -> Dict[str, Dict]:
        """Load predefined syncopation patterns."""
        return {
            'basic_offbeat': {
                'description': 'Simple off-beat accents',
                'positions': [0.5, 1.5, 2.5, 3.5],  # Off-beats in 4/4
                'intensity': 'light'
            },
            'salsa_tumbao': {
                'description': 'Salsa tumbao syncopation',
                'positions': [0.5, 1.0, 1.5, 2.5, 3.0, 3.5],
                'intensity': 'moderate'
            },
            'funk_ghost_notes': {
                'description': 'Funk ghost note syncopation',
                'positions': [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
                'intensity': 'heavy'
            },
            'jazz_comping': {
                'description': 'Jazz chord comping syncopation',
                'positions': [0.33, 0.67, 1.33, 1.67, 2.33, 2.67, 3.33, 3.67],
                'intensity': 'moderate'
            },
            'brazilian_samba': {
                'description': 'Brazilian samba syncopation patterns',
                'positions': [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75],
                'intensity': 'heavy'
            }
        }

    def _load_swing_profiles(self) -> Dict[str, Dict]:
        """Load swing timing profiles for different intensities."""
        return {
            'straight': {'ratio': 1.0, 'description': 'No swing'},
            'light': {'ratio': 1.1, 'description': 'Subtle swing'},
            'moderate': {'ratio': 1.2, 'description': 'Moderate swing'},
            'heavy': {'ratio': 1.33, 'description': 'Heavy swing (triplet feel)'},
            'extreme': {'ratio': 1.5, 'description': 'Extreme swing'}
        }

    def apply_syncopation(self, base_pattern: RhythmPattern, genre: str,
                         syncopation_level: float) -> RhythmPattern:
        """Apply sophisticated syncopation to a rhythm pattern.

        Args:
            base_pattern: Base pattern to syncopate
            genre: Musical genre for style selection
            syncopation_level: Intensity of syncopation (0.0-1.0)

        Returns:
            RhythmPattern: Syncopated rhythm pattern
        """
        if syncopation_level < 0.1:
            return base_pattern

        # Select appropriate syncopation pattern
        sync_pattern = self._select_syncopation_pattern(genre, syncopation_level)

        # Apply syncopation to the pattern
        syncopated_notes = self._apply_syncopation_to_notes(
            base_pattern.notes, sync_pattern, syncopation_level
        )

        return RhythmPattern(
            notes=syncopated_notes,
            tempo=base_pattern.tempo,
            time_signature=base_pattern.time_signature,
            complexity=min(1.0, base_pattern.complexity + syncopation_level * 0.3),
            groove_intensity=min(1.0, base_pattern.groove_intensity + syncopation_level * 0.2)
        )

    def _select_syncopation_pattern(self, genre: str, intensity: float) -> Dict:
        """Select appropriate syncopation pattern based on genre and intensity."""
        genre_patterns = {
            'jazz': ['basic_offbeat', 'jazz_comping'],
            'funk': ['funk_ghost_notes', 'basic_offbeat'],
            'latin': ['salsa_tumbao', 'brazilian_samba'],
            'rock': ['basic_offbeat'],
            'electronic': ['funk_ghost_notes', 'basic_offbeat'],
            'classical': [],  # Minimal syncopation
            'folk': ['basic_offbeat']
        }

        available_patterns = genre_patterns.get(genre, ['basic_offbeat'])

        if not available_patterns:
            return self.syncopation_patterns['basic_offbeat']

        pattern_name = random.choice(available_patterns)
        return self.syncopation_patterns[pattern_name]

    def _apply_syncopation_to_notes(self, notes: List[Note], sync_pattern: Dict,
                                  intensity: float) -> List[Note]:
        """Apply syncopation pattern to existing notes."""
        syncopated_notes = notes.copy()

        # Add syncopated accents
        pattern_duration = 4.0  # Assume 4/4 time
        sync_positions = sync_pattern['positions']

        for position in sync_positions:
            if random.random() < intensity:  # Probability based on intensity
                # Create syncopated note
                pitch = self._select_syncopation_pitch(sync_pattern)
                velocity = self._calculate_syncopation_velocity(intensity)

                # Add small timing variation for groove
                timing_variation = random.uniform(-0.05, 0.05)
                actual_position = position + timing_variation

                if 0 <= actual_position < pattern_duration:
                    sync_note = Note(pitch, 0.125, velocity, actual_position)
                    syncopated_notes.append(sync_note)

        return syncopated_notes

    def _select_syncopation_pitch(self, sync_pattern: Dict) -> int:
        """Select appropriate pitch for syncopated notes."""
        pattern_type = sync_pattern.get('description', '').lower()

        if 'ghost' in pattern_type:
            return random.choice([37, 38])  # Ghost snares
        elif 'comping' in pattern_type:
            return random.choice([42, 44, 46])  # Hi-hats
        elif 'tumbao' in pattern_type:
            return random.choice([39, 40])  # Claps/snaps
        else:
            return random.choice([42, 44])  # Hi-hats (default)

    def _calculate_syncopation_velocity(self, intensity: float) -> int:
        """Calculate velocity for syncopated notes."""
        base_velocity = 50
        intensity_bonus = int(intensity * 30)  # Up to +30 velocity for high intensity
        variation = random.randint(-10, 10)

        return max(1, min(127, base_velocity + intensity_bonus + variation))

    def generate_swing_feel(self, base_pattern: RhythmPattern, swing_intensity: float) -> RhythmPattern:
        """Generate swing feel with variable intensity.

        Args:
            base_pattern: Base pattern to apply swing to
            swing_intensity: Swing intensity (0.0-1.0)

        Returns:
            RhythmPattern: Pattern with swing timing applied
        """
        if swing_intensity < 0.1:
            return base_pattern

        # Select swing profile
        profile_name = self._select_swing_profile(swing_intensity)
        profile = self.swing_profiles[profile_name]

        # Apply swing timing to notes
        swung_notes = []

        for note in base_pattern.notes:
            swung_note = self._apply_swing_to_note(note, profile, swing_intensity)
            swung_notes.append(swung_note)

        return RhythmPattern(
            notes=swung_notes,
            tempo=base_pattern.tempo,
            time_signature=base_pattern.time_signature,
            complexity=base_pattern.complexity,
            groove_intensity=min(1.0, base_pattern.groove_intensity + swing_intensity * 0.3)
        )

    def _select_swing_profile(self, intensity: float) -> str:
        """Select swing profile based on intensity."""
        if intensity < 0.3:
            return 'light'
        elif intensity < 0.6:
            return 'moderate'
        elif intensity < 0.8:
            return 'heavy'
        else:
            return 'extreme'

    def _apply_swing_to_note(self, note: Note, swing_profile: Dict, intensity: float) -> Note:
        """Apply swing timing to an individual note."""
        swing_ratio = swing_profile['ratio']

        # Calculate beat position within the measure
        beat_position = note.start_time % 1.0  # Position within the beat

        # Apply swing to off-beats (positions 0.5, 1.5, etc.)
        if abs(beat_position - 0.5) < 0.1:  # Close to off-beat
            # Calculate swing delay
            swing_delay = (swing_ratio - 1.0) * 0.5 * intensity
            note.start_time += swing_delay

        return note


class AdvancedRhythmEngine:
    """Advanced rhythm pattern generation engine with polyrhythm and syncopation support."""

    def __init__(self):
        self.polyrhythm_engine = PolyrhythmEngine()
        self.syncopation_engine = SyncopationEngine()
        self.rhythm_patterns = self._load_rhythm_pattern_templates()
        self.tension_resolution_systems = self._load_tension_resolution_systems()

    def _load_rhythm_pattern_templates(self) -> Dict[str, Dict]:
        """Load rhythm pattern templates organized by genre and complexity."""
        return {
            'jazz': {
                'simple': {'description': 'Basic jazz swing', 'swing_intensity': 0.6, 'syncopation': 0.3},
                'complex': {'description': 'Complex jazz with polyrhythms', 'swing_intensity': 0.7, 'syncopation': 0.5},
                'dense': {'description': 'Dense jazz with heavy syncopation', 'swing_intensity': 0.8, 'syncopation': 0.7}
            },
            'funk': {
                'simple': {'description': 'Basic funk groove', 'swing_intensity': 0.3, 'syncopation': 0.4},
                'complex': {'description': 'Complex funk with ghost notes', 'swing_intensity': 0.4, 'syncopation': 0.6},
                'dense': {'description': 'Dense funk with heavy ghost notes', 'swing_intensity': 0.5, 'syncopation': 0.8}
            },
            'latin': {
                'simple': {'description': 'Basic latin rhythm', 'swing_intensity': 0.2, 'syncopation': 0.5},
                'complex': {'description': 'Complex latin with polyrhythms', 'swing_intensity': 0.3, 'syncopation': 0.7},
                'dense': {'description': 'Dense latin with layered rhythms', 'swing_intensity': 0.4, 'syncopation': 0.8}
            }
        }

    def _load_tension_resolution_systems(self) -> Dict[str, Dict]:
        """Load rhythmic tension and resolution systems."""
        return {
            'build_release': {
                'description': 'Gradual build to release',
                'tension_curve': [0.2, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.3],  # 8 steps
                'resolution_point': 5
            },
            'wave_pattern': {
                'description': 'Wave-like tension fluctuations',
                'tension_curve': [0.3, 0.6, 0.9, 0.6, 0.3, 0.6, 0.9, 0.4],
                'resolution_point': 7
            },
            'climax_resolution': {
                'description': 'Build to climax then resolve',
                'tension_curve': [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.4, 0.1],
                'resolution_point': 6
            }
        }

    def generate_complex_rhythm(self, genre: str, complexity: str,
                              time_signature: Tuple[int, int] = (4, 4),
                              tempo: float = 120.0) -> RhythmPattern:
        """Generate complex rhythmic patterns with polyrhythm and syncopation.

        Args:
            genre: Musical genre ('jazz', 'funk', 'latin', etc.)
            complexity: Complexity level ('simple', 'complex', 'dense')
            time_signature: Time signature as (beats_per_bar, beat_unit)
            tempo: Tempo in BPM

        Returns:
            RhythmPattern: Complex rhythmic pattern
        """
        # Get genre-specific parameters
        genre_params = self.rhythm_patterns.get(genre, self.rhythm_patterns.get('jazz', {}))

        # Select appropriate subgenre parameters
        subgenre_key = complexity if complexity in genre_params else 'simple'
        params = genre_params[subgenre_key]

        # Create base pattern
        base_pattern = self._create_base_rhythm_pattern(genre, complexity, time_signature, tempo)

        # Apply polyrhythms for complex/dense patterns
        if complexity in ['complex', 'dense']:
            base_pattern = self.polyrhythm_engine.apply_polyrhythm(base_pattern, genre, complexity)

        # Apply syncopation
        syncopation_level = params.get('syncopation', 0.3)
        base_pattern = self.syncopation_engine.apply_syncopation(base_pattern, genre, syncopation_level)

        # Apply groove variations and swing
        swing_intensity = params.get('swing_intensity', 0.5)
        base_pattern = self.syncopation_engine.generate_swing_feel(base_pattern, swing_intensity)

        # Apply groove-specific timing variations
        base_pattern = self._apply_groove_variations(base_pattern, genre, complexity)

        return base_pattern

    def _create_base_rhythm_pattern(self, genre: str, complexity: str,
                                  time_signature: Tuple[int, int], tempo: float) -> RhythmPattern:
        """Create a base rhythm pattern for further processing."""
        notes = []
        beats_per_bar, beat_unit = time_signature
        pattern_duration = beats_per_bar

        # Create basic rhythmic foundation based on genre
        if genre == 'jazz':
            notes = self._create_jazz_rhythm_base(complexity, pattern_duration)
        elif genre == 'funk':
            notes = self._create_funk_rhythm_base(complexity, pattern_duration)
        elif genre == 'latin':
            notes = self._create_latin_rhythm_base(complexity, pattern_duration)
        else:
            # Generic backbeat pattern
            notes = self._create_generic_rhythm_base(complexity, pattern_duration)

        complexity_value = {'simple': 0.3, 'complex': 0.6, 'dense': 0.9}.get(complexity, 0.5)

        return RhythmPattern(
            notes=notes,
            tempo=tempo,
            time_signature=time_signature,
            complexity=complexity_value,
            groove_intensity=0.5
        )

    def _create_jazz_rhythm_base(self, complexity: str, duration: float) -> List[Note]:
        """Create basic jazz rhythm foundation."""
        notes = []

        if complexity == 'simple':
            # Basic swing pattern
            for beat in range(int(duration)):
                # Downbeat kick
                notes.append(Note(36, 0.25, 90, beat))
                # Off-beat snare
                notes.append(Note(38, 0.25, 80, beat + 0.5))
        else:
            # More complex jazz pattern
            for beat in range(int(duration)):
                notes.append(Note(36, 0.25, 85, beat))  # Kick
                notes.append(Note(38, 0.25, 75, beat + 0.5))  # Snare
                if complexity == 'dense':
                    notes.append(Note(42, 0.125, 60, beat + 0.25))  # Hi-hat
                    notes.append(Note(42, 0.125, 55, beat + 0.75))

        return notes

    def _create_funk_rhythm_base(self, complexity: str, duration: float) -> List[Note]:
        """Create basic funk rhythm foundation."""
        notes = []

        for beat in range(int(duration)):
            notes.append(Note(36, 0.25, 95, beat))  # Kick on downbeat
            notes.append(Note(38, 0.25, 70, beat + 0.5))  # Snare on off-beat

            if complexity in ['complex', 'dense']:
                # Add ghost notes
                notes.append(Note(38, 0.125, 45, beat + 0.25))  # Ghost snare
                notes.append(Note(38, 0.125, 40, beat + 0.75))
                notes.append(Note(42, 0.125, 65, beat + 0.125))  # Hi-hat
                notes.append(Note(42, 0.125, 60, beat + 0.375))

        return notes

    def _create_latin_rhythm_base(self, complexity: str, duration: float) -> List[Note]:
        """Create basic latin rhythm foundation."""
        notes = []

        # Simple clave pattern
        clave_pattern = [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]  # 16th notes

        for i, hit in enumerate(clave_pattern):
            if hit:
                time_pos = i * 0.25  # 16th note timing
                if time_pos < duration:
                    # Alternate between different percussion sounds
                    if i % 4 == 0:
                        pitch = 39  # Hand clap
                        velocity = 85
                    elif i % 4 == 2:
                        pitch = 40  # Electric snare
                        velocity = 75
                    else:
                        pitch = 48  # Hi bongo
                        velocity = 70

                    notes.append(Note(pitch, 0.125, velocity, time_pos))

        if complexity in ['complex', 'dense']:
            # Add more complex latin percussion
            for beat in range(int(duration)):
                notes.append(Note(36, 0.25, 80, beat))  # Bass drum
                notes.append(Note(51, 0.125, 65, beat + 0.5))  # Ride cymbal

        return notes

    def _create_generic_rhythm_base(self, complexity: str, duration: float) -> List[Note]:
        """Create generic rhythm foundation with backbeat."""
        notes = []

        for beat in range(int(duration)):
            notes.append(Note(36, 0.25, 90, beat))  # Kick
            notes.append(Note(38, 0.25, 80, beat + 0.5))  # Snare

            if complexity in ['complex', 'dense']:
                notes.append(Note(42, 0.125, 60, beat + 0.25))  # Hi-hat
                notes.append(Note(42, 0.125, 55, beat + 0.75))

        return notes

    def _apply_groove_variations(self, pattern: RhythmPattern, genre: str, complexity: str) -> RhythmPattern:
        """Apply genre-specific groove timing variations."""
        groove_variations = {
            'jazz': {'timing_variation': 0.03, 'velocity_humanization': 0.15},
            'funk': {'timing_variation': 0.02, 'velocity_humanization': 0.12},
            'latin': {'timing_variation': 0.025, 'velocity_humanization': 0.18},
            'rock': {'timing_variation': 0.015, 'velocity_humanization': 0.1},
            'electronic': {'timing_variation': 0.005, 'velocity_humanization': 0.05}
        }

        params = groove_variations.get(genre, groove_variations['jazz'])

        # Adjust timing variation based on complexity
        complexity_multiplier = {'simple': 0.5, 'complex': 1.0, 'dense': 1.2}.get(complexity, 1.0)
        timing_variation = params['timing_variation'] * complexity_multiplier
        velocity_variation = params['velocity_humanization'] * complexity_multiplier

        # Apply variations to notes
        varied_notes = []
        for note in pattern.notes:
            # Timing variation
            timing_offset = random.uniform(-timing_variation, timing_variation)
            note.start_time += timing_offset

            # Velocity humanization
            velocity_offset = int(random.uniform(-velocity_variation, velocity_variation) * 127)
            note.velocity = max(1, min(127, note.velocity + velocity_offset))

            varied_notes.append(note)

        return RhythmPattern(
            notes=varied_notes,
            tempo=pattern.tempo,
            time_signature=pattern.time_signature,
            complexity=pattern.complexity,
            groove_intensity=min(1.0, pattern.groove_intensity + 0.2)
        )

    def apply_rhythmic_tension_resolution(self, pattern: RhythmPattern,
                                       system_name: str = 'build_release') -> RhythmPattern:
        """Apply complex rhythmic tension and resolution systems.

        Args:
            pattern: Base rhythm pattern
            system_name: Name of tension-resolution system to apply

        Returns:
            RhythmPattern: Pattern with tension-resolution dynamics
        """
        system = self.tension_resolution_systems.get(system_name)
        if not system:
            return pattern

        tension_curve = system['tension_curve']
        resolution_point = system['resolution_point']

        # Divide pattern into segments
        pattern_duration = 4.0  # Assume 4 beats
        segment_duration = pattern_duration / len(tension_curve)

        tension_applied_notes = []

        for note in pattern.notes:
            # Determine which tension segment this note falls into
            segment_index = int(note.start_time / segment_duration)
            if segment_index >= len(tension_curve):
                segment_index = len(tension_curve) - 1

            tension_level = tension_curve[segment_index]

            # Apply tension-based modifications
            modified_note = self._apply_tension_to_note(note, tension_level, segment_index, resolution_point)
            tension_applied_notes.append(modified_note)

        return RhythmPattern(
            notes=tension_applied_notes,
            tempo=pattern.tempo,
            time_signature=pattern.time_signature,
            complexity=min(1.0, pattern.complexity + 0.1),
            groove_intensity=pattern.groove_intensity
        )

    def _apply_tension_to_note(self, note: Note, tension_level: float,
                             segment_index: int, resolution_point: int) -> Note:
        """Apply tension-based modifications to a note."""
        modified_note = Note(note.pitch, note.duration, note.velocity, note.start_time)

        # Velocity modification based on tension
        tension_velocity_boost = int(tension_level * 20)  # Up to +20 velocity for high tension
        if segment_index >= resolution_point:
            # Resolution phase - slightly reduce velocity
            tension_velocity_boost -= 10

        modified_note.velocity = max(1, min(127, modified_note.velocity + tension_velocity_boost))

        # Duration modification for tension build
        if segment_index < resolution_point:
            # Build tension - slightly shorten notes
            modified_note.duration *= 0.9
        else:
            # Resolution - slightly lengthen notes
            modified_note.duration *= 1.1

        # For high tension segments, add slight timing anticipation or delay
        if tension_level > 0.7:
            timing_adjustment = random.choice([-0.02, 0.02])  # Anticipation or delay
            modified_note.start_time += timing_adjustment

        return modified_note