"""
Advanced Arrangement Engine for sophisticated musical arrangement techniques.

Provides counterpoint generation, call-and-response patterns, layering techniques,
and orchestration capabilities for complex musical arrangements.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from .instrumentation_manager import (
    InstrumentationManager, Arrangement, TimbreType, RegisterType,
    ArticulationType, instrumentation_manager
)
from structures.data_structures import Note, Chord, Pattern, PatternType


class ArrangementTechnique(Enum):
    """Types of arrangement techniques."""
    COUNTERPOINT = "counterpoint"
    CALL_RESPONSE = "call_response"
    LAYERING = "layering"
    ORCHESTRATION = "orchestration"
    CANON = "canon"
    FUGUE = "fugue"
    OSTINATO = "ostinato"


@dataclass
class ArrangementRule:
    """Rules for a specific arrangement technique."""
    technique: ArrangementTechnique
    max_voices: int = 4
    voice_spacing: int = 7  # semitones between voices
    rhythmic_offset: float = 0.0  # rhythmic displacement
    dynamic_balance: str = "balanced"
    spatial_distribution: str = "stereo"


@dataclass
class Voice:
    """Represents a musical voice in an arrangement."""
    name: str
    register: RegisterType
    timbre: TimbreType
    pattern: Pattern
    instrument: Optional[str] = None
    pan: float = 0.0
    volume: float = 0.7
    delay: float = 0.0  # timing delay in beats


class AdvancedArrangementEngine:
    """Handles complex arrangement techniques and orchestration."""

    def __init__(self):
        self.arrangement_rules = self._initialize_arrangement_rules()
        self.quality_evaluator = ArrangementQualityEvaluator()

    def _initialize_arrangement_rules(self) -> Dict[str, ArrangementRule]:
        """Initialize rules for different arrangement techniques."""
        return {
            "counterpoint": ArrangementRule(
                technique=ArrangementTechnique.COUNTERPOINT,
                max_voices=4,
                voice_spacing=7,
                rhythmic_offset=0.5,
                dynamic_balance="hierarchical",
                spatial_distribution="wide"
            ),
            "call_response": ArrangementRule(
                technique=ArrangementTechnique.CALL_RESPONSE,
                max_voices=2,
                voice_spacing=12,
                rhythmic_offset=2.0,
                dynamic_balance="alternating",
                spatial_distribution="ping_pong"
            ),
            "layering": ArrangementRule(
                technique=ArrangementTechnique.LAYERING,
                max_voices=8,
                voice_spacing=0,
                rhythmic_offset=0.0,
                dynamic_balance="layered",
                spatial_distribution="centered"
            ),
            "orchestration": ArrangementRule(
                technique=ArrangementTechnique.ORCHESTRATION,
                max_voices=12,
                voice_spacing=5,
                rhythmic_offset=0.25,
                dynamic_balance="orchestral",
                spatial_distribution="orchestral"
            )
        }

    def apply_arrangement_technique(self, base_pattern: Pattern, technique: str,
                                   genre: str = "general", complexity: str = "medium") -> Pattern:
        """Apply a specific arrangement technique to a pattern."""
        if technique == "counterpoint":
            return self._apply_counterpoint(base_pattern, genre, complexity)
        elif technique == "call_response":
            return self._apply_call_response(base_pattern, genre, complexity)
        elif technique == "layering":
            return self._apply_layering(base_pattern, genre, complexity)
        elif technique == "orchestration":
            return self._apply_orchestration(base_pattern, genre, complexity)
        else:
            return base_pattern

    def _apply_counterpoint(self, pattern: Pattern, genre: str, complexity: str) -> Pattern:
        """Apply counterpoint arrangement technique."""
        rule = self.arrangement_rules["counterpoint"]

        # Create multiple voices
        num_voices = min(rule.max_voices, 3 if complexity == "simple" else 5)

        voices = []
        base_notes = [note for note in pattern.notes if note.velocity > 0]

        for i in range(num_voices):
            voice_pattern = Pattern(pattern_type=PatternType.MELODY, notes=[], chords=[])
            voice_register = self._get_register_for_voice(i, num_voices)

            for j, note in enumerate(base_notes):
                # Create counterpoint note
                counterpoint_note = self._create_counterpoint_note(
                    note, i, j, voice_register, genre
                )
                if counterpoint_note:
                    voice_pattern.notes.append(counterpoint_note)

            # Apply voice-specific timing
            voice_delay = i * rule.rhythmic_offset
            for note in voice_pattern.notes:
                note.start_time += voice_delay

            voices.append(Voice(
                name=f"counterpoint_voice_{i+1}",
                register=voice_register,
                timbre=self._get_timbre_for_voice(i, genre),
                pattern=voice_pattern,
                pan=self._calculate_voice_pan(i, num_voices, "wide"),
                volume=0.8 - (i * 0.1),  # decreasing volume for higher voices
                delay=voice_delay
            ))

        # Combine voices into final pattern
        return self._combine_voices(voices)

    def _apply_call_response(self, pattern: Pattern, genre: str, complexity: str) -> Pattern:
        """Apply call and response arrangement technique."""
        rule = self.arrangement_rules["call_response"]

        # Split pattern into call and response sections
        notes = sorted(pattern.notes, key=lambda n: n.start_time)

        if len(notes) < 4:
            return pattern  # Not enough material for call/response

        mid_point = len(notes) // 2
        call_notes = notes[:mid_point]
        response_notes = notes[mid_point:]

        # Create call voice
        call_pattern = Pattern(pattern_type=PatternType.MELODY, notes=[], chords=[])
        call_pattern.notes = call_notes

        # Create response voice with variation
        response_pattern = Pattern(pattern_type=PatternType.MELODY, notes=[], chords=[])
        for note in response_notes:
            response_note = Note(
                pitch=note.pitch + random.choice([-2, -1, 0, 1, 2]),  # slight pitch variation
                start_time=note.start_time + rule.rhythmic_offset,
                duration=note.duration,
                velocity=int(note.velocity * 0.8),  # softer response
                channel=note.channel
            )
            response_pattern.notes.append(response_note)

        # Create voices
        voices = [
            Voice(
                name="call_voice",
                register=RegisterType.MID,
                timbre=TimbreType.BRIGHT,
                pattern=call_pattern,
                pan=-0.5,  # left side
                volume=0.8,
                delay=0.0
            ),
            Voice(
                name="response_voice",
                register=RegisterType.MID_HIGH,
                timbre=TimbreType.WARM,
                pattern=response_pattern,
                pan=0.5,  # right side
                volume=0.7,
                delay=rule.rhythmic_offset
            )
        ]

        return self._combine_voices(voices)

    def _apply_layering(self, pattern: Pattern, genre: str, complexity: str) -> Pattern:
        """Apply layering arrangement technique."""
        rule = self.arrangement_rules["layering"]

        # Create multiple layers with different characteristics
        num_layers = min(rule.max_voices, 4 if complexity == "simple" else 6)

        voices = []
        base_notes = [note for note in pattern.notes if note.velocity > 0]

        for i in range(num_layers):
            layer_pattern = Pattern(pattern_type=PatternType.MELODY, notes=[], chords=[])

            # Filter notes for this layer based on rhythm or pitch
            layer_notes = self._filter_notes_for_layer(base_notes, i, num_layers)

            for note in layer_notes:
                # Apply layer-specific modifications
                layer_note = self._create_layer_note(note, i, genre)
                layer_pattern.notes.append(layer_note)

            voices.append(Voice(
                name=f"layer_{i+1}",
                register=self._get_register_for_layer(i),
                timbre=self._get_timbre_for_layer(i, genre),
                pattern=layer_pattern,
                pan=self._calculate_layer_pan(i, num_layers),
                volume=0.6 + (i * 0.1),  # increasing volume for higher layers
                delay=i * 0.125  # slight delay between layers
            ))

        return self._combine_voices(voices)

    def _apply_orchestration(self, pattern: Pattern, genre: str, complexity: str) -> Pattern:
        """Apply orchestration arrangement technique."""
        rule = self.arrangement_rules["orchestration"]

        # Create orchestral arrangement with different instrument families
        instrument_families = self._get_orchestral_families(genre)
        num_sections = min(len(instrument_families), rule.max_voices)

        voices = []
        base_notes = [note for note in pattern.notes if note.velocity > 0]

        for i in range(num_sections):
            section_pattern = Pattern(pattern_type=PatternType.MELODY, notes=[], chords=[])
            family = instrument_families[i]

            # Assign notes to this orchestral section
            section_notes = self._assign_notes_to_section(base_notes, i, num_sections)

            for note in section_notes:
                # Apply orchestral modifications
                orchestral_note = self._create_orchestral_note(note, family)
                section_pattern.notes.append(orchestral_note)

            voices.append(Voice(
                name=f"orchestral_{family['name']}",
                register=family['register'],
                timbre=family['timbre'],
                pattern=section_pattern,
                instrument=family['instrument'],
                pan=family['pan'],
                volume=family['volume'],
                delay=i * rule.rhythmic_offset
            ))

        return self._combine_voices(voices)

    def _create_counterpoint_note(self, base_note: Note, voice_index: int,
                                note_index: int, target_register: RegisterType,
                                genre: str) -> Optional[Note]:
        """Create a counterpoint note based on the base note."""
        # Interval choices for counterpoint (thirds, sixths, etc.)
        intervals = [3, 4, 5, 6, 7, 9]  # diatonic intervals

        # Choose interval based on voice and position
        interval_choice = intervals[voice_index % len(intervals)]
        direction = 1 if voice_index % 2 == 0 else -1

        new_pitch = base_note.pitch + (interval_choice * direction)

        # Adjust for register
        register_offset = self._get_register_offset(target_register)
        new_pitch += register_offset

        # Keep within reasonable MIDI range
        new_pitch = max(21, min(108, new_pitch))

        return Note(
            pitch=new_pitch,
            start_time=base_note.start_time,
            duration=base_note.duration,
            velocity=int(base_note.velocity * (0.9 - voice_index * 0.1)),
            channel=voice_index + 1
        )

    def _create_layer_note(self, base_note: Note, layer_index: int, genre: str) -> Note:
        """Create a note for a specific layer."""
        # Apply layer-specific modifications
        pitch_mod = 0
        if layer_index > 0:
            pitch_mod = layer_index * 12  # octaves for higher layers

        velocity_mod = 1.0 - (layer_index * 0.1)  # decreasing velocity for higher layers

        return Note(
            pitch=base_note.pitch + pitch_mod,
            start_time=base_note.start_time + (layer_index * 0.0625),  # slight timing offset
            duration=base_note.duration,
            velocity=int(base_note.velocity * velocity_mod),
            channel=layer_index + 1
        )

    def _create_orchestral_note(self, base_note: Note, family: Dict[str, Any]) -> Note:
        """Create a note for an orchestral section."""
        # Apply family-specific modifications
        pitch_mod = family.get('pitch_offset', 0)
        velocity_mod = family.get('velocity_mod', 1.0)

        return Note(
            pitch=base_note.pitch + pitch_mod,
            start_time=base_note.start_time,
            duration=base_note.duration * family.get('duration_mod', 1.0),
            velocity=int(base_note.velocity * velocity_mod),
            channel=family.get('channel', 1)
        )

    def _combine_voices(self, voices: List[Voice]) -> Pattern:
        """Combine multiple voices into a single pattern."""
        combined_pattern = Pattern(pattern_type=PatternType.MELODY, notes=[], chords=[])

        for voice in voices:
            # Apply voice-specific modifications
            for note in voice.pattern.notes:
                modified_note = Note(
                    pitch=note.pitch,
                    start_time=note.start_time + voice.delay,
                    duration=note.duration,
                    velocity=int(note.velocity * voice.volume),
                    channel=note.channel
                )
                combined_pattern.notes.append(modified_note)

        # Sort notes by start time
        combined_pattern.notes.sort(key=lambda n: n.start_time)

        return combined_pattern

    def _get_register_for_voice(self, voice_index: int, total_voices: int) -> RegisterType:
        """Get appropriate register for a counterpoint voice."""
        registers = [RegisterType.LOW, RegisterType.MID_LOW, RegisterType.MID,
                    RegisterType.MID_HIGH, RegisterType.HIGH]
        return registers[min(voice_index, len(registers) - 1)]

    def _get_register_for_layer(self, layer_index: int) -> RegisterType:
        """Get register for a layer."""
        registers = [RegisterType.LOW, RegisterType.MID_LOW, RegisterType.MID,
                    RegisterType.MID_HIGH, RegisterType.HIGH]
        return registers[min(layer_index, len(registers) - 1)]

    def _get_timbre_for_voice(self, voice_index: int, genre: str) -> TimbreType:
        """Get timbre for a voice based on genre."""
        if genre == "jazz":
            timbres = [TimbreType.WARM, TimbreType.MELLOW, TimbreType.BRIGHT]
        elif genre == "classical":
            timbres = [TimbreType.WARM, TimbreType.MELLOW, TimbreType.DARK]
        else:
            timbres = [TimbreType.BRIGHT, TimbreType.WARM, TimbreType.DARK]

        return timbres[voice_index % len(timbres)]

    def _get_timbre_for_layer(self, layer_index: int, genre: str) -> TimbreType:
        """Get timbre for a layer."""
        timbres = [TimbreType.WARM, TimbreType.BRIGHT, TimbreType.DARK, TimbreType.MELLOW]
        return timbres[layer_index % len(timbres)]

    def _calculate_voice_pan(self, voice_index: int, total_voices: int, distribution: str) -> float:
        """Calculate pan position for a voice."""
        if distribution == "wide":
            return -1.0 + (voice_index / max(1, total_voices - 1)) * 2.0
        elif distribution == "centered":
            return -0.5 + (voice_index / max(1, total_voices - 1))
        else:
            return 0.0

    def _calculate_layer_pan(self, layer_index: int, total_layers: int) -> float:
        """Calculate pan position for a layer."""
        return -0.8 + (layer_index / max(1, total_layers - 1)) * 1.6

    def _get_register_offset(self, register: RegisterType) -> int:
        """Get pitch offset for a register."""
        offsets = {
            RegisterType.LOW: -24,
            RegisterType.MID_LOW: -12,
            RegisterType.MID: 0,
            RegisterType.MID_HIGH: 12,
            RegisterType.HIGH: 24
        }
        return offsets.get(register, 0)

    def _filter_notes_for_layer(self, notes: List[Note], layer_index: int, total_layers: int) -> List[Note]:
        """Filter notes for a specific layer."""
        if total_layers == 1:
            return notes

        # Simple rhythmic filtering - alternate layers get different notes
        filtered_notes = []
        for i, note in enumerate(notes):
            if i % total_layers == layer_index:
                filtered_notes.append(note)

        return filtered_notes

    def _get_orchestral_families(self, genre: str) -> List[Dict[str, Any]]:
        """Get orchestral instrument families for a genre."""
        if genre == "classical":
            return [
                {
                    "name": "strings",
                    "instrument": "violin",
                    "register": RegisterType.MID_HIGH,
                    "timbre": TimbreType.WARM,
                    "pan": -0.3,
                    "volume": 0.8,
                    "pitch_offset": 0,
                    "velocity_mod": 1.0,
                    "duration_mod": 1.0,
                    "channel": 1
                },
                {
                    "name": "woodwinds",
                    "instrument": "flute",
                    "register": RegisterType.HIGH,
                    "timbre": TimbreType.MELLOW,
                    "pan": 0.3,
                    "volume": 0.7,
                    "pitch_offset": 0,
                    "velocity_mod": 0.9,
                    "duration_mod": 0.8,
                    "channel": 2
                },
                {
                    "name": "brass",
                    "instrument": "trumpet",
                    "register": RegisterType.MID,
                    "timbre": TimbreType.BRIGHT,
                    "pan": -0.1,
                    "volume": 0.6,
                    "pitch_offset": 0,
                    "velocity_mod": 0.8,
                    "duration_mod": 0.9,
                    "channel": 3
                }
            ]
        else:
            # Generic families for other genres
            return [
                {
                    "name": "melody",
                    "instrument": "piano",
                    "register": RegisterType.MID,
                    "timbre": TimbreType.WARM,
                    "pan": 0.0,
                    "volume": 0.8,
                    "pitch_offset": 0,
                    "velocity_mod": 1.0,
                    "duration_mod": 1.0,
                    "channel": 1
                },
                {
                    "name": "harmony",
                    "instrument": "strings",
                    "register": RegisterType.MID_LOW,
                    "timbre": TimbreType.MELLOW,
                    "pan": -0.2,
                    "volume": 0.6,
                    "pitch_offset": 0,
                    "velocity_mod": 0.8,
                    "duration_mod": 1.2,
                    "channel": 2
                },
                {
                    "name": "bass",
                    "instrument": "bass",
                    "register": RegisterType.LOW,
                    "timbre": TimbreType.DARK,
                    "pan": 0.0,
                    "volume": 0.7,
                    "pitch_offset": 0,
                    "velocity_mod": 0.9,
                    "duration_mod": 1.0,
                    "channel": 3
                }
            ]

    def _assign_notes_to_section(self, notes: List[Note], section_index: int, total_sections: int) -> List[Note]:
        """Assign notes to an orchestral section."""
        if total_sections == 1:
            return notes

        # Distribute notes across sections
        assigned_notes = []
        for i, note in enumerate(notes):
            if i % total_sections == section_index:
                assigned_notes.append(note)

        return assigned_notes


class ArrangementQualityEvaluator:
    """Evaluates the quality of musical arrangements."""

    def evaluate_arrangement(self, arrangement: Arrangement) -> Dict[str, float]:
        """Evaluate an arrangement's quality across multiple dimensions."""
        scores = {}

        # Spatial distribution quality
        scores["spatial_balance"] = self._evaluate_spatial_balance(arrangement)

        # Dynamic balance quality
        scores["dynamic_balance"] = self._evaluate_dynamic_balance(arrangement)

        # Timbral variety
        scores["timbral_variety"] = self._evaluate_timbral_variety(arrangement)

        # Voice independence (for counterpoint)
        scores["voice_independence"] = self._evaluate_voice_independence(arrangement)

        # Overall balance
        scores["overall_balance"] = sum(scores.values()) / len(scores)

        return scores

    def _evaluate_spatial_balance(self, arrangement: Arrangement) -> float:
        """Evaluate spatial distribution balance."""
        pans = []
        for spatial in arrangement.spatial_config.values():
            pans.append(spatial.get("pan", 0.0))

        if not pans:
            return 0.5

        # Check for good stereo distribution
        left_count = sum(1 for p in pans if p < -0.3)
        right_count = sum(1 for p in pans if p > 0.3)
        center_count = sum(1 for p in pans if -0.3 <= p <= 0.3)

        # Ideal: balanced L/R with some center
        balance_score = min(left_count, right_count) / max(1, (left_count + right_count) / 2)
        return min(1.0, balance_score + (center_count * 0.1))

    def _evaluate_dynamic_balance(self, arrangement: Arrangement) -> float:
        """Evaluate dynamic balance."""
        volumes = []
        for dynamic in arrangement.dynamic_config.values():
            volumes.append(dynamic.get("volume", 0.7))

        if not volumes:
            return 0.5

        # Check volume distribution
        avg_volume = sum(volumes) / len(volumes)
        variance = sum((v - avg_volume) ** 2 for v in volumes) / len(volumes)

        # Lower variance is better (more balanced)
        return max(0.0, 1.0 - (variance * 4))

    def _evaluate_timbral_variety(self, arrangement: Arrangement) -> float:
        """Evaluate timbral variety."""
        # This would need instrument characteristics data
        # For now, return neutral score
        return 0.7

    def _evaluate_voice_independence(self, arrangement: Arrangement) -> float:
        """Evaluate voice independence in counterpoint."""
        # This would analyze melodic independence
        # For now, return neutral score
        return 0.6


# Global instance
advanced_arrangement_engine = AdvancedArrangementEngine()
arrangement_quality_evaluator = ArrangementQualityEvaluator()