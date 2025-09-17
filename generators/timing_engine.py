"""
Advanced Timing and Microtiming Engine

This module implements sophisticated timing algorithms that go beyond basic swing
to include human-like microtiming variations, tempo fluctuations, and expressive timing.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Callable, Any
from structures.data_structures import Pattern, Note, Chord


class TempoCurve:
    """Manages tempo variations throughout a piece."""

    def __init__(self, base_tempo: float):
        """
        Initialize the tempo curve.

        Args:
            base_tempo: Base tempo in BPM
        """
        self.base_tempo = base_tempo
        self.curve_points = []  # List of (action, start_beat, end_beat, target_multiplier) tuples

    def add_acceleration(self, start_beat: float, end_beat: float, target_multiplier: float):
        """Add tempo acceleration over a range."""
        self.curve_points.append(('accelerate', start_beat, end_beat, target_multiplier))

    def add_ritardando(self, start_beat: float, end_beat: float, target_multiplier: float):
        """Add tempo slowdown over a range."""
        self.curve_points.append(('ritardando', start_beat, end_beat, target_multiplier))

    def get_tempo_multiplier_at_beat(self, beat: float) -> float:
        """Get the tempo multiplier at a specific beat."""
        # Default to 1.0 (no change)
        multiplier = 1.0

        # Find applicable curve points
        for action, start_beat, end_beat, target_multiplier in self.curve_points:
            if start_beat <= beat <= end_beat:
                # Calculate interpolation factor
                if end_beat == start_beat:
                    factor = 1.0
                else:
                    factor = (beat - start_beat) / (end_beat - start_beat)

                if action == 'accelerate':
                    # Exponential acceleration
                    multiplier *= (1.0 + (target_multiplier - 1.0) * factor)
                elif action == 'ritardando':
                    # Logarithmic slowdown
                    if factor < 1.0:
                        multiplier *= (1.0 + (target_multiplier - 1.0) * math.log(1.0 + 9.0 * factor) / math.log(10.0))
                    else:
                        multiplier *= target_multiplier

        return multiplier

    def apply_to_pattern(self, pattern: Pattern) -> Pattern:
        """Apply tempo curve to all timing in the pattern."""
        modified_notes = []
        modified_chords = []

        # Process individual notes
        for note in pattern.notes:
            # Get tempo multiplier at note start time
            tempo_multiplier = self.get_tempo_multiplier_at_beat(note.start_time)

            # Adjust note timing based on tempo curve
            # For acceleration, notes start slightly early
            # For ritardando, notes start slightly late
            timing_offset = 0.0
            if tempo_multiplier > 1.0:
                # Acceleration - slight anticipation
                timing_offset = -0.01 * (tempo_multiplier - 1.0)
            elif tempo_multiplier < 1.0:
                # Ritardando - slight delay
                timing_offset = 0.01 * (1.0 - tempo_multiplier)

            modified_note = Note(
                pitch=note.pitch,
                start_time=note.start_time + timing_offset,
                duration=note.duration * tempo_multiplier,  # Duration scales with tempo
                velocity=note.velocity
            )
            modified_notes.append(modified_note)

        # Process chords
        for chord in pattern.chords:
            modified_chord_notes = []
            for note in chord.notes:
                tempo_multiplier = self.get_tempo_multiplier_at_beat(note.start_time)
                timing_offset = 0.0
                if tempo_multiplier > 1.0:
                    timing_offset = -0.01 * (tempo_multiplier - 1.0)
                elif tempo_multiplier < 1.0:
                    timing_offset = 0.01 * (1.0 - tempo_multiplier)

                modified_note = Note(
                    pitch=note.pitch,
                    start_time=note.start_time + timing_offset,
                    duration=note.duration * tempo_multiplier,
                    velocity=note.velocity
                )
                modified_chord_notes.append(modified_note)

            modified_chords.append(Chord(modified_chord_notes, chord.start_time))

        return Pattern(pattern.pattern_type, modified_notes, modified_chords)


class MicroTimingProfile:
    """Defines microtiming characteristics for different genres/moods."""

    def __init__(self, name: str):
        """
        Initialize a microtiming profile.

        Args:
            name: Profile name (e.g., 'jazz_swing', 'funk_groove')
        """
        self.name = name
        self.timing_offsets = {}  # voice -> offset_function mapping
        self.groove_intensity = 0.5  # Overall groove intensity (0.0-1.0)
        self.swing_ratio = 0.67  # Swing ratio for off-beats (0.5 = triplet feel, 0.67 = standard swing)

    def add_voice_offset(self, voice: str, offset_function: Callable[[float, int], float]):
        """
        Add timing offset function for a specific voice.

        Args:
            voice: Voice identifier (e.g., 'melody', 'bass', 'drums')
            offset_function: Function that takes (beat_position, velocity) and returns timing offset in beats
        """
        self.timing_offsets[voice] = offset_function

    def apply_to_note(self, note: Note, voice: str, beat_position: float) -> Note:
        """
        Apply microtiming to a note.

        Args:
            note: The note to modify
            voice: Voice identifier
            beat_position: Position within the beat (0.0-1.0)

        Returns:
            Modified note with microtiming applied
        """
        if voice in self.timing_offsets:
            offset = self.timing_offsets[voice](beat_position, note.velocity)
            # Apply groove intensity scaling
            offset *= self.groove_intensity
            modified_note = Note(
                pitch=note.pitch,
                start_time=note.start_time + offset,
                duration=note.duration,
                velocity=note.velocity
            )
            return modified_note
        return note

    def apply_swing(self, beat_position: float) -> float:
        """
        Apply swing timing to a beat position.

        Args:
            beat_position: Position within beat (0.0-1.0)

        Returns:
            Modified beat position with swing
        """
        if beat_position >= 0.5:  # Off-beat
            # Apply swing ratio to off-beats
            swing_offset = (beat_position - 0.5) * (self.swing_ratio - 0.5) / 0.5
            return 0.5 + swing_offset
        return beat_position


class ExpressiveTimingEngine:
    """Engine for generating expressive timing variations."""

    def __init__(self, breath_intensity: float = 0.3, phrasing_intensity: float = 0.2):
        """
        Initialize expressive timing engine.

        Args:
            breath_intensity: Intensity of breath simulation (0.0-1.0)
            phrasing_intensity: Intensity of phrasing timing (0.0-1.0)
        """
        self.breath_intensity = breath_intensity
        self.phrasing_intensity = phrasing_intensity
        self.performance_anxiety = 0.1  # Base anxiety level
        self.phrase_positions = {}  # Track phrase positions for each voice

    def apply_breath_simulation(self, note: Note, time_since_last_note: float) -> Note:
        """
        Apply breath simulation timing.

        Simulates human breathing patterns where longer pauses create slight rushing.

        Args:
            note: Note to modify
            time_since_last_note: Time since last note in this voice

        Returns:
            Modified note
        """
        if time_since_last_note > 1.0:  # Significant pause
            # Slight anticipation after long pause (breath intake)
            breath_offset = -self.breath_intensity * min(time_since_last_note / 4.0, 1.0) * 0.02
        else:
            # Slight delay after short pause (breath recovery)
            breath_offset = self.breath_intensity * 0.01

        return Note(
            pitch=note.pitch,
            start_time=note.start_time + breath_offset,
            duration=note.duration,
            velocity=note.velocity
        )

    def apply_phrasing(self, note: Note, phrase_position: float, phrase_length: float) -> Note:
        """
        Apply phrasing timing variations.

        Args:
            note: Note to modify
            phrase_position: Position within phrase (0.0-1.0)
            phrase_length: Total phrase length in beats

        Returns:
            Modified note
        """
        # Apply slight ritardando at phrase ends
        if phrase_position > 0.7:  # Last 30% of phrase
            phrase_offset = self.phrasing_intensity * (phrase_position - 0.7) / 0.3 * 0.015
        else:
            phrase_offset = 0.0

        return Note(
            pitch=note.pitch,
            start_time=note.start_time + phrase_offset,
            duration=note.duration,
            velocity=note.velocity
        )

    def apply_performance_anxiety(self, note: Note, section_intensity: str) -> Note:
        """
        Apply performance anxiety timing variations.

        Simulates how musicians play differently under pressure.

        Args:
            note: Note to modify
            section_intensity: Section intensity ('low', 'medium', 'high')

        Returns:
            Modified note
        """
        intensity_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 1.5}.get(section_intensity, 1.0)
        anxiety_level = self.performance_anxiety * intensity_multiplier

        # High anxiety causes slight rushing
        anxiety_offset = -anxiety_level * 0.01 * random.uniform(0.5, 1.5)

        # Occasionally add hesitation
        if random.random() < anxiety_level * 0.1:
            anxiety_offset += anxiety_level * 0.02

        return Note(
            pitch=note.pitch,
            start_time=note.start_time + anxiety_offset,
            duration=note.duration,
            velocity=note.velocity
        )


class AdvancedTimingEngine:
    """Advanced timing and microtiming engine."""

    def __init__(self, base_tempo: float = 120.0):
        """
        Initialize the advanced timing engine.

        Args:
            base_tempo: Base tempo in BPM
        """
        self.base_tempo = base_tempo
        self.tempo_curve = TempoCurve(base_tempo)
        self.microtiming_profiles = self._load_microtiming_profiles()
        self.expressive_engine = ExpressiveTimingEngine()
        self.ensemble_coordinator = EnsembleTimingCoordinator()

    def _load_microtiming_profiles(self) -> Dict[str, MicroTimingProfile]:
        """Load predefined microtiming profiles for different genres."""
        profiles = {}

        # Jazz profile - complex swing with variable intensity
        jazz_profile = MicroTimingProfile("jazz_swing")
        jazz_profile.groove_intensity = 0.7
        jazz_profile.swing_ratio = 0.67

        def jazz_melody_offset(beat_pos: float, velocity: int) -> float:
            # Jazz melodies have anticipatory timing
            if beat_pos >= 0.5:  # Off-beats
                return -0.01 * (velocity / 127.0)  # Louder notes anticipate more
            return 0.005  # Slight delay on downbeats

        def jazz_rhythm_offset(beat_pos: float, velocity: int) -> float:
            # Swing feel with variable intensity
            if beat_pos >= 0.5:
                swing_amount = 0.015 * (velocity / 127.0)
                return beat_pos * swing_amount
            return 0.0

        jazz_profile.add_voice_offset('melody', jazz_melody_offset)
        jazz_profile.add_voice_offset('rhythm', jazz_rhythm_offset)
        profiles['jazz'] = jazz_profile

        # Funk profile - tight but groovy timing
        funk_profile = MicroTimingProfile("funk_groove")
        funk_profile.groove_intensity = 0.6
        funk_profile.swing_ratio = 0.55  # Less swing, more groove

        def funk_bass_offset(beat_pos: float, velocity: int) -> float:
            # Funk bass is rock-solid with slight groove
            return 0.005 * math.sin(beat_pos * 2 * math.pi)

        def funk_rhythm_offset(beat_pos: float, velocity: int) -> float:
            # Precise but groovy
            groove = 0.01 * math.sin(beat_pos * 4 * math.pi)
            return groove * (velocity / 127.0)

        funk_profile.add_voice_offset('bass', funk_bass_offset)
        funk_profile.add_voice_offset('rhythm', funk_rhythm_offset)
        profiles['funk'] = funk_profile

        # Rock profile - aggressive timing with fills
        rock_profile = MicroTimingProfile("rock_aggressive")
        rock_profile.groove_intensity = 0.8
        rock_profile.swing_ratio = 0.5  # Minimal swing

        def rock_rhythm_offset(beat_pos: float, velocity: int) -> float:
            # Aggressive timing with slight anticipation on accents
            if velocity > 100:  # Accent notes
                return -0.008
            return 0.002

        rock_profile.add_voice_offset('rhythm', rock_rhythm_offset)
        profiles['rock'] = rock_profile

        # Classical profile - precise timing with minimal variation
        classical_profile = MicroTimingProfile("classical_precise")
        classical_profile.groove_intensity = 0.1  # Very minimal variation
        classical_profile.swing_ratio = 0.5  # No swing

        def classical_timing_offset(beat_pos: float, velocity: int) -> float:
            # Minimal timing variation for classical precision
            return 0.001 * (random.random() - 0.5)

        classical_profile.add_voice_offset('melody', classical_timing_offset)
        classical_profile.add_voice_offset('harmony', classical_timing_offset)
        profiles['classical'] = classical_profile

        return profiles

    def apply_timing_variations(self, pattern: Pattern, genre: str, mood: str,
                              section_intensity: str = 'medium') -> Pattern:
        """
        Apply comprehensive timing variations to a pattern.

        Args:
            pattern: Pattern to modify
            genre: Music genre
            mood: Music mood
            section_intensity: Section intensity ('low', 'medium', 'high')

        Returns:
            Modified pattern with timing variations
        """
        # Apply tempo curve
        pattern = self.tempo_curve.apply_to_pattern(pattern)

        # Apply microtiming based on genre and mood
        microtiming_profile = self._select_microtiming_profile(genre, mood)
        pattern = self._apply_microtiming(pattern, microtiming_profile)

        # Apply expressive timing
        pattern = self._apply_expressive_timing(pattern, section_intensity)

        # Apply ensemble coordination
        pattern = self.ensemble_coordinator.apply_coordination(pattern)

        return pattern

    def _select_microtiming_profile(self, genre: str, mood: str) -> MicroTimingProfile:
        """Select appropriate microtiming profile."""
        # Primary selection by genre
        if genre in self.microtiming_profiles:
            return self.microtiming_profiles[genre]

        # Fallback to jazz for complex genres
        if genre in ['fusion', 'latin', 'world']:
            return self.microtiming_profiles.get('jazz', self.microtiming_profiles['classical'])

        # Default to classical for unknown genres
        return self.microtiming_profiles.get('classical', list(self.microtiming_profiles.values())[0])

    def _apply_microtiming(self, pattern: Pattern, profile: MicroTimingProfile) -> Pattern:
        """Apply microtiming profile to pattern."""
        modified_notes = []
        modified_chords = []

        for note in pattern.notes:
            voice = pattern.pattern_type.value.lower()  # Use pattern type as voice identifier
            beat_position = note.start_time % 1.0  # Position within beat

            # Apply swing first
            swung_position = profile.apply_swing(beat_position)

            # Apply voice-specific microtiming
            modified_note = profile.apply_to_note(note, voice, swung_position)
            modified_notes.append(modified_note)

        # Apply timing to chords as well
        for chord in pattern.chords:
            modified_chord_notes = []
            for note in chord.notes:
                voice = pattern.pattern_type.value.lower()
                beat_position = note.start_time % 1.0
                swung_position = profile.apply_swing(beat_position)
                modified_note = profile.apply_to_note(note, voice, swung_position)
                modified_chord_notes.append(modified_note)
            modified_chords.append(Chord(modified_chord_notes, chord.start_time))

        return Pattern(pattern.pattern_type, modified_notes, modified_chords)

    def _apply_expressive_timing(self, pattern: Pattern, section_intensity: str) -> Pattern:
        """Apply expressive timing variations."""
        modified_notes = []
        modified_chords = []

        # Track timing for each voice (using pattern type as voice identifier)
        last_note_times = {}

        for note in sorted(pattern.notes, key=lambda n: n.start_time):
            voice = pattern.pattern_type.value.lower()

            # Apply breath simulation
            time_since_last = 0.0
            if voice in last_note_times:
                time_since_last = note.start_time - last_note_times[voice]
            note = self.expressive_engine.apply_breath_simulation(note, time_since_last)

            # Apply phrasing (simplified - assume 4-bar phrases)
            phrase_position = (note.start_time % 16.0) / 16.0  # Position in 4-bar phrase
            note = self.expressive_engine.apply_phrasing(note, phrase_position, 16.0)

            # Apply performance anxiety
            note = self.expressive_engine.apply_performance_anxiety(note, section_intensity)

            modified_notes.append(note)
            last_note_times[voice] = note.start_time

        # Apply to chords
        for chord in pattern.chords:
            modified_chord_notes = []
            for note in chord.notes:
                voice = pattern.pattern_type.value.lower()
                time_since_last = 0.0
                if voice in last_note_times:
                    time_since_last = note.start_time - last_note_times[voice]

                note = self.expressive_engine.apply_breath_simulation(note, time_since_last)
                phrase_position = (note.start_time % 16.0) / 16.0
                note = self.expressive_engine.apply_phrasing(note, phrase_position, 16.0)
                note = self.expressive_engine.apply_performance_anxiety(note, section_intensity)
                modified_chord_notes.append(note)

            modified_chords.append(Chord(modified_chord_notes, chord.start_time))

        return Pattern(pattern.pattern_type, modified_notes, modified_chords)


class EnsembleTimingCoordinator:
    """Coordinates timing across ensemble voices for realistic performance."""

    def __init__(self):
        """Initialize ensemble coordinator."""
        self.voice_delays = {}  # Voice -> typical delay mapping
        self.coordination_strength = 0.3  # How strongly voices coordinate

    def apply_coordination(self, pattern: Pattern) -> Pattern:
        """
        Apply ensemble coordination timing.

        Ensures voices don't all start at exactly the same time,
        simulating human ensemble playing.
        """
        modified_notes = []
        modified_chords = []

        for note in pattern.notes:
            voice = pattern.pattern_type.value.lower()

            # Get typical delay for this voice
            if voice not in self.voice_delays:
                # Assign random delays for different voices
                self.voice_delays[voice] = random.uniform(-0.005, 0.005)

            delay = self.voice_delays[voice] * self.coordination_strength

            modified_note = Note(
                pitch=note.pitch,
                start_time=note.start_time + delay,
                duration=note.duration,
                velocity=note.velocity
            )
            modified_notes.append(modified_note)

        # Apply to chords
        for chord in pattern.chords:
            modified_chord_notes = []
            for note in chord.notes:
                voice = pattern.pattern_type.value.lower()
                if voice not in self.voice_delays:
                    self.voice_delays[voice] = random.uniform(-0.005, 0.005)

                delay = self.voice_delays[voice] * self.coordination_strength
                modified_note = Note(
                    pitch=note.pitch,
                    start_time=note.start_time + delay,
                    duration=note.duration,
                    velocity=note.velocity
                )
                modified_chord_notes.append(modified_note)
            modified_chords.append(Chord(modified_chord_notes, chord.start_time))

        return Pattern(pattern.pattern_type, modified_notes, modified_chords)