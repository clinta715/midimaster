"""
Bass Generator Module
 
This module contains the BassGenerator class responsible for generating
bass lines that provide harmonic foundation using chord roots.
"""

import random
from typing import TYPE_CHECKING, List

from structures.data_structures import Note, Pattern, PatternType
from generators.generator_utils import get_velocity_for_mood, initialize_key_and_scale

if TYPE_CHECKING:
    from generators.generator_context import GeneratorContext


class BassGenerator:
    """Generates bass patterns using chord roots from the progression.
 
    The BassGenerator creates bass lines that typically follow the root notes
    of the chord progression, providing the low-end foundation for the harmony.
    Bass notes are typically longer in duration and lower in pitch.
    """
 
    def __init__(self, context: 'GeneratorContext'):
        """
        Initialize the BassGenerator.
 
        Args:
            context: Shared GeneratorContext containing music theory and configuration
        """
        self.context = context
 
    def generate(self, num_bars: int) -> Pattern:
        """Generate a bass pattern.
 
        Creates a bass line that typically follows the root notes of the
        chord progression, providing the low-end foundation for the harmony.
        Bass notes are typically longer in duration and lower in pitch.
 
        Args:
            num_bars: Number of bars to generate
 
        Returns:
            Pattern object containing the bass notes
        """
        # Ensure key and scale are established
        if not self.context.scale_pitches:
            initialize_key_and_scale(self.context)
 
        notes = []
        chords = []
        start_time = 0.0
 
        # Get chord progressions for bass line
        # If user key/mode is specified, use transposed progressions
        if self.context.user_key and self.context.user_mode:
            progressions = self.context.genre_rules.get_transposed_chord_progressions(
                self.context.user_key, self.context.user_mode
            )
            if not progressions:
                progressions = [['Am', 'F', 'C', 'G']]  # Fallback to A minor progression
        else:
            progressions = self.context.genre_rules.get_chord_progressions()
            if not progressions:
                progressions = [['I', 'IV', 'V', 'I']]
        selected_progression = random.choice(progressions)
 
        # Generate bass notes (typically root notes of chords)
        # Use density manager to determine number of bass notes per bar
        for i in range(num_bars):
            # Get the chord for this bar
            chord_symbol = selected_progression[i % len(selected_progression)]

            # Get the root note of the chord for bass
            try:
                if self.context.user_key and self.context.user_mode:
                    # Using transposed progressions with chord names
                    chord_pitches = self._parse_chord_symbol(chord_symbol)
                else:
                    # Using Roman numerals
                    chord_pitches = self.context.music_theory.get_chord_pitches_from_roman(
                        chord_symbol, f"{self.context.current_key} {self.context.current_scale}"
                    )

                # Use the root note (first pitch) for bass
                if chord_pitches:
                    root_pitch = chord_pitches[0]
                    # Bass notes are typically an octave lower for proper register
                    bass_pitch = root_pitch - 12
                    # Ensure bass pitch is in scale (find closest scale pitch if not)
                    if bass_pitch not in self.context.scale_pitches:
                        # Find the closest pitch in scale_pitches
                        scale_pitches = sorted(self.context.scale_pitches)
                        closest_pitch = min(scale_pitches, key=lambda x: abs(x - bass_pitch))
                        bass_pitch = closest_pitch
                else:
                    # Fallback to scale-based bass note if chord generation fails
                    if self.context.scale_pitches:
                        low_pitches = [p for p in self.context.scale_pitches if p < 60]
                        if low_pitches:
                            bass_pitch = random.choice(low_pitches)
                        else:
                            # No low register scale tones available; choose nearest scale tone to low C
                            scale_pitches = sorted(self.context.scale_pitches)
                            bass_pitch = min(scale_pitches, key=lambda x: abs(x - 48))
                    else:
                        bass_pitch = 48  # Low C fallback
            except Exception as e:
                print(f"Error getting bass note for {chord_symbol}: {e}")
                # Fallback to scale-based bass note
                if self.context.scale_pitches:
                    low_pitches = [p for p in self.context.scale_pitches if p < 60]
                    if low_pitches:
                        bass_pitch = random.choice(low_pitches)
                    else:
                        # No low register scale tones available; choose nearest scale tone to low C
                        scale_pitches = sorted(self.context.scale_pitches)
                        bass_pitch = min(scale_pitches, key=lambda x: abs(x - 48))
                else:
                    bass_pitch = 48  # Low C fallback
 
            # Determine number of bass notes for this bar based on density
            bass_notes_per_bar = self.context.density_manager.get_bass_note_count()
            duration = 4.0 / bass_notes_per_bar  # Split bar evenly among bass notes
 
            # Create bass notes for this bar
            for j in range(bass_notes_per_bar):
                note_start = start_time + (j * duration)
 
                # Only place note based on density probability
                if self.context.density_manager.should_place_note(note_start, num_bars * 4):
                    # Adjust velocity based on mood and metric position
                    base_velocity = get_velocity_for_mood(self.context.mood)
                    
                    # Emphasize notes on strong beats (e.g., beat 1 and 3 in 4/4)
                    beat_in_bar = note_start % 4 # Assuming 4 beats per bar
                    if beat_in_bar == 0: # Strong beat 1
                        velocity = min(127, base_velocity + 15)
                    elif beat_in_bar == 2: # Medium strong beat 3
                        velocity = min(127, base_velocity + 5)
                    else: # Weak beats
                        velocity = base_velocity
 
                    # Add a small random variation for humanization
                    velocity = max(0, min(127, velocity + random.randint(-5, 5)))

                    # Ensure bass pitch is in scale (snap to nearest if not)
                    if bass_pitch not in self.context.scale_pitches and self.context.scale_pitches:
                        scale_pitches = sorted(self.context.scale_pitches)
                        bass_pitch = min(scale_pitches, key=lambda x: abs(x - bass_pitch))
                    assert bass_pitch in self.context.scale_pitches, f"Bass pitch {bass_pitch} not in scale {self.context.scale_pitches}"
                    note = Note(bass_pitch, duration, velocity, note_start)
                    notes.append(note)
 
            start_time += 4.0  # Advance to next bar
 
        # Return the bass pattern
        return Pattern(PatternType.BASS, notes, chords)

    def _parse_chord_symbol(self, chord_symbol: str) -> List[int]:
        """Parse a chord symbol like 'Am', 'Fmaj7', 'Bb' into MIDI pitches.

        Args:
            chord_symbol: Chord name (e.g., 'Am', 'Fmaj7')

        Returns:
            List of MIDI pitch values for the chord
        """
        from music_theory import MusicTheory, Note, ChordType

        # Parse root note
        chord_symbol = chord_symbol.strip()

        # Handle root note
        if chord_symbol.startswith(('A#', 'Bb', 'C#', 'D#', 'F#', 'G#')):
            if chord_symbol.startswith('A#'):
                root_name = 'A#'
                remainder = chord_symbol[2:]
            elif chord_symbol.startswith('Bb'):
                root_name = 'Bb'
                remainder = chord_symbol[2:]
            elif chord_symbol.startswith('C#'):
                root_name = 'C#'
                remainder = chord_symbol[2:]
            elif chord_symbol.startswith('D#'):
                root_name = 'D#'
                remainder = chord_symbol[2:]
            elif chord_symbol.startswith('F#'):
                root_name = 'F#'
                remainder = chord_symbol[2:]
            elif chord_symbol.startswith('G#'):
                root_name = 'G#'
                remainder = chord_symbol[2:]
        else:
            root_name = chord_symbol[0]
            remainder = chord_symbol[1:]

        # Map to Note enum
        note_map = {
            'C': Note.C, 'C#': Note.C_SHARP, 'Db': Note.C_SHARP,
            'D': Note.D, 'D#': Note.D_SHARP, 'Eb': Note.D_SHARP,
            'E': Note.E, 'F': Note.F, 'F#': Note.F_SHARP, 'Gb': Note.F_SHARP,
            'G': Note.G, 'G#': Note.G_SHARP, 'Ab': Note.G_SHARP,
            'A': Note.A, 'A#': Note.A_SHARP, 'Bb': Note.A_SHARP,
            'B': Note.B
        }

        root_note = note_map.get(root_name)
        if root_note is None:
            return []

        # Determine chord type
        remainder = remainder.lower()
        if remainder in ['m', 'min', 'minor']:
            chord_type = ChordType.MINOR
        elif remainder in ['7']:
            chord_type = ChordType.DOMINANT_7TH
        elif remainder in ['maj7', 'major7']:
            chord_type = ChordType.MAJOR_7TH
        elif remainder in ['m7', 'min7', 'minor7']:
            chord_type = ChordType.MINOR_7TH
        elif remainder in ['dim', 'diminished']:
            chord_type = ChordType.DIMINISHED
        elif remainder in ['aug', 'augmented']:
            chord_type = ChordType.AUGMENTED
        elif remainder in ['dim7', 'diminished7']:
            chord_type = ChordType.DIMINISHED_7TH
        elif remainder in ['sus2']:
            chord_type = ChordType.SUS2
        elif remainder in ['sus4']:
            chord_type = ChordType.SUS4
        else:  # Default to major
            chord_type = ChordType.MAJOR

        # Generate chord pitches
        return MusicTheory.build_chord(root_note, chord_type)