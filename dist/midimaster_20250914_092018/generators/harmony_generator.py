"""
Harmony Generator Module

This module contains the HarmonyGenerator class responsible for generating
harmonically rich chord progressions based on genre rules and musical context.
"""

import random
from typing import TYPE_CHECKING, List

from structures.data_structures import Note, Pattern, PatternType, Chord
from generators.generator_utils import get_velocity_for_mood, initialize_key_and_scale

if TYPE_CHECKING:
    from generators.generator_context import GeneratorContext


class HarmonyGenerator:
    """Generates harmony patterns using Roman numeral chord progressions.

    The HarmonyGenerator creates chord progressions based on genre-specific
    chord sequences expressed in Roman numeral notation, voiced appropriately
    and timed to create harmonic support for the melody.
    """

    def __init__(self, context: 'GeneratorContext'):
        """
        Initialize the HarmonyGenerator.

        Args:
            context: Shared GeneratorContext containing music theory and configuration
        """
        self.context = context

    def generate(self, num_bars: int, chord_complexity: str = 'medium', harmonic_variance: str = 'medium') -> Pattern:
        """Generate a harmony pattern.

        Creates chord progressions based on the genre's typical chord sequences
        expressed in Roman numeral notation. Chords are voiced appropriately
        and timed to create harmonic support for the melody.

        Args:
            num_bars: Number of bars to generate
            chord_complexity: Complexity level of chords ('simple', 'medium', 'complex')
            harmonic_variance: Level of harmonic movement between chords ('close', 'medium', 'distant')

        Returns:
            Pattern object containing the harmony chords
        """
        # Validate chord complexity parameter
        valid_complexities = ['simple', 'medium', 'complex']
        # Validate harmonic variance parameter
        valid_variances = ['close', 'medium', 'distant']
        if harmonic_variance not in valid_variances:
            raise ValueError(f"harmonic_variance must be one of {valid_variances}")
        if chord_complexity not in valid_complexities:
            raise ValueError(f"chord_complexity must be one of {valid_complexities}")

        # Ensure key and scale are established
        if not self.context.scale_pitches:
            initialize_key_and_scale(self.context)

        notes = []
        chords = []
        start_time = 0.0

        # Get chord progressions from genre rules
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

        # Randomly select one progression from the available options
        selected_progression = random.choice(progressions)

        print(f"Selected chord progression: {selected_progression}")
        # Filter progressions by harmonic variance and select one
        if harmonic_variance != 'medium':  # Only filter if not default
            key_scale_string = f"{self.context.current_key} {self.context.current_scale}"
            filtered_progressions = self.context.music_theory.filter_progressions_by_distance(
                progressions, key_scale_string, harmonic_variance
            )
            if filtered_progressions:
                progressions = filtered_progressions

        # Generate chords based on the selected progression
        # One chord per bar is the typical approach
        for i in range(num_bars):
            # Get the chord for this bar (cycle through progression if needed)
            chord_symbol = selected_progression[i % len(selected_progression)]

            # Convert chord symbol to pitches (either Roman numeral or chord name)
            try:
                if self.context.user_key and self.context.user_mode:
                    # Using transposed progressions with chord names
                    chord_pitches = self._parse_chord_symbol(chord_symbol)
                else:
                    # Using Roman numerals
                    chord_pitches = self.context.music_theory.get_chord_pitches_from_roman(
                        chord_symbol, f"{self.context.current_key} {self.context.current_scale}"
                    )

                print(f"Chord {chord_symbol} -> pitches: {chord_pitches}")

                # Filter chord pitches to only include those in the current scale
                if chord_pitches:
                    original_count = len(chord_pitches)
                    chord_pitches = [p for p in chord_pitches if p in self.context.scale_pitches]
                    if len(chord_pitches) < original_count:
                        print(f"Warning: Filtered {original_count - len(chord_pitches)} pitches from chord {chord_symbol} to adhere to scale")
                    if not chord_pitches:
                        print(f"Warning: Chord {chord_symbol} has no pitches in scale {self.context.scale_pitches[:7]}...")

                # Apply random inversion for voice leading
                if chord_pitches:
                    inversion_level = random.randint(0, min(2, len(chord_pitches) - 1)) # Limit to 0, 1, or 2
                    chord_pitches = self.context.music_theory.get_chord_inversion(chord_pitches, inversion_level)

                # Create chord if pitches were successfully generated and not empty after filtering
                if chord_pitches:
                    # Determine chord voicing size based on chord complexity and density
                    max_notes = len(chord_pitches)
                    if chord_complexity == 'simple':
                        voicing_size = min(2, max_notes)  # Just root and basic harmony
                    elif chord_complexity == 'medium':
                        voicing_size = min(3, max_notes)  # Root, third, fifth
                    else:  # complex
                        voicing_size = max_notes  # Full chord voicing

                    # Apply density management to further refine voicing
                    voicing_size = self.context.density_manager.get_chord_voicing_size(voicing_size)

                    # Select subset of pitches based on determined voicing size
                    if voicing_size < len(chord_pitches):
                        chord_pitches = chord_pitches[:voicing_size]

                    # Update chord duration to be more musical (slightly shorter than full bar)
                    chord_duration = 3.5

                    # Create chord notes with simultaneous start time and proper duration
                    chord_notes = []
                    for j, pitch in enumerate(chord_pitches):
                        # Adjust velocity based on mood and metric position
                        base_velocity = get_velocity_for_mood(self.context.mood)
                        
                        # Emphasize notes on strong beats (e.g., beat 1 and 3 in 4/4)
                        beat_in_bar = start_time % 4 # Assuming 4 beats per bar
                        if beat_in_bar == 0: # Strong beat 1
                            velocity = min(127, base_velocity + 20)
                        elif beat_in_bar == 2: # Medium strong beat 3
                            velocity = min(127, base_velocity + 10)
                        else: # Weak beats
                            velocity = base_velocity

                        # Add a small random variation for humanization
                        velocity = max(0, min(127, velocity + random.randint(-5, 5)))

                        # All chord notes start simultaneously
                        assert pitch in self.context.scale_pitches, f"Harmony chord pitch {pitch} not in scale {self.context.scale_pitches}"
                        note = Note(pitch, chord_duration, velocity, start_time)
                        chord_notes.append(note)

                    # Create the chord object and add it to the pattern
                    chord = Chord(chord_notes, start_time)
                    chords.append(chord)
                else:
                    print(f"Warning: Could not generate chord for {chord_symbol}")
            except Exception as e:
                print(f"Error generating chord {chord_symbol}: {e}")

            # Advance start time by one bar (4 beats)
            start_time += 4.0

        # Return the harmony pattern
        return Pattern(PatternType.HARMONY, notes, chords)

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