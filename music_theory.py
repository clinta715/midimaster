"""
Music Theory Module for MIDI Master

This module provides music theory utilities for scale construction, chord generation,
and harmonic analysis. It supports both classical music theory and contemporary
concepts needed for automatic music generation.
"""

from enum import Enum
from typing import List, Tuple, Optional, Dict
from structures.data_structures import Note as MidiNote
import math


class MusicNote(Enum):
    """Musical notes with their MIDI pitch values (middle C = C4 = 60)."""
    C = 60
    C_SHARP = 61
    D_FLAT = 61
    D = 62
    D_SHARP = 63
    E_FLAT = 63
    E = 64
    F = 65
    F_SHARP = 66
    G_FLAT = 66
    G = 67
    G_SHARP = 68
    A_FLAT = 68
    A = 69
    A_SHARP = 70
    B_FLAT = 70
    B = 71

# Alias for backward compatibility
Note = MusicNote


class ScaleType(Enum):
    """Common musical scales."""
    MAJOR = "major"
    MINOR_NATURAL = "minor_natural"
    MINOR_HARMONIC = "minor_harmonic"
    MINOR_MELODIC = "minor_melodic"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    IONIAN = "ionian"
    LOCRIAN = "locrian"
    PENTATONIC_MAJOR = "pentatonic_major"
    PENTATONIC_MINOR = "pentatonic_minor"


class ChordType(Enum):
    """Common chord types."""
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    SUS2 = "sus2"
    SUS4 = "sus4"
    DOMINANT_7TH = "dominant_7th"
    MAJOR_7TH = "major_7th"
    MINOR_7TH = "minor_7th"
    DIMINISHED_7TH = "diminished_7th"


class MusicTheory:
    """Core music theory utilities for the MIDI Master application."""

    # Scale intervals (in semitones from root)
    SCALE_INTERVALS = {
        ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
        ScaleType.MINOR_NATURAL: [0, 2, 3, 5, 7, 8, 10],
        ScaleType.MINOR_HARMONIC: [0, 2, 3, 5, 7, 8, 11],
        ScaleType.MINOR_MELODIC: [0, 2, 3, 5, 7, 9, 11],
        ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
        ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
        ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
        ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
        ScaleType.AEOLIAN: [0, 2, 3, 5, 7, 8, 10],
        ScaleType.IONIAN: [0, 2, 4, 5, 7, 9, 11],
        ScaleType.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
        ScaleType.PENTATONIC_MAJOR: [0, 2, 4, 7, 9],
        ScaleType.PENTATONIC_MINOR: [0, 3, 5, 7, 10]
    }

    # Chord intervals (in semitones from root)
    CHORD_INTERVALS = {
        ChordType.MAJOR: [0, 4, 7],
        ChordType.MINOR: [0, 3, 7],
        ChordType.DIMINISHED: [0, 3, 6],
        ChordType.AUGMENTED: [0, 4, 8],
        ChordType.SUS2: [0, 2, 7],
        ChordType.SUS4: [0, 5, 7],
        ChordType.DOMINANT_7TH: [0, 4, 7, 10],
        ChordType.MAJOR_7TH: [0, 4, 7, 11],
        ChordType.MINOR_7TH: [0, 3, 7, 10],
        ChordType.DIMINISHED_7TH: [0, 3, 6, 9]
    }

    # Roman numeral to scale degree mapping
    ROMAN_NUMERALS = {
        # Major key roman numerals
        'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5, 'VII': 6,
        'i': 0, 'ii': 1, 'iii': 2, 'iv': 3, 'v': 4, 'vi': 5, 'vii': 6,
        # Minor key variations
        'bII': 1, 'bIII': 2, 'bVI': 5, 'bVII': 6,
        'VIIdim': 6, 'ii7': 1, 'iii7': 2, 'IV7': 3, 'V7': 4, 'vi7': 5,
        'vii7': 6
    }

    @staticmethod
    def get_scale_pitches_from_string(scale_str: str, octave_range: int = 2) -> List[int]:
        """
        Parse a scale string and return all pitches in the scale across octaves.

        Args:
            scale_str: String like "C major" or "A minor"
            octave_range: Number of octaves to generate (default: 2)

        Returns:
            List of MIDI pitch values for the scale
        """
        try:
            root_note, scale_type = MusicTheory.parse_scale_string(scale_str)
            return MusicTheory.build_scale(root_note, scale_type, octave_range)
        except Exception as e:
            print(f"Error parsing scale string '{scale_str}': {e}")
            return []

    @staticmethod
    def parse_scale_string(scale_str: str) -> Tuple[Note, ScaleType]:
        """
        Parse a scale string into root note and scale type.

        Args:
            scale_str: String like "C major" or "A minor"

        Returns:
            Tuple of (root_note, scale_type)

        Raises:
            ValueError: If the scale string is invalid
        """
        parts = scale_str.lower().split()
        if len(parts) < 2:
            raise ValueError("Scale string must contain root note and scale type")

        # Parse root note
        root_name = parts[0].upper()
        if root_name.endswith('B'):
            if root_name == 'B':
                root_note = Note.B
            else:
                # Handle flats like Ab, Bb, etc.
                note_part = root_name[:-1]
                root_note = getattr(Note, f"{note_part}_FLAT", None)
        elif root_name.endswith('#'):
            if root_name == 'C#':
                root_note = Note.C_SHARP
            else:
                # Handle sharps like F#, G#, etc.
                note_part = root_name[:-1]
                root_note = getattr(Note, f"{note_part}_SHARP", None)
        else:
            root_note = getattr(Note, root_name, None)

        if root_note is None:
            raise ValueError(f"Unknown root note: {root_name}")

        # Parse scale type
        scale_phrase = ' '.join(parts[1:])
        scale_type = None
        for st, interval in MusicTheory.SCALE_INTERVALS.items():
            if scale_phrase == st.value.lower():
                scale_type = st
                break

        # Handle abbreviated scale names (e.g., "minor" -> "minor_natural")
        if scale_type is None:
            if scale_phrase == "minor":
                scale_type = ScaleType.MINOR_NATURAL
            else:
                raise ValueError(f"Unknown scale type: {scale_phrase}")

        return root_note, scale_type

    @staticmethod
    def build_scale(root_note: Note, scale_type: ScaleType, octave_range: int = 2) -> List[int]:
        """
        Build a scale starting from the given root note.

        Args:
            root_note: Root note of the scale
            scale_type: Type of scale to build
            octave_range: Number of octaves to generate (default: 2)

        Returns:
            List of MIDI pitch values for the scale
        """
        intervals = MusicTheory.SCALE_INTERVALS[scale_type]
        pitches = []

        for octave in range(octave_range):
            octave_offset = octave * 12
            for interval in intervals:
                pitch = root_note.value + octave_offset + interval
                if 0 <= pitch <= 127:  # MIDI pitch range
                    pitches.append(pitch)

        return pitches

    @staticmethod
    def build_chord(root_note: Note, chord_type: ChordType, octave: int = 0) -> List[int]:
        """
        Build a chord from root note and chord type.

        Args:
            root_note: Root note of the chord
            chord_type: Type of chord to build
            octave: Octave offset from root octave (default: 0)

        Returns:
            List of MIDI pitch values for the chord
        """
        intervals = MusicTheory.CHORD_INTERVALS[chord_type]
        pitches = []

        octave_offset = octave * 12
        for interval in intervals:
            pitch = root_note.value + octave_offset + interval
            if 0 <= pitch <= 127:  # MIDI pitch range
                pitches.append(pitch)

        return pitches

    @staticmethod
    def get_chord_inversion(pitches: List[int], inversion: int) -> List[int]:
        """
        Get a specific inversion of a chord.

        Args:
            pitches: List of MIDI pitch values for the chord
            inversion: The inversion to apply (0 for root, 1 for 1st, etc.)

        Returns:
            List of MIDI pitch values for the inverted chord
        """
        if not pitches or inversion == 0:
            return pitches

        num_notes = len(pitches)
        inversion = inversion % num_notes

        inverted_pitches = sorted(pitches)
        for i in range(inversion):
            root = inverted_pitches.pop(0)
            inverted_pitches.append(root + 12)

        return inverted_pitches

    @staticmethod
    def get_chord_pitches_from_roman(roman_numeral: str, key_scale_string: str) -> List[int]:
        """
        Get chord pitches from Roman numeral notation in a given key.

        Args:
            roman_numeral: Roman numeral like "I", "V7", "vi", etc.
            key_scale_string: Key and scale like "C major" or "A minor"

        Returns:
            List of MIDI pitch values for the chord
        """
        try:
            # Parse the key and scale
            root_note, scale_type = MusicTheory.parse_scale_string(key_scale_string)

            # Get scale degree from roman numeral
            scale_degree = MusicTheory.ROMAN_NUMERALS.get(roman_numeral)
            if scale_degree is None:
                return []  # Unknown roman numeral

            # Determine chord type from roman numeral
            chord_type = MusicTheory._get_chord_type_from_roman(roman_numeral)

            # Calculate the actual root pitch for this chord
            scale_intervals = MusicTheory.SCALE_INTERVALS[scale_type]
            if scale_degree < len(scale_intervals):
                root_offset = scale_intervals[scale_degree]
                # Convert the pitch to a Note enum value (C = 60, C# = 61, etc.)
                pitch = (root_note.value + root_offset) % 12 + 60
                chord_root = None
                for note in Note:
                    if note.value == pitch:
                        chord_root = note
                        break
                if chord_root is None:
                    return []  # Invalid pitch, can't create chord

                return MusicTheory.build_chord(chord_root, chord_type)

        except Exception as e:
            print(f"Error generating chord from roman numeral '{roman_numeral}' in '{key_scale_string}': {e}")

        return []

    @staticmethod
    def _get_chord_type_from_roman(roman_numeral: str) -> ChordType:
        """Determine chord type from Roman numeral notation."""
        if roman_numeral.endswith('7'):
            if roman_numeral.lower().endswith('7dim'):
                return ChordType.DIMINISHED_7TH
            elif roman_numeral.lower().startswith('v'):
                return ChordType.DOMINANT_7TH
            elif roman_numeral.lower().startswith('i'):
                return ChordType.MINOR_7TH
            else:
                return ChordType.DOMINANT_7TH
        elif roman_numeral.lower().startswith('v'):
            return ChordType.DOMINANT_7TH
        elif roman_numeral.endswith('dim'):
            return ChordType.DIMINISHED
        elif roman_numeral.islower():
            return ChordType.MINOR
        else:
            return ChordType.MAJOR

    @staticmethod
    def calculate_harmonic_distance(from_roman: str, to_roman: str, key_scale_string: str) -> float:
        """
        Calculate harmonic distance between two chords in a key.

        Args:
            from_roman: Starting chord Roman numeral
            to_roman: Ending chord Roman numeral
            key_scale_string: Key and scale string

        Returns:
            Float representing harmonic distance (0.0 = same chord, higher = more distant)
        """
        try:
            from_pitches = set(MusicTheory.get_chord_pitches_from_roman(from_roman, key_scale_string))
            to_pitches = set(MusicTheory.get_chord_pitches_from_roman(to_roman, key_scale_string))

            if not from_pitches or not to_pitches:
                return 10.0  # Unknown chords are very distant

            # Common notes
            common_notes = len(from_pitches.intersection(to_pitches))

            # Total unique notes
            total_notes = len(from_pitches.union(to_pitches))

            # Distance is proportion of different notes
            distance = 1.0 - (common_notes / total_notes)

            # Add penalty for modal distance (changes in chord quality)
            from_type = MusicTheory._get_chord_type_from_roman(from_roman).value
            to_type = MusicTheory._get_chord_type_from_roman(to_roman).value
            if from_type != to_type:
                distance += 0.5

            return distance

        except:
            return 10.0  # Unknown chords are very distant

    @staticmethod
    def validate_key_mode(key: str, mode: str) -> bool:
        """
        Validate if a key and mode combination is valid.

        Args:
            key: Root note (e.g., 'C', 'A#')
            mode: Scale type (e.g., 'major', 'dorian')

        Returns:
            True if the combination is valid, False otherwise
        """
        try:
            scale_str = f"{key} {mode}"
            MusicTheory.parse_scale_string(scale_str)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_all_roots() -> List[str]:
        """
        Get all chromatic root notes.

        Returns:
            List of all possible root notes from C to B including sharps/flats
        """
        return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    @staticmethod
    def get_all_modes() -> List[str]:
        """
        Get all available scale modes.

        Returns:
            List of all scale type names as strings
        """
        return [scale_type.value for scale_type in ScaleType]

    @staticmethod
    def filter_progressions_by_distance(progressions: List[List[str]],
                                       key_scale_string: str,
                                       distance_level: str) -> List[List[str]]:
        """
        Filter chord progressions by harmonic distance.

        Args:
            progressions: List of progression lists (e.g., [["I", "IV", "V"]])
            key_scale_string: Key and scale string
            distance_level: "close", "medium", or "distant"

        Returns:
            Filtered list of progressions
        """
        distance_ranges = {
            "close": (0.0, 0.3),
            "medium": (0.3, 0.7),
            "distant": (0.7, 10.0)
        }

        min_dist, max_dist = distance_ranges.get(distance_level, (0.0, 10.0))
        filtered = []

        for progression in progressions:
            if len(progression) < 2:
                filtered.append(progression)
                continue

            total_distance = 0.0
            transition_count = 0

            for i in range(len(progression) - 1):
                distance = MusicTheory.calculate_harmonic_distance(
                    progression[i], progression[i + 1], key_scale_string
                )
                total_distance += distance
                transition_count += 1

            avg_distance = total_distance / transition_count if transition_count > 0 else 0.0

            if min_dist <= avg_distance <= max_dist:
                filtered.append(progression)

        return filtered


# Example usage and testing
if __name__ == "__main__":
    print("Music Theory Module Test")
    print("=" * 40)

    # Test scale construction
    c_major = MusicTheory.get_scale_pitches_from_string("C major", 2)
    print(f"C Major scale: {c_major}")

    # Test chord construction
    c_major_chord = MusicTheory.build_chord(Note.C, ChordType.MAJOR)
    print(f"C Major chord: {c_major_chord}")

    # Test Roman numeral chords
    i_chord = MusicTheory.get_chord_pitches_from_roman("I", "C major")
    print(f"I chord in C major: {i_chord}")

    v7_chord = MusicTheory.get_chord_pitches_from_roman("V7", "C major")
    print(f"V7 chord in C major: {v7_chord}")

    # Test minor key chords
    i_minor = MusicTheory.get_chord_pitches_from_roman("i", "C minor")
    print(f"i chord in C minor: {i_minor}")

    bVII_chord = MusicTheory.get_chord_pitches_from_roman("bVII", "C minor")
    print(f"bVII chord in C minor: {bVII_chord}")

    # Test harmonic distance
    print("\nHarmonic Distance Tests:")
    iv_distance = MusicTheory.calculate_harmonic_distance("I", "IV", "C major")
    vi_distance = MusicTheory.calculate_harmonic_distance("I", "vi", "C major")
    bvi_distance = MusicTheory.calculate_harmonic_distance("I", "bVI", "C major")
    print(".1f")
    print(".1f")
    print(".1f")

    # Test progression distance
    close_prog = ["I", "V", "IV"]
    med_prog = ["I", "iii", "vi"]
    distant_prog = ["I", "VI", "iii", "VII"]

    print("\nProgression Distance Tests:")
    print(".1f")
    print(".1f")
    print(".1f")

    print(f"\nFiltering progressions for 'close' distance: {close_prog}")
    filtered = MusicTheory.filter_progressions_by_distance(
        [close_prog, med_prog, distant_prog], "C major", "close")
    print(f"Filtered: {filtered}")