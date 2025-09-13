"""
MIDI output handler for the MIDI Master music generation program.

This module is responsible for converting the internal musical representation
into standard MIDI files that can be played in any MIDI-compatible software
or Digital Audio Workstation (DAW).

The module uses the mido library to create and write MIDI files, handling
the conversion of notes, chords, and timing information into the MIDI format.
"""
import random
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido library not available. MIDI output will not work.")

from structures.data_structures import Note, Chord, Pattern, PatternType
from structures.song_skeleton import SongSkeleton
from genres.genre_rules import GenreRules


class MidiOutput:
    """Handles output of generated music to MIDI files.

    This class converts the internal representation of musical elements
    (notes, chords, patterns, and song structure) into standard MIDI format.

    The conversion process:
    1. Creates a new MIDI file with appropriate tempo and time signature
    2. Processes each section and its patterns
    3. Converts notes and chords to MIDI note events
    4. Writes the MIDI file to disk

    The resulting MIDI files can be imported into any DAW or played with
    standard MIDI players.
    """

    def __init__(self):
        """Initialize the MidiOutput.

        Checks that the required mido library is available for MIDI operations.

        Raises:
            ImportError: If the mido library is not available
        """
        if not MIDO_AVAILABLE:
            raise ImportError("mido library is required for MIDI output")

    def create_temp_directory(self, base_temp_dir: Optional[str] = None) -> Path:
        """
        Create a temporary directory for output files.

        Args:
            base_temp_dir: Base temporary directory path, uses system default if None

        Returns:
            Path to created temporary directory
        """
        if base_temp_dir:
            temp_base = Path(base_temp_dir) / "midi_master_temp"
        else:
            temp_base = Path(tempfile.gettempdir()) / "midi_master_temp"

        temp_base.mkdir(parents=True, exist_ok=True)
        return temp_base

    def generate_temp_filename(self, prefix: str = "preview", genre: str = "",
                              mood: str = "", temp_dir: Optional[Path] = None) -> Path:
        """
        Generate a temporary filename for preview files.

        Args:
            prefix: Filename prefix
            genre: Music genre
            mood: Music mood
            temp_dir: Temporary directory path

        Returns:
            Full path to temporary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if genre and mood:
            filename = f"{prefix}_{genre}_{mood}_{timestamp}.mid"
        else:
            filename = f"{prefix}_{timestamp}.mid"

        if temp_dir:
            return temp_dir / filename
        else:
            return Path(tempfile.mktemp(suffix='.mid', prefix=prefix))

    def check_file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists at the given path.

        Args:
            file_path: Path to check

        Returns:
            bool: True if file exists, False otherwise
        """
        return Path(file_path).exists()

    def copy_to_final_location(self, temp_path: str, final_path: str) -> bool:
        """
        Copy a temporary file to its final location.

        Args:
            temp_path: Path to temporary file
            final_path: Path to final destination

        Returns:
            bool: True if copy was successful, False if target file exists (needs overwrite confirmation)

        Raises:
            IOError: If copy operation fails for other reasons
        """
        final_path_obj = Path(final_path)
        final_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Check if target file exists
        if final_path_obj.exists():
            return False  # File exists, needs overwrite confirmation

        try:
            shutil.copy2(temp_path, final_path)
            return True
        except IOError as e:
            raise IOError(f"Failed to copy file to final location: {e}")

    def ensure_directory_exists(self, path: str) -> None:
        """
        Ensure that a directory exists, creating it if necessary.

        Args:
            path: Directory path to ensure exists

        Raises:
            OSError: If directory creation fails
        """
        Path(path).mkdir(parents=True, exist_ok=True)

    def validate_output_path(self, path: str) -> Tuple[bool, str]:
        """
        Validate an output path for writing.

        Args:
            path: Path to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        path_obj = Path(path)

        # Check if parent directory exists or can be created
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False, f"Cannot create directory: {path_obj.parent}"

        # Check if we can write to the location (by trying to create parent)
        if path_obj.parent.exists() and not os.access(path_obj.parent, os.W_OK):
            return False, f"No write permission for directory: {path_obj.parent}"

        # Check filename validity
        if not path_obj.name:
            return False, "Filename cannot be empty"

        if path_obj.name.startswith('.'):
            return False, "Filename cannot start with a dot"

        # Check for invalid characters in filename
        invalid_chars = '<>:"/\\|?*'
        if any(char in path_obj.name for char in invalid_chars):
            return False, f"Filename contains invalid characters: {invalid_chars}"

        # Check file extension
        if path_obj.suffix.lower() != '.mid':
            return False, "File must have .mid extension"

        return True, ""

    def save_to_midi(self, song_skeleton: SongSkeleton, output_path: str, genre_rules: Optional[GenreRules] = None, separate_files: bool = False):
        """
        Save a song to a MIDI file.

        This method creates a MIDI file representation of the song and writes
        it to disk. It handles tempo, time signature, and converts all musical
        elements to MIDI events.

        Args:
            song_skeleton: The song to save (contains all musical content)
            output_path: The full path to save to (should end with .mid)
            genre_rules: The genre rules used for generation, for applying swing etc.
            separate_files: If True, save patterns to separate MIDI files per instrument
        """
        # Ensure directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if separate_files:
            return self.save_to_separate_midi_files(song_skeleton, output_path.replace('.mid', ''), genre_rules)

        # Create a new MIDI file with one track
        midi_file = mido.MidiFile()
        track = mido.MidiTrack()
        midi_file.tracks.append(track)

        # Set tempo based on song skeleton
        tempo = mido.bpm2tempo(song_skeleton.tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        # Add time signature (4/4 default is standard for most genres)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # Initialize current position tracker for proper MIDI timing
        current_time = 0

        # Process each section and its patterns
        current_time_beats = 0.0
        for section_type, patterns in song_skeleton.sections.items():
            # Calculate the duration of the current section in beats
            section_duration_beats = 0.0
            for pattern in patterns:
                # Assuming patterns are roughly aligned to bars, 4 beats per bar
                # This is a simplification; a more robust solution would track actual pattern lengths
                section_duration_beats = max(section_duration_beats, len(pattern.notes) * 0.25) # Rough estimate

            # Add MIDI CC automation based on section type
            if section_type.value == 'pre_chorus':
                # Gradual increase in expression (CC11) to build tension
                for i in range(1, 11):
                    track.append(mido.Message('control_change', channel=0, control=11, value=i*10, time=mido.bpm2tempo(song_skeleton.tempo) // 10))
            elif section_type.value == 'chorus':
                # Reset expression and add some modulation (CC1) for richness
                track.append(mido.Message('control_change', channel=0, control=11, value=100, time=0))
                track.append(mido.Message('control_change', channel=0, control=1, value=64, time=0))
            elif section_type.value == 'outro':
                # Gradual decrease in expression
                for i in range(10, 0, -1):
                    track.append(mido.Message('control_change', channel=0, control=11, value=i*10, time=mido.bpm2tempo(song_skeleton.tempo) // 10))

            for pattern in patterns:
                current_time = self._add_pattern_to_track(track, pattern, current_time, pattern.pattern_type.value, genre_rules)

        # Also add any standalone patterns that aren't in sections
        for pattern in song_skeleton.patterns:
            # Check if pattern is already in a section to avoid duplication
            in_section = False
            for section_patterns in song_skeleton.sections.values():
                if pattern in section_patterns:
                    in_section = True
                    break

            # Only add patterns that aren't already in sections
            if not in_section:
                current_time = self._add_pattern_to_track(track, pattern, current_time, pattern.pattern_type.value, genre_rules)

        # Save the MIDI file to the specified output path
        midi_file.save(output_path)

    def save_to_separate_midi_files(self, song_skeleton, base_filename, genre_rules=None):
        """
        Save patterns to separate MIDI files per instrument type.

        Args:
            song_skeleton: The song skeleton containing patterns
            base_filename: Base name for output files (without extension)
            genre_rules: Optional genre rules for applying swing etc.
        """
        patterns_by_type = defaultdict(list)
        for pattern in song_skeleton.patterns:
            patterns_by_type[pattern.pattern_type].append(pattern)

        instruments = {
            PatternType.MELODY: 'melody',
            PatternType.HARMONY: 'harmony',
            PatternType.BASS: 'bass',
            PatternType.RHYTHM: 'rhythm'
        }

        for ptype, patterns in patterns_by_type.items():
            if not patterns:
                continue
            midi_file = mido.MidiFile()
            track = mido.MidiTrack()
            midi_file.tracks.append(track)

            tempo = mido.bpm2tempo(song_skeleton.tempo)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

            # Get time signature for this pattern type, default to 4/4
            numerator, denominator = song_skeleton.get_time_signature(ptype)
            track.append(mido.MetaMessage('time_signature', numerator=numerator, denominator=denominator, time=0))

            program = self._get_program_for_instrument(instruments[ptype])
            track.append(mido.Message('program_change', channel=0, program=program, time=0))

            current_time = 0
            for pattern in sorted(patterns, key=lambda p: min((n.start_time for n in p.notes), default=float('inf'))):
                current_time = self._add_pattern_to_track(track, pattern, current_time, instruments[ptype], genre_rules)

            filename = f"{base_filename}_{instruments[ptype]}.mid"
            midi_file.save(filename)

    def _add_pattern_to_track(self, track, pattern: Pattern, current_time: int, section_type: str, genre_rules: Optional[GenreRules] = None):
        """
        Add a pattern to a MIDI track.

        This method processes all notes and chords in a pattern and adds
        them to the MIDI track as appropriate MIDI events.

        Args:
            track: The MIDI track to add to
            pattern: The pattern to add (contains notes and chords)
            current_time: Current time position in ticks
            section_type: Type of section (melody, harmony, bass, rhythm)
            genre_rules: The genre rules used for generation, for applying swing etc.

        Returns:
            Updated current time after processing the pattern
        """
        # Get appropriate MIDI channel for different musical elements
        channel = self._get_channel_for_section(section_type)

        # Process individual notes in the pattern
        for note in pattern.notes:
            current_time = self._add_note_to_track(track, note, current_time, channel, genre_rules)

        # Process chords in the pattern
        # Each chord contains multiple notes that play simultaneously
        for chord in pattern.chords:
            current_time = self._add_chord_to_track(track, chord, current_time, channel, genre_rules)

        return current_time

    def _get_channel_for_section(self, section_type: str) -> int:
        """
        Get MIDI channel assignment for different section types.

        Args:
            section_type: Type of musical section

        Returns:
            MIDI channel number (0-15)
        """
        channel_map = {
            'melody': 0,      # Main melody on channel 0
            'harmony': 1,     # Harmony on channel 1
            'bass': 2,        # Bass on channel 2
            'rhythm': 9,      # Drums on channel 9 (standard GM drum channel)
            'standalone': 0   # Default to channel 0
        }
        return channel_map.get(section_type, 0)

    def _get_program_for_instrument(self, instrument: str) -> int:
        """
        Get MIDI program number for different instruments.

        Args:
            instrument: Type of instrument ('melody', 'harmony', 'bass', 'rhythm')

        Returns:
            MIDI program number (0-127)
        """
        programs = {
            'melody': 0,    # Grand Piano
            'harmony': 0,   # Grand Piano
            'bass': 32,     # Acoustic Bass
            'rhythm': 0     # Grand Piano (for now)
        }
        return programs.get(instrument, 0)

    def _add_chord_to_track(self, track, chord: Chord, current_time: int, channel: int, genre_rules: Optional[GenreRules] = None):
        """
        Add a chord to a MIDI track as simultaneous note events.

        Args:
            track: The MIDI track to add to
            chord: The chord to add
            current_time: Current time position in ticks
            channel: MIDI channel to use
            genre_rules: The genre rules used for generation, for applying swing etc.

        Returns:
            Updated current time after the chord
        """
        if not chord.notes:
            return current_time

        # Calculate timing for the chord
        ticks_per_beat = 480  # Standard MIDI resolution

        # Find the earliest and latest note times in the chord
        chord_start_ticks = min(int(note.start_time * ticks_per_beat) for note in chord.notes)
        chord_end_ticks = max(int((note.start_time + note.duration) * ticks_per_beat) for note in chord.notes)

        # Calculate delta time from current position
        delta_time = chord_start_ticks - current_time

        # Add all note_on messages simultaneously (time=0 for subsequent notes)
        first_note = True
        for note in chord.notes:
            # Ensure minimum duration
            duration_ticks = int(note.duration * ticks_per_beat)
            if duration_ticks < 1:
                duration_ticks = 1

            # First note uses delta time from last event
            time_delta = max(0, delta_time) if first_note else 0
            first_note = False

            track.append(mido.Message(
                'note_on',
                channel=channel,
                note=note.pitch,
                velocity=note.velocity,
                time=time_delta
            ))

        # Add note_off messages (all at time 0 since they end simultaneously)
        chord_duration_ticks = chord_end_ticks - chord_start_ticks
        if chord_duration_ticks < 1:
            chord_duration_ticks = 1

        for note in chord.notes:
            track.append(mido.Message(
                'note_off',
                channel=channel,
                note=note.pitch,
                velocity=0,
                time=0 if note != chord.notes[0] else chord_duration_ticks
            ))

        return chord_end_ticks

    def _add_note_to_track(self, track, note: Note, current_time: int, channel: int, genre_rules: Optional[GenreRules] = None):
        """
        Add a note to a MIDI track.

        This method converts a Note object into MIDI note_on and note_off
        events with appropriate timing and adds them to the track.

        Args:
            track: The MIDI track to add to
            note: The note to add
            current_time: Current time position in ticks for relative timing
            channel: MIDI channel to use
            genre_rules: The genre rules used for generation, for applying swing etc.

        Returns:
            Updated current time after the note
        """
        # Calculate timing in MIDI ticks with proper tempo conversion
        ticks_per_beat = 480  # Standard MIDI resolution

        # Convert note timing from beats to MIDI ticks
        start_ticks = int(note.start_time * ticks_per_beat)
        duration_ticks = int(note.duration * ticks_per_beat)

        # Apply swing and micro-timing imperfections
        if genre_rules:
            beat_position = note.start_time % 1.0
            swing_factor = genre_rules.get_beat_characteristics().get('swing_factor', 0.5)

            # Apply swing to off-beats (8th notes)
            if 0.45 < beat_position < 0.55 or 0.95 < beat_position < 1.05:
                swing_offset = int((swing_factor - 0.5) * ticks_per_beat * 0.5)
                start_ticks += swing_offset

            # Apply micro-timing imperfections
            timing_imperfection = int(random.uniform(-0.02, 0.02) * ticks_per_beat)
            start_ticks += timing_imperfection


        # Calculate delta time from current position
        delta_time = start_ticks - current_time

        # Ensure minimum duration (prevent zero-length notes)
        if duration_ticks < 1:
            duration_ticks = 1

        # Add note on message
        track.append(mido.Message(
            'note_on',
            channel=channel,
            note=note.pitch,
            velocity=note.velocity,
            time=max(0, delta_time)  # Ensure non-negative delta time
        ))

        # Add note off message
        track.append(mido.Message(
            'note_off',
            channel=channel,
            note=note.pitch,
            velocity=0,
            time=duration_ticks
        ))

        # Return updated current time
        return start_ticks + duration_ticks