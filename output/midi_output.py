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
from typing import List, Optional, Tuple, Any, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from generators.generator_context import GeneratorContext
from collections import defaultdict
from datetime import datetime
import logging
from core.filename_templater import format_filename as templ_format

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido library not available. MIDI output will not work.")

from structures.data_structures import Note, Chord, Pattern, PatternType
from structures.song_skeleton import SongSkeleton
from genres.genre_rules import GenreRules

try:
    from output.metadata_manager import MetadataManager, ProjectMetadata, DAWType
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False
    MetadataManager = None
    ProjectMetadata = None
    DAWType = None
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

    def _sanitize_filename_component(self, value: str) -> str:
        """
        Sanitize a filename component (not a full path).
        - Lowercase
        - Replace spaces with underscores
        - Remove/replace invalid characters
        - Avoid Windows reserved names
        """
        value = str(value or "").strip().lower().replace(" ", "_")

        # Remove invalid characters for cross-platform safety
        invalid_chars = '<>:"/\\|?*'
        for ch in invalid_chars:
            value = value.replace(ch, "_")

        # Keep only safe characters (alnum, underscore, dash)
        safe = []
        for ch in value:
            if ch.isalnum() or ch in ("_", "-"):
                safe.append(ch)
            else:
                # drop other punctuation
                pass
        value = "".join(safe).strip("_-")

        # Avoid Windows reserved names for basenames
        reserved = {
            "con", "prn", "aux", "nul",
            *{f"com{i}" for i in range(1, 10)},
            *{f"lpt{i}" for i in range(1, 10)},
        }
        if value in reserved or value == "":
            value = (value + "_file") if value else "file"

        # Limit length of a component to be safe within 255 overall
        return value[:100]

    def generate_output_filename(
        self,
        genre: str,
        mood: str,
        tempo: int,
        time_signature: str,
        output_folder: Optional[str] = None,
    ) -> str:
        """
        Generate a default output filename from settings and the current date.

        Format: {genre}_{mood}_{tempo}_{time_sig}_{timestamp}.mid
        Example: pop_happy_120_4-4_20250915_210038.mid

        Args:
            genre: Genre name
            mood: Mood name
            tempo: Tempo in BPM
            time_signature: Time signature as string e.g. "4/4"
            output_folder: Optional folder; defaults to 'output/' if not provided

        Returns:
            Full path (as string) to the suggested output .mid file.
        """
        g = self._sanitize_filename_component(genre)
        m = self._sanitize_filename_component(mood)
        t = int(tempo) if isinstance(tempo, (int, float, str)) and str(tempo).isdigit() else tempo

        ts = time_signature.replace('/', '-')
        ts = self._sanitize_filename_component(ts)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{g}_{m}_{t}_{ts}_{timestamp}.mid"

        folder = Path(output_folder) if output_folder else Path("output")
        # Ensure folder exists only when actually saving; here we only return the path.
        return str(folder / filename)

    def get_unique_filename(self, file_path: str) -> str:
        """
        Generate a unique filename by appending a counter if the file exists.

        Args:
            file_path: The desired file path

        Returns:
            A unique file path that does not exist yet
        """
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if not path_obj.exists():
            return str(path_obj)

        stem = path_obj.stem
        suffix = path_obj.suffix
        dir_path = path_obj.parent
        counter = 1
        while True:
            new_stem = f"{stem}_{counter}"
            new_path = dir_path / f"{new_stem}{suffix}"
            if not new_path.exists():
                return str(new_path)
            counter += 1

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

    def save_to_midi(self, song_skeleton: SongSkeleton, output_path: Optional[str] = None, genre_rules: Optional[GenreRules] = None, separate_files: bool = False, context: Optional['GeneratorContext'] = None, genre: Optional[str] = None, mood: Optional[str] = None, tempo: Optional[int] = None, time_signature: str = "4/4", filename_template: Optional[str] = None, template_settings: Optional[Dict[str, Any]] = None, template_context: Optional[Dict[str, Any]] = None, base_output_dir: Optional[str] = None):
        """
        Save a song to a MIDI file.

        This method creates a MIDI file representation of the song and writes
        it to disk. It handles tempo, time signature, and converts all musical
        elements to MIDI events.

        Args:
            song_skeleton: The song to save (contains all musical content)
            output_path: The full path to save to (should end with .mid). If None, auto-generate.
            genre_rules: The genre rules used for generation, for applying swing etc.
            separate_files: If True, save patterns to separate MIDI files per instrument
            genre: Genre for auto filename (required if output_path is None)
            mood: Mood for auto filename (required if output_path is None)
            tempo: Tempo for auto filename (required if output_path is None)
            time_signature: Time signature for auto filename (default "4/4")
        """
        # Generate or make unique output_path
        if output_path is None:
            # If a template is provided, use it to construct the output path; otherwise legacy behavior
            if filename_template:
                base_dir = base_output_dir or "output"
                # Build minimal settings/context if not provided explicitly
                settings = template_settings or {
                    "genre": genre or "",
                    "mood": mood or "",
                    "tempo": tempo or "",
                    "bars": getattr(song_skeleton, "bars", ""),
                }
                ctx = dict(template_context or {})
                # For combined file naming
                ctx.setdefault("stem", "combined")
                output_path = str(templ_format(filename_template, settings, ctx, base_dir=base_dir))
                logging.info(f"Templated filename: {output_path}")
            else:
                if genre is None or mood is None or tempo is None:
                    raise ValueError("Genre, mood, and tempo are required for automatic filename generation.")
                output_path = self.generate_output_filename(genre, mood, tempo, time_signature)
                output_path = self.get_unique_filename(output_path)
                logging.info(f"Auto-generated unique filename: {output_path}")
        else:
            output_path = self.get_unique_filename(output_path)
            logging.info(f"Using provided filename (made unique): {output_path}")

        # Ensure directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if separate_files:
            base_filename = str(output_path_obj.with_suffix(''))
            # Route separate stems through templater when template provided; otherwise preserve legacy base naming
            self.save_to_separate_midi_files(
                song_skeleton,
                base_filename,
                genre_rules,
                context,
                filename_template=filename_template,
                template_settings=template_settings or {
                    "genre": genre or "",
                    "mood": mood or "",
                    "tempo": tempo or "",
                    "bars": getattr(song_skeleton, "bars", ""),
                } if filename_template else None,
                template_context=template_context,
                base_output_dir=base_output_dir or "output",
            )
            return

        # Create a new MIDI file with one track
        midi_file = mido.MidiFile()
        track = mido.MidiTrack()
        midi_file.tracks.append(track)

        # Set tempo based on song skeleton
        tempo = mido.bpm2tempo(song_skeleton.tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        bpm = song_skeleton.tempo
        
        # Add time signature (4/4 default is standard for most genres)
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        # Initialize current position tracker for proper MIDI timing
        current_time = 0

        # Process each section and its patterns with proper section offsets
        ticks_per_beat = 480
        current_time_beats = 0.0
        for section_type, patterns in song_skeleton.sections:
            # Decide target length for this section (in beats)
            section_duration_beats = self._get_section_target_length_beats(section_type)
    
            # Move to the exact section start and add a MIDI marker (for DAW navigation)
            section_start_ticks = int(current_time_beats * ticks_per_beat)
            delta_to_section = section_start_ticks - current_time
            try:
                track.append(mido.MetaMessage('marker', text=str(section_type.value), time=max(0, delta_to_section)))
            except Exception:
                # Some players may not support marker meta; ignore if fails
                pass
            # Align internal current_time to section start
            current_time = max(current_time, section_start_ticks)
    
            # Add MIDI CC automation based on section type
            if section_type.value == 'pre_chorus':
                for i in range(1, 11):
                    track.append(mido.Message('control_change', channel=0, control=11, value=i*10, time=mido.bpm2tempo(song_skeleton.tempo) // 10))
            elif section_type.value == 'chorus':
                track.append(mido.Message('control_change', channel=0, control=11, value=100, time=0))
                track.append(mido.Message('control_change', channel=0, control=1, value=64, time=0))
            elif section_type.value == 'outro':
                for i in range(10, 0, -1):
                    track.append(mido.Message('control_change', channel=0, control=11, value=i*10, time=mido.bpm2tempo(song_skeleton.tempo) // 10))
    
            # Add patterns with section start offset (in beats) and clamp to section length
            for pattern in patterns:
                current_time = self._add_pattern_to_track(
                    track,
                    pattern,
                    current_time,
                    pattern.pattern_type.value,
                    genre_rules,
                    context=context,
                    bpm=bpm,
                    section_start_beats=current_time_beats,
                    section_duration_beats=section_duration_beats
                )
    
            # Advance section start position by its target duration
            current_time_beats += section_duration_beats
    
        # Also add any standalone patterns that aren't in sections (append after sections)
        for pattern in song_skeleton.patterns:
            # Check if pattern is already in a section to avoid duplication
            in_section = False
            for _, section_patterns in song_skeleton.sections:
                if pattern in section_patterns:
                    in_section = True
                    break
    
            # Only add patterns that aren't already in sections
            if not in_section:
                # Place after the last section
                section_start_ticks = int(current_time_beats * ticks_per_beat)
                delta_to_section = section_start_ticks - current_time
                if delta_to_section > 0:
                    track.append(mido.MetaMessage('text', text='standalone', time=delta_to_section))
                    current_time = section_start_ticks
    
                current_time = self._add_pattern_to_track(
                    track,
                    pattern,
                    current_time,
                    pattern.pattern_type.value,
                    genre_rules,
                    context=context,
                    bpm=bpm,
                    section_start_beats=current_time_beats,
                    section_duration_beats=self._estimate_pattern_duration_beats(pattern)
                )
                # Advance offset by this pattern's duration estimate
                current_time_beats += self._estimate_pattern_duration_beats(pattern)

        # Save the MIDI file to the specified output path
        logging.info(f"Saved combined MIDI to: {output_path}")
        midi_file.save(output_path)

    def save_to_separate_midi_files(self, song_skeleton, base_filename: str, genre_rules=None, context=None, filename_template: Optional[str] = None, template_settings: Optional[Dict[str, Any]] = None, template_context: Optional[Dict[str, Any]] = None, base_output_dir: Optional[str] = None):
        """
        Save patterns to separate MIDI files per instrument type.

        Args:
            song_skeleton: The song skeleton containing patterns
            base_filename: Base name for output files (without extension)
            genre_rules: Optional genre rules for applying swing etc.
            context: Optional generator context for performance
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
                current_time = self._add_pattern_to_track(track, pattern, current_time, instruments[ptype], genre_rules, context=context, bpm=song_skeleton.tempo)

            if filename_template:
                settings = template_settings or {}
                ctx = dict(template_context or {})
                ctx.setdefault("stem", instruments[ptype])
                base_dir = base_output_dir or "output"
                out_path = templ_format(filename_template, settings, ctx, base_dir=base_dir)
                filename_path = str(out_path)
            else:
                filename_path = self.get_unique_filename(f"{base_filename}_{instruments[ptype]}.mid")
            logging.info(f"Saved {instruments[ptype]} stem to: {filename_path}")
            midi_file.save(filename_path)

    def _add_pattern_to_track(
        self,
        track,
        pattern: Pattern,
        current_time: int,
        section_type: str,
        genre_rules: Optional[GenreRules] = None,
        context: Optional['GeneratorContext'] = None,
        bpm: Optional[int] = None,
        section_start_beats: float = 0.0,
        section_duration_beats: Optional[float] = None
    ):
        """
        Add a pattern to a MIDI track with section start offset and optional clamping.
    
        Notes and chords are cloned with adjusted start_time = local + section_start_beats
        and durations clamped to not exceed section_duration_beats.
        """
        # Get appropriate MIDI channel for different musical elements
        channel = self._get_channel_for_section(section_type)
    
        # Helper to clamp duration within section bounds
        def _clamp_duration(local_start: float, dur: float) -> float:
            if section_duration_beats is None:
                return dur
            remaining = max(0.0, section_duration_beats - local_start)
            return max(0.0, min(dur, remaining))
    
        # Process individual notes in the pattern (clone, offset, clamp)
        for note in getattr(pattern, 'notes', []) or []:
            local_start = float(getattr(note, 'start_time', 0.0))
            if section_duration_beats is not None and local_start >= section_duration_beats:
                continue
            dur = _clamp_duration(local_start, float(getattr(note, 'duration', 0.0)))
            if dur <= 0.0:
                continue
            adj_note = Note(int(note.pitch), dur, int(note.velocity), local_start + section_start_beats)
            current_time = self._add_note_to_track(track, adj_note, current_time, channel, genre_rules, context=context, bpm=bpm)
    
        # Process chords in the pattern (clone all chord notes, offset, clamp)
        for chord in getattr(pattern, 'chords', []) or []:
            # Determine chord local start as earliest note start
            if getattr(chord, 'notes', []):
                chord_local_start = min(float(n.start_time) for n in chord.notes)
                chord_end = max(float(n.start_time + n.duration) for n in chord.notes)
                chord_total_dur = chord_end - chord_local_start
            else:
                chord_local_start = float(getattr(chord, 'start_time', 0.0))
                try:
                    chord_total_dur = float(chord.duration())
                except Exception:
                    chord_total_dur = 0.0
    
            if section_duration_beats is not None and chord_local_start >= section_duration_beats:
                continue
    
            # Clamp chord to section length
            clamped_total = _clamp_duration(chord_local_start, chord_total_dur)
            if clamped_total <= 0.0:
                continue
    
            # Build cloned chord notes
            cloned_notes: List[Note] = []
            for n in getattr(chord, 'notes', []) or []:
                n_local = float(n.start_time)
                if section_duration_beats is not None and n_local >= section_duration_beats:
                    continue
                n_dur = _clamp_duration(n_local, float(n.duration))
                if n_dur <= 0.0:
                    continue
                cloned_notes.append(Note(int(n.pitch), n_dur, int(n.velocity), n_local + section_start_beats))
    
            if not cloned_notes:
                continue
    
            cloned_chord = Chord(cloned_notes, start_time=chord_local_start + section_start_beats)
            current_time = self._add_chord_to_track(track, cloned_chord, current_time, channel, genre_rules)
    
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
    
    def _get_section_target_length_beats(self, section_type) -> float:
        """
        Target length per section type (beats). Defaults to 8 bars (32 beats).
        """
        mapping = {
            'intro': 32.0,         # 8 bars
            'verse': 64.0,         # 16 bars
            'pre_chorus': 16.0,    # 4 bars
            'chorus': 64.0,        # 16 bars
            'post_chorus': 16.0,   # 4 bars
            'bridge': 32.0,        # 8 bars
            'solo': 64.0,          # 16 bars
            'fill': 8.0,           # 2 bars
            'outro': 32.0          # 8 bars
        }
        key = getattr(section_type, "value", str(section_type))
        return mapping.get(key, 32.0)
    
    def _estimate_pattern_duration_beats(self, pattern: Pattern) -> float:
        """
        Estimate pattern duration in beats by scanning note and chord endpoints.
        """
        end_beats = 0.0
        try:
            for n in getattr(pattern, 'notes', []) or []:
                end_beats = max(end_beats, float(n.start_time) + float(n.duration))
            for ch in getattr(pattern, 'chords', []) or []:
                if getattr(ch, 'notes', []):
                    for n in ch.notes:
                        end_beats = max(end_beats, float(n.start_time) + float(n.duration))
                else:
                    # Fallback: use chord start + duration() if available
                    try:
                        end_beats = max(end_beats, float(getattr(ch, 'start_time', 0.0)) + float(ch.duration()))
                    except Exception:
                        pass
        except Exception:
            pass
        if end_beats <= 0.0:
            end_beats = 16.0  # conservative default (4 bars)
        return end_beats

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

    def _add_note_to_track(self, track, note: Note, current_time: int, channel: int, genre_rules: Optional[GenreRules] = None, context: Optional['GeneratorContext'] = None, bpm: Optional[int] = None):
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

        # Defaults for performance shaping
        perf = getattr(context, "performance", None) if context else None
        swing_mode = getattr(perf, "swing_mode", "eighth") if perf else "eighth"
        # Swing factor source: genre rules by default
        swing_factor = 0.5
        if genre_rules:
            try:
                swing_factor = genre_rules.get_beat_characteristics().get('swing_factor', 0.5)
            except Exception:
                swing_factor = 0.5

        # Apply swing based on mode
        # Eighth-note swing: delay every second 8th; Sixteenth: delay odd 16ths; Triplet handled like eighth for simplicity
        beat_pos = note.start_time % 1.0
        if abs(swing_factor - 0.5) > 1e-6:
            if swing_mode.lower().startswith("sixteenth"):
                # Which 16th within the beat?
                sixteenth_len = 0.25
                idx16 = int(round((beat_pos / sixteenth_len)))  # 0..4 approx
                if idx16 % 2 == 1:
                    swing_offset = int((swing_factor - 0.5) * ticks_per_beat * 0.25)  # quarter-beat offset
                    start_ticks += swing_offset
            else:
                # Eighth/triplet swing as default: adjust the off-8ths near 0.5 within the beat
                if 0.45 < beat_pos < 0.55 or 0.95 < beat_pos < 1.05:
                    swing_offset = int((swing_factor - 0.5) * ticks_per_beat * 0.5)
                    start_ticks += swing_offset

        # Apply grid bias and micro-timing humanization (ms) if available
        if bpm is None:
            # Fallback: small fixed humanize if tempo unknown
            timing_imperfection = int(random.uniform(-0.02, 0.02) * ticks_per_beat)
            start_ticks += timing_imperfection
        else:
            # Convert ms to ticks: ticks_per_ms = ticks_per_beat * (BPM / 60000)
            ticks_per_ms = ticks_per_beat * (float(bpm) / 60000.0)
            micro_ms = float(getattr(perf, "micro_timing_range_ms", 0.0)) if perf else 0.0
            bias_ms = float(getattr(perf, "grid_bias_ms", 0.0)) if perf else 0.0
            humanize_ms = random.uniform(-micro_ms, micro_ms) if micro_ms > 0 else 0.0
            total_ms = bias_ms + humanize_ms
            if abs(total_ms) > 1e-6:
                start_ticks += int(total_ms * ticks_per_ms)

        # Apply articulation and length variance
        if perf:
            # Length variance
            length_var = float(getattr(perf, "note_length_variance", 0.0))
            if length_var > 0.0:
                scale = 1.0 + random.uniform(-length_var, length_var)
                duration_ticks = max(1, int(duration_ticks * scale))

            # Articulation probabilities
            stacc_p = float(getattr(perf, "staccato_prob", 0.0))
            tenu_p = float(getattr(perf, "tenuto_prob", 0.0))
            marc_p = float(getattr(perf, "marcato_prob", 0.0))
            # legato_prob exists but true overlapping requires note context; skip overlap here
            r = random.random()
            acc_boost = 0
            if r < stacc_p:
                dur_scale = float(getattr(perf, "staccato_scale", 0.6))
                duration_ticks = max(1, int(duration_ticks * dur_scale))
            elif r < stacc_p + tenu_p:
                dur_scale = float(getattr(perf, "tenuto_scale", 1.0))
                duration_ticks = max(1, int(duration_ticks * dur_scale))
            elif r < stacc_p + tenu_p + marc_p:
                acc_boost = int(getattr(perf, "marcato_velocity_boost", 12))

            # Velocity profile shaping (per-phrase arch)
            velocity_val = int(note.velocity)
            try:
                vp = getattr(perf, "velocity_profile", None)
                if isinstance(vp, dict):
                    shape = str(vp.get("shape", "arch")).lower()
                    intensity = float(vp.get("intensity", 0.3))
                    phrase_len = float(vp.get("phrase_length_beats", 4.0))
                    if phrase_len > 0 and shape == "arch":
                        t = (note.start_time % phrase_len) / phrase_len  # 0..1
                        # Arch curve in [-1, 1] centered at 0.5, scaled modestly
                        arch = (-4 * (t - 0.5) ** 2 + 1.0)  # peak at 1, min ~0
                        mult = 1.0 + (intensity * arch * 0.2)  # max ~ +20% at peak
                        velocity_val = max(0, min(127, int(velocity_val * mult)))
                # Apply marcato accent if set
                if acc_boost:
                    velocity_val = max(0, min(127, velocity_val + acc_boost))
            except Exception:
                velocity_val = int(note.velocity)
        else:
            velocity_val = int(note.velocity)


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
            velocity=velocity_val,
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