# MIDI Master API Documentation

This document provides detailed API documentation for all modules and classes in the MIDI Master music generation program.

## Table of Contents

1. [Main Module](#main-module)
2. [Music Theory Module](#music-theory-module)
3. [Data Structures](#data-structures)
4. [Genres Module](#genres-module)
5. [Generators Module](#generators-module)
6. [Output Module](#output-module)
7. [Song Skeleton](#song-skeleton)

## Main Module

### main.py

The entry point for the MIDI Master application.

#### Functions

##### main()
Parses command-line arguments and orchestrates the music generation process.

```python
def main() -> None
```

Parameters:
- None

Returns:
- None

## Music Theory Module

### music_theory.py

Provides utilities for music theory calculations including scale construction, chord building, and Roman numeral mapping.

#### Classes

##### Note(Enum)
Musical notes with MIDI pitch values.

Values:
- C = 60
- C_SHARP = 61
- D_FLAT = 61
- D = 62
- D_SHARP = 63
- E_FLAT = 63
- E = 64
- F = 65
- F_SHARP = 6
- G_FLAT = 66
- G = 67
- G_SHARP = 68
- A_FLAT = 68
- A = 69
- A_SHARP = 70
- B_FLAT = 70
- B = 71

##### ScaleType(Enum)
Common musical scales.

Values:
- MAJOR = "major"
- MINOR = "minor"
- DORIAN = "dorian"
- PHRYGIAN = "phrygian"
- LYDIAN = "lydian"
- MIXOLYDIAN = "mixolydian"
- LOCRIAN = "locrian"
- MELODIC_MINOR = "melodic_minor"
- HARMONIC_MINOR = "harmonic_minor"
- BLUES = "blues"
- PENTATONIC_MAJOR = "pentatonic_major"
- PENTATONIC_MINOR = "pentatonic_minor"

##### ChordType(Enum)
Common chord types.

Values:
- MAJOR = "major"
- MINOR = "minor"
- DIMINISHED = "diminished"
- AUGMENTED = "augmented"
- MAJOR7 = "major7"
- MINOR7 = "minor7"
- DOMINANT7 = "dominant7"
- HALF_DIMINISHED7 = "half_diminished7"
- FULLY_DIMINISHED7 = "fully_diminished7"
- MINOR_MAJOR7 = "minor_major7"

##### MusicTheory
Music theory utilities for scale and chord construction.

###### Static Methods

####### parse_scale_string(scale_str)
Parse a scale string like 'C major' into note and scale type.

```python
@staticmethod
def parse_scale_string(scale_str: str) -> Tuple[Note, ScaleType]
```

Parameters:
- scale_str (str): Scale string (e.g., 'C major', 'A minor')

Returns:
- Tuple[Note, ScaleType]: Tuple of (root_note, scale_type)

####### build_scale(root_note, scale_type, octave_range)
Build a scale with MIDI pitch values.

```python
@staticmethod
def build_scale(root_note: Note, scale_type: ScaleType, octave_range: int = 2) -> List[int]
```

Parameters:
- root_note (Note): Root note of the scale
- scale_type (ScaleType): Type of scale
- octave_range (int): Number of octaves to span (default: 2)

Returns:
- List[int]: List of MIDI pitch values in the scale

####### build_chord(root_note, chord_type, octave)
Build a chord with MIDI pitch values.

```python
@staticmethod
def build_chord(root_note: Note, chord_type: ChordType, octave: int = 0) -> List[int]
```

Parameters:
- root_note (Note): Root note of the chord
- chord_type (ChordType): Type of chord
- octave (int): Octave offset (0 = root octave, default: 0)

Returns:
- List[int]: List of MIDI pitch values in the chord

####### roman_numeral_to_chord(roman_numeral, key_type, key_root)
Convert a Roman numeral chord symbol to actual chord pitches.

```python
@staticmethod
def roman_numeral_to_chord(roman_numeral: str, key_type: str, key_root: Note) -> List[int]
```

Parameters:
- roman_numeral (str): Roman numeral (e.g., "I", "V7", "ii")
- key_type (str): "major" or "minor"
- key_root (Note): Root note of the key

Returns:
- List[int]: List of MIDI pitch values for the chord

####### get_scale_pitches_from_string(scale_str, octave_range)
Convenience method to get scale pitches from a string.

```python
@staticmethod
def get_scale_pitches_from_string(scale_str: str, octave_range: int = 2) -> List[int]
```

Parameters:
- scale_str (str): Scale string (e.g., 'C major')
- octave_range (int): Number of octaves (default: 2)

Returns:
- List[int]: List of MIDI pitch values

####### get_chord_pitches_from_roman(roman_numeral, key_scale_str)
Convenience method to get chord pitches from Roman numeral and key.

```python
@staticmethod
def get_chord_pitches_from_roman(roman_numeral: str, key_scale_str: str) -> List[int]
```

Parameters:
- roman_numeral (str): Roman numeral (e.g., 'I', 'V7')
- key_scale_str (str): Key scale string (e.g., 'C major')

Returns:
- List[int]: List of MIDI pitch values

####### select_random_from_scale(scale_pitches, current_pitch, step_range)
Select a random pitch from a scale, optionally biased toward current pitch.

```python
@staticmethod
def select_random_from_scale(scale_pitches: List[int], current_pitch: Optional[int] = None, step_range: int = 4) -> int
```

Parameters:
- scale_pitches (List[int]): Available pitches in the scale
- current_pitch (Optional[int]): Current pitch for step-based selection (default: None)
- step_range (int): Maximum step size in semitones (default: 4)

Returns:
- int: Selected MIDI pitch value

## Data Structures

### structures/data_structures.py

Contains the core data structures used to represent musical elements.

#### Classes

##### Note
Represents a musical note with pitch, duration, and velocity.

###### Constructor

```python
def __init__(self, pitch: int, duration: float, velocity: int = 64, start_time: float = 0.0)
```

Parameters:
- pitch (int): MIDI pitch value (0-127)
- duration (float): Duration in beats
- velocity (int): Note velocity (0-127, default 64)
- start_time (float): Start time in beats from beginning of piece (default 0.0)

###### Methods

####### __repr__()
String representation of the Note.

```python
def __repr__(self) -> str
```

Returns:
- str: String representation

####### __eq__(other)
Equality comparison for Note objects.

```python
def __eq__(self, other) -> bool
```

Parameters:
- other: Object to compare with

Returns:
- bool: True if equal, False otherwise

##### Chord
Represents a chord as a collection of Notes played simultaneously.

###### Constructor

```python
def __init__(self, notes: List[Note], start_time: float = 0.0)
```

Parameters:
- notes (List[Note]): List of Notes that make up the chord
- start_time (float): Start time in beats from beginning of piece (default 0.0)

###### Methods

####### __repr__()
String representation of the Chord.

```python
def __repr__(self) -> str
```

Returns:
- str: String representation

####### add_note(note)
Add a note to the chord.

```python
def add_note(self, note: Note) -> None
```

Parameters:
- note (Note): Note to add

Returns:
- None

####### duration()
Get the duration of the chord (duration of the longest note).

```python
def duration(self) -> float
```

Returns:
- float: Duration of the chord

##### PatternType(Enum)
Types of musical patterns.

Values:
- MELODY = "melody"
- HARMONY = "harmony"
- RHYTHM = "rhythm"
- BASS = "bass"

##### Pattern
Represents a musical pattern (melody, harmony, rhythm, or bass line).

###### Constructor

```python
def __init__(self, pattern_type: PatternType, notes: List[Note], chords: List[Chord])
```

Parameters:
- pattern_type (PatternType): Type of pattern
- notes (List[Note]): List of individual notes
- chords (List[Chord]): List of chords

###### Methods

####### __repr__()
String representation of the Pattern.

```python
def __repr__(self) -> str
```

Returns:
- str: String representation

####### add_note(note)
Add a note to the pattern.

```python
def add_note(self, note: Note) -> None
```

Parameters:
- note (Note): Note to add

Returns:
- None

####### add_chord(chord)
Add a chord to the pattern.

```python
def add_chord(self, chord: Chord) -> None
```

Parameters:
- chord (Chord): Chord to add

Returns:
- None

####### get_all_notes()
Get all notes from both individual notes and chords.

```python
def get_all_notes(self) -> List[Note]
```

Returns:
- List[Note]: All notes in the pattern

##### SectionType(Enum)
Types of song sections.

Values:
- INTRO = "intro"
- VERSE = "verse"
- CHORUS = "chorus"
- BRIDGE = "bridge"
- OUTRO = "outro"

## Genres Module

### genres/genre_rules.py

Contains genre-specific rules for music generation.

#### Classes

##### GenreRules
Base class for genre-specific rules.

###### Methods

####### get_rules()
Get the rules for this genre.

```python
def get_rules(self) -> Dict[str, Any]
```

Returns:
- Dict[str, Any]: Dictionary of genre rules

####### get_scales()
Get the scales typically used in this genre.

```python
def get_scales(self) -> List[str]
```

Returns:
- List[str]: List of scale strings

####### get_chord_progressions()
Get common chord progressions for this genre.

```python
def get_chord_progressions(self) -> List[List[str]]
```

Returns:
- List[List[str]]: List of chord progressions

####### get_rhythm_patterns()
Get typical rhythm patterns for this genre.

```python
def get_rhythm_patterns(self) -> List[Dict[str, Any]]
```

Returns:
- List[Dict[str, Any]]: List of rhythm patterns

####### get_typical_structure()
Get the typical song structure for this genre.

```python
def get_typical_structure(self) -> List[str]
```

Returns:
- List[str]: List of section types

####### get_instrumentation()
Get typical instruments used in this genre.

```python
def get_instrumentation(self) -> List[str]
```

Returns:
- List[str]: List of instruments

##### PopRules(GenreRules)
Rules for pop music.

##### RockRules(GenreRules)
Rules for rock music.

##### JazzRules(GenreRules)
Rules for jazz music.

##### ElectronicRules(GenreRules)
Rules for electronic music.

##### HipHopRules(GenreRules)
Rules for hip-hop music.

##### ClassicalRules(GenreRules)
Rules for classical music.

### genres/genre_factory.py

Factory for creating genre-specific rules.

#### Classes

##### GenreFactory
Factory for creating genre-specific rules.

###### Static Methods

####### create_genre_rules(genre)
Create genre-specific rules based on the genre.

```python
@staticmethod
def create_genre_rules(genre: str) -> Dict[str, Any]
```

Parameters:
- genre (str): The genre to create rules for

Returns:
- Dict[str, Any]: Dictionary of genre-specific rules

## Generators Module

### generators/pattern_generator.py

Generates musical patterns based on genre rules and mood.

#### Classes

##### PatternGenerator
Generates musical patterns based on genre rules and mood.

###### Constructor

```python
def __init__(self, genre_rules: Dict[str, Any], mood: str)
```

Parameters:
- genre_rules (Dict[str, Any]): Dictionary of genre-specific rules
- mood (str): Mood for the music generation

###### Methods

####### generate_patterns(song_skeleton, num_bars)
Generate musical patterns for a song.

```python
def generate_patterns(self, song_skeleton: SongSkeleton, num_bars: int) -> List[Pattern]
```

Parameters:
- song_skeleton (SongSkeleton): The song structure to generate patterns for
- num_bars (int): Number of bars to generate

Returns:
- List[Pattern]: List of generated patterns

####### _establish_key_and_scale()
Establish the key and scale based on genre rules.

```python
def _establish_key_and_scale(self) -> None
```

Parameters:
- None

Returns:
- None

####### _generate_melody_pattern(num_bars)
Generate a melody pattern using scale-based pitch selection.

```python
def _generate_melody_pattern(self, num_bars: int) -> Pattern
```

Parameters:
- num_bars (int): Number of bars to generate

Returns:
- Pattern: Generated melody pattern

####### _generate_harmony_pattern(num_bars)
Generate a harmony pattern using Roman numeral chord progressions.

```python
def _generate_harmony_pattern(self, num_bars: int) -> Pattern
```

Parameters:
- num_bars (int): Number of bars to generate

Returns:
- Pattern: Generated harmony pattern

####### _generate_bass_pattern(num_bars)
Generate a bass pattern using chord roots from the progression.

```python
def _generate_bass_pattern(self, num_bars: int) -> Pattern
```

Parameters:
- num_bars (int): Number of bars to generate

Returns:
- Pattern: Generated bass pattern

####### _generate_rhythm_pattern(num_bars)
Generate a rhythm pattern.

```python
def _generate_rhythm_pattern(self, num_bars: int) -> Pattern
```

Parameters:
- num_bars (int): Number of bars to generate

Returns:
- Pattern: Generated rhythm pattern

####### _get_velocity_for_mood()
Get appropriate velocity based on mood.

```python
def _get_velocity_for_mood(self) -> int
```

Parameters:
- None

Returns:
- int: Velocity value based on mood

## Output Module

### output/midi_output.py

Handles output of generated music to MIDI files.

#### Classes

##### MidiOutput
Handles output of generated music to MIDI files.

###### Constructor

```python
def __init__(self)
```

Parameters:
- None

###### Methods

####### save_to_midi(song_skeleton, filename, genre_rules, separate_files)
Save a song to a MIDI file.

```python
def save_to_midi(self, song_skeleton: SongSkeleton, filename: str, genre_rules: Optional[GenreRules] = None, separate_files: bool = False) -> None
```

Parameters:
- song_skeleton (SongSkeleton): The song to save
- filename (str): The filename to save to
- genre_rules (Optional[GenreRules]): The genre rules used for generation, for applying swing etc.
- separate_files (bool): If True, save patterns to separate MIDI files per instrument

Returns:
- None

####### save_to_separate_midi_files(song_skeleton, base_filename, genre_rules)
Save patterns to separate MIDI files per instrument type.

```python
def save_to_separate_midi_files(self, song_skeleton: SongSkeleton, base_filename: str, genre_rules: Optional[GenreRules] = None) -> None
```

Parameters:
- song_skeleton (SongSkeleton): The song skeleton containing patterns
- base_filename (str): Base name for output files (without extension)
- genre_rules (Optional[GenreRules]): Optional genre rules for applying swing etc.

Returns:
- None

This method creates separate MIDI files for each instrument type:
- melody.mid (melody line)
- harmony.mid (chord accompaniment)
- bass.mid (bass line)
- rhythm.mid (percussion/rhythm)

####### _get_program_for_instrument(instrument)
Get MIDI program number for different instruments.

```python
def _get_program_for_instrument(self, instrument: str) -> int
```

Parameters:
- instrument (str): Type of instrument ('melody', 'harmony', 'bass', 'rhythm')

Returns:
- int: MIDI program number (0-127)

####### _add_pattern_to_track(track, pattern)
Add a pattern to a MIDI track.

```python
def _add_pattern_to_track(self, track, pattern: Pattern) -> None
```

Parameters:
- track: The MIDI track to add to
- pattern (Pattern): The pattern to add

Returns:
- None

####### _add_pattern_to_track(track, pattern, current_time, section_type, genre_rules)
Add a pattern to a MIDI track.

```python
def _add_pattern_to_track(self, track, pattern: Pattern, current_time: int, section_type: str, genre_rules: Optional[GenreRules] = None) -> int
```

Parameters:
- track: The MIDI track to add to
- pattern (Pattern): The pattern to add (contains notes and chords)
- current_time (int): Current time position in ticks
- section_type (str): Type of section (melody, harmony, bass, rhythm)
- genre_rules (Optional[GenreRules]): The genre rules used for generation, for applying swing etc.

Returns:
- int: Updated current time after processing the pattern

####### _get_channel_for_section(section_type)
Get MIDI channel assignment for different section types.

```python
def _get_channel_for_section(self, section_type: str) -> int
```

Parameters:
- section_type (str): Type of musical section

Returns:
- int: MIDI channel number (0-15)

####### _add_chord_to_track(track, chord, current_time, channel, genre_rules)
Add a chord to a MIDI track as simultaneous note events.

```python
def _add_chord_to_track(self, track, chord: Chord, current_time: int, channel: int, genre_rules: Optional[GenreRules] = None) -> int
```

Parameters:
- track: The MIDI track to add to
- chord (Chord): The chord to add
- current_time (int): Current time position in ticks
- channel (int): MIDI channel to use
- genre_rules (Optional[GenreRules]): The genre rules used for generation, for applying swing etc.

Returns:
- int: Updated current time after the chord

####### _add_note_to_track(track, note, current_time, channel, genre_rules)
Add a note to a MIDI track.

```python
def _add_note_to_track(self, track, note: Note, current_time: int, channel: int, genre_rules: Optional[GenreRules] = None) -> int
```

Parameters:
- track: The MIDI track to add to
- note (Note): The note to add
- current_time (int): Current time position in ticks for relative timing
- channel (int): MIDI channel to use
- genre_rules (Optional[GenreRules]): The genre rules used for generation, for applying swing etc.

Returns:
- int: Updated current time after the note

## Song Skeleton

### structures/song_skeleton.py

Represents the overall structure of a song.

#### Classes

##### SongSkeleton
Represents the overall structure of a song.

###### Constructor

```python
def __init__(self, genre: str, tempo: int, mood: str)
```

Parameters:
- genre (str): Music genre
- tempo (int): Tempo in BPM
- mood (str): Mood of the song

###### Methods

####### __repr__()
String representation of the SongSkeleton.

```python
def __repr__(self) -> str
```

Returns:
- str: String representation

####### add_pattern(pattern)
Add a pattern to the song.

```python
def add_pattern(self, pattern: Pattern) -> None
```

Parameters:
- pattern (Pattern): Pattern to add

Returns:
- None

####### add_section(section_type, patterns)
Add a section with its patterns to the song.

```python
def add_section(self, section_type: SectionType, patterns: List[Pattern]) -> None
```

Parameters:
- section_type (SectionType): Type of section
- patterns (List[Pattern]): List of patterns for the section

Returns:
- None

####### build_arrangement(patterns)
Build a basic song arrangement.

```python
def build_arrangement(self, patterns: List[Pattern]) -> None
```

Parameters:
- patterns (List[Pattern]): List of patterns to arrange

Returns:
- None
## Core Utilities (Templating, Presets, Rhythms DB, Config)

This section documents the new helper APIs introduced for filename templating, settings presets, rhythms DB resolution/validation, and effective settings resolution. See source modules:
- [core/filename_templater.py](core/filename_templater.py)
- [core/settings_preset_manager.py](core/settings_preset_manager.py)
- [core/rhythms_db_resolver.py](core/rhythms_db_resolver.py)
- [core/config_loader.py](core/config_loader.py)

### FilenameTemplater

- [python.FilenameTemplater.format_filename()](core/filename_templater.py:170)

  Signature:
  ```python
  def format_filename(
      template: str,
      settings: Dict[str, Any],
      context: Optional[Dict[str, Any]] = None,
      base_dir: Union[str, Path] = "output",
      timestamp_fmt: str = "%Y%m%d_%H%M%S",
  ) -> Path
  ```
  Description: Render a filename from a template with placeholders and ensure safety/uniqueness. Supports path subdirectories and appends .mid if missing. Creates parent dirs and resolves collisions.

- [python.FilenameTemplater.resolve_placeholders()](core/filename_templater.py:78)

  Signature:
  ```python
  def resolve_placeholders(
      settings: Dict[str, Any],
      context: Optional[Dict[str, Any]] = None,
      timestamp_fmt: str = "%Y%m%d_%H%M%S",
  ) -> Dict[str, str]
  ```
  Description: Build mapping for placeholders {genre,mood,tempo,bars,timestamp,stem,run_index,unique_id} with sanitization.

- [python.FilenameTemplater.validate_template()](core/filename_templater.py:118)

  Signature:
  ```python
  def validate_template(template: str) -> Tuple[bool, str]
  ```
  Description: Validate placeholder names used in a template. Returns (True,"") when valid, else (False,"reason").

Related:
- Sanitization helper [python.sanitize_component()](core/filename_templater.py:24)

### SettingsPresetManager

Class: SettingsPresetManager ([core/settings_preset_manager.py](core/settings_preset_manager.py))

- [python.SettingsPresetManager.save_preset()](core/settings_preset_manager.py:47)

  ```python
  def save_preset(self, name: str, settings_dict: Dict[str, Any]) -> bool
  ```
  Description: Normalize/validate and persist a named preset to configs/presets/{slug}.json, updating index.json.

- [python.SettingsPresetManager.load_preset()](core/settings_preset_manager.py:87)

  ```python
  def load_preset(self, name: str) -> Optional[Dict[str, Any]]
  ```
  Description: Load a preset by name using index.json mapping or slug fallback.

- [python.SettingsPresetManager.list_presets()](core/settings_preset_manager.py:108)

  ```python
  def list_presets(self) -> List[str]
  ```
  Description: List known preset names (index-first, fallback to scanning preset files).

- [python.SettingsPresetManager.delete_preset()](core/settings_preset_manager.py:127)

  ```python
  def delete_preset(self, name: str) -> bool
  ```
  Description: Delete preset file and index entry, returns True if anything removed.

Other helpers:
- Validation [python.SettingsPresetManager.validate_preset()](core/settings_preset_manager.py:174)

### RhythmsDbResolver

Class: RhythmsDbResolver ([core/rhythms_db_resolver.py](core/rhythms_db_resolver.py))

- [python.RhythmsDbResolver.get_rhythms_db_path()](core/rhythms_db_resolver.py:69)

  ```python
  def get_rhythms_db_path(self, override: Optional[Union[str, os.PathLike]] = None) -> Path
  ```
  Description: Resolve rhythms DB directory via priority: override → env MIDIMASTER_RHYTHMS_DB → configs/settings.json (fallback to temp_settings.json) → ./reference_midis. Always returns absolute Path.

- [python.RhythmsDbResolver.validate_path()](core/rhythms_db_resolver.py:98)

  ```python
  def validate_path(self, path: Path) -> Tuple[bool, str]
  ```
  Description: Validate existence, directory, readability, and presence of at least one .mid (skips macOS AppleDouble).

- [python.RhythmsDbResolver.set_config_path()](core/rhythms_db_resolver.py:135)

  ```python
  def set_config_path(self, path: Path) -> None
  ```
  Description: Persist rhythms_db_path to configs/settings.json (merging if present). Caller should validate first.

### Config Loading and Effective Settings

Module: [core/config_loader.py](core/config_loader.py)

- [python.load_settings_json()](core/config_loader.py:24)

  ```python
  def load_settings_json(path: str | Path = "configs/settings.json") -> Dict[str, Any]
  ```
  Description: Read optional user settings JSON: keys include filename_template, rhythms_db_path, default_preset. Returns {} on failure.

- [python.resolve_effective_settings()](core/config_loader.py:83)

  ```python
  def resolve_effective_settings(cli_args) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]
  ```
  Description: Compute effective generation settings with priority:
  1) built-in defaults
  2) configs/settings.json
  3) configs/temp_settings.json (legacy rhythms_db_path)
  4) preset via --load-preset or default_preset; normalized/validated then merged
  5) explicit CLI overrides including --filename-template and --rhythms-db
  Returns (effective_settings, filename_template_or_None, rhythms_db_abs_path_str).

### MIDI Output integration

Templating is applied by MIDI output when a template is provided:

- [python.MidiOutput.save_to_midi()](output/midi_output.py:312)
- [python.MidiOutput.save_to_separate_midi_files()](output/midi_output.py:479)

See also the templating reference:
- [docs/filename_templating.md](filename_templating.md)