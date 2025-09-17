# MIDI Master - Song Sketch Generator

MIDI Master is a generalized music generator designed to create MIDI-based sketches and starting ideas for songs. Whether you're a songwriter, producer, or musician looking for inspiration, MIDI Master quickly generates original musical compositions across multiple genres to kickstart your creative process.

## What Makes MIDI Master Unique

MIDI Master focuses on **creative ideation** - generating musical foundations that spark your imagination rather than polished finished tracks. Each generation provides:

- **Musical sketches** in pop, rock, jazz, electronic, hip-hop, and classical styles
- **Song starting points** with chord progressions, melody lines, bass patterns, and rhythms
- **Creative inspiration** through algorithmic composition based on music theory
- **Flexible MIDI output** ready for import into any Digital Audio Workstation (DAW)

## Features
## Features

- Generates music in multiple genres: pop, rock, jazz, electronic, hip-hop, and classical
- Creates melody, harmony, rhythm, and bass patterns
- Builds song arrangements with proper section structures
- Outputs playable MIDI files that can be imported into any DAW
- Command-line interface for easy customization
- Genre-specific rules for authentic musical styles
- Mood-based variations in velocity and note selection
- **Per-track time signatures** for polymetric and complex rhythmic compositions
- Extensible architecture for adding new genres and features

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Steps

1. Clone or download this repository:
   ```
   git clone https://github.com/your-username/midi-master.git
   cd midi-master
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the program with the following command:
```
python main.py [options]
```

### Options

- `--genre`: Music genre to generate (pop, rock, jazz, electronic, hip-hop, classical) - default: pop
- `--tempo`: Tempo in BPM - default: 120
- `--mood`: Mood of the music (happy, sad, energetic, calm) - default: happy
- `--output`: Output MIDI file name - default: output.mid
- `--bars`: Number of bars to generate - default: 16
- `--separate-files`: Save patterns to separate MIDI files per instrument (melody, harmony, bass, rhythm) instead of one combined file
- `--melody-time-signature`: Time signature for melody track (e.g., 4/4, 3/4, 6/8) - default: 4/4
- `--harmony-time-signature`: Time signature for harmony track (e.g., 4/4, 3/4, 6/8) - default: 4/4
- `--bass-time-signature`: Time signature for bass track (e.g., 4/4, 3/4, 6/8) - default: 4/4
- `--rhythm-time-signature`: Time signature for rhythm track (e.g., 4/4, 3/4, 6/8) - default: 4/4

### Examples

Generate a pop song with default settings:
```
python main.py
```

Generate a jazz song with a slow tempo:
```
python main.py --genre jazz --tempo 80 --output jazz_song.mid
```

Generate an energetic rock song:
```
python main.py --genre rock --mood energetic --bars 32 --output rock_song.mid
```

Generate a calm classical piece:
```
python main.py --genre classical --mood calm --tempo 60 --output classical_calm.mid
```

Generate an electronic track with specific parameters:
```
python main.py --genre electronic --mood energetic --tempo 128 --bars 64 --output electronic_beat.mid
```

Generate separate MIDI files per instrument:
```
python main.py --genre jazz --separate-files --output mysong.mid
```
This will produce mysong_melody.mid, mysong_harmony.mid, mysong_bass.mid, and mysong_rhythm.mid files.

Generate a polymetric composition with different time signatures per track:
```
python main.py --genre jazz --separate-files --output polymetric_jazz.mid \
               --melody-time-signature 4/4 \
               --harmony-time-signature 3/4 \
               --bass-time-signature 6/8 \
               --rhythm-time-signature 4/4
```
This creates a complex rhythmic texture where each instrument plays in a different meter but at the same tempo.

## Project Structure

```
midi-master/
├── main.py                 # Entry point of the application
├── music_theory.py         # Music theory utilities and calculations
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── structures/             # Data structures (Note, Chord, Pattern, SongSkeleton)
│   ├── __init__.py
│   ├── data_structures.py
│   └── song_skeleton.py
├── genres/                 # Genre-specific rules and factory
│   ├── __init__.py
│   ├── genre_rules.py
│   └── genre_factory.py
├── generators/             # Pattern generation algorithms
│   ├── __init__.py
│   └── pattern_generator.py
├── output/                 # MIDI output handling
│   ├── __init__.py
│   └── midi_output.py
└── examples/               # Example scripts (to be created)
    └── genre_demos/
```

## How It Works

### 1. Data Structures

The program uses several classes to represent musical elements:

- **Note**: Represents a musical note with pitch, duration, and velocity
- **Chord**: Represents a chord as a collection of Notes played simultaneously
- **Pattern**: Represents a musical pattern (melody, harmony, rhythm, or bass line)
- **SongSkeleton**: Represents the overall structure of a song with sections

### 2. Genre Rules

Each genre has specific rules for:
- Scales (keys and modes)
- Chord progressions (Roman numeral patterns)
- Rhythm patterns (note duration sequences)
- Typical song structures (section arrangements)
- Instrumentation (typical instruments used)

### 3. Pattern Generation

Based on genre rules and mood, the program generates musical patterns:
- **Melody**: Single-note lines using scale-based pitch selection
- **Harmony**: Chord progressions using Roman numeral system
- **Bass**: Low-register patterns typically following chord roots
- **Rhythm**: Percussion-like patterns with genre-specific timing

### 4. Song Arrangement

Patterns are arranged into sections (verse, chorus, bridge, etc.) to create a complete song structure based on genre-specific conventions.

### 5. MIDI Output

The final song is exported as a MIDI file using the mido library, which can be imported into any Digital Audio Workstation (DAW) for further editing or playback.

## Per-Track Time Signatures

MIDI Master supports **per-track time signatures**, allowing you to create polymetric and complex rhythmic compositions where different instruments play in different meters simultaneously.

### How It Works

When using the `--separate-files` option, each generated MIDI file (melody.mid, harmony.mid, bass.mid, rhythm.mid) can have its own time signature. This enables:

- **Polymetric compositions**: Instruments playing in different meters at the same tempo
- **Complex rhythmic textures**: Layered rhythms with different underlying pulse structures
- **Experimental music**: Break conventional time signature constraints

### Usage

Specify different time signatures for each track using the command-line options:

```bash
python main.py --genre jazz --separate-files \
               --melody-time-signature 4/4 \
               --harmony-time-signature 3/4 \
               --bass-time-signature 6/8 \
               --rhythm-time-signature 4/4 \
               --output polymetric.mid
```

### DAW Integration Tips

When working with polymetric MIDI files in your DAW:

1. **Import all tracks**: Load all four MIDI files into your DAW
2. **Set tempo matching**: Ensure your DAW's tempo matches the generated tempo (usually 120 BPM)
3. **Time alignment**: Align the tracks by their start times, not bar lines
4. **Creative routing**: Send each track to different effects or instruments for layered sound design

### Supported Time Signatures

Common time signatures supported:
- 4/4 (common time)
- 3/4 (waltz time)
- 6/8 (compound time)
- 2/4, 5/4, 7/4, 9/8, etc.

## Music Theory Implementation

### Scales and Keys

The program uses a comprehensive system of scales including:
- Major and minor scales
- Modal scales (Dorian, Phrygian, Lydian, Mixolydian, Locrian)
- Specialized scales (Blues, Pentatonic)

### Chord Progressions

Chord progressions are defined using Roman numeral notation:
- Major keys: I, ii, iii, IV, V, vi, vii°
- Minor keys: i, ii°, III, iv, v, VI, VII
- Extended chords: 7th chords, diminished chords

### Rhythm Patterns

Rhythm patterns are defined as sequences of note durations that create characteristic feels for each genre:
- Straight eighths vs. swing eighths
- Syncopated patterns
- Genre-specific rhythmic feels

## Extending the Program

You can extend the program by:

### Adding New Genres

1. Create a new class in `genres/genre_rules.py` that inherits from `GenreRules`
2. Implement the required methods:
   - `get_scales()`: Return a list of scale strings
   - `get_chord_progressions()`: Return a list of Roman numeral progressions
   - `get_rhythm_patterns()`: Return a list of rhythm pattern dictionaries
   - `get_typical_structure()`: Return a list of section types
   - `get_instrumentation()`: Return a list of typical instruments
3. Add the new genre to the genre map in `genres/genre_factory.py`

### Enhancing Pattern Generation

Modify `generators/pattern_generator.py` to:
- Add new pattern generation algorithms
- Implement more sophisticated musical rules
- Add new pattern types beyond melody, harmony, rhythm, and bass

### Adding New Data Structures

Create new classes in `structures/data_structures.py` for:
- Additional musical elements (arpeggios, scales, etc.)
- Enhanced pattern types
- More complex song structures

### Modifying Song Arrangement

Enhance `structures/song_skeleton.py` to:
- Create more sophisticated arrangement algorithms
- Add support for more section types
- Implement genre-specific arrangement rules

## Using MIDI Output in DAWs

The generated MIDI files can be imported into any Digital Audio Workstation:

### Popular DAWs

- **Ableton Live**: Drag and drop the MIDI file into a MIDI track
- **Logic Pro**: Import > MIDI File
- **Pro Tools**: Import > MIDI
- **FL Studio**: Import > MIDI file
- **Cubase**: File > Import > MIDI File

### Tips for DAW Integration

1. Assign appropriate virtual instruments to different MIDI tracks
2. Adjust the tempo to match your DAW project
3. Use the MIDI data as a starting point for further composition
4. Layer additional tracks on top of the generated MIDI
5. Edit note velocities and timing for more human feel

## Example Outputs

The repository includes several example MIDI files:
- `output.mid`: Default pop generation
- `jazz_calm.mid`: Jazz piece with calm mood
- `electronic_energetic.mid`: Electronic track with energetic mood

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'mido'**
   - Solution: Install the required dependency with `pip install mido`

2. **Generated MIDI file sounds strange**
   - Check that your DAW's virtual instruments are properly assigned
   - Try different genres and moods for variety
   - Adjust tempo to better match the genre

3. **Program crashes with error messages**
   - Ensure you're using Python 3.6 or higher
   - Check that all dependencies are installed
   - Verify command-line arguments are valid

4. **No output file is created**
   - Check that you have write permissions in the directory
   - Verify the output file path is valid
   - Ensure the program completes without errors

### Getting Help

If you encounter issues not covered here:
1. Check the console output for error messages
2. Verify all installation steps were completed
3. Ensure command-line arguments are correctly formatted
4. Check that your Python environment is properly configured

## Contributing

Contributions to MIDI Master are welcome! You can contribute by:
- Adding new genres
- Improving existing algorithms
- Fixing bugs
- Enhancing documentation
- Creating example files

Please fork the repository and submit a pull request with your changes.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Uses the [mido](https://mido.readthedocs.io/) library for MIDI file handling
- Inspired by algorithmic composition techniques
- Built with music theory principles from classical and contemporary sources
## What’s New

The latest release adds three capabilities across CLI and GUI:
- Filename templating for output paths with placeholders and subdirectories
- Settings presets (save/load/list/delete) with on-disk storage
- Rhythms DB path resolution and validation via CLI, env, and config

See the detailed reference:
- [docs/filename_templating.md](docs/filename_templating.md)
- [docs/api.md](docs/api.md)

## Quick Start: New CLI Workflows

- Save a preset
  ```
  python main.py --genre jazz --tempo 90 --mood calm --bars 4 --save-preset mellow_jazz
  ```
  Uses preset manager [python.SettingsPresetManager.save_preset()](core/settings_preset_manager.py:47) behind the scenes.

- List presets
  ```
  python main.py --list-presets
  ```
  Handled in CLI at [main.py](main.py:291) and powered by [python.SettingsPresetManager.list_presets()](core/settings_preset_manager.py:108).

- Load a preset and override with CLI flags
  ```
  python main.py --load-preset mellow_jazz --tempo 120
  ```
  Resolution/merge performed by [python.resolve_effective_settings()](core/config_loader.py:83).

- Override rhythms database (CLI)
  ```
  python main.py --rhythms-db ./reference_midis/midi8
  ```
  Resolved/validated via:
  - [python.RhythmsDbResolver.get_rhythms_db_path()](core/rhythms_db_resolver.py:69)
  - [python.RhythmsDbResolver.validate_path()](core/rhythms_db_resolver.py:98)
  CLI validation occurs at [main.py](main.py:311).

- Use filename template
  ```
  python main.py --filename-template "runs/{genre}_{mood}_{tempo}_{bars}_{stem}"
  ```
  Template is validated by [python.validate_template()](core/filename_templater.py:118) and rendered by [python.format_filename()](core/filename_templater.py:170) when saving via [python.MidiOutput.save_to_midi()](output/midi_output.py:312).

Placeholders supported include {genre}, {mood}, {tempo}, {bars}, {timestamp}, {stem}, {run_index}, {unique_id}. See [docs/filename_templating.md](docs/filename_templating.md).

## Settings Resolution and Merge Order

Effective settings are computed by [python.resolve_effective_settings()](core/config_loader.py:83) using this priority (low → high):
1) Built-in defaults (genre, mood, tempo, bars)
2) User config [python.load_settings_json()](core/config_loader.py:24) (filename_template, rhythms_db_path, default_preset)
3) Legacy temp settings (configs/temp_settings.json) for rhythms_db_path
4) Preset load via `--load-preset` or settings.json `default_preset` (validated with [python.SettingsPresetManager.validate_preset()](core/settings_preset_manager.py:174))
5) Explicit CLI overrides, including:
   - Core fields (genre, mood, tempo, bars)
   - `--filename-template` (validated in CLI at [main.py](main.py:305))
   - `--rhythms-db` (validated in CLI at [main.py](main.py:311))

Preset operations:
- Save: [python.SettingsPresetManager.save_preset()](core/settings_preset_manager.py:47)
- Load: [python.SettingsPresetManager.load_preset()](core/settings_preset_manager.py:87)
- List: [python.SettingsPresetManager.list_presets()](core/settings_preset_manager.py:108)
- Delete: [python.SettingsPresetManager.delete_preset()](core/settings_preset_manager.py:127)

Rhythms DB resolution order: override → env → config → default
- Override: `--rhythms-db`
- Env: `MIDIMASTER_RHYTHMS_DB`
- Config: configs/settings.json (fallback to configs/temp_settings.json)
- Default: ./reference_midis
Implemented by:
- [python.RhythmsDbResolver.get_rhythms_db_path()](core/rhythms_db_resolver.py:69)
- [python.RhythmsDbResolver.validate_path()](core/rhythms_db_resolver.py:98)
- [python.RhythmsDbResolver.set_config_path()](core/rhythms_db_resolver.py:135)

## GUI Usage Highlights

Session Settings group (see [gui/parameter_controls.py](gui/parameter_controls.py)):
- Presets (dropdown + Save/Load/Delete)
  - Uses [python.SettingsPresetManager.save_preset()](core/settings_preset_manager.py:47), [python.SettingsPresetManager.load_preset()](core/settings_preset_manager.py:87), [python.SettingsPresetManager.delete_preset()](core/settings_preset_manager.py:127)
  - Files stored under configs/presets (index at configs/presets/index.json)
- Rhythms DB path
  - Browse and Apply with optional “Set as default” persistence to configs/settings.json via [python.RhythmsDbResolver.set_config_path()](core/rhythms_db_resolver.py:135)
- Filename template
  - Template field with live validation and preview using GUI helpers:
    - [python.validate_template()](core/filename_templater.py:118)
    - [python.resolve_placeholders()](core/filename_templater.py:78)
    - Preview builder [python.build_preview_filename()](gui/settings_helpers.py:39) (pure string; no I/O)
  - On generation, actual saving applies full sanitization/uniqueness via [python.format_filename()](core/filename_templater.py:170) in [python.MidiOutput.save_to_midi()](output/midi_output.py:312).

For an in-depth templating guide, see [docs/filename_templating.md](docs/filename_templating.md).