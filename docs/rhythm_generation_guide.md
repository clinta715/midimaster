# Rhythm Generation Guide

## Overview

The MIDI Master now features a sophisticated rhythm generation system with per-voice drum templates, dynamic variations, and genre-authentic patterns. This guide covers the new drum template API and rhythm variation controls.

## Drum Template System

### Template Structure

Each drum pattern template is a dictionary with the following structure:

```python
{
    'name': 'pattern_name',           # Unique identifier
    'steps_per_bar': 16,              # Steps per bar (16 = 16th notes)
    'voices': {                       # Per-voice step indices
        'kick': [0, 4, 8, 12],        # Kick drum steps
        'snare': [4, 12],             # Snare steps
        'ghost_snare': [3, 5, 11, 13], # Ghost snare steps
        'ch': [0, 2, 4, 6, 8, 10, 12, 14],  # Closed hats
        'oh': [2, 6, 10, 14],         # Open hats
        'clap': [4, 12],              # Claps (layer with snare)
        'rim': [5, 13],               # Rim shots
        'shaker': [0, 2, 4, 6, 8, 10, 12, 14]  # Shakers
    },
    'weight': 1.0                     # Selection probability weight
}
```

### Supported Voices

| Voice | GM MIDI Note | Description |
|-------|--------------|-------------|
| kick | 36 | Bass drum |
| snare | 38 | Main snare |
| ghost_snare | 38 | Quiet snare for groove |
| rim | 37 | Rim click accent |
| clap | 39 | Hand clap |
| ch | 42 | Closed hi-hat |
| oh | 46 | Open hi-hat |
| ride | 51 | Ride cymbal |
| crash | 49 | Crash cymbal |
| shaker | Custom | Shaker/percussion |

## Genre-Specific Templates

### Hip-Hop

#### Trap
- Fast 16th note hi-hats throughout
- Heavy kick on 1, 2.5, 4
- Layered snares and claps
- Ghost snares for groove

```python
# Example trap pattern
{
    'name': 'trap_basic_A',
    'steps_per_bar': 16,
    'voices': {
        'kick': [0, 4, 8, 12, 14],      # Heavy kick pattern
        'snare': [4, 12],               # Backbeat
        'ch': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],  # Fast hats
        'oh': [2, 6, 10, 14],           # Open hats on offbeats
        'ghost_snare': [3, 5, 11, 13],   # Ghost notes
        'clap': [4, 12]                 # Layered with snare
    }
}
```

#### Reggaeton/Latin Trap
- Dem bow kick pattern (1, 2.5, 4)
- Offbeat hi-hats
- Shakers for Latin groove
- Ghost snares on 16ths

#### Boom Bap
- Classic boom-bap kick/snare
- Even 16th note hats
- Ghost snares for vintage feel
- Rim shots for accent

### Electronic/DnB

#### DnB/Jungle
- 16-step grid with complex breakbeats
- Heavy sub-bass kicks
- Ghost snares for rolling fills
- Fast hi-hat patterns

#### Techno
- Four-on-the-floor kick
- Straight 16th note hats
- Minimal ghost elements
- Industrial feel

## Rhythm Variation Engine

### Variation Controls

| Parameter | Range | Description |
|-----------|-------|-------------|
| pattern_strength | 0.0-1.0 | How strictly to follow original pattern |
| swing_percent | 0.0-1.0 | Amount of swing feel |
| fill_frequency | 0.0-0.5 | How often to add rhythmic fills |
| ghost_note_level | 0.0-2.0 | Ghost note intensity multiplier |

### Variation Features

#### Bar-Level Pattern Switching
- Switches between A/B patterns every 4 bars
- Controlled by `pattern_strength` parameter
- Allows dynamic rhythm evolution

#### Fill Generation
- Adds fills every N bars based on `fill_frequency`
- Increases ghost snares during fills
- Adds snare rolls and extra accents

#### Ghost Note Variation
- Modifies ghost note probability based on `ghost_note_level`
- Higher levels = more prominent ghost notes
- Fill bars get extra ghost notes

#### Hat Thinning
- Reduces hi-hat density during micro-drops
- Micro-drops occur every 8 bars
- Creates tension and release

#### Swing Timing
- Applies swing feel to even-numbered steps
- Controlled by `swing_percent` parameter
- Affects snare and hat timing

## Command-Line Usage

### Basic Rhythm Control

```bash
# Generate trap with high pattern strength, moderate swing
python main.py --genre hip-hop --subgenre trap --pattern-strength 0.9 --swing-percent 0.6

# Create DnB with frequent fills and prominent ghost notes
python main.py --genre electronic --subgenre jungle --fill-frequency 0.33 --ghost-note-level 1.5

# Generate techno with straight timing, no fills
python main.py --genre electronic --subgenre techno --swing-percent 0.5 --fill-frequency 0.0
```

### Advanced Examples

```bash
# Latin trap with authentic reggaeton patterns
python main.py --genre hip-hop --subgenre reggaeton --tempo 92 --pattern-strength 0.8 --fill-frequency 0.25

# Boom bap with vintage feel
python main.py --genre hip-hop --subgenre boom_bap --tempo 95 --ghost-note-level 1.2 --swing-percent 0.55

# DnB with rolling fills
python main.py --genre electronic --subgenre liquid --fill-frequency 0.4 --ghost-note-level 1.8
```

## API Integration

### Adding New Drum Templates

1. Extend the `get_drum_patterns()` method in your genre rules class
2. Add subgenre-specific templates
3. Include weight values for pattern selection
4. Test with different variation settings

```python
def get_drum_patterns(self, subgenre: Optional[str] = None) -> List[Dict[str, Any]]:
    if subgenre == 'your_subgenre':
        return [
            {
                'name': 'your_pattern',
                'steps_per_bar': 16,
                'voices': {
                    'kick': [0, 4, 8, 12],
                    'snare': [4, 12],
                    'ch': [0, 2, 4, 6, 8, 10, 12, 14],
                    'ghost_snare': [3, 11]
                },
                'weight': 1.0
            }
        ]
    return []  # Fallback to base implementation
```

### Extending Variation Engine

The `RhythmVariationEngine` class can be extended with new variation types:

```python
class CustomVariationEngine(RhythmVariationEngine):
    def apply_custom_variation(self, voices: Dict[str, List[int]]) -> Dict[str, List[int]]:
        # Add your custom variation logic here
        return voices
```

## Performance Optimization

### Template Selection
- Templates are selected once per generation
- Weighted random selection for variety
- Pattern switching occurs at bar boundaries

### Memory Usage
- Templates stored as lightweight dictionaries
- No audio processing during pattern generation
- Efficient step-based note placement

### Real-time Generation
- Supports real-time pattern generation
- Low latency for interactive applications
- Configurable complexity levels

## Troubleshooting

### Common Issues

**No drum patterns found**
- Check that `get_drum_patterns()` returns a non-empty list
- Verify subgenre parameter is correct
- Ensure base class fallback is implemented

**Poor grid alignment**
- Check `beat_characteristics` tempo settings
- Verify step timing calculations
- Use beat analyzer to diagnose timing issues

**Variation not working**
- Confirm variation parameters are within valid ranges
- Check that rhythm generator accepts variation parameters
- Verify bar-level logic is correctly implemented

### Debug Output

Enable debug output to see pattern selection and timing:

```python
# In rhythm_generator.py, debug output is enabled by default
print(f"Using drum pattern: {tmpl.get('name', 'unnamed')}")
print(f"Generated {len(notes)} notes")
```

## Future Enhancements

### Planned Features
- MIDI file import for custom template creation
- Real-time pattern morphing
- AI-assisted pattern generation
- Multi-layered rhythmic complexity

### Community Contributions
- Submit new genre templates via pull requests
- Share custom variation engines
- Report rhythm generation issues

## References

- [GM MIDI Drum Map](https://www.midi.org/specifications-old/item/gm-level-1-sound-set)
- [Drum Pattern Theory](https://en.wikipedia.org/wiki/Drum_beat)
- [Rhythm and Meter](https://en.wikipedia.org/wiki/Meter_(music))

---

*This guide covers the rhythm generation system as of version 2.1. For the latest updates, check the project repository.*
## Configuring the Rhythms Database

The Rhythms Database is the corpus of reference MIDI files used for rhythm extraction and pattern guidance. You can point the app to any directory that contains at least one valid .mid file (recursively). Configuration works the same across CLI and GUI.

Resolution order (highest priority last applied):
1) CLI override: `--rhythms-db`
2) Environment variable: `MIDIMASTER_RHYTHMS_DB`
3) User config: configs/settings.json (fallback to configs/temp_settings.json)
4) Project default: ./reference_midis

Implemented by:
- [python.RhythmsDbResolver.get_rhythms_db_path()](core/rhythms_db_resolver.py:69)
- Integration in effective settings: [python.resolve_effective_settings()](core/config_loader.py:83) (resolver used at [core/config_loader.py](core/config_loader.py:165))
- CLI validation: [main.py](main.py:311)

Validation rules
- Directory must exist and be readable
- Must contain at least one .mid file (search is recursive; hidden files like "._*" are skipped)
- Validated by [python.RhythmsDbResolver.validate_path()](core/rhythms_db_resolver.py:98)

Persisting a default
Use the GUI checkbox “Set as default” or call the API to write configs/settings.json. This is additive and non-breaking.
- GUI persist logic calls [python.RhythmsDbResolver.set_config_path()](core/rhythms_db_resolver.py:135)
- GUI helpers:
  - [python.validate_rhythms_path()](gui/settings_helpers.py:93)
  - [python.persist_rhythms_default()](gui/settings_helpers.py:105)
- GUI controls and handlers:
  - Session Settings → “Rhythms Database” group
  - Path field and Browse: [gui/parameter_controls.py](gui/parameter_controls.py:231)
  - Apply and “Set as default” checkbox: [gui/parameter_controls.py](gui/parameter_controls.py:247)

CLI usage examples
- Override the rhythms DB for a single run:
  ```
  python main.py --rhythms-db ./reference_midis/midi8
  ```
  If invalid, the CLI prints a reason and exits. See validation hook at [main.py](main.py:311).

- Combine with presets:
  ```
  python main.py --load-preset mellow_jazz --rhythms-db ./reference_midis/midi99
  ```

Environment variable
- Set once per shell (Windows CMD):
  ```
  set MIDIMASTER_RHYTHMS_DB=.\reference_midis\midi8
  python main.py --genre jazz
  ```
- PowerShell:
  ```
  $env:MIDIMASTER_RHYTHMS_DB = ".\reference_midis\midi8"
  python main.py --genre jazz
  ```
- Unix-like shells:
  ```
  export MIDIMASTER_RHYTHMS_DB=./reference_midis/midi8
  python main.py --genre jazz
  ```

GUI usage
- Open the “Session Settings” group → “Rhythms Database”
- Paste or Browse to select a folder
- Click “Apply”
- Optionally enable “Set as default” to persist to configs/settings.json via [python.RhythmsDbResolver.set_config_path()](core/rhythms_db_resolver.py:135)
- Invalid selections will show a reason (uses [python.validate_rhythms_path()](gui/settings_helpers.py:93))

Folder structure expectations
- Minimum requirement: The directory (or its subdirectories) contains at least one .mid file
- There is no specific hierarchy required; common structures:
  ```
  reference_midis/
    midi8/
      Artist - Track A.mid
      Artist - Track B.mid
    midi99/
      ._Faded Memories_Em_74BPM.mid   (hidden AppleDouble file, ignored)
      First Love_Gm_120BPM.mid
  ```
- Hidden/system artifacts like “._*” are ignored automatically in scanning

Programmatic references
- Resolver API:
  - [python.RhythmsDbResolver.get_rhythms_db_path()](core/rhythms_db_resolver.py:69)
  - [python.RhythmsDbResolver.validate_path()](core/rhythms_db_resolver.py:98)
  - [python.RhythmsDbResolver.set_config_path()](core/rhythms_db_resolver.py:135)
- Configuration loader:
  - [python.load_settings_json()](core/config_loader.py:24)
  - [python.resolve_effective_settings()](core/config_loader.py:83)