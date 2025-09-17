# MidiMaster GUI Enhancement Specification

## Overview

This specification details the design for enhancing the MidiMaster GUI to provide an interactive graphical interface that replicates and extends the command-line parameters. The enhanced GUI will feature a parameter matrix visualization, intuitive controls for all CLI parameters, and integrated generation and playback capabilities.

## Current State Analysis

### CLI Parameters
- **Genre**: `['pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical']`, default: 'pop'
- **Mood**: `['happy', 'sad', 'energetic', 'calm']`, default: 'happy'
- **Tempo**: Integer range 80-160, default: 120
- **Bars**: Integer, default: 16
- **Density**: `['minimal', 'sparse', 'balanced', 'dense', 'complex']`, default: 'balanced'
- **Output**: String filename, default: 'output.mid'

### Current GUI Structure
- Basic PyQt6 window (800x600)
- Placeholder labels for controls, piano roll, and playback
- Single Generate button (currently non-functional)
- No parameter input controls

## UI Layout Design

### Main Window Structure
```
+----------------------------------------------------------+
| MIDI Master v2.0 - Enhanced GUI                          |
+----------------------------------------------------------+
| [Menu Bar: File | Generate | Playback | View | Help]     |
+----------------------------------------------------------+
| Parameter Controls Panel          | Parameter Matrix     |
|                                   | Visualization        |
| +-------------------------------+ +--------------------+ |
| | Genre Selection               | | [Genre] [Mood]     | |
| | [Pop ▼] [Jazz ▼] etc.         | |                    | |
| +-------------------------------+ | [Tempo Range]      | |
| | Mood Selection                | | [80]---[160]       | |
| | [Happy ▼] [Sad ▼] etc.        | |                    | |
| +-------------------------------+ | [Density Levels]   | |
| | Numeric Controls              | | Min Spl Bl Dns Cmp | |
| | Tempo: [120] [Slider]         | +--------------------+ |
| | Bars: [16] [Slider]           |                       |
| +-------------------------------+                       |
| | Density Presets               |                       |
| | [Minimal] [Sparse] [Balanced] |                       |
| | [Dense] [Complex]             |                       |
+----------------------------------------------------------+
| Preview/Output Panel                                     |
| +------------------------------------------------------+ |
| | Piano Roll/Matrix Preview                            | |
| | [Grid showing generated MIDI]                        | |
| | [Timeline with note events]                          | |
| +------------------------------------------------------+ |
| | Playback Controls                                     | |
| | [▶] [⏸] [⏹] [Volume] [Loop] [Save] [Export]          | |
| +------------------------------------------------------+ |
+----------------------------------------------------------+
```

### Layout Components

1. **Parameter Controls Panel** (Left side, ~40% width)
   - Genre selection (dropdown/combobox)
   - Mood selection (dropdown/combobox)
   - Tempo slider (80-160 BPM)
   - Bars slider (4-32 bars)
   - Density presets (radio buttons)

2. **Parameter Matrix Visualization** (Right side, ~60% width)
   - Interactive grid showing parameter combinations
   - Cells represent genre vs tempo, mood vs density combinations
   - Color-coded cells indicating compatibility/recommendations
   - Click cells to apply parameter combinations

3. **Preview/Output Panel** (Bottom, full width)
   - Piano roll visualization of generated MIDI
   - Timeline showing note events, chords, rhythm
   - Zoom controls, scroll capabilities

4. **Playback Controls** (Bottom toolbar)
   - Play/Pause/Stop buttons
   - Volume slider
   - Loop toggle
   - Save/Export buttons

## Component Specifications

### PyQt6 Widgets Mapping

| CLI Parameter | Widget Type | Configuration | Validation |
|---------------|-------------|---------------|-----------|
| genre | QComboBox | Items: ['pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical'] | Always valid (choices enforced) |
| mood | QComboBox | Items: ['happy', 'sad', 'energetic', 'calm'] | Always valid (choices enforced) |
| tempo | QSlider + QSpinBox | Range: 80-160, Step: 1 | Min: 80, Max: 160 |
| bars | QSlider + QSpinBox | Range: 4-32, Step: 1 | Min: 4, Max: 32 |
| density | QButtonGroup (Radio) | Options: ['minimal', 'sparse', 'balanced', 'dense', 'complex'] | Always valid (radio group) |
| output | QLineEdit | Default: 'output.mid' | Check valid filename characters |

### Parameter Matrix Design

#### Matrix Structure
- **Primary Axis**: Genre (6 options)
- **Secondary Axis**: Tempo ranges (grouped: 80-100, 101-120, 121-140, 141-160)
- **Cell Content**: Shows compatibility indicators and mood/density combinations
- **Interaction**: Click cells to auto-populate parameter controls

#### Matrix Features
- Color coding: Green (optimal), Yellow (good), Red (challenging)
- Tooltips showing recommended mood/density for each combination
- Hover effects showing preview combinations
- Selection indicators for current parameter settings

### Piano Roll/Matrix Preview Component

#### Features
- Timeline-based visualization (horizontal scroll)
- Vertical piano keyboard (MIDI notes 21-108)
- Note rectangles showing pitch, start time, duration
- Color coding by instrument type:
  - Melody: Blue
  - Harmony: Green
  - Bass: Orange
  - Rhythm: Red (drums on separate track)
- Zoom controls (horizontal/vertical)
- Grid lines showing beat/bar divisions

#### Interaction
- Click notes to select/edit
- Drag to adjust timing/pitch
- Right-click for context menu (delete, properties)
- Scroll and zoom with mouse wheel

## Integration Plan

### Generation Integration

#### Direct Import Approach
```python
# Import main generation components
from main import main as generate_music
from generators.pattern_orchestrator import PatternOrchestrator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory
from generators.density_manager import create_density_manager_from_preset

# In GUI generate handler
def on_generate_clicked():
    try:
        # Collect GUI parameters
        params = {
            'genre': self.genre_combo.currentText().lower(),
            'mood': self.mood_combo.currentText().lower(),
            'tempo': self.tempo_slider.value(),
            'bars': self.bars_slider.value(),
            'density': self.get_selected_density(),
            'output': self.output_edit.text()
        }

        # Create generation objects
        genre_rules = GenreFactory.create_genre_rules(params['genre'])
        song_skeleton = SongSkeleton(params['genre'], params['tempo'], params['mood'])
        density_manager = create_density_manager_from_preset(params['density'])

        # Generate patterns
        pattern_generator = PatternOrchestrator(
            genre_rules, params['mood'],
            note_density=density_manager.note_density,
            rhythm_density=density_manager.rhythm_density,
            chord_density=density_manager.chord_density,
            bass_density=density_manager.bass_density
        )

        patterns = pattern_generator.generate_patterns(song_skeleton, params['bars'])
        song_skeleton.build_arrangement(patterns)

        # Save and preview
        midi_output = MidiOutput()
        midi_output.save_to_midi(song_skeleton, params['output'], genre_rules)

        # Update preview
        self.update_piano_roll_preview(song_skeleton)

        self.status_label.setText(f"Generated: {params['output']}")

    except Exception as e:
        self.show_error_dialog(f"Generation failed: {str(e)}")
```

#### Subprocess Approach (Alternative)
```python
import subprocess
import json

def generate_via_subprocess(params):
    # Convert GUI params to command line args
    cmd = ['python', 'main.py']
    cmd.extend(['--genre', params['genre']])
    cmd.extend(['--mood', params['mood']])
    cmd.extend(['--tempo', str(params['tempo'])])
    cmd.extend(['--bars', str(params['bars'])])
    cmd.extend(['--density', params['density']])
    cmd.extend(['--output', params['output']])

    # Run generation
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr
```

### Playback Integration

#### Using pygame.midi (Simple)
```python
import pygame.midi
import time

class MidiPlayer:
    def __init__(self):
        pygame.midi.init()
        self.output = pygame.midi.Output(0)
        self.output.set_instrument(0)

    def play_midi_file(self, filename):
        # Parse MIDI file and send events to output
        # Implementation would read mido.MidiFile and send messages
        pass

    def stop(self):
        self.output.note_off(60, 127)  # All notes off
```

#### Using dawdreamer (Advanced)
```python
import dawdreamer as daw

def render_and_play_with_dawdreamer(song_skeleton, plugin_path):
    # Create DAW session
    session = daw.RenderEngine(sample_rate=44100, block_size=512)

    # Load instrument plugin
    instrument = session.add_processor(plugin_path)

    # Load MIDI and render audio
    # Implementation would convert song_skeleton to MIDI events
    # and route through plugin for audio rendering
    pass
```

## Validation and Error Handling

### Input Validation
- **Genre/Mood**: Enforced by combobox choices (no validation needed)
- **Tempo**: Range validation (80-160), visual feedback on invalid values
- **Bars**: Range validation (4-32), prevent values outside bounds
- **Density**: Radio button group ensures one selection
- **Output**: Filename validation, check for valid characters and extension

### Error Handling Categories
1. **Parameter Validation Errors**: Invalid ranges, missing selections
2. **Generation Errors**: Pattern generation failures, memory issues
3. **File I/O Errors**: Permission issues, disk space, invalid paths
4. **Playback Errors**: MIDI device unavailable, audio rendering failures

### Error Display
- Status bar for non-critical errors
- Modal dialogs for critical errors
- Color-coded feedback (red for errors, yellow for warnings)
- Detailed error messages with suggested fixes

## Wireframe Description

### High-Level Layout
```
┌─────────────────────────────────────────────────────────┐
│ MIDI Master Enhanced GUI                                │
├─────────────────┬───────────────────────────────────────┤
│ Parameters      │ Parameter Matrix                      │
│ ├─────────────┤ │ ├───────────────────────────────────┤ │
│ │ Genre: [Pop▼] │ │ │ Genre │ Tempo  │ Mood │ Density │ │
│ │ Mood:  [Happy]│ │ ├───────────────────────────────────┤ │
│ │ Tempo: [120]──┼─┼ │ Pop   │ 80-100 │ ...   │ ...     │ │
│ │ Bars:  [16]───┼─┼ │ Rock  │101-120 │ ...   │ ...     │ │
│ │ Density: ●Bal │ │ │ Jazz  │121-140 │ ...   │ ...     │ │
│ └─────────────┴─┼─┼ │ ...   │141-160 │ ...   │ ...     │ │
│                 │ └───────────────────────────────────┘ │
└─────────────────┴───────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Piano Roll Preview                                       │
│ ├─────────────────────────────────────────────────────┤ │
│ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ │
│ │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ │
│ │ ▓▓████████████  ████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ │
│ │ ████████▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────┬──────────────────┬────────────────────┤
│ [▶] [⏸] [⏹]     │ Volume: [───●──]  │ [Save] [Export]     │
└─────────────────┴──────────────────┴────────────────────┘
```

### Component Details

1. **Parameter Panel** (Left, 300px width)
   - Vertical layout with grouped controls
   - Consistent spacing and alignment
   - Labels above controls for clarity

2. **Matrix Panel** (Right, 500px width)
   - Grid layout with headers
   - Scrollable if needed for more parameters
   - Interactive cells with hover effects

3. **Preview Panel** (Bottom, full width, 300px height)
   - Large piano roll area with scroll/zoom
   - Timeline at top, keyboard at left
   - Note rectangles with color coding

4. **Control Bar** (Bottom, full width, 50px height)
   - Playback buttons aligned left
   - Volume slider center
   - Action buttons right

## Pseudocode for Core Integration

### Main Window Initialization
```python
class EnhancedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDI Master v2.0")
        self.setGeometry(100, 100, 1000, 800)

        # Initialize parameter defaults
        self.current_params = {
            'genre': 'pop',
            'mood': 'happy',
            'tempo': 120,
            'bars': 16,
            'density': 'balanced',
            'output': 'output.mid'
        }

        # Create UI components
        self.setup_parameter_controls()
        self.setup_matrix_visualization()
        self.setup_preview_panel()
        self.setup_playback_controls()

        # Connect signals
        self.connect_signals()

        # Load default preview
        self.generate_preview()
```

### Parameter Change Handler
```python
def on_parameter_changed(self, param_name, value):
    # Update current parameters
    self.current_params[param_name] = value

    # Update matrix highlighting
    self.matrix_view.update_highlight(self.current_params)

    # Auto-generate preview if auto-preview enabled
    if self.auto_preview_enabled:
        self.generate_preview()
```

### Generation Handler
```python
def generate_music(self):
    try:
        # Show progress dialog
        progress = QProgressDialog("Generating music...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        # Collect parameters
        params = self.collect_parameters()

        # Generate in background thread
        self.generation_thread = GenerationWorker(params)
        self.generation_thread.progress.connect(progress.setValue)
        self.generation_thread.finished.connect(self.on_generation_complete)
        self.generation_thread.error.connect(self.on_generation_error)

        self.generation_thread.start()

    except Exception as e:
        self.show_error_dialog(f"Generation setup failed: {str(e)}")
```

### Preview Update
```python
def update_preview(self, song_skeleton):
    # Clear current preview
    self.piano_roll_scene.clear()

    # Add notes from song skeleton
    for section in song_skeleton.sections.values():
        for pattern in section:
            for note in pattern.notes:
                self.add_note_to_preview(note, pattern.pattern_type)

    # Fit view and update
    self.piano_roll_view.fitInView(self.piano_roll_scene.sceneRect(), Qt.KeepAspectRatio)
```

## Implementation Notes

### Dependencies
- PyQt6 (for GUI framework)
- mido (for MIDI file handling)
- pygame (for basic MIDI playback)
- dawdreamer (optional, for advanced audio rendering)

### Architecture Considerations
- Separate UI thread from generation thread
- Use signals/slots for thread communication
- Implement undo/redo for parameter changes
- Support for saving/loading parameter presets
- Plugin architecture for additional parameter types

### Performance Considerations
- Generate previews with reduced complexity
- Cache generated patterns for quick preview updates
- Implement progressive loading for large MIDI files
- Background processing for generation tasks

### File Structure Changes
- `gui/main_window.py` - Enhanced main window class
- `gui/parameter_matrix.py` - Matrix visualization component
- `gui/piano_roll_view.py` - Piano roll preview component
- `gui/playback_controller.py` - Playback control logic
- `gui/generation_worker.py` - Background generation worker

This specification provides a comprehensive design for transforming the basic MidiMaster GUI into a feature-rich, interactive application that maintains full compatibility with the existing CLI while adding powerful visual controls and real-time preview capabilities.