# GUI-Backend Integration Audit

## Current GUI Structure

### Main Window Architecture
- **Framework**: PyQt6-based QMainWindow (1200x800)
- **Layout**: Tabbed interface with 5 main tabs
- **Menu System**: QMenuBar with File, Edit, View, Generate, Help menus
- **Toolbar**: QToolBar with common actions
- **Status Bar**: QStatusBar for user feedback

### GUI Components by Tab

#### Parameters Tab
- **Genre Selection**: QComboBox populated from GenreFactory.get_available_genres()
- **Tempo Control**: QSpinBox (60-200 BPM)
- **Mood Selection**: QComboBox (calm, happy, sad, energetic)
- **Density Selection**: QComboBox (sparse, balanced, dense)
- **Separate Files**: QCheckBox for multi-stem output
- **Parameter Updates**: Emits signals for real-time config updates

#### Matrix Tab
- **Interactive Grid**: QGridLayout with MatrixCell (QPushButton) for each genre-tempo combination
- **Color Coding**: Green (optimal), Yellow (good), Red (challenging) compatibility indicators
- **Key Selection**: QComboBox from MusicTheory.get_all_roots()
- **Mode Selection**: QComboBox from MusicTheory.get_all_modes()
- **Tooltips**: Show recommended mood/density combinations
- **Click Handling**: Auto-populates parameter controls

#### Preview Tab
- **Piano Roll**: QGraphicsView with QGraphicsScene for MIDI visualization
- **Timeline**: Horizontal scroll with beat/bar divisions
- **Piano Keyboard**: Vertical display (MIDI notes 21-108)
- **Note Rendering**: Color-coded rectangles (Melody: Blue, Harmony: Green, Bass: Orange, Rhythm: Red)
- **Playhead**: Animated position indicator during playback
- **Zoom Controls**: Horizontal/vertical scaling

#### Playback Tab
- **Transport Controls**: QPushButton for Play/Pause/Stop
- **Volume Control**: QSlider (0-100)
- **Time Display**: QLabel showing MM:SS format
- **Loop Toggle**: QPushButton (checkable)
- **MIDI Playback**: pygame.midi integration with threading

#### Plugins Tab
- **Plugin Selection**: QComboBox for available plugins
- **Plugin Management**: QPushButton for Scan and Load operations
- **Parameter Controls**: Dynamic QSlider/QComboBox widgets for plugin parameters
- **Audio Rendering**: QPushButton to render with selected plugin
- **Progress Tracking**: QProgressBar for rendering progress
- **Output Selection**: QFileDialog for audio file output

### Additional Components
- **ConfigManager**: JSON-based configuration persistence
- **GenerationWorker**: QThread for background MIDI generation
- **File Dialogs**: QFileDialog for save/load operations
- **Message Boxes**: QMessageBox for user notifications

## Backend Capabilities

### Generation Engine
- **Genre Support**: pop, rock, jazz, electronic, hip-hop, classical, dnb/drum-and-bass
- **Pattern Orchestration**: Multi-pattern generation with PatternOrchestrator
- **Density Management**: Configurable note/rhythm density via DensityManager
- **Mood-Based Generation**: Emotional parameter influence on patterns
- **Key/Mode Support**: Full music theory integration with scales and modes
- **Tempo Range**: 60-200 BPM with timing calculations
- **Multi-Stem Output**: Separate MIDI files for melody, harmony, bass, rhythm

### Music Theory Utilities
- **Scale Generation**: All standard scales and modes
- **Chord Progressions**: Harmonic structure analysis and generation
- **Note Manipulation**: Transposition, inversion, and transformation functions
- **Key Detection**: Automatic key/scale analysis from MIDI

### Analysis Features
- **MIDI Analysis**: File parsing and pattern extraction
- **Beat Analysis**: Rhythm and timing pattern detection
- **Harmonic Analysis**: Chord progression and key detection
- **Pattern Analysis**: Structural analysis of generated content

### Configuration System
- **Parameter Persistence**: JSON-based config storage and loading
- **Preset Management**: Save/load parameter combinations
- **Rhythm Configuration**: Advanced timing and groove settings
- **Validation**: Parameter range and compatibility checking

### Output Systems
- **MIDI Output**: Standard MIDI file generation with mido
- **Audio Rendering**: Plugin-based audio output with pedalboard/dawdreamer
- **Multi-Format Support**: MIDI, WAV, and other audio formats
- **Stem Separation**: Individual track extraction and export

### Plugin Architecture
- **Plugin Hosting**: VST/AudioUnit support via pedalboard
- **Parameter Enumeration**: Dynamic parameter discovery and control
- **Audio Processing**: Real-time and offline rendering capabilities
- **Effect Integration**: Full plugin parameter automation

## Connection Mapping

### Parameter Flow
```
GUI Controls → ParameterConfig → GenerationWorker → PatternOrchestrator → SongSkeleton → MidiOutput
```

### Specific Mappings
- **Genre QComboBox** → `GenreFactory.create_genre_rules()` → Pattern selection rules
- **Tempo QSpinBox** → `SongSkeleton(tempo=...)` → Timing calculations for all patterns
- **Mood QComboBox** → `PatternOrchestrator(mood=...)` → Emotional pattern filtering
- **Density QComboBox** → `DensityManager.create_density_manager_from_preset()` → Note/rhythm density scaling
- **Key/Mode QComboBox** → `MusicTheory.get_scale_notes()` → Note selection constraints
- **MatrixCell Click** → Parameter auto-population → Trigger generation with recommended settings
- **Generate QAction** → `GenerationWorker.run()` → Background generation process
- **Playback Controls** → `PlaybackController` → MIDI event extraction and pygame.midi output
- **Plugin Parameters** → `PluginHost.parameter_updates` → Audio effect processing
- **File Operations** → `ConfigManager.save/load_config()` → JSON persistence

### Signal/Slot Connections
- Parameter changes emit `parameters_updated` signal
- Generation completion updates `PianoRollView`
- Matrix selection triggers parameter updates
- Playback position updates piano roll playhead
- Plugin loading populates parameter controls

## Identified Gaps and Risks

### Functionality Gaps
1. **Missing Analysis UI**: No interface for MIDI analysis capabilities (analyzers/)
2. **Limited Parameter Controls**: Missing bars control, advanced timing parameters
3. **No Output File Selection**: Generation uses hardcoded "output.mid"
4. **No Real-time Preview**: Preview only updates after full generation
5. **Missing Advanced Parameters**: No UI for swing_percent, fill_frequency, ghost_note_level
6. **Limited Piano Roll Features**: View-only, no editing capabilities
7. **No Preset Management**: Beyond basic config save/load
8. **Missing Progress Feedback**: No progress bars for generation
9. **No Stem Visualization**: Cannot preview individual stems separately
10. **Limited Error Recovery**: Basic error handling without detailed diagnostics

### Integration Risks
1. **Threading Issues**: Potential race conditions in GenerationWorker
2. **Dependency Failures**: pygame.midi, mido, pedalboard may not be available
3. **Memory Management**: Large generations may cause GUI freezing
4. **File I/O Errors**: Unhandled permission or disk space issues
5. **Parameter Validation**: Inconsistent validation between GUI and backend
6. **Exception Handling**: Uncaught exceptions in background threads
7. **Resource Cleanup**: MIDI devices and audio resources not properly released
8. **Cross-platform Compatibility**: PyQt6 and audio libraries may behave differently

### Performance Risks
1. **GUI Blocking**: Long generations without proper progress indication
2. **Memory Leaks**: Improper cleanup of QThread objects and resources
3. **Event Queue Overflow**: Rapid parameter changes may overwhelm update signals
4. **Plugin Instability**: Audio plugins may crash or hang the application
5. **Large File Handling**: No size limits or streaming for large MIDI files

### User Experience Risks
1. **Silent Failures**: Generation errors not clearly communicated to user
2. **Inconsistent State**: GUI may not reflect actual backend state
3. **Confusing Workflows**: Complex parameter relationships not clearly explained
4. **Limited Feedback**: No visual feedback for long-running operations
5. **Accessibility Issues**: No keyboard navigation or screen reader support

## Recommendations

### High Priority
1. Add progress dialog for generation operations
2. Implement comprehensive error handling with user-friendly messages
3. Add output file selection dialog
4. Create analysis tab for MIDI file analysis features
5. Add bars parameter control to match CLI functionality

### Medium Priority
6. Implement real-time preview updates
7. Add preset management system for parameter combinations
8. Enhance piano roll with basic editing capabilities
9. Add validation feedback for parameter combinations
10. Implement proper resource cleanup and threading management

### Low Priority
11. Add advanced parameter controls (swing, fill frequency, etc.)
12. Implement stem-specific visualization and controls
13. Add keyboard shortcuts and accessibility features
14. Create user documentation and tutorials
15. Add export capabilities for different audio formats