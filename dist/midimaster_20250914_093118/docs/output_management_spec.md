# MIDI Master GUI Output Management and Configuration Specification

## Overview

This document specifies the design for enhancing the MIDI Master GUI application with improved output management and configuration saving capabilities. The enhancements focus on providing users with better control over file output, render-to-file workflows, and persistent configuration management.

## Existing Architecture Context

### Current GUI Structure (main_window.py)
- **PyQt6-based main window** with parameter controls (genre, mood, tempo, bars, density)
- **Parameter matrix** for visual parameter selection
- **Piano roll preview** for real-time visualization
- **Playback controls** for audio playback
- **Single output filename field** in parameters section
- **"Generate Music" button** triggers full generation and save

### Current Generation Flow (generation_worker.py)
- **Background QThread** for non-blocking generation
- **Parameter validation** before generation
- **Pattern generation** and song skeleton building
- **Direct MIDI save** using MidiOutput class
- **Progress signaling** and error handling

### Current Output Handling (midi_output.py)
- **Single file output** or separate instrument files
- **Mido library** for MIDI file creation
- **Channel mapping** for different instruments
- **Tempo and time signature** metadata

## Feature Specifications

### 1. Output Filename and Folder Specification

#### UI Design

**New UI Elements:**
```
Parameter Controls Group Box:
├── Genre: [ComboBox]
├── Mood: [ComboBox]
├── Tempo: [Slider] [SpinBox] BPM
├── Bars: [Slider] [SpinBox]
└── Density: [RadioButtons]

Output Management Group Box:
├── Output Folder: [LineEdit] [Browse Button]
├── Filename: [LineEdit] [Auto-generate Button]
├── Preview: [Checkbox] "Use auto-generated filename"
└── Default Folder: [Button] "Reset to default"

[Generate Preview] [Render to File] [Save Config]
```

**Layout Integration:**
- Add new "Output Management" group box below existing parameter controls
- Maintain current left panel width (300-400px) with expanded height
- Output folder field with browse dialog (QFileDialog.getExistingDirectory)
- Filename field with validation feedback
- Auto-generate button populates filename based on current parameters

#### Filename Auto-Generation Logic

**Format:** `{genre}_{mood}_{tempo}_{bars}_{timestamp}.mid`

**Examples:**
- `jazz_energetic_120_16_20241219_143022.mid`
- `pop_happy_100_8_20241219_143500.mid`

**Validation Rules:**
- Filename must be non-empty and contain only valid filesystem characters
- Maximum length: 255 characters
- Reserved names check (CON, PRN, AUX, etc. on Windows)
- Extension validation (.mid required)

#### Folder Management

**Default Behavior:**
- Default folder: `output/` subdirectory in project root
- Auto-create `output/` if it doesn't exist
- Fallback: current working directory if `output/` creation fails

**User Customization:**
- Browse dialog for custom folder selection
- Path validation (existence, writability, sufficient space)
- Relative path support (relative to project root)
- "Reset to default" button restores default path

#### Integration Points

**main_window.py Modifications:**
- Add `_create_output_management_panel()` method
- Connect browse buttons to `QFileDialog` slots
- Add filename validation in `_validate_parameters()`
- Update `current_params` dict with `output_folder` key
- Modify `_on_output_changed()` to handle folder + filename

**generation_worker.py Modifications:**
- Update parameter validation to include folder path
- Use full path construction: `os.path.join(folder, filename)`
- Add folder creation logic before MIDI save

**midi_output.py Modifications:**
- Accept full output path in `save_to_midi()` method
- Maintain backward compatibility with existing filename-only calls

### 2. Render-to-File Mechanism

#### UI Flow Design

**Three-Phase Workflow:**
1. **Preview Phase:** Quick generation with temporary output
2. **Refinement Phase:** User adjusts parameters with live preview
3. **Final Render Phase:** Save to specified filename/folder

**Button States:**
```
[Generate Preview] - Enabled: Always
[Render to File]  - Enabled: After successful preview generation
                  - Disabled: During generation, if no preview exists
```

**Status Indicators:**
- Status bar messages: "Preview generated", "Rendering to file...", "File saved successfully"
- Progress bar for both preview and render operations
- Visual feedback: Button text changes during operations

#### File Management Architecture

**Temporary File Strategy:**
```
temp/
├── preview_[timestamp].mid
├── preview_[timestamp]_melody.mid
├── preview_[timestamp]_harmony.mid
├── preview_[timestamp]_bass.mid
└── preview_[timestamp]_rhythm.mid
```

**Cleanup Strategy:**
- Auto-cleanup on application exit
- Configurable retention period (default: 24 hours)
- Manual cleanup option in settings
- Size-based cleanup (delete oldest files when temp folder exceeds size limit)

#### Data Flow

```
User Click "Generate Preview":
├── Validate parameters
├── Generate patterns (reduced complexity)
├── Save to temp file
├── Update piano roll preview
└── Enable "Render to File" button

User Click "Render to File":
├── Copy temp file to final location
├── Update status: "Render complete"
├── Log file save operation
└── Reset preview state
```

#### Integration Points

**main_window.py Modifications:**
- Separate `_generate_preview()` and `_render_to_file()` methods
- Add `_preview_file_path` instance variable
- Modify generation worker to accept `preview_mode` flag
- Update button states based on preview availability
- Add status messages for render operations

**generation_worker.py Modifications:**
- Add `preview_mode` parameter to constructor
- Skip full complexity processing in preview mode
- Return temporary file path instead of final save
- Add `render_to_file()` method for final save

### 3. Configurable Scratch/Temp File Location

#### Configuration UI

**Settings Dialog:**
```
Temp File Configuration:
├── Temp Directory: [LineEdit] [Browse Button]
├── Auto-cleanup: [Checkbox] "Enable automatic cleanup"
├── Retention Period: [SpinBox] hours
├── Max Size: [SpinBox] MB
└── [Apply] [Cancel] [Reset to Default]
```

**Integration:**
- Add "Settings" menu item to main window menu bar
- Settings persist across application sessions
- Default temp location: system temp directory + "midi_master_temp"

#### File System Operations

**Directory Creation:**
- Check write permissions before setting temp directory
- Create directory structure if it doesn't exist
- Handle permission errors gracefully
- Fallback to system temp if custom location fails

**Cleanup Operations:**
- Background thread for cleanup to avoid UI blocking
- Delete files older than retention period
- Delete files when total size exceeds limit
- Preserve most recent files during cleanup

**Error Handling:**
- Disk space validation before operations
- Permission error dialogs
- Automatic fallback to system temp
- Logging of cleanup operations

#### Configuration Persistence

**Storage Format:** JSON configuration file
```
{
  "temp_directory": "C:\\Users\\user\\AppData\\Local\\Temp\\midi_master_temp",
  "auto_cleanup": true,
  "retention_hours": 24,
  "max_size_mb": 500,
  "last_cleanup": "2024-12-19T14:30:22Z"
}
```

**File Location:** `config/temp_settings.json` in project directory

### 4. Configuration Save/Load System

#### Configuration Data Structure

**Comprehensive Parameter Set:**
```json
{
  "version": "1.0",
  "name": "Jazz Energetic Session",
  "description": "Settings for energetic jazz compositions",
  "timestamp": "2024-12-19T14:30:22Z",
  "parameters": {
    "genre": "jazz",
    "mood": "energetic",
    "tempo": 120,
    "bars": 16,
    "density": "balanced",
    "output_folder": "output/",
    "auto_preview": true,
    "separate_files": false
  },
  "matrix_selections": {
    "selected_cells": [
      {"genre": "jazz", "tempo_range": [110, 130], "mood": "energetic"}
    ]
  },
  "ui_state": {
    "window_size": [1200, 900],
    "splitter_sizes": [350, 850]
  }
}
```

#### UI Design

**Configuration Management Panel:**
```
Configuration Group Box:
├── Config Name: [LineEdit]
├── Description: [LineEdit]
├── [Save Configuration] [Load Configuration] [Browse Configs]
└── Recent Configs: [ComboBox]
```

**Load Dialog:**
- List view of saved configurations with metadata
- Preview of configuration parameters
- Load/Cancel buttons
- Delete configuration option

#### File Management

**Configuration Directory:** `configs/` in project root
**Naming Convention:** `config_{genre}_{mood}_{timestamp}.json`
**Backup Strategy:** Create backup of current settings before loading new config

#### Integration Points

**main_window.py Modifications:**
- Add configuration management methods
- Update parameter loading to include new fields
- Add configuration validation
- Integrate with existing parameter change handlers

## Data Flow Diagrams

### Preview Generation Flow
```
User Parameter Change
       ↓
Auto-preview enabled?
       ↓
  Generate Preview Worker
       ↓
Validate Parameters
       ↓
Create Song Skeleton
       ↓
Generate Patterns (reduced)
       ↓
Save to Temp File
       ↓
Update Piano Roll
       ↓
Enable Render Button
```

### Final Render Flow
```
User Clicks "Render to File"
       ↓
Validate Output Path
       ↓
Copy Temp File to Final
       ↓
Update Status Bar
       ↓
Reset Preview State
       ↓
Log Completion
```

### Configuration Management Flow
```
Save Configuration:
User Enters Name
       ↓
Collect All Parameters
       ↓
Serialize to JSON
       ↓
Save to configs/ directory
       ↓
Update Recent Configs List

Load Configuration:
User Selects Config
       ↓
Load JSON File
       ↓
Validate Version Compatibility
       ↓
Apply Parameters to UI
       ↓
Trigger Preview Update
```

## Edge Cases and Validation

### File System Edge Cases
1. **Path Too Long:** Windows MAX_PATH limitation (260 characters)
2. **Permission Denied:** Target directory not writable
3. **Disk Full:** Insufficient space for output file
4. **Network Path:** UNC paths or mapped drives
5. **Special Characters:** Unicode filenames, reserved names
6. **Concurrent Access:** Multiple instances writing to same location

### Parameter Validation
1. **Invalid Tempo:** Outside 80-160 BPM range
2. **Invalid Bars:** Outside 4-32 range
3. **Empty Filename:** Missing or whitespace-only filename
4. **Invalid Extension:** Filename without .mid extension
5. **Path Injection:** Attempted directory traversal attacks

### Error Handling Strategies
- **Graceful Degradation:** Fall back to defaults when custom paths fail
- **User Feedback:** Clear error messages with suggested actions
- **Recovery Options:** Allow user to choose alternative paths
- **Logging:** Detailed error logging for debugging

## Best Practices for File I/O

### Python File I/O Patterns
1. **Use pathlib for path operations:**
   ```python
   from pathlib import Path
   config_path = Path("configs") / "settings.json"
   ```

2. **Atomic file operations:**
   ```python
   # Write to temporary file then rename
   temp_path = final_path.with_suffix('.tmp')
   with open(temp_path, 'w') as f:
       f.write(content)
   temp_path.replace(final_path)
   ```

3. **Resource management with context managers:**
   ```python
   with open(filepath, 'w', encoding='utf-8') as f:
       json.dump(data, f, indent=2)
   ```

### Thread Safety Considerations
1. **Main thread for UI updates only**
2. **Background threads for file I/O operations**
3. **Signal-slot mechanism for thread communication**
4. **Mutex protection for shared resources**

### Performance Optimizations
1. **Lazy loading of configuration files**
2. **Caching of frequently accessed paths**
3. **Batch file operations where possible**
4. **Progress reporting for long operations**

## Implementation Roadmap

### Phase 1: Core Output Management
- Implement output folder specification UI
- Add filename auto-generation
- Basic path validation
- Integration with existing save logic

### Phase 2: Render-to-File Mechanism
- Separate preview and render workflows
- Temporary file management
- Status indicators and progress feedback
- Cleanup system implementation

### Phase 3: Advanced Configuration
- Configuration save/load system
- Settings persistence
- Temp directory configuration
- Advanced validation and error handling

### Phase 4: Polish and Testing
- UI/UX refinements
- Comprehensive error handling
- Performance optimization
- User testing and feedback integration

## Dependencies and Requirements

### Python Libraries
- **PyQt6:** GUI framework (already in use)
- **pathlib:** Modern path handling (Python 3.4+ standard)
- **json:** Configuration serialization (Python standard)
- **tempfile:** Temporary file management (Python standard)
- **shutil:** File operations (Python standard)

### System Requirements
- **File System:** Read/write access to project directory
- **Permissions:** Ability to create directories and files
- **Space:** Sufficient disk space for temp files and outputs
- **Unicode Support:** For international filename support

## Migration Strategy

### Backward Compatibility
- Existing `output.mid` default maintained
- Optional new features - existing workflow unchanged
- Configuration files versioned for future compatibility
- Graceful handling of missing configuration files

### Data Migration
- Automatic detection of legacy configurations
- Migration prompts for users with existing setups
- Backup creation before configuration changes
- Rollback capability for failed migrations

---

*This specification provides a comprehensive design for the output management and configuration enhancements. Implementation should follow PyQt6 best practices and maintain the application's responsive, non-blocking nature.*