# MIDI File Visualizer

A powerful tool that creates professional Digital Audio Workstation (DAW) style visualizations of MIDI files in HTML format with interactive piano roll views.

## Features

- 🏗️ **Professional DAW Interface**: Clean, modern design mimicking industry-standard DAW software
- 🎹 **Piano Roll View**: Horizontal timeline with piano keys and note blocks
- 🎨 **Visual Note Differentiation**: Different colors for white keys vs black keys
- ⏯️ **Interactive Playback**: Animated playhead with play/pause controls
- 🔍 **Zoom Controls**: Zoom in/out functionality for detailed analysis
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices
- ⌨️ **Keyboard Shortcuts**: Space (play/pause), R (reset), +/- (zoom)
- ⚡ **Performance Optimized**: Handles large MIDI files efficiently
- 🎵 **Tempo Detection**: Automatically detects and displays tempo from MIDI files

## Installation

The tool requires Python 3 and the `mido` library, which is already included in this project:

```bash
cd "/path/to/midimaster"
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python midi_visualizer.py your_song.mid
```

This creates `your_song_visual.html` in the current directory.

### Advanced Options

```bash
# Specify custom output file and zoom level
python midi_visualizer.py jazz_calm.mid --output jazz_visual.html --zoom 1.5

# Set custom title
python midi_visualizer.py pop_energetic.mid --title "Pop Energetic Track"
```

### Command Line Options

- `input_file`: Path to MIDI file (required)
- `--output`, `-o`: Output HTML file path (optional)
- `--zoom`, `-z`: Initial zoom level (default: 1.0)
- `--title`, `-t`: Custom title for visualization (optional)

## Examples

### Example 1: Create visualization with default settings

```bash
python midi_visualizer.py output.mid
# Creates: output_visual.html
```


### Example 2: High zoom for detailed analysis
```bash
python midi_visualizer.py jazz_calm.mid --zoom 2.0 --title "Jazz Analysis"
# Creates: jazz_calm_visual.html with 2x zoom
```


### Example 3: Custom output location
```bash
python midi_visualizer.py pop_energetic.mid --output ~/Desktop/pop_visual.html
# Creates: ~/Desktop/pop_visual.html
```

## Interactive Features

Once opened in a web browser, the visualization provides:


### Controls
- **▶ Play/Pause**: Starts/stops playhead animation
- **⏮ Reset**: Returns playhead to beginning
- **+ Zoom In**: Increases zoom level
- **− Zoom Out**: Decreases zoom level


### Note Interaction
- **Click Notes**: Shows detailed note information (pitch, velocity, timing)
- **Hover Effects**: Notes highlight on mouse over
- **Visual Encoding**: Note opacity reflects velocity


### Keyboard Shortcuts
- **Space**: Play/pause
- **R**: Reset to beginning
- **+ / =**: Zoom in
- **-**: Zoom out

## Data Displayed

Each note displays:
- **Pitch**: MIDI note number and note name (e.g., "60 (C4)")
- **Velocity**: Note strength (0-127)
- **Start Time**: When note begins (in seconds)
- **Duration**: Length of note (in seconds)

## Supported MIDI Files

- Standard MIDI files (.mid extension)
- Multi-track and multi-channel MIDI files
- Various tempos and time signatures
- Note velocity and channel information

## Technical Details

### Architecture
- **Backend**: Python with mido library for MIDI parsing
- **Frontend**: HTML5/CSS3/JavaScript for visualization
- **Styling**: Professional dark theme with gradients and animations
- **Responsive**: CSS media queries for mobile compatibility

### Performance
- Optimized for files with 200+ notes
- Efficient event handling and rendering
- Smooth animations at 60fps when possible

### Browser Compatibility
- Chrome/Chromium 80+
- Firefox 78+
- Safari 14+
- Edge 80+

## Troubleshooting

### Common Issues

**"Error: mido library not available"**
```bash
pip install mido
```

**"Input file does not exist"**
- Ensure MIDI file has .mid extension
- Check file path for typos
- Use absolute paths if needed

**Visualization appears empty**
- MIDI file may not contain note events
- Check MIDI file in a DAW software first
- Try a different MIDI file from the test_outputs directory

**Chrome won't open HTML file**
- Chrome has restrictions on local file access
- Use Firefox, Safari, or Edge instead
- Or run a local server: `python -m http.server`

### Getting Help

If you encounter issues:
1. Check the console for error messages
2. Try with a known working MIDI file from test_outputs/
3. Verify Python version (3.6+ required)

## Sample Visualizations

The project includes several sample MIDI files across different genres:


### Jazz Visualizations

```bash
python midi_visualizer.py test_outputs/midi_files/jazz_energetic_sparse_tempo160_bars16_run1.mid --zoom 1.5
```

### Pop/Rock Visualizations
```bash
python midi_visualizer.py test_outputs/midi_files/pop_energetic_dense_tempo120_bars16_run1.mid
```

## Integration

This visualizer works perfectly with the MIDI Master music generation program:

```bash
# Generate music
python main.py --genre jazz --tempo 120 --mood calm --output my_song.mid

# Visualize it
python midi_visualizer.py my_song.mid --title "AI Generated Jazz"
```

## Contributing

To extend the visualizer:

1. **Add new features**: Edit the `MidiVisualizer` class
2. **Improve styling**: Modify the CSS in `generate_piano_roll_html()`
3. **Add interactions**: Enhance the JavaScript functionality
4. **Performance**: Optimize for larger files if needed

## License

This tool is part of the MIDI Master project and follows the same licensing terms.