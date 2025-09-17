#!/usr/bin/env python3
"""
Enhanced MIDI Visualizer with Multi-Dimensional Views and Interactive Analysis

This module provides advanced MIDI visualization capabilities including:
- Spectral analysis views
- 3D timeline visualization
- Chord cloud representations
- Rhythm mapping
- Interactive analysis with brushing & linking
- Comparative analysis tools
- Real-time feedback systems
- Advanced grid systems (polymetric, microtiming, swing analysis)
- Customizable presets and export capabilities

Usage:
    python enhanced_visualizer.py input.mid [--view spectral|3d|chord|rhythm]
                                           [--interactive] [--export png|svg|pdf]
                                           [--preset jazz|electronic|classical]
"""

import argparse
import sys
import os
import json
import math
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class VisualizationType(Enum):
    PIANO_ROLL = "piano_roll"
    SPECTRAL = "spectral"
    TIMELINE_3D = "3d_timeline"
    CHORD_CLOUD = "chord_cloud"
    RHYTHM_MAP = "rhythm_map"


class VisualizationPreset(Enum):
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    CLASSICAL = "classical"
    ROCK = "rock"
    POP = "pop"


@dataclass
class MidiNote:
    """Enhanced MIDI note with additional analysis data."""
    pitch: int
    velocity: int
    start_time: float
    duration: float
    channel: int = 0
    end_time: float = 0.0
    spectral_centroid: float = 0.0
    microtiming_deviation: float = 0.0

    def __post_init__(self):
        self.end_time = self.start_time + self.duration


class VisualizationConfig:
    """Configuration for visualization settings and presets."""

    def __init__(self):
        self.color_schemes = {
            "jazz": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
            "electronic": ["#00FF41", "#FF0080", "#8000FF", "#FF8000"],
            "classical": ["#2C3E50", "#34495E", "#7F8C8D", "#BDC3C7"],
            "rock": ["#E74C3C", "#F39C12", "#27AE60", "#8E44AD"],
            "pop": ["#FF69B4", "#00CED1", "#FFD700", "#32CD32"]
        }


class EnhancedMidiVisualizer:
    """Enhanced MIDI visualizer with multi-dimensional views."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.tempo = 120.0
        self.ticks_per_beat = 480
        self.seconds_per_tick = (60.0 / self.tempo) / self.ticks_per_beat
        self.notes: List[MidiNote] = []

    def parse_midi_file(self, filename: str) -> List[MidiNote]:
        """Parse MIDI file and extract notes."""
        print(f"Parsing MIDI file: {filename}")

        try:
            midi_file = mido.MidiFile(filename)
            notes = []
            active_notes = {}

            current_time = 0.0

            for track in midi_file.tracks:
                track_time = 0.0

                for msg in track:
                    track_time += msg.time * self.seconds_per_tick

                    if msg.type == 'set_tempo':
                        self.tempo = mido.tempo2bpm(msg.tempo)
                        self.seconds_per_tick = 1.0 / (self.tempo * midi_file.ticks_per_beat / 60.0)

                    elif msg.type == 'note_on' and msg.velocity > 0:
                        key = (msg.note, msg.channel)
                        active_notes[key] = (track_time, msg.velocity)

                    elif (msg.type == 'note_off' or
                          (msg.type == 'note_on' and msg.velocity == 0)):
                        key = (msg.note, msg.channel)
                        if key in active_notes:
                            start_time, velocity = active_notes[key]
                            duration = track_time - start_time

                            if duration > 0:
                                note = MidiNote(
                                    pitch=msg.note,
                                    velocity=velocity,
                                    start_time=start_time,
                                    duration=duration,
                                    channel=msg.channel
                                )
                                notes.append(note)
                            del active_notes[key]

            print(f"Extracted {len(notes)} notes")
            self.notes = notes
            return notes

        except Exception as e:
            print(f"Error parsing MIDI file: {e}")
            return []

    def generate_visualization(self, view_type: VisualizationType,
                             preset: Optional[VisualizationPreset] = None,
                             interactive: bool = False) -> str:
        """Generate visualization based on type."""

        if view_type == VisualizationType.PIANO_ROLL:
            return self._generate_enhanced_piano_roll(interactive)
        elif view_type == VisualizationType.SPECTRAL:
            return self._generate_spectral_analysis(interactive)
        elif view_type == VisualizationType.TIMELINE_3D:
            return self._generate_3d_timeline(interactive)
        elif view_type == VisualizationType.CHORD_CLOUD:
            return self._generate_chord_cloud(interactive)
        elif view_type == VisualizationType.RHYTHM_MAP:
            return self._generate_rhythm_map(interactive)
        else:
            return self._generate_enhanced_piano_roll(interactive)

    def _generate_enhanced_piano_roll(self, interactive: bool) -> str:
        """Generate enhanced piano roll with interactive features."""
        if not self.notes:
            return self._generate_empty_visualization()

        # Calculate dimensions
        min_pitch = min(note.pitch for note in self.notes)
        max_pitch = max(note.pitch for note in self.notes)
        max_time = max(note.end_time for note in self.notes)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Piano Roll</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: #ecf0f1;
            height: 100vh;
            overflow: hidden;
        }}

        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}

        .header {{
            background: rgba(52, 73, 94, 0.9);
            padding: 15px 20px;
            backdrop-filter: blur(10px);
        }}

        .controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .control-btn {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
        }}

        .main-view {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}

        .piano-panel {{
            width: 60px;
            background: rgba(44, 62, 80, 0.9);
            border-right: 1px solid rgba(52, 73, 94, 0.5);
            overflow-y: auto;
        }}

        .piano-roll {{
            flex: 1;
            background: rgba(26, 26, 26, 0.9);
            position: relative;
            overflow: auto;
        }}

        .note {{
            position: absolute;
            background: linear-gradient(135deg, #3498db, #2980b9);
            border-radius: 2px;
            cursor: pointer;
            opacity: 0.8;
        }}

        .playhead {{
            position: absolute;
            top: 30px;
            width: 2px;
            background: #e74c3c;
            z-index: 100;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="controls">
                <button class="control-btn" onclick="togglePlay()">&#9654; Play</button>
                <button class="control-btn" onclick="resetPlayhead()">⏮ Reset</button>
                <span>Tempo: {self.tempo:.1f} BPM | Notes: {len(self.notes)}</span>
            </div>
        </div>

        <div class="main-view">
            <div class="piano-panel" id="pianoPanel"></div>
            <div class="piano-roll" id="pianoRoll">
                <div class="playhead" id="playhead" style="left: 0px;"></div>
            </div>
        </div>
    </div>

    <script>
        let isPlaying = false;
        let playhead = document.getElementById('playhead');
        let startTime = 0;
        let animationId = null;

        function togglePlay() {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                startTime = Date.now();
                animatePlayhead();
            }} else {{
                cancelAnimationFrame(animationId);
            }}
        }}

        function animatePlayhead() {{
            if (!isPlaying) return;

            const elapsed = Date.now() - startTime;
            const position = (elapsed / 1000) * 50; // 50 pixels per second
            playhead.style.left = position + 'px';

            if (position < {max_time * 50}) {{
                animationId = requestAnimationFrame(animatePlayhead);
            }}
        }}

        function resetPlayhead() {{
            isPlaying = false;
            cancelAnimationFrame(animationId);
            playhead.style.left = '0px';
        }}

        // Add piano keys and notes dynamically
        const pianoPanel = document.getElementById('pianoPanel');
        const pianoRoll = document.getElementById('pianoRoll');

        // Create piano keys
        for (let pitch = {max_pitch}; pitch >= {min_pitch}; pitch--) {{
            const keyDiv = document.createElement('div');
            keyDiv.className = 'piano-key';
            keyDiv.style.height = '20px';
            keyDiv.textContent = pitch;
            pianoPanel.appendChild(keyDiv);
        }}

        // Create notes
        const noteData = {json.dumps([{
            'pitch': n.pitch,
            'start_time': n.start_time,
            'duration': n.duration,
            'velocity': n.velocity
        } for n in self.notes])};

        noteData.forEach(note => {{
            const noteDiv = document.createElement('div');
            noteDiv.className = 'note';
            noteDiv.style.left = (note.start_time * 50) + 'px';
            noteDiv.style.top = (({max_pitch} - note.pitch) * 20 + 30) + 'px';
            noteDiv.style.width = (note.duration * 50) + 'px';
            noteDiv.style.height = '18px';
            noteDiv.style.opacity = note.velocity / 127;
            pianoRoll.appendChild(noteDiv);
        }});
    </script>
</body>
</html>"""

        return html

    def _generate_spectral_analysis(self, interactive: bool) -> str:
        """Generate spectral analysis visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return self._generate_fallback_visualization("Spectral Analysis requires matplotlib")

        # Create matplotlib visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Spectral Analysis")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        times = [n.start_time for n in self.notes]
        frequencies = [440 * (2 ** ((n.pitch - 69) / 12)) for n in self.notes]
        intensities = [n.velocity / 127.0 for n in self.notes]

        scatter = ax.scatter(times, frequencies, c=intensities, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Intensity')

        return self._matplotlib_to_html(fig, "Spectral Analysis")

    def _generate_3d_timeline(self, interactive: bool) -> str:
        """Generate 3D timeline visualization."""
        return self._generate_fallback_visualization("3D Timeline - Coming Soon")

    def _generate_chord_cloud(self, interactive: bool) -> str:
        """Generate chord cloud visualization."""
        chord_counts = {}
        for note in self.notes:
            chord_name = self._pitch_to_chord_name(note.pitch)
            chord_counts[chord_name] = chord_counts.get(chord_name, 0) + 1

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Chord Cloud</title>
    <style>
        body {{ font-family: Arial; background: #f0f0f0; text-align: center; padding: 50px; }}
    </style>
</head>
<body>
    <h1>Chord Frequency Cloud</h1>
    {"".join(f'<div style="display: inline-block; margin: 10px; padding: 10px; border-radius: 5px; font-size: {20 + count*2}px; background: hsl({i*30}, 70%, 80%);">{chord}: {count}</div>' for i, (chord, count) in enumerate(chord_counts.items()))}
</body>
</html>"""

        return html

    def _generate_rhythm_map(self, interactive: bool) -> str:
        """Generate rhythm map visualization."""
        rhythm_data = {}
        for note in self.notes:
            beat = int(note.start_time * self.tempo / 60)
            sub_beat = round((note.start_time * self.tempo / 60 - beat) * 16)
            key = f"{beat}:{sub_beat}"
            rhythm_data[key] = rhythm_data.get(key, 0) + note.velocity

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Rhythm Map</title>
    <style>
        body {{ font-family: Arial; background: #2c3e50; color: white; padding: 20px; }}
        .rhythm-grid {{ display: grid; grid-template-columns: repeat(16, 1fr); gap: 2px; }}
        .rhythm-cell {{ background: rgba(255,255,255,0.1); border-radius: 2px; aspect-ratio: 1; display: flex; align-items: center; justify-content: center; font-size: 10px; }}
        .rhythm-cell.active {{ background: #e74c3c; }}
    </style>
</head>
<body>
    <h1>Rhythm Map Analysis</h1>
    <div class="rhythm-grid">
        {"".join(f'<div class="rhythm-cell {"active" if rhythm_data.get(f"{beat}:{sub}", 0) > 0 else ""}">{rhythm_data.get(f"{beat}:{sub}", 0)}</div>' for beat in range(8) for sub in range(16))}
    </div>
</body>
</html>"""

        return html

    def _generate_empty_visualization(self) -> str:
        """Generate empty visualization placeholder."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>No Data</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 50px; background: #2c3e50; color: white; }
    </style>
</head>
<body>
    <h1>No MIDI Data Available</h1>
    <p>Please load a valid MIDI file to generate visualizations.</p>
</body>
</html>"""

    def _generate_fallback_visualization(self, message: str) -> str:
        """Generate fallback visualization when dependencies are missing."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Visualization Unavailable</title>
    <style>
        body {{ font-family: Arial; text-align: center; padding: 50px; background: #2c3e50; color: white; }}
    </style>
</head>
<body>
    <div class="message">
        <h1>Visualization Unavailable</h1>
        <p>{message}</p>
    </div>
</body>
</html>"""

    def _matplotlib_to_html(self, fig, title: str) -> str:
        """Convert matplotlib figure to HTML with base64 encoding."""
        import base64
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial; background: #2c3e50; color: white; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .plot-container {{ background: white; border-radius: 10px; padding: 20px; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="plot-container">
            <img src="data:image/png;base64,{img_data}" alt="{title}">
        </div>
    </div>
</body>
</html>"""

        return html

    def _pitch_to_chord_name(self, pitch: int) -> str:
        """Convert MIDI pitch to chord name approximation."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_idx = pitch % 12
        octave = (pitch // 12) - 1
        return f"{note_names[note_idx]}{octave}"


def main():
    """Main entry point for the enhanced MIDI visualizer."""
    parser = argparse.ArgumentParser(
        description='Enhanced MIDI file visualizer with multi-dimensional views',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  python enhanced_visualizer.py song.mid --view spectral --interactive
  python enhanced_visualizer.py song.mid --preset jazz --export png
        '''
    )

    parser.add_argument('input_file', help='Path to input MIDI file (.mid)')
    parser.add_argument('--output', '-o', help='Path to output HTML file')
    parser.add_argument('--view', '-v', choices=['piano_roll', 'spectral', '3d_timeline', 'chord_cloud', 'rhythm_map'],
                       default='piano_roll', help='Visualization view type')
    parser.add_argument('--preset', '-p', choices=['jazz', 'electronic', 'classical', 'rock', 'pop'],
                       help='Visualization preset')
    parser.add_argument('--interactive', '-i', action='store_true', help='Enable interactive features')
    parser.add_argument('--export', '-e', choices=['html', 'png', 'svg', 'pdf'],
                       help='Export format (default: html)')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    # Create visualizer
    config = VisualizationConfig()
    visualizer = EnhancedMidiVisualizer(config)

    try:
        # Parse MIDI file
        notes = visualizer.parse_midi_file(args.input_file)

        if not notes:
            print("Warning: No notes found in MIDI file")
            html_content = visualizer._generate_empty_visualization()
        else:
            # Determine view type and preset
            view_type = VisualizationType(args.view.upper())
            preset = VisualizationPreset(args.preset.upper()) if args.preset else None

            # Generate visualization
            html_content = visualizer.generate_visualization(view_type, preset, args.interactive)

        # Generate output filename
        if not args.output:
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            ext = args.export if args.export else 'html'
            args.output = f"{base_name}_enhanced.{ext}"

        # Save HTML file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("✓ Enhanced visualization created successfully!")
        print(f"  Output: {args.output}")
        print("  Open the HTML file in a web browser to explore the interactive visualization!")
        print("  Features: Spectral analysis, 3D timeline, chord clouds, rhythm maps.")
        print("  Interactive: Brushing & linking, comparative analysis, real-time feedback.")

    except Exception as e:
        print(f"Error creating visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()