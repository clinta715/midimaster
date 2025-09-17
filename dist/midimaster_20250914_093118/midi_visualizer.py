#!/usr/bin/env python3
"""
MIDI File Visualizer for DAW-style visualization in HTML format.

This module creates professional Digital Audio Workstation (DAW) style
visualizations of MIDI files in HTML format with piano roll views.

Usage:
    python midi_visualizer.py input.mid [--output visualization.html] [--zoom 1.0]

Example:
    python midi_visualizer.py my_song.mid --output my_song_visual.html --zoom 1.5
"""

import argparse
import sys
import os
from typing import List, Tuple, Dict
import math

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Error: mido library is required. Install with: pip install mido")
    sys.exit(1)


class MidiNote:
    """Represents a single note with timing information."""

    def __init__(self, pitch: int, velocity: int, start_time: float, duration: float, channel: int = 0):
        self.pitch = pitch
        self.velocity = velocity
        self.start_time = start_time
        self.duration = duration
        self.channel = channel
        self.end_time = start_time + duration


class MidiVisualizer:
    """Handles MIDI file parsing and HTML visualization generation."""

    # Piano key mapping
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    WHITE_KEYS = [0, 2, 4, 5, 7, 9, 11]  # White key indices within octave
    BLACK_KEYS = [1, 3, 6, 8, 10]  # Black key indices within octave

    def __init__(self):
        self.tempo = 120  # Default tempo
        self.ticks_per_beat = 480  # Default MIDI resolution
        self.seconds_per_tick = 0.5 / self.tempo  # Default timing

    def parse_midi_file(self, filename: str) -> List[MidiNote]:
        """Parse a MIDI file and extract note events."""
        print(f"Parsing MIDI file: {filename}")

        try:
            midi_file = mido.MidiFile(filename)
            notes = []

            # Track note events (start and end times)
            active_notes = {}  # (pitch, channel) -> (start_time, velocity)

            current_time = 0.0

            # Process each track
            for track in midi_file.tracks:
                track_time = 0.0

                for msg in track:
                    track_time += msg.time * self.seconds_per_tick

                    if msg.type == 'set_tempo':
                        self.tempo = mido.tempo2bpm(msg.tempo)
                        self.seconds_per_tick = 1.0 / (self.tempo * midi_file.ticks_per_beat / 60.0) if midi_file.ticks_per_beat > 0 else self.seconds_per_tick

                    elif msg.type == 'note_on' and msg.velocity > 0:
                        # Note start
                        key = (msg.note, msg.channel)
                        active_notes[key] = (track_time, msg.velocity)

                    elif (msg.type == 'note_off' or
                          (msg.type == 'note_on' and msg.velocity == 0)):
                        # Note end
                        key = (msg.note, msg.channel)
                        if key in active_notes:
                            start_time, velocity = active_notes[key]
                            duration = track_time - start_time

                            if duration > 0:
                                notes.append(MidiNote(
                                    pitch=msg.note,
                                    velocity=velocity,
                                    start_time=start_time,
                                    duration=duration,
                                    channel=msg.channel
                                ))
                            del active_notes[key]

            print(f"Extracted {len(notes)} notes from MIDI file")
            return notes

        except Exception as e:
            print(f"Error parsing MIDI file: {e}")
            return []

    def pitch_to_note_name(self, pitch: int) -> str:
        """Convert MIDI pitch to note name (e.g., C4, F#3)."""
        note_idx = pitch % 12
        octave = (pitch // 12) - 1
        return f"{self.NOTE_NAMES[note_idx]}{octave}"

    def is_white_key(self, pitch: int) -> bool:
        """Check if a pitch corresponds to a white key."""
        return (pitch % 12) in self.WHITE_KEYS

    def generate_piano_roll_html(self, notes: List[MidiNote], zoom: float = 1.0, title: str = "MIDI Visualization") -> str:
        """Generate HTML piano roll visualization."""

        if not notes:
            return self.generate_empty_html(title)

        # Calculate dimensions
        min_pitch = min(note.pitch for note in notes)
        max_pitch = max(note.pitch for note in notes)
        total_duration = max(note.end_time for note in notes)

        # Add some padding
        pitch_padding = 2
        time_padding = total_duration * 0.1

        min_pitch -= pitch_padding
        max_pitch += pitch_padding
        total_duration += time_padding

        # Piano roll dimensions
        note_height = 8 * zoom
        pixels_per_second = 50 * zoom
        piano_key_width = 60
        timeline_height = 30
        canvas_width = int(total_duration * pixels_per_second) + piano_key_width
        canvas_height = int((max_pitch - min_pitch + 1) * note_height) + timeline_height

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2c3e50, #34495e);
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
            background: #34495e;
            padding: 10px 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            z-index: 10;
        }}

        h1 {{
            font-size: 1.2em;
            font-weight: 600;
            color: #ecf0f1;
        }}

        .controls {{
            display: flex;
            gap: 10px;
            margin-top: 5px;
            flex-wrap: wrap;
        }}

        .control-btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background-color 0.2s;
        }}

        .control-btn:hover {{
            background: #2980b9;
        }}

        .main-view {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}

        .piano-panel {{
            width: {piano_key_width}px;
            background: #2c3e50;
            border-right: 1px solid #34495e;
            overflow-y: auto;
            flex-shrink: 0;
        }}

        .timeline {{
            height: {timeline_height}px;
            background: #34495e;
            border-bottom: 1px solid #2c3e50;
            position: relative;
            overflow: hidden;
        }}

        .time-ruler {{
            position: relative;
            height: 100%;
            background: linear-gradient(to right, #34495e 0%, #2c3e50 100%);
        }}

        .time-label {{
            position: absolute;
            top: 5px;
            font-size: 10px;
            color: #bdc3c7;
            user-select: none;
        }}

        .piano-roll {{
            flex: 1;
            background: #1a1a1a;
            position: relative;
            overflow: auto;
            border-left: 1px solid #34495e;
        }}

        .grid {{
            position: absolute;
            top: {timeline_height}px;
            left: 0;
            width: {canvas_width}px;
            height: {canvas_height - timeline_height}px;
        }}

        .grid-line {{
            position: absolute;
            background: rgba(52, 73, 94, 0.3);
        }}

        .vertical-line {{
            width: 1px;
            height: 100%;
        }}

        .horizontal-line {{
            height: 1px;
            width: 100%;
        }}

        .note {{
            position: absolute;
            background: linear-gradient(135deg, #3498db, #2980b9);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 2px;
            cursor: pointer;
            transition: opacity 0.2s;
            overflow: hidden;
        }}

        .note:hover {{
            opacity: 0.8;
            box-shadow: 0 0 10px rgba(52, 152, 219, 0.5);
        }}

        .note.white-key {{
            background: linear-gradient(135deg, #2ecc71, #27ae60);
        }}

        .note.black-key {{
            background: linear-gradient(135deg, #e67e22, #d35400);
        }}

        .note.selected {{
            box-shadow: 0 0 15px rgba(241, 196, 15, 0.8);
        }}

        .playhead {{
            position: absolute;
            top: {timeline_height}px;
            width: 2px;
            height: {canvas_height - timeline_height}px;
            background: #e74c3c;
            z-index: 100;
            box-shadow: 0 0 10px rgba(231, 76, 60, 0.5);
        }}

        .note-label {{
            font-size: 10px;
            color: white;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.8);
            padding: 2px;
            pointer-events: none;
        }}

        .piano-key {{
            height: {note_height:.1f}px;
            border-bottom: 1px solid rgba(52, 73, 94, 0.3);
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            font-size: 11px;
            color: #bdc3c7;
            user-select: none;
        }}

        .white-key {{
            background: #34495e;
        }}

        .black-key {{
            background: #2c3e50;
            color: #ecf0f1;
        }}

        @media (max-width: 768px) {{
            .controls {{
                justify-content: center;
            }}

            .piano-panel {{
                width: 40px;
            }}

            .control-btn {{
                padding: 4px 8px;
                font-size: 0.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="controls">
                <button class="control-btn" onclick="togglePlay()">&#9654; Play</button>
                <button class="control-btn" onclick="resetPlayhead()">⏮ Reset</button>
                <button class="control-btn" onclick="zoomIn()">+</button>
                <button class="control-btn" onclick="zoomOut()">−</button>
                <span style="margin-left: 10px; font-size: 0.9em;">
                    Tempo: {self.tempo:.1f} BPM | Notes: {len(notes)} | Duration: {total_duration:.2f}s
                </span>
            </div>
        </div>

        <div class="main-view">
            <div class="piano-panel" id="pianoPanel">
                <div class="timeline"></div>"""

        # Add piano keys
        html += "<div class=\"piano-keys\" id=\"pianoKeys\">"
        for pitch in range(max_pitch, min_pitch - 1, -1):
            key_class = "white-key" if self.is_white_key(pitch) else "black-key"
            note_name = self.pitch_to_note_name(pitch)
            html += f'<div class="piano-key {key_class}" data-pitch="{pitch}">{note_name}</div>'
        html += "</div>\n        </div>\n"

        # Piano roll area
        html += f"""        <div class="piano-roll" id="pianoRoll">
            <div class="grid" id="grid">\n"""

        # Generate grid lines (every beat)
        beats = max(1, int(total_duration * (self.tempo / 60)))
        for beat in range(beats + 1):
            beat_time = beat / (self.tempo / 60)
            x_pos = int(beat_time * pixels_per_second)
            html += f'<div class="grid-line vertical-line" style="left: {x_pos}px;"></div>'

            # Time labels every few beats
            if beat % 4 == 0:
                bar = beat // 4
                beat_in_bar = beat % 4
                html += f'<div class="time-label" style="left: {x_pos + 5}px;">{bar}:{beat_in_bar}</div>'

        # Generate horizontal grid lines (every octave)
        for pitch in range(min_pitch, max_pitch + 1):
            if pitch % 12 == 0:  # C notes
                y_pos = int((max_pitch - pitch) * note_height) + timeline_height
                html += f'<div class="grid-line horizontal-line" style="top: {y_pos}px;"></div>'

        # Add notes (limit for performance)
        for i, note in enumerate(notes[:min(200, len(notes))]):
            if min_pitch <= note.pitch <= max_pitch:
                x = int(note.start_time * pixels_per_second)
                y = int((max_pitch - note.pitch) * note_height) + timeline_height
                width = max(1, int(note.duration * pixels_per_second))
                height = int(note_height * 0.8)

                key_class = "white-key" if self.is_white_key(note.pitch) else "black-key"
                opacity = min(1.0, note.velocity / 127.0 * 0.8 + 0.2)

                note_label = self.pitch_to_note_name(note.pitch)
                selected_class = " selected" if i < 3 else ""  # Highlight first few notes

                html += f'''<div class="note {key_class}{selected_class}"
                    style="left: {x}px; top: {y}px; width: {width}px; height: {height}px; opacity: {opacity};"
                    data-pitch="{note.pitch}" data-velocity="{note.velocity}"
                    onclick="showNoteInfo({note.pitch}, {note.velocity}, {note.start_time:.3f}, {note.duration:.3f})"
                    title="{note_label} - Vel:{note.velocity} - Start:{note.start_time:.2f}s - Dur:{note.duration:.2f}s">
                    <div class="note-label">{note_label}</div>
                </div>'''

        html += f'''
            </div>
            <div class="playhead" id="playhead" style="left: 0px;"></div>
        </div>
    </div>

    <script>
        let isPlaying = false;
        let playhead = document.getElementById('playhead');
        let startTime = 0;
        let animationId = null;
        let currentZoom = {zoom};
        let grid = document.getElementById('grid');
        let pianoRoll = document.getElementById('pianoRoll');

        function togglePlay() {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                startTime = Date.now() - (parseInt(playhead.style.left) / {pixels_per_second} * 1000);
                animatePlayhead();
                document.querySelector('.control-btn:first-child').innerHTML = '⏸ Pause';
            }} else {{
                cancelAnimationFrame(animationId);
                document.querySelector('.control-btn:first-child').innerHTML = '▶ Play';
            }}
        }}

        function animatePlayhead() {{
            if (!isPlaying) return;

            const elapsed = Date.now() - startTime;
            const position = (elapsed / 1000) * {pixels_per_second};
            const maxPosition = {canvas_width} - {piano_key_width};

            if (position > maxPosition) {{
                togglePlay();
                return;
            }}

            playhead.style.left = position + 'px';

            // Auto-scroll
            const rollRect = pianoRoll.getBoundingClientRect();
            const headRect = playhead.getBoundingClientRect();
            if (headRect.left > rollRect.width * 0.8) {{
                pianoRoll.scrollLeft = position - rollRect.width * 0.5;
            }}

            animationId = requestAnimationFrame(animatePlayhead);
        }}

        function resetPlayhead() {{
            isPlaying = false;
            cancelAnimationFrame(animationId);
            playhead.style.left = '0px';
            pianoRoll.scrollLeft = 0;
            document.querySelector('.control-btn:first-child').innerHTML = '▶ Play';
        }}

        function showNoteInfo(pitch, velocity, startTime, duration) {{
            alert(`Pitch: ${{pitch}} (${{getNoteName(pitch)}})\\nVelocity: ${{velocity}}\\nStart: ${{startTime.toFixed(3)}}s\\nDuration: ${{duration.toFixed(3)}}s`);
        }}

        function getNoteName(pitch) {{
            const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
            const noteIdx = pitch % 12;
            const octave = Math.floor(pitch / 12) - 1;
            return notes[noteIdx] + octave;
        }}

        function zoomIn() {{
            currentZoom *= 1.2;
            updateZoom();
        }}

        function zoomOut() {{
            currentZoom /= 1.2;
            currentZoom = Math.max(0.1, currentZoom);
            updateZoom();
        }}

        function updateZoom() {{
            grid.style.transform = `scale(${{currentZoom}})`;
            grid.style.transformOrigin = '0 0';
            pianoRoll.style.overflow = 'auto';
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.code === 'Space') {{
                e.preventDefault();
                togglePlay();
            }} else if (e.code === 'KeyR') {{
                resetPlayhead();
            }} else if (e.code === 'Equal' || e.code === 'NumpadAdd') {{
                zoomIn();
            }} else if (e.code === 'Minus' || e.code === 'NumpadSubtract') {{
                zoomOut();
            }}
        }});

        // Initialize
        pianoRoll.addEventListener('scroll', function() {{
            document.getElementById('pianoPanel').scrollTop = pianoRoll.scrollTop;
        }});

        // Set initial grid scale
        grid.style.transform = `scale(${{currentZoom}})`;
        grid.style.transformOrigin = '0 0';
    </script>
</body>
</html>'''

        return html

    def generate_empty_html(self, title: str) -> str:
        """Generate HTML for empty/no-notes case."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: #ecf0f1;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            text-align: center;
        }}
        .message {{
            background: #34495e;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        h1 {{ color: #e74c3c; margin-bottom: 20px; }}
        p {{ font-size: 18px; }}
    </style>
</head>
<body>
<div class="message">
    <h1>No Notes Found</h1>
    <p>The MIDI file either doesn't exist or contains no note events.</p>
    <p>Please check the file path and try again.</p>
</div>
</body>
</html>"""


def main():
    """Main entry point for the MIDI visualizer."""
    parser = argparse.ArgumentParser(
        description='Create DAW-style MIDI file visualizations in HTML format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python midi_visualizer.py song.mid
  python midi_visualizer.py song.mid --output visualization.html --zoom 1.5
  python midi_visualizer.py song.mid --output song_visual.html
        '''
    )

    parser.add_argument('input_file', help='Path to input MIDI file (.mid)')
    parser.add_argument(
        '--output', '-o',
        help='Path to output HTML file (default: input_file_visual.html)'
    )
    parser.add_argument(
        '--zoom', '-z',
        type=float,
        default=1.0,
        help='Initial zoom level (default: 1.0)'
    )
    parser.add_argument(
        '--title', '-t',
        help='Title for the visualization (default: filename)'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    if not args.input_file.lower().endswith('.mid'):
        print("Warning: Input file should be a MIDI file (.mid extension)")

    # Generate output filename
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = f"{base_name}_visual.html"

    # Generate title
    if not args.title:
        args.title = os.path.splitext(os.path.basename(args.input_file))[0].replace('_', ' ').title()

    # Create visualizer and process file
    visualizer = MidiVisualizer()
    try:
        notes = visualizer.parse_midi_file(args.input_file)

        if not notes:
            print("Warning: No notes found in MIDI file")
            html_content = visualizer.generate_empty_html(args.title)
        else:
            html_content = visualizer.generate_piano_roll_html(notes, args.zoom, args.title)

        # Write HTML file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✓ Visualization created: {args.output}")
        print("  Open the HTML file in a web browser to view the piano roll!")
        print("  Keyboard shortcuts: Space (play/pause), R (reset), +/- (zoom)")

    except Exception as e:
        print(f"Error creating visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()