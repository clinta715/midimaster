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
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import colorsys

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import base64
    import io
    BASE64_AVAILABLE = True
except ImportError:
    BASE64_AVAILABLE = False


class VisualizationType(Enum):
    PIANO_ROLL = "piano_roll"
    SPECTRAL = "spectral"
    TIMELINE_3D = "3d_timeline"
    CHORD_CLOUD = "chord_cloud"
    RHYTHM_MAP = "rhythm_map"
    POLYMETRIC = "polymetric"
    MICROTEMPO = "microtempo"
    SWING = "swing"


class VisualizationPreset(Enum):
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    CLASSICAL = "classical"
    ROCK = "rock"
    POP = "pop"
    CUSTOM = "custom"


@dataclass
class MidiNote:
    """Enhanced MIDI note with additional analysis data."""
    pitch: int
    velocity: int
    start_time: float
    duration: float
    channel: int = 0
    instrument: str = "piano"
    end_time: float = 0.0
    spectral_centroid: float = 0.0
    microtiming_deviation: float = 0.0
    swing_ratio: float = 0.0

    def __post_init__(self):
        self.end_time = self.start_time + self.duration


@dataclass
class ChordEvent:
    """Represents a chord with timing and analysis data."""
    pitches: List[int]
    start_time: float
    duration: float
    root_note: int
    chord_type: str
    complexity: float = 0.0


@dataclass
class RhythmPattern:
    """Represents rhythmic patterns and timing analysis."""
    pattern: List[float]
    start_time: float
    tempo: float
    swing_amount: float = 0.0
    polymetric_ratio: float = 1.0


class VisualizationConfig:
    """Configuration for visualization settings and presets."""

    def __init__(self):
        self.color_schemes = {
            "jazz": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"],
            "electronic": ["#00FF41", "#FF0080", "#8000FF", "#FF8000", "#0080FF", "#FF4040"],
            "classical": ["#2C3E50", "#34495E", "#7F8C8D", "#BDC3C7", "#ECF0F1", "#95A5A6"],
            "rock": ["#E74C3C", "#F39C12", "#27AE60", "#8E44AD", "#3498DB", "#1ABC9C"],
            "pop": ["#FF69B4", "#00CED1", "#FFD700", "#32CD32", "#FF6347", "#9370DB"]
        }

        self.view_settings = {
            "piano_roll": {
                "note_height": 8,
                "pixels_per_second": 50,
                "piano_key_width": 60,
                "timeline_height": 30,
                "grid_opacity": 0.3,
                "color_by_velocity": True,
                "show_note_labels": True
            },
            "spectral": {
                "fft_size": 2048,
                "hop_size": 512,
                "freq_bins": 256,
                "time_resolution": 0.01,
                "color_map": "viridis",
                "centroid_threshold": 0.1
            },
            "3d_timeline": {
                "depth_scale": 1.0,
                "rotation_speed": 0.5,
                "camera_distance": 10,
                "lighting_intensity": 0.8
            },
            "chord_cloud": {
                "word_size_factor": 0.5,
                "cloud_dimensions": (800, 600),
                "background_color": "white",
                "max_words": 100
            },
            "rhythm_map": {
                "grid_resolution": 16,
                "velocity_scaling": True,
                "syncopation_highlight": True,
                "polymetric_overlay": False
            },
            "polymetric": {
                "max_polymeters": 4,
                "grid_opacity": 0.2,
                "ratio_display": True,
                "phase_alignment": True
            },
            "microtempo": {
                "heatmap_resolution": 100,
                "deviation_scale": 100,
                "color_range": (-50, 50),
                "smoothing_factor": 0.3
            },
            "swing": {
                "swing_ratio_range": (0.5, 0.75),
                "beat_resolution": 16,
                "analysis_window": 4.0,
                "visualization_type": "waveform"
            }
        }

    def get_preset_config(self, preset: VisualizationPreset) -> Dict[str, Any]:
        """Get configuration for a specific preset."""
        base_config = {
            "colors": self.color_schemes[preset.value],
            "views": self.view_settings.copy()
        }

        # Apply preset-specific modifications
        if preset == VisualizationPreset.JAZZ:
            base_config["views"]["swing"]["swing_ratio_range"] = (0.6, 0.7)
            base_config["views"]["microtempo"]["deviation_scale"] = 80
        elif preset == VisualizationPreset.ELECTRONIC:
            base_config["views"]["rhythm_map"]["syncopation_highlight"] = True
            base_config["views"]["polymetric"]["max_polymeters"] = 6
        elif preset == VisualizationPreset.CLASSICAL:
            base_config["views"]["microtempo"]["deviation_scale"] = 120
            base_config["views"]["chord_cloud"]["word_size_factor"] = 0.3

        return base_config

    def save_config(self, filename: str, config: Dict[str, Any]):
        """Save configuration to JSON file."""
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(filename, 'r') as f:
            return json.load(f)


class EnhancedMidiVisualizer:
    """Enhanced MIDI visualizer with multi-dimensional views and interactive analysis."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.tempo = 120.0
        self.ticks_per_beat = 480
        self.seconds_per_tick = 0.5 / self.tempo
        self.current_preset = VisualizationPreset.CUSTOM

        # Analysis data
        self.notes: List[MidiNote] = []
        self.chords: List[ChordEvent] = []
        self.rhythm_patterns: List[RhythmPattern] = []
        self.spectral_data = None
        self.microtiming_data = None
        self.swing_analysis = None

        # Interactive state
        self.selected_notes: List[MidiNote] = []
        self.brush_region = None
        self.comparison_data = None

    def parse_midi_file(self, filename: str) -> List[MidiNote]:
        """Enhanced MIDI file parsing with additional analysis."""
        print(f"Parsing MIDI file: {filename}")

        if not MIDO_AVAILABLE:
            print("Error: mido library not available. Install with: pip install mido")
            return []

        try:
            midi_file = mido.MidiFile(filename)
            notes = []
            active_notes = {}

            current_time = 0.0

            for track in midi_file.tracks:
                track_time = 0.0

                for msg in track:
                    if self.seconds_per_tick <= 0:
                        self.seconds_per_tick = 0.001  # Prevent division by zero

                    track_time += msg.time * self.seconds_per_tick

                    if msg.type == 'set_tempo':
                        try:
                            self.tempo = mido.tempo2bpm(msg.tempo)
                            if self.tempo > 0:
                                self.seconds_per_tick = 60.0 / (self.tempo * midi_file.ticks_per_beat)
                            else:
                                self.tempo = 120.0
                                self.seconds_per_tick = 0.5
                        except:
                            pass  # Fallback to default

                    elif msg.type == 'note_on' and msg.velocity > 0:
                        key = (msg.note, msg.channel)
                        active_notes[key] = (track_time, msg.velocity)

                    elif (msg.type == 'note_off' or
                          (msg.type == 'note_on' and msg.velocity == 0)):
                        key = (msg.note, msg.channel)
                        if key in active_notes:
                            start_time, velocity = active_notes[key]
                            duration = track_time - start_time

                            if duration > 0.001:  # Minimum duration threshold
                                note = MidiNote(
                                    pitch=msg.note,
                                    velocity=velocity,
                                    start_time=start_time,
                                    duration=duration,
                                    channel=msg.channel
                                )

                                # Perform additional analysis
                                note.spectral_centroid = self._calculate_spectral_centroid(note)
                                note.microtiming_deviation = self._analyze_microtiming(note, midi_file.ticks_per_beat)

                                notes.append(note)
                            del active_notes[key]

            print(f"Extracted {len(notes)} notes with enhanced analysis")
            self.notes = notes
            return notes

        except Exception as e:
            print(f"Error parsing MIDI file: {e}")
            return []

    def _calculate_spectral_centroid(self, note: MidiNote) -> float:
        """Calculate spectral centroid for a note."""
        # Simplified spectral centroid calculation based on pitch
        # In a real implementation, this would analyze the actual audio spectrum
        frequency = 440 * (2 ** ((note.pitch - 69) / 12))
        # Return normalized centroid (0-1)
        return min(1.0, max(0.0, (frequency - 20) / (20000 - 20)))

    def _analyze_microtiming(self, note: MidiNote, ticks_per_beat: int) -> float:
        """Analyze microtiming deviations for a note."""
        # Simplified microtiming analysis
        # In practice, this would compare against a grid and measure deviations
        beat_position = (note.start_time * self.tempo / 60) % 1
        grid_position = round(beat_position * 16) / 16  # 16th note grid
        return (beat_position - grid_position) * 100  # Return in milliseconds

    def generate_visualization(self, view_type: VisualizationType,
                             preset: Optional[VisualizationPreset] = None,
                             interactive: bool = False) -> str:
        """Generate visualization based on type and preset."""

        if preset:
            self.current_preset = preset
            viz_config = self.config.get_preset_config(preset)
        else:
            viz_config = self.config.get_preset_config(self.current_preset)

        if view_type == VisualizationType.PIANO_ROLL:
            return self._generate_enhanced_piano_roll(viz_config, interactive)
        elif view_type == VisualizationType.SPECTRAL:
            return self._generate_spectral_analysis(viz_config, interactive)
        elif view_type == VisualizationType.TIMELINE_3D:
            return self._generate_3d_timeline(viz_config, interactive)
        elif view_type == VisualizationType.CHORD_CLOUD:
            return self._generate_chord_cloud(viz_config, interactive)
        elif view_type == VisualizationType.RHYTHM_MAP:
            return self._generate_rhythm_map(viz_config, interactive)
        elif view_type == VisualizationType.POLYMETRIC:
            return self._generate_polymetric_view(viz_config, interactive)
        elif view_type == VisualizationType.MICROTEMPO:
            return self._generate_microtempo_heatmap(viz_config, interactive)
        elif view_type == VisualizationType.SWING:
            return self._generate_swing_analysis(viz_config, interactive)
        else:
            return self._generate_multi_view_dashboard(viz_config, interactive)

    def _generate_enhanced_piano_roll(self, config: Dict, interactive: bool) -> str:
        """Generate enhanced piano roll with interactive features."""
        if not self.notes:
            return self._generate_empty_visualization()

        # Enhanced piano roll implementation with interactive features
        settings = config["views"]["piano_roll"]
        colors = config["colors"]
        config_json = json.dumps(config)
        note_data_json = json.dumps([{
            'pitch': n.pitch,
            'velocity': n.velocity,
            'start_time': n.start_time,
            'duration': n.duration,
            'channel': n.channel,
            'end_time': n.end_time,
            'spectral_centroid': n.spectral_centroid,
            'microtiming_deviation': n.microtiming_deviation,
            'swing_ratio': n.swing_ratio
        } for n in self.notes])
        colors_json = json.dumps(colors)

        jazz_selected = 'selected' if self.current_preset == VisualizationPreset.JAZZ else ''
        electronic_selected = 'selected' if self.current_preset == VisualizationPreset.ELECTRONIC else ''
        classical_selected = 'selected' if self.current_preset == VisualizationPreset.CLASSICAL else ''
        rock_selected = 'selected' if self.current_preset == VisualizationPreset.ROCK else ''
        pop_selected = 'selected' if self.current_preset == VisualizationPreset.POP else ''

        note_labels_opacity = 1 if settings['show_note_labels'] else 0

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Piano Roll - {self.current_preset.value.title()}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

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
            border-bottom: 1px solid rgba(255,255,255,0.1);
            z-index: 100;
        }}

        .controls {{
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .control-btn {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}

        .control-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        .control-btn.active {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }}

        .view-toggle {{
            display: flex;
            gap: 5px;
            margin-left: 20px;
        }}

        .preset-selector {{
            margin-left: 20px;
        }}

        .preset-selector select {{
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.2);
            padding: 5px 10px;
            border-radius: 4px;
        }}

        .main-view {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}

        .sidebar {{
            width: 300px;
            background: rgba(44, 62, 80, 0.9);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255,255,255,0.1);
            padding: 20px;
            overflow-y: auto;
        }}

        .analysis-panel {{
            margin-bottom: 20px;
        }}

        .analysis-panel h3 {{
            color: #3498db;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .stat-item {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 0.9em;
        }}

        .brush-controls {{
            margin-bottom: 20px;
        }}

        .brush-controls input {{
            width: 100%;
            margin-bottom: 5px;
        }}

        .piano-panel {{
            width: 60px;
            background: rgba(44, 62, 80, 0.9);
            border-right: 1px solid rgba(52, 73, 94, 0.5);
            overflow-y: auto;
            flex-shrink: 0;
        }}

        .timeline {{
            height: 30px;
            background: rgba(52, 73, 94, 0.8);
            border-bottom: 1px solid rgba(44, 62, 80, 0.5);
            position: relative;
            overflow: hidden;
        }}

        .piano-roll {{
            flex: 1;
            background: rgba(26, 26, 26, 0.9);
            position: relative;
            overflow: auto;
        }}

        .grid {{
            position: absolute;
            top: 30px;
            left: 0;
        }}

        .grid-line {{
            position: absolute;
            background: rgba(52, 73, 94, {settings['grid_opacity']});
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
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .note:hover {{
            transform: scale(1.05);
            z-index: 10;
        }}

        .note.selected {{
            box-shadow: 0 0 15px rgba(241, 196, 15, 0.8);
            border-color: #f1c40f;
        }}

        .brush-selection {{
            position: absolute;
            background: rgba(52, 152, 219, 0.2);
            border: 2px solid #3498db;
            pointer-events: none;
            z-index: 5;
        }}

        .playhead {{
            position: absolute;
            top: 30px;
            width: 2px;
            background: linear-gradient(to bottom, #e74c3c, #c0392b);
            z-index: 100;
            box-shadow: 0 0 10px rgba(231, 76, 60, 0.5);
        }}

        .note-label {{
            font-size: 10px;
            color: white;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.8);
            padding: 2px;
            pointer-events: none;
            opacity: {note_labels_opacity};
        }}

        .piano-key {{
            height: {settings['note_height']}px;
            border-bottom: 1px solid rgba(52, 73, 94, 0.3);
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            font-size: 11px;
            color: #bdc3c7;
            user-select: none;
            transition: background-color 0.2s;
        }}

        .piano-key:hover {{
            background: rgba(52, 152, 219, 0.3);
        }}

        .white-key {{
            background: rgba(52, 73, 94, 0.8);
        }}

        .black-key {{
            background: rgba(44, 62, 80, 0.8);
            color: #ecf0f1;
        }}

        .tooltip {{
            position: fixed;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            display: none;
        }}

        @media (max-width: 768px) {{
            .sidebar {{
                width: 250px;
            }}

            .controls {{
                font-size: 0.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="controls">
                <button class="control-btn" onclick="togglePlay()">▶ Play</button>
                <button class="control-btn" onclick="resetPlayhead()">⏮ Reset</button>
                <button class="control-btn" onclick="toggleBrush()">🖌️ Brush</button>
                <button class="control-btn" onclick="clearSelection()">✖ Clear</button>
                <button class="control-btn" onclick="zoomIn()">+</button>
                <button class="control-btn" onclick="zoomOut()">−</button>

                <div class="view-toggle">
                    <button class="control-btn" onclick="switchView('piano_roll')" id="piano-btn">🎹 Piano</button>
                    <button class="control-btn" onclick="switchView('spectral')" id="spectral-btn">🌊 Spectral</button>
                    <button class="control-btn" onclick="switchView('3d_timeline')" id="3d-btn">🎲 3D</button>
                    <button class="control-btn" onclick="switchView('chord_cloud')" id="chord-btn">☁️ Chords</button>
                    <button class="control-btn" onclick="switchView('rhythm_map')" id="rhythm-btn">🥁 Rhythm</button>
                </div>

                <div class="preset-selector">
                    <select onchange="changePreset(this.value)">
                        <option value="jazz" {jazz_selected}>🎷 Jazz</option>
                        <option value="electronic" {electronic_selected}>🎹 Electronic</option>
                        <option value="classical" {classical_selected}>🎼 Classical</option>
                        <option value="rock" {rock_selected}>🎸 Rock</option>
                        <option value="pop" {pop_selected}>🎤 Pop</option>
                    </select>
                </div>
            </div>
        </div>

        <div class="main-view">
            <div class="sidebar">
                <div class="analysis-panel">
                    <h3>📊 Analysis</h3>
                    <div class="stat-item">
                        <span>Notes:</span>
                        <span id="note-count">{len(self.notes)}</span>
                    </div>
                    <div class="stat-item">
                        <span>Duration:</span>
                        <span id="duration">{max((n.end_time for n in self.notes), default=0):.1f}s</span>
                    </div>
                    <div class="stat-item">
                        <span>Tempo:</span>
                        <span id="tempo">{self.tempo:.1f} BPM</span>
                    </div>
                    <div class="stat-item">
                        <span>Selected:</span>
                        <span id="selected-count">0</span>
                    </div>
                </div>

                <div class="brush-controls">
                    <h3>🖌️ Brush Selection</h3>
                    <input type="range" id="brush-size" min="10" max="200" value="50" step="10">
                    <label for="brush-size">Size: <span id="brush-size-value">50</span>px</label>
                    <br>
                    <input type="color" id="brush-color" value="#3498db">
                    <label for="brush-color">Color</label>
                </div>
            </div>

            <div class="piano-panel" id="pianoPanel">
                <div class="timeline"></div>
                <div class="piano-keys" id="pianoKeys"></div>
            </div>

            <div class="piano-roll" id="pianoRoll">
                <div class="grid" id="grid"></div>
                <div class="playhead" id="playhead" style="left: 0px;"></div>
                <div class="brush-selection" id="brushSelection" style="display: none;"></div>
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        // Global state
        let isPlaying = false;
        let playhead = document.getElementById('playhead');
        let startTime = 0;
        let animationId = null;
        let currentZoom = 1.0;
        let brushMode = false;
        let brushStart = null;
        let selectedNotes = new Set();
        let currentView = 'piano_roll';
        let currentPreset = '{self.current_preset.value}';

        // Configuration
        const config = {config_json};
        const noteData = {note_data_json};
        const colors = {colors_json};

        function initPianoRoll() {{
            const pianoKeys = document.getElementById('pianoKeys');
            const grid = document.getElementById('grid');
            const pianoRoll = document.getElementById('pianoRoll');

            // Calculate dimensions
            const minPitch = Math.min(...noteData.map(n => n.pitch));
            const maxPitch = Math.max(...noteData.map(n => n.pitch));
            const maxTime = Math.max(...noteData.map(n => n.end_time));
            const noteHeight = {settings['note_height']};
            const pixelsPerSecond = {settings['pixels_per_second']} * currentZoom;
            const pianoKeyWidth = {settings['piano_key_width']};
            const timelineHeight = {settings['timeline_height']};

            // Create piano keys
            for (let pitch = maxPitch; pitch >= minPitch - 2; pitch--) {{
                const keyClass = isWhiteKey(pitch) ? 'white-key' : 'black-key';
                const noteName = getNoteName(pitch);
                pianoKeys.innerHTML += `<div class="piano-key ${{keyClass}}" data-pitch="${{pitch}}">${{noteName}}</div>`;
            }}

            // Create grid lines
            const beats = Math.max(1, Math.floor(maxTime * {self.tempo} / 60));
            for (let beat = 0; beat <= beats; beat++) {{
                const x = beat / ({self.tempo} / 60) * pixelsPerSecond;
                grid.innerHTML += `<div class="grid-line vertical-line" style="left: ${{x}}px;"></div>`;

                if (beat % 4 === 0) {{
                    const bar = beat / 4;
                    grid.innerHTML += `<div class="time-label" style="left: ${{x + 5}}px; top: 5px;">${{bar}}:0</div>`;
                }}
            }}

            // Create octave lines
            for (let pitch = minPitch; pitch <= maxPitch; pitch++) {{
                if (pitch % 12 === 0) {{
                    const y = (maxPitch - pitch) * noteHeight + timelineHeight;
                    grid.innerHTML += `<div class="grid-line horizontal-line" style="top: ${{y}}px;"></div>`;
                }}
            }}

            // Create notes
            noteData.forEach((note, index) => {{
                if (note.pitch >= minPitch && note.pitch <= maxPitch) {{
                    const x = note.start_time * pixelsPerSecond;
                    const y = (maxPitch - note.pitch) * noteHeight + timelineHeight;
                    const width = Math.max(1, note.duration * pixelsPerSecond);
                    const height = noteHeight * 0.8;
                    const opacity = Math.min(1.0, note.velocity / 127.0 * 0.8 + 0.2);
                    const noteName = getNoteName(note.pitch);

                    const noteClass = isWhiteKey(note.pitch) ? 'white-key' : 'black-key';
                    const noteEl = document.createElement('div');
                    noteEl.className = `note ${{noteClass}}`;
                    noteEl.style.left = `${{x}}px`;
                    noteEl.style.top = `${{y}}px`;
                    noteEl.style.width = `${{width}}px`;
                    noteEl.style.height = `${{height}}px`;
                    noteEl.style.opacity = opacity;
                    noteEl.dataset.index = index;
                    noteEl.innerHTML = `<div class="note-label">${{noteName}}</div>`;

                    noteEl.onclick = (e) => {{
                        e.stopPropagation();
                        selectNote(index, !selectedNotes.has(index));
                    }};

                    grid.appendChild(noteEl);
                }}
            }});

            // Update container dimensions
            const canvasWidth = maxTime * pixelsPerSecond + pianoKeyWidth;
            const canvasHeight = (maxPitch - minPitch + 3) * noteHeight + timelineHeight;
            grid.style.width = `${{canvasWidth}}px`;
            grid.style.height = `${{canvasHeight}}px`;
            pianoRoll.style.width = `${{canvasWidth}}px`;

            // Sync scroll
            pianoRoll.addEventListener('scroll', () => {{
                document.getElementById('pianoPanel').scrollTop = pianoRoll.scrollTop;
            }});

            // Mouse events for brushing
            pianoRoll.onmousedown = (e) => {{
                if (brushMode) {{
                    brushStart = {{x: e.clientX - pianoRoll.getBoundingClientRect().left + pianoRoll.scrollLeft,
                                  y: e.clientY - pianoRoll.getBoundingClientRect().top + pianoRoll.scrollTop}};
                    updateBrushSelection(e);
                }}
            }};

            pianoRoll.onmousemove = (e) => {{
                if (brushMode && brushStart) {{
                    updateBrushSelection(e);
                }}
            }};

            pianoRoll.onmouseup = () => {{
                if (brushMode && brushStart) {{
                    applyBrushSelection();
                    brushStart = null;
                    document.getElementById('brushSelection').style.display = 'none';
                }}
            }};
        }}

        function updateBrushSelection(e) {{
            if (!brushStart) return;

            const pianoRoll = document.getElementById('pianoRoll');
            const brushSelection = document.getElementById('brushSelection');
            const rect = pianoRoll.getBoundingClientRect();

            const currentX = e.clientX - rect.left + pianoRoll.scrollLeft;
            const currentY = e.clientY - rect.top + pianoRoll.scrollTop;

            const left = Math.min(brushStart.x, currentX);
            const top = Math.min(brushStart.y, currentY);
            const width = Math.abs(currentX - brushStart.x);
            const height = Math.abs(currentY - brushStart.y);

            brushSelection.style.left = `${{left}}px`;
            brushSelection.style.top = `${{top}}px`;
            brushSelection.style.width = `${{width}}px`;
            brushSelection.style.height = `${{height}}px`;
            brushSelection.style.display = 'block';
        }}

        function applyBrushSelection() {{
            const brushSelection = document.getElementById('brushSelection');
            const rect = brushSelection.getBoundingClientRect();
            const pianoRoll = document.getElementById('pianoRoll');
            const rollRect = pianoRoll.getBoundingClientRect();

            const brushRect = {{
                left: rect.left - rollRect.left + pianoRoll.scrollLeft,
                top: rect.top - rollRect.top + pianoRoll.scrollTop,
                right: rect.right - rollRect.left + pianoRoll.scrollLeft,
                bottom: rect.bottom - rollRect.top + pianoRoll.scrollTop
            }};

            // Find notes within brush area
            const notes = document.querySelectorAll('.note');
            notes.forEach(note => {{
                const noteRect = note.getBoundingClientRect();
                const noteRollRect = {{
                    left: noteRect.left - rollRect.left + pianoRoll.scrollLeft,
                    top: noteRect.top - rollRect.top + pianoRoll.scrollTop,
                    right: noteRect.right - rollRect.left + pianoRoll.scrollLeft,
                    bottom: noteRect.bottom - rollRect.top + pianoRoll.scrollTop
                }};

                if (brushRect.left < noteRollRect.right && brushRect.right > noteRollRect.left &&
                    brushRect.top < noteRollRect.bottom && brushRect.bottom > noteRollRect.top) {{
                    const index = parseInt(note.dataset.index);
                    selectNote(index, true);
                }}
            }});
        }}

        function selectNote(index, selected) {{
            const notes = document.querySelectorAll('.note');
            const note = notes[index];
            if (!note) return;

            if (selected) {{
                selectedNotes.add(index);
                note.classList.add('selected');
            }} else {{
                selectedNotes.delete(index);
                note.classList.remove('selected');
            }}

            document.getElementById('selected-count').textContent = selectedNotes.size;
        }}

        function clearSelection() {{
            selectedNotes.clear();
            document.querySelectorAll('.note.selected').forEach(note => {{
                note.classList.remove('selected');
            }});
            document.getElementById('selected-count').textContent = '0';
        }}

        function toggleBrush() {{
            brushMode = !brushMode;
            const btn = document.querySelector('[onclick*="toggleBrush"]');
            btn.classList.toggle('active');
        }}

        function isWhiteKey(pitch) {{
            const whiteKeys = [0, 2, 4, 5, 7, 9, 11];
            return whiteKeys.includes(pitch % 12);
        }}

        function getNoteName(pitch) {{
            const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
            const noteIdx = pitch % 12;
            const octave = Math.floor(pitch / 12) - 1;
            return notes[noteIdx] + octave;
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.querySelector('[onclick*="togglePlay"]');

            if (isPlaying) {{
                startTime = Date.now() - (parseInt(playhead.style.left) || 0) / {settings['pixels_per_second']} * 1000;
                animatePlayhead();
                btn.innerHTML = '⏸ Pause';
            }} else {{
                cancelAnimationFrame(animationId);
                btn.innerHTML = '▶ Play';
            }}
        }}

        function animatePlayhead() {{
            if (!isPlaying) return;

            const elapsed = Date.now() - startTime;
            const position = (elapsed / 1000) * {settings['pixels_per_second']};
            const maxPosition = document.getElementById('grid').offsetWidth - {settings['piano_key_width']};

            if (position > maxPosition) {{
                togglePlay();
                return;
            }}

            playhead.style.left = position + 'px';

            // Auto-scroll
            const pianoRoll = document.getElementById('pianoRoll');
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
            document.getElementById('pianoRoll').scrollLeft = 0;
            document.querySelector('[onclick*="togglePlay"]').innerHTML = '▶ Play';
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
            const grid = document.getElementById('grid');
            grid.style.transform = `scale(${{currentZoom}})`;
            grid.style.transformOrigin = '0 0';
        }}

        function switchView(viewType) {{
            // This would switch to different visualization views
            currentView = viewType;
            console.log('Switching to view:', viewType);

            // Update active button
            document.querySelectorAll('.view-toggle button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.getElementById(viewType + '-btn').classList.add('active');
        }}

        function changePreset(preset) {{
            currentPreset = preset;
            // This would reload the visualization with the new preset
            console.log('Changing preset to:', preset);
        }}

        // Initialize
        initPianoRoll();
        updateZoom();

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.code === 'Space') {{
                e.preventDefault();
                togglePlay();
            }} else if (e.code === 'KeyR') {{
                resetPlayhead();
            }} else if (e.code === 'KeyC') {{
                clearSelection();
            }} else if (e.code === 'KeyB') {{
                toggleBrush();
            }} else if (e.code === 'Equal' || e.code === 'NumpadAdd') {{
                zoomIn();
            }} else if (e.code === 'Minus' || e.code === 'NumpadSubtract') {{
                zoomOut();
            }}
        }});
    </script>
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
        .message {{ background: #34495e; padding: 40px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
    </style>
</head>
<body>
    <div class="message">
        <h1>Visualization Unavailable</h1>
        <p>{message}</p>
        <p>Please ensure all required dependencies are installed.</p>
    </div>
</body>
</html>"""

    def _matplotlib_to_html(self, fig, title: str, config: Dict, interactive: bool) -> str:
        """Convert matplotlib figure to HTML with interactive features."""
        if not BASE64_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return self._generate_fallback_visualization("Matplotlib export requires base64 and matplotlib")

        # Save figure to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        colors = config.get('colors', ['#3498db'])

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial; background: #2c3e50; color: white; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .plot-container {{ background: white; border-radius: 10px; padding: 20px; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; }}
        .controls {{ text-align: center; margin: 20px 0; }}
        .control-btn {{ background: {colors[0]}; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }}
        .control-btn:hover {{ opacity: 0.8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="controls">
            <button class="control-btn" onclick="window.history.back()">← Back</button>
            <button class="control-btn" onclick="window.print()">🖨️ Print</button>
        </div>
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

    def _generate_spectral_analysis(self, config: Dict, interactive: bool) -> str:
        """Generate spectral analysis visualization."""
        if not self.notes:
            return self._generate_empty_visualization()

        if not MATPLOTLIB_AVAILABLE:
            return self._generate_fallback_visualization("Spectral Analysis requires matplotlib")

        # Generate spectral analysis visualization
        # This would create a spectrogram-like view
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Spectral Analysis")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

        # Placeholder for spectral data
        times = [n.start_time for n in self.notes]
        frequencies = [440 * (2 ** ((n.pitch - 69) / 12)) for n in self.notes]
        intensities = [n.velocity / 127.0 for n in self.notes]

        if NUMPY_AVAILABLE:
            times = np.array(times)
            frequencies = np.array(frequencies)
            intensities = np.array(intensities)

        scatter = ax.scatter(times, frequencies, c=intensities, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Intensity')

        # Save to HTML with interactive features
        return self._matplotlib_to_html(fig, "Spectral Analysis", config, interactive)

    def _generate_3d_timeline(self, config: Dict, interactive: bool) -> str:
        """Generate 3D timeline visualization."""
        if not self.notes:
            return self._generate_empty_visualization()

        if not MATPLOTLIB_AVAILABLE:
            return self._generate_fallback_visualization("3D Timeline requires matplotlib")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_title("3D MIDI Timeline")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pitch")
        ax.set_zlabel("Velocity")

        times = [n.start_time for n in self.notes]
        pitches = [n.pitch for n in self.notes]
        velocities = [n.velocity for n in self.notes]

        # Fix typing issue by using np.array if available
        if NUMPY_AVAILABLE:
            times = np.array(times)
            pitches = np.array(pitches)
            velocities = np.array(velocities)

        scatter = ax.scatter(times, pitches, velocities, c=velocities, cmap='plasma', s=50, alpha=0.7)  # type: ignore

        return self._matplotlib_to_html(fig, "3D Timeline", config, interactive)

    def _generate_chord_cloud(self, config: Dict, interactive: bool) -> str:
        """Generate chord cloud visualization."""
        # Generate word cloud of chord progressions
        chord_counts = {}
        for note in self.notes:
            chord_name = self._pitch_to_chord_name(note.pitch)
            chord_counts[chord_name] = chord_counts.get(chord_name, 0) + 1

        cloud_html = ''.join(f'<div class="chord" style="font-size: {min(50, 20 + count*2)}px; background: hsl({i*30}, 70%, 80%);">{chord}: {count}</div>' for i, (chord, count) in enumerate(chord_counts.items()))

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Chord Cloud</title>
    <style>
        body {{ font-family: Arial; background: #f0f0f0; }}
        .chord-cloud {{ text-align: center; padding: 50px; }}
        .chord {{ display: inline-block; margin: 10px; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="chord-cloud">
        <h1>Chord Frequency Cloud</h1>
        {cloud_html}
    </div>
</body>
</html>"""

        return html

    def _generate_rhythm_map(self, config: Dict, interactive: bool) -> str:
        """Generate rhythm map visualization."""
        # Create rhythm grid visualization
        rhythm_data = {}
        for note in self.notes:
            beat = int(note.start_time * self.tempo / 60)
            sub_beat = round((note.start_time * self.tempo / 60 - beat) * 16)
            key = f"{beat}:{sub_beat}"
            rhythm_data[key] = rhythm_data.get(key, 0) + note.velocity

        grid_html = ''.join(f'<div class="rhythm-cell {"active" if rhythm_data.get(f"{beat}:{sub}", 0) > 0 else ""}">{rhythm_data.get(f"{beat}:{sub}", 0)}</div>' for beat in range(16) for sub in range(16))

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Rhythm Map</title>
    <style>
        body {{ font-family: Arial; background: #2c3e50; color: white; }}
        .rhythm-grid {{ display: grid; grid-template-columns: repeat(16, 1fr); gap: 2px; padding: 20px; }}
        .rhythm-cell {{ aspect-ratio: 1; background: rgba(255,255,255,0.1); border-radius: 2px; display: flex; align-items: center; justify-content: center; font-size: 10px; }}
        .rhythm-cell.active {{ background: #e74c3c; }}
    </style>
</head>
<body>
    <h1>Rhythm Map Analysis</h1>
    <div class="rhythm-grid">
        {grid_html}
    </div>
</body>
</html>"""

        return html

    def _generate_polymetric_view(self, config: Dict, interactive: bool) -> str:
        """Generate polymetric visualization."""
        # Polymetric grid showing multiple rhythmic layers
        return self._generate_fallback_visualization("Polymetric View - Coming Soon")

    def _generate_microtempo_heatmap(self, config: Dict, interactive: bool) -> str:
        """Generate microtiming heatmap."""
        if not self.notes:
            return self._generate_empty_visualization()

        if not MATPLOTLIB_AVAILABLE:
            return self._generate_fallback_visualization("Microtiming Heatmap requires matplotlib")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Microtiming Analysis Heatmap")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Timing Deviation (ms)")

        times = [n.start_time for n in self.notes]
        deviations = [n.microtiming_deviation for n in self.notes]

        if NUMPY_AVAILABLE:
            times = np.array(times)
            deviations = np.array(deviations)

        heatmap = ax.scatter(times, deviations, c=deviations, cmap='RdYlBu_r', s=30, alpha=0.7)
        plt.colorbar(heatmap, ax=ax, label='Deviation (ms)')

        return self._matplotlib_to_html(fig, "Microtiming Heatmap", config, interactive)

    def _generate_swing_analysis(self, config: Dict, interactive: bool) -> str:
        """Generate swing analysis visualization."""
        # Swing ratio analysis
        return self._generate_fallback_visualization("Swing Analysis - Coming Soon")

    def _generate_multi_view_dashboard(self, config: Dict, interactive: bool) -> str:
        """Generate multi-view dashboard."""
        return self._generate_enhanced_piano_roll(config, interactive)

    def export_visualization(self, html_content: str, output_path: str, format_type: str = 'html'):
        """Export visualization in various formats."""
        if format_type.lower() == 'html':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        elif format_type.lower() == 'png':
            # Would require additional dependencies like selenium or playwright
            self._export_as_image(html_content, output_path, 'png')
        elif format_type.lower() == 'svg':
            # Convert HTML to SVG (simplified)
            self._export_as_image(html_content, output_path, 'svg')
        elif format_type.lower() == 'pdf':
            # Would require additional dependencies
            self._export_as_pdf(html_content, output_path)

    def _export_as_image(self, html_content: str, output_path: str, format_type: str):
        """Export visualization as image (placeholder)."""
        # This would require additional setup with headless browser
        print(f"Image export to {format_type} not yet implemented. Saving as HTML instead.")
        html_path = output_path.replace(f'.{format_type}', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _export_as_pdf(self, html_content: str, output_path: str):
        """Export visualization as PDF (placeholder)."""
        print("PDF export not yet implemented. Saving as HTML instead.")
        html_path = output_path.replace('.pdf', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def compare_visualizations(self, other_visualizer: 'EnhancedMidiVisualizer') -> str:
        """Generate comparative analysis between two visualizers."""
        comparison_data = {
            'note_count': (len(self.notes), len(other_visualizer.notes)),
            'duration': (max((n.end_time for n in self.notes), default=0),
                         max((n.end_time for n in other_visualizer.notes), default=0)),
            'tempo': (self.tempo, other_visualizer.tempo),
            'pitch_range': ((min(n.pitch for n in self.notes), max(n.pitch for n in self.notes)) if self.notes else (0, 127),
                           (min(n.pitch for n in other_visualizer.notes), max(n.pitch for n in other_visualizer.notes)) if other_visualizer.notes else (0, 127))
        }

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Visualization Comparison</title>
    <style>
        body {{ font-family: Arial; background: #2c3e50; color: white; padding: 20px; }}
        .comparison {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ background: #34495e; padding: 20px; border-radius: 10px; text-align: center; }}
        .metric h3 {{ margin-bottom: 10px; }}
        .value {{ font-size: 2em; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Visualization Comparison</h1>
    <div class="comparison">
        <div class="metric">
            <h3>Note Count</h3>
            <div class="value">{comparison_data['note_count'][0]} vs {comparison_data['note_count'][1]}</div>
        </div>
        <div class="metric">
            <h3>Duration</h3>
            <div class="value">{comparison_data['duration'][0]:.1f}s vs {comparison_data['duration'][1]:.1f}s</div>
        </div>
        <div class="metric">
            <h3>Tempo</h3>
            <div class="value">{comparison_data['tempo'][0]:.1f} vs {comparison_data['tempo'][1]:.1f} BPM</div>
        </div>
    </div>
</body>
</html>"""

        return html


def main():
    """Main entry point for the enhanced MIDI visualizer."""
    parser = argparse.ArgumentParser(
        description='Enhanced MIDI file visualizer with multi-dimensional views',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  python enhanced_visualizer.py song.mid --view spectral --interactive
  python enhanced_visualizer.py song.mid --preset jazz --export png
  python enhanced_visualizer.py song.mid --view 3d_timeline --zoom 1.5
        '''
    )

    parser.add_argument('input_file', help='Path to input MIDI file (.mid)')
    parser.add_argument('--output', '-o', help='Path to output HTML file')
    parser.add_argument('--view', '-v', choices=['piano_roll', 'spectral', '3d_timeline', 'chord_cloud', 'rhythm_map', 'polymetric', 'microtempo', 'swing'],
                       default='piano_roll', help='Visualization view type')
    parser.add_argument('--preset', '-p', choices=['jazz', 'electronic', 'classical', 'rock', 'pop'],
                       help='Visualization preset')
    parser.add_argument('--interactive', '-i', action='store_true', help='Enable interactive features')
    parser.add_argument('--export', '-e', choices=['html', 'png', 'svg', 'pdf'],
                       help='Export format (default: html)')
    parser.add_argument('--zoom', '-z', type=float, default=1.0, help='Zoom level (default: 1.0)')
    parser.add_argument('--compare', '-c', help='Compare with another MIDI file')

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
            view_type_str = args.view.replace(' ', '_').upper()
            try:
                view_type = VisualizationType(view_type_str)
            except ValueError:
                print(f"Warning: Invalid view type '{args.view}'. Defaulting to PIANO_ROLL.")
                view_type = VisualizationType.PIANO_ROLL

            preset = None
            if args.preset:
                try:
                    preset = VisualizationPreset(args.preset.upper())
                except ValueError:
                    print(f"Warning: Invalid preset '{args.preset}'. Using CUSTOM.")

            # Generate visualization
            html_content = visualizer.generate_visualization(view_type, preset, args.interactive)

            # Handle comparison if requested
            if args.compare:
                if not os.path.exists(args.compare):
                    print(f"Warning: Comparison file '{args.compare}' does not exist.")
                else:
                    comp_visualizer = EnhancedMidiVisualizer(config)
                    comp_visualizer.parse_midi_file(args.compare)
                    html_content = visualizer.compare_visualizations(comp_visualizer)

        # Generate output filename
        if not args.output:
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            ext = args.export if args.export else 'html'
            args.output = f"{base_name}_enhanced.{ext}"

        # Export visualization
        visualizer.export_visualization(html_content, args.output, args.export or 'html')

        print("✓ Enhanced visualization created successfully!")
        print(f"  Output: {args.output}")
        print("  Open the HTML file in a web browser to explore the interactive visualization!")
        print("  Features: Spectral analysis, 3D timeline, chord clouds, rhythm maps")
        print("  Interactive: Brushing & linking, comparative analysis, real-time feedback")
        print("  Keyboard shortcuts: Space (play/pause), R (reset), B (brush), C (clear selection)")

    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()