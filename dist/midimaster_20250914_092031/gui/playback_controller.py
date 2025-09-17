"""
Playback Controller for MIDI Master GUI

This module provides playback controls for MIDI files with support for
basic MIDI playback using pygame (if available) or mido for MIDI parsing.
"""

import sys
import os
import threading
import time
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QVBoxLayout
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import playback libraries
try:
    import pygame
    import pygame.midi
    pygame.midi.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available, using basic MIDI playback")

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("mido not available, MIDI playback disabled")


class PlaybackController(QWidget):
    """
    Widget providing playback controls for MIDI files.
    """

    # Signals
    playRequested = pyqtSignal()
    pauseRequested = pyqtSignal()
    stopRequested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.song_skeleton = None
        self.is_playing = False
        self.is_paused = False
        self.current_time = 0
        self.playback_thread = None
        self.playback_timer = None

        self._init_ui()

        if not MIDO_AVAILABLE:
            self._disable_playback("MIDI library (mido) not available")

    def _init_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)

        # Playback buttons
        buttons_layout = QVBoxLayout()

        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self._on_play_clicked)
        buttons_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.clicked.connect(self._on_pause_clicked)
        self.pause_button.setEnabled(False)
        buttons_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)

        layout.addLayout(buttons_layout)

        # Volume control
        volume_layout = QVBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        volume_layout.addWidget(self.volume_slider)
        layout.addLayout(volume_layout)

        # Time display
        time_layout = QVBoxLayout()
        time_layout.addWidget(QLabel("Time:"))
        self.time_label = QLabel("00:00")
        time_layout.addWidget(self.time_label)
        layout.addLayout(time_layout)

        # Loop toggle
        self.loop_checkbox = QPushButton("🔄 Loop")
        self.loop_checkbox.setCheckable(True)
        self.loop_checkbox.clicked.connect(self._on_loop_toggled)
        layout.addWidget(self.loop_checkbox)

        layout.addStretch()

    def set_song_skeleton(self, song_skeleton):
        """Set the song skeleton for playback."""
        self.song_skeleton = song_skeleton
        self._update_ui_state()

    def _update_ui_state(self):
        """Update button states based on current playback status."""
        has_song = self.song_skeleton is not None

        self.play_button.setEnabled(has_song and MIDO_AVAILABLE)
        self.pause_button.setEnabled(self.is_playing)
        self.stop_button.setEnabled(self.is_playing or self.is_paused)

        if has_song and MIDO_AVAILABLE:
            self.play_button.setText("▶ Play" if not self.is_playing else "⏸ Pause")
        else:
            self.play_button.setText("▶ Play (No Song)" if not has_song else "▶ Play (No MIDI)")

    def _on_play_clicked(self):
        """Handle play button click."""
        if not self.song_skeleton:
            return

        if self.is_playing:
            self.pause()
        else:
            self.play()

    def _on_pause_clicked(self):
        """Handle pause button click."""
        self.pause()

    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.stop()

    def _on_volume_changed(self, value):
        """Handle volume slider change."""
        if PYGAME_AVAILABLE and self.is_playing:
            # Note: pygame volume control might not work as expected for MIDI
            # Volume control for pygame.midi is not directly available
            pass

    def _on_loop_toggled(self, checked):
        """Handle loop toggle."""
        # Loop functionality would need additional implementation
        pass

    def play(self):
        """Start playback."""
        if not self.song_skeleton or not MIDO_AVAILABLE:
            return

        self.playRequested.emit()

        if self.is_paused:
            # Resume playback
            self.is_paused = False
            self.is_playing = True
        else:
            # Start new playback
            self.is_playing = True
            self.is_paused = False
            self.current_time = 0

            # Start playback thread
            self.playback_thread = threading.Thread(target=self._playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()

            # Start timer for UI updates
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self._update_time_display)
            self.playback_timer.start(100)  # Update every 100ms

        self._update_ui_state()

    def pause(self):
        """Pause playback."""
        if not self.is_playing:
            return

        self.pauseRequested.emit()
        self.is_paused = True
        self.is_playing = False

        if self.playback_timer:
            self.playback_timer.stop()

        self._update_ui_state()

    def stop(self):
        """Stop playback."""
        self.stopRequested.emit()

        self.is_playing = False
        self.is_paused = False
        self.current_time = 0

        if self.playback_timer:
            self.playback_timer.stop()
            self.playback_timer = None

        self.playback_thread = None
        self._update_time_display()
        self._update_ui_state()

    def _playback_worker(self):
        """Background worker for MIDI playback."""
        try:
            if not self.song_skeleton:
                return

            # Simple MIDI playback using timing
            # This is a basic implementation - real MIDI playback would be more complex

            # Calculate total duration (rough estimate) using section type targets
            def _get_section_target_length_beats(section_type) -> float:
                mapping = {
                    'intro': 16.0,         # 4 bars
                    'verse': 32.0,         # 8 bars
                    'pre_chorus': 16.0,    # 4 bars
                    'chorus': 32.0,        # 8 bars
                    'post_chorus': 16.0,   # 4 bars
                    'bridge': 24.0,        # 6 bars
                    'solo': 32.0,          # 8 bars
                    'fill': 4.0,           # 1 bar
                    'outro': 16.0          # 4 bars
                }
                key = getattr(section_type, "value", str(section_type))
                return mapping.get(key, 32.0)

            total_beats = 0.0
            for section_type, patterns in self.song_skeleton.sections:
                total_beats += _get_section_target_length_beats(section_type)

            # Assume 120 BPM default, calculate duration in seconds
            tempo = getattr(self.song_skeleton, 'tempo', 120)
            beats_per_second = tempo / 60.0
            total_duration = total_beats / beats_per_second

            # Simulate playback timing
            start_time = time.time()
            while self.is_playing and not self.is_paused:
                elapsed = time.time() - start_time
                self.current_time = min(elapsed, total_duration)

                if elapsed >= total_duration:
                    if self.loop_checkbox.isChecked():
                        # Loop back to beginning
                        start_time = time.time()
                        self.current_time = 0
                    else:
                        # Stop at end
                        break

                time.sleep(0.01)  # Small sleep to prevent busy waiting

            # Playback finished
            if self.is_playing:  # Only if not manually stopped
                self.stop()

        except Exception as e:
            print(f"Playback error: {e}")
            self.stop()

    def _update_time_display(self):
        """Update the time display label."""
        if self.is_playing or self.is_paused:
            minutes = int(self.current_time // 60)
            seconds = int(self.current_time % 60)
            self.time_label.setText(f"{minutes:02d}:{seconds:02d}")
        else:
            self.time_label.setText("00:00")

    def _disable_playback(self, reason):
        """Disable playback functionality."""
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.volume_slider.setEnabled(False)
        self.time_label.setText(reason)

        # Update button text to indicate disabled state
        self.play_button.setText("▶ Play (Disabled)")
        self.pause_button.setText("⏸ Pause (Disabled)")
        self.stop_button.setText("⏹ Stop (Disabled)")

    def get_midi_events(self):
        """Extract MIDI events from song skeleton for playback."""
        if not self.song_skeleton:
            return []

        events = []
        current_time = 0

        # Helper for section target length (beats)
        def _get_section_target_length_beats(section_type) -> float:
            mapping = {
                'intro': 16.0, 'verse': 32.0, 'pre_chorus': 16.0, 'chorus': 32.0,
                'post_chorus': 16.0, 'bridge': 24.0, 'solo': 32.0, 'fill': 4.0, 'outro': 16.0
            }
            key = getattr(section_type, "value", str(section_type))
            return mapping.get(key, 32.0)

        for section_type, patterns in self.song_skeleton.sections:
            if not patterns:
                continue

            for pattern in patterns:
                if not hasattr(pattern, 'notes') or not pattern.notes:
                    continue

                for note in pattern.notes:
                    # Note on event
                    events.append({
                        'type': 'note_on',
                        'time': current_time + note.start_time,
                        'note': note.pitch,
                        'velocity': note.velocity,
                        'channel': self._get_channel_for_pattern(pattern.pattern_type)
                    })

                    # Note off event
                    events.append({
                        'type': 'note_off',
                        'time': current_time + note.start_time + note.duration,
                        'note': note.pitch,
                        'velocity': 0,
                        'channel': self._get_channel_for_pattern(pattern.pattern_type)
                    })

            # Move to next section by target length
            current_time += _get_section_target_length_beats(section_type)

        # Sort events by time
        events.sort(key=lambda x: x['time'])
        return events

    def _get_channel_for_pattern(self, pattern_type):
        """Get MIDI channel for different pattern types."""
        channel_map = {
            'melody': 0,      # Piano
            'harmony': 1,     # Piano (different channel)
            'bass': 2,        # Bass
            'rhythm': 9       # Drums (channel 10 in MIDI, but 9 in 0-based)
        }

        # Handle both string and PatternType enum
        if hasattr(pattern_type, 'value'):
            pattern_name = pattern_type.value
        else:
            pattern_name = str(pattern_type).lower()

        return channel_map.get(pattern_name, 0)