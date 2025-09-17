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
    PYGAME_MIDI_AVAILABLE = True
except ImportError:
    PYGAME_MIDI_AVAILABLE = False
    print("pygame.midi not available, using simulated playback")

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
    positionUpdated = pyqtSignal(float)  # current beat position

    def __init__(self):
        self.pygame_midi_available = PYGAME_MIDI_AVAILABLE
        super().__init__()
        self.song_skeleton = None
        self.is_playing = False
        self.is_paused = False
        self.current_beat = 0.0
        self.playback_thread = None
        self.playback_timer = None
        self.midi_output = None
        self.active_notes = {}  # (channel, note): start_time
        self.current_volume = 70  # 0-100
        self.tempo = 120  # default BPM
        self.events = []
        self.start_time = 0.0
        self.paused_time = 0.0

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
        if song_skeleton:
            self.tempo = getattr(song_skeleton, 'tempo', 120)
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
        self.current_volume = value
        if self.pygame_midi_available and self.midi_output and self.is_playing:
            # Send MIDI volume change (CC 7) for all channels
            volume_midi = int((value / 100.0) * 127)
            for channel in range(16):
                self.midi_output.write_short(0xB0 | channel, 7, volume_midi)  # CC 7: volume

    def _on_loop_toggled(self, checked):
        """Handle loop toggle."""
        # Loop functionality handled in worker
        pass

    def play(self):
        """Start playback."""
        if not self.song_skeleton or not MIDO_AVAILABLE:
            return

        self.playRequested.emit()

        if self.is_paused:
            # Resume from paused position
            self.is_paused = False
            self.is_playing = True
            self.start_time += time.time() - self.paused_time
            if self.playback_timer:
                self.playback_timer.start(100)
        else:
            # Start new playback
            self.is_playing = True
            self.is_paused = False
            self.current_beat = 0.0
            self.events = self.get_midi_events()
            self.active_notes.clear()

            if self.pygame_midi_available:
                try:
                    self.midi_output = pygame.midi.Output(0)  # Default device
                    # Send initial volume for all channels
                    volume_midi = int((self.current_volume / 100.0) * 127)
                    for channel in range(16):
                        self.midi_output.write_short(0xB0 | channel, 7, volume_midi)
                except Exception as e:
                    print(f"MIDI output init error: {e}")
                    self.pygame_midi_available = False  # Fallback

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
        if self.is_playing:
            self.is_paused = True
            self.is_playing = False
            self.paused_time = time.time()
            if self.pygame_midi_available and self.midi_output:
                # Send note off for active notes
                for (ch, note), _ in self.active_notes.items():
                    self.midi_output.write_short(0x80 | ch, note, 0)
                self.active_notes.clear()
            if self.playback_timer:
                self.playback_timer.stop()

        self._update_ui_state()

    def stop(self):
        """Stop playback."""
        self.stopRequested.emit()

        self.is_playing = False
        self.is_paused = False
        self.current_beat = 0.0

        if self.playback_timer:
            self.playback_timer.stop()
            self.playback_timer = None

        if self.pygame_midi_available and self.midi_output:
            # All notes off for all channels
            for channel in range(16):
                self.midi_output.write_short(0xB0 | channel, 123, 0)  # All notes off
            self.midi_output.close()
            self.midi_output = None
        self.active_notes.clear()
        self.playback_thread = None
        self._update_time_display()
        self._update_ui_state()

    def _playback_worker(self):
        """Background worker for MIDI playback."""
        try:
            if not self.events:
                return

            # Calculate total duration in seconds
            max_time = max(e['time'] for e in self.events) if self.events else 0
            beat_duration = 60.0 / self.tempo
            total_duration = max_time * beat_duration

            self.start_time = time.time()
            last_update = self.start_time

            event_idx = 0
            while self.is_playing and not self.is_paused:
                current_time = time.time()
                elapsed = current_time - self.start_time
                self.current_beat = elapsed / beat_duration

                # Update position periodically
                if current_time - last_update >= 0.1:  # 100ms
                    self.positionUpdated.emit(self.current_beat)
                    last_update = current_time

                # Process events up to current time
                target_event_time = self.current_beat
                while event_idx < len(self.events) and self.events[event_idx]['time'] <= target_event_time:
                    event = self.events[event_idx]
                    if self.pygame_midi_available and self.midi_output:
                        if event['type'] == 'note_on' and event['velocity'] > 0:
                            self.midi_output.note_on(event['channel'], event['note'], event['velocity'])
                            self.active_notes[(event['channel'], event['note'])] = current_time
                        elif event['type'] == 'note_off' or (event['type'] == 'note_on' and event['velocity'] == 0):
                            self.midi_output.note_off(event['channel'], event['note'], 0)
                            self.active_notes.pop((event['channel'], event['note']), None)
                    event_idx += 1

                # Check if finished
                if self.current_beat >= max_time:
                    if self.loop_checkbox.isChecked():
                        # Loop
                        event_idx = 0
                        self.current_beat = 0.0
                        self.start_time = current_time
                        self.active_notes.clear()
                        # Restart volume if needed
                        if self.pygame_midi_available and self.midi_output:
                            volume_midi = int((self.current_volume / 100.0) * 127)
                            for channel in range(16):
                                self.midi_output.write_short(0xB0 | channel, 7, volume_midi)
                    else:
                        break

                time.sleep(0.001)  # Minimal sleep

            # Finished naturally
            if self.is_playing:
                self.positionUpdated.emit(self.current_beat)
                self.stop()

        except Exception as e:
            print(f"Playback error: {e}")
            self.stop()

    def _update_time_display(self):
        """Update the time display label."""
        if self.is_playing or self.is_paused:
            beat_duration = 60.0 / self.tempo
            elapsed_seconds = self.current_beat * beat_duration
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
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
        current_time = 0.0

        # Volume scaling
        volume_scale = self.current_volume / 100.0

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
                    velocity = int(note.velocity * volume_scale)
                    channel = self._get_channel_for_pattern(pattern.pattern_type)

                    # Note on event
                    events.append({
                        'type': 'note_on',
                        'time': current_time + note.start_time,
                        'note': note.pitch,
                        'velocity': velocity,
                        'channel': channel
                    })

                    # Note off event
                    events.append({
                        'type': 'note_off',
                        'time': current_time + note.start_time + note.duration,
                        'note': note.pitch,
                        'velocity': 0,
                        'channel': channel
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