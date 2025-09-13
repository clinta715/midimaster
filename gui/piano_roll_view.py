"""
Piano Roll View for MIDI Master GUI

This module provides a graphical piano roll visualization component
for displaying MIDI note events in a timeline view.
"""

import sys
import os
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PyQt6.QtGui import QPen, QColor, QFont, QPainter
from PyQt6.QtCore import Qt

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PianoRollView(QGraphicsView):
    """
    A graphical view for displaying MIDI notes in a piano roll format.

    Shows notes as rectangles on a timeline with piano keyboard on the side.
    Supports zooming and scrolling for navigation.
    """

    def __init__(self):
        """Initialize the piano roll view."""
        super().__init__()

        # Create graphics scene
        scene = QGraphicsScene()
        self._scene = scene
        self.setScene(scene)

        # Set up view properties
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Initialize display parameters
        self.pixels_per_beat = 40
        self.pixels_per_note = 12
        self.zoom_x = 1.0
        self.zoom_y = 1.0

        # Note range (MIDI notes 21-108 covers most useful range)
        self.min_note = 21  # A0
        self.max_note = 108  # C8

        # Color mapping for different pattern types
        self.pattern_colors = {
            'melody': QColor(100, 150, 255),    # Blue
            'harmony': QColor(100, 255, 150),   # Green
            'bass': QColor(255, 150, 100),      # Orange
            'rhythm': QColor(255, 100, 150)     # Red
        }

    def update_preview(self, song_skeleton):
        """
        Update the piano roll display with notes from a song skeleton.

        Args:
            song_skeleton: SongSkeleton object containing the generated music
        """
        # Clear current display
        scene = self.scene()
        if scene is None:
            return

        scene.clear()

        if not song_skeleton:
            return

        # Calculate total beats from song skeleton
        sections_count = len(song_skeleton.sections)
        beats_per_section = 16  # Assume 16 beats per section
        total_beats = sections_count * beats_per_section

        # Draw background grid
        self._draw_grid(scene, total_beats, self.min_note, self.max_note)

        # Draw notes from all sections
        current_time = 0
        for section_name, section in song_skeleton.sections.items():
            if not section:
                continue

            # Draw notes from each pattern in the section
            for pattern in section:
                if hasattr(pattern, 'notes') and pattern.notes:
                    self._draw_pattern_notes(scene, pattern, current_time)

            # Move to next section
            current_time += beats_per_section

        # Fit view to content
        scene_rect = scene.sceneRect()
        if not scene_rect.isEmpty():
            self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _draw_pattern_notes(self, scene, pattern, time_offset):
        """
        Draw notes from a single pattern.

        Args:
            pattern: Pattern object containing notes
            time_offset: Time offset for this pattern in beats
        """
        color = self.pattern_colors.get(pattern.pattern_type.value, QColor(150, 150, 150))

        for note in pattern.notes:
            # Calculate note position and size
            x = (time_offset + note.start_time) * self.pixels_per_beat * self.zoom_x
            y = (self.max_note - note.pitch) * self.pixels_per_note * self.zoom_y
            width = note.duration * self.pixels_per_beat * self.zoom_x
            height = self.pixels_per_note * self.zoom_y

            # Create note rectangle
            note_rect = QGraphicsRectItem(x, y, width, height)
            note_rect.setBrush(color)
            note_rect.setPen(QPen(color.darker(), 1))
            scene.addItem(note_rect)

    def _draw_grid(self, scene, total_beats, min_note, max_note):
        """Draw the background grid."""
        # Vertical lines (beats)
        for beat in range(0, int(total_beats) + 1, 4):  # Every 4 beats (measure)
            x = beat * self.pixels_per_beat * self.zoom_x
            pen = QPen(QColor(200, 200, 200), 2 if beat % 16 == 0 else 1)
            scene.addLine(x, 0, x, (max_note - min_note + 2) * self.pixels_per_note * self.zoom_y, pen)

        # Horizontal lines (notes)
        for note in range(min_note, max_note + 1):
            y = (max_note - note) * self.pixels_per_note * self.zoom_y
            pen = QPen(QColor(220, 220, 220), 1)
            scene.addLine(0, y, total_beats * self.pixels_per_beat * self.zoom_x, y, pen)

            # Add note labels every octave
            if note % 12 == 0:
                note_name = self._midi_note_to_name(note)
                text_item = scene.addText(note_name)
                text_item.setPos(-40, y - 10)
                font = text_item.font()
                font.setPointSize(8)
                text_item.setFont(font)

    def _midi_note_to_name(self, midi_note):
        """
        Convert MIDI note number to note name.

        Args:
            midi_note: MIDI note number (0-127)

        Returns:
            String representation of the note (e.g., "C4", "F#3")
        """
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note_index = midi_note % 12
        return f"{note_names[note_index]}{octave}"