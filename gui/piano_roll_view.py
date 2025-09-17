"""
Piano Roll View for MIDI Master GUI
 
This module provides a graphical piano roll visualization component
for displaying MIDI note events in a timeline view.
"""

import sys
import os
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QSizePolicy
from PyQt6.QtGui import QPen, QColor, QFont, QPainter, QWheelEvent, QKeyEvent
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
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMinimumHeight(350)
 
        # Initialize display parameters
        self.pixels_per_beat = 40
        self.pixels_per_note = 12
        self.zoom_x = 1.0
        self.zoom_y = 1.0
 
        # Cached content for redraws on zoom
        self._last_song = None
        self._total_beats = 0
        self.total_beats = 0
 
        # Playhead
        self.playhead = None
        self.playhead_visible = False
 
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
        self.playhead = None
 
        if not song_skeleton:
            # Cache cleared when no song provided
            self._last_song = None
            self._total_beats = 0
            self.total_beats = 0
            return
 
        # Cache for future redraws (zoom changes)
        self._last_song = song_skeleton
 
        # Calculate total beats from song skeleton (sum target section lengths)
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
        for section_type, patterns in song_skeleton.sections:
            total_beats += _get_section_target_length_beats(section_type)
        self._total_beats = total_beats
        self.total_beats = total_beats
 
        # Draw background grid
        self._draw_grid(scene, total_beats, self.min_note, self.max_note)
 
        # Draw notes from all sections
        current_time = 0.0
        for section_type, patterns in song_skeleton.sections:
            if not patterns:
                continue
 
            # Draw notes from each pattern in the section
            for pattern in patterns:
                if hasattr(pattern, 'notes') and pattern.notes:
                    self._draw_pattern_notes(scene, pattern, current_time)
 
            # Move to next section by mapped target length
            current_time += _get_section_target_length_beats(section_type)
 
        # Update scene rect; avoid auto-fitting to preserve user zoom/scroll
        scene.setSceneRect(scene.itemsBoundingRect())
 
        # Re-position playhead if visible
        if self.playhead_visible:
            self.update_playhead(0.0)
 
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
 
    # -------------------------
    # Zoom and interaction API
    # -------------------------
 
    def get_zoom_percent(self) -> int:
        """Get current zoom level as a percentage."""
        return int(round(self.zoom_x * 100))
 
    def set_zoom_percent(self, percent: float):
        """Set zoom level (both axes) as a percentage (e.g., 100 = 1.0x)."""
        clamped = max(25.0, min(400.0, float(percent)))
        self.zoom_x = clamped / 100.0
        self.zoom_y = clamped / 100.0
        self._redraw_from_cache()
 
    def zoom_in(self, factor: float = 1.2):
        """Incrementally zoom in."""
        self.set_zoom_percent(self.get_zoom_percent() * factor)
 
    def zoom_out(self, factor: float = 1.2):
        """Incrementally zoom out."""
        self.set_zoom_percent(self.get_zoom_percent() / factor)
 
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.set_zoom_percent(100.0)
 
    def fit_to_scene(self):
        """
        Fit the entire scene into view.
        Note: This uses view transform scaling and won't update internal zoom_x/y.
        """
        scene = self.scene()
        if not scene:
            return
        rect = scene.itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
 
    def wheelEvent(self, event: QWheelEvent):
        """
        Ctrl + Mouse Wheel to zoom. Regular wheel scrolls.
        """
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in(1.1)
            else:
                self.zoom_out(1.1)
            event.accept()
        else:
            super().wheelEvent(event)
 
    def keyPressEvent(self, event: QKeyEvent):
        """
        Keyboard shortcuts:
        + / = : Zoom in
        -      : Zoom out
        0      : Reset zoom to 100%
        """
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_in(1.2)
            event.accept()
            return
        if key == Qt.Key.Key_Minus:
            self.zoom_out(1.2)
            event.accept()
            return
        if key == Qt.Key.Key_0:
            self.reset_zoom()
            event.accept()
            return
        super().keyPressEvent(event)
 
    def _redraw_from_cache(self):
        """Redraw the scene using the cached song skeleton."""
        if self._last_song is not None:
            self.update_preview(self._last_song)
            if self.playhead_visible:
                self.update_playhead(0.0)
 
    def update_playhead(self, beat_position: float):
        """Update the playhead position during playback."""
        scene = self.scene()
        if scene is None:
            self.playhead_visible = False
            return
        if beat_position > self.total_beats:
            self.hide_playhead()
            return
 
        if self.playhead is None:
            self.playhead = scene.addLine(0, 0, 0, 0, QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
 
        # To satisfy Pylance type narrowing, assign to local variable
        line_item = self.playhead
        if line_item is None:
            print("Warning: playhead creation failed unexpectedly")
            return
 
        x = beat_position * self.pixels_per_beat * self.zoom_x
        height = (self.max_note - self.min_note + 1) * self.pixels_per_note * self.zoom_y
        print(f"Updating playhead to position {x} (height: {height})")
        line_item.setLine(x, 0, x, height)
        self.playhead_visible = True
 
    def show_playhead(self):
        """Show the playhead at the beginning."""
        self.playhead_visible = True
        self.update_playhead(0.0)
 
    def hide_playhead(self):
        """Hide the playhead."""
        self.playhead_visible = False
        scene = self.scene()
        if self.playhead and scene is not None:
            scene.removeItem(self.playhead)
        self.playhead = None