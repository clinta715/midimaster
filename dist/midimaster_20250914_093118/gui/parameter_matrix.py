"""
Parameter Matrix Component for MIDI Master GUI

This module provides an interactive parameter matrix visualization that allows
users to explore genre-tempo combinations and quickly apply parameter presets.
"""

import sys
import os
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QGroupBox, QScrollArea, QFrame, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
from typing import Optional, List

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_theory import MusicTheory


class ParameterMatrix(QWidget):
    """
    Interactive parameter matrix showing genre vs tempo combinations with
    color-coded cells indicating compatibility and recommendations.
    """

    # Signal emitted when user selects parameters from matrix
    parameterSelected = pyqtSignal(str, tuple, str, str)  # genre, tempo_range, mood, density

    def __init__(self, genres: Optional[List[str]] = None):
        super().__init__()
        self.current_params = {}
        if genres is None:
            from genres.genre_factory import GenreFactory
            self.genres = GenreFactory.get_available_genres()
        else:
            self.genres = genres
        print("ParameterMatrix genres loaded:", self.genres)  # Log for validation
        self._init_ui()
        self._populate_matrix()

    def _init_ui(self):
        """Initialize the matrix user interface."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Genre vs Tempo Matrix")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(header)

        # Create scroll area for matrix
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)
        layout.addWidget(scroll_area)

        # Matrix container
        matrix_widget = QWidget()
        self.matrix_layout = QGridLayout(matrix_widget)
        self.matrix_layout.setSpacing(2)

        # Add headers
        self._add_headers()

        scroll_area.setWidget(matrix_widget)

        # Key/Mode parameters
        self._add_key_mode_controls(layout)

        # Legend
        self._add_legend(layout)

    def _add_headers(self):
        """Add row and column headers to the matrix."""
        # Corner label
        corner_label = QLabel("")
        corner_label.setMinimumSize(100, 30)
        self.matrix_layout.addWidget(corner_label, 0, 0)

        # Tempo headers (columns)
        tempo_ranges = [
            (80, 100), (101, 120), (121, 140), (141, 160)
        ]

        for col, (tempo_min, tempo_max) in enumerate(tempo_ranges, 1):
            header = QLabel(f"{tempo_min}-{tempo_max}")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.setStyleSheet("""
                QLabel {
                    font-weight: bold;
                    background-color: #e0e0e0;
                    border: 1px solid #ccc;
                    padding: 5px;
                    min-width: 80px;
                }
            """)
            self.matrix_layout.addWidget(header, 0, col)

        # Genre headers (rows)
        for row, genre in enumerate(self.genres, 1):
            header = QLabel(genre.capitalize())
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.setStyleSheet("""
                QLabel {
                    font-weight: bold;
                    background-color: #e0e0e0;
                    border: 1px solid #ccc;
                    padding: 5px;
                    min-height: 40px;
                }
            """)
            self.matrix_layout.addWidget(header, row, 0)

    def _populate_matrix(self):
        """Populate the matrix with interactive cells."""
        tempo_ranges = [
            (80, 100), (101, 120), (121, 140), (141, 160)
        ]

        for row, genre in enumerate(self.genres, 1):
            for col, tempo_range in enumerate(tempo_ranges, 1):
                cell = MatrixCell(genre, tempo_range)
                cell.clicked.connect(self._on_cell_clicked)

                # Set initial styling based on compatibility
                self._style_cell(cell, genre, tempo_range)

                self.matrix_layout.addWidget(cell, row, col)

    def _add_legend(self, parent_layout):
        """Add a legend explaining the color coding."""
        legend_group = QGroupBox("Legend")
        legend_layout = QHBoxLayout(legend_group)

        # Create legend items
        legend_items = [
            ("Optimal", "#90EE90"),      # Light green
            ("Good", "#FFFFE0"),         # Light yellow
            ("Challenging", "#FFB6C1"),  # Light red
            ("Selected", "#4A90E2")      # Blue
        ]

        for text, color in legend_items:
            item_layout = QHBoxLayout()

            # Color indicator
            color_label = QLabel()
            color_label.setFixedSize(20, 20)
            color_label.setStyleSheet(f"background-color: {color}; border: 1px solid #ccc;")
            item_layout.addWidget(color_label)

            # Text label
            text_label = QLabel(text)
            item_layout.addWidget(text_label)

            legend_layout.addLayout(item_layout)

        legend_layout.addStretch()
        parent_layout.addWidget(legend_group)

    def _add_key_mode_controls(self, parent_layout):
        """Add key/mode parameter controls."""
        key_mode_group = QGroupBox("Key/Mode Parameters")
        key_mode_layout = QHBoxLayout(key_mode_group)

        # Key selection
        key_layout = QVBoxLayout()
        key_layout.addWidget(QLabel("Key:"))
        self.key_combo = QComboBox()
        music_theory = MusicTheory()
        self.key_combo.addItem("")  # Empty option for fallback
        self.key_combo.addItems(music_theory.get_all_roots())
        key_layout.addWidget(self.key_combo)
        key_mode_layout.addLayout(key_layout)

        # Mode selection
        mode_layout = QVBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("")  # Empty option for fallback
        self.mode_combo.addItems(music_theory.get_all_modes())
        mode_layout.addWidget(self.mode_combo)
        key_mode_layout.addLayout(mode_layout)

        parent_layout.addWidget(key_mode_group)

    def _style_cell(self, cell, genre, tempo_range):
        """Style a matrix cell based on genre-tempo compatibility."""
        tempo_min, tempo_max = tempo_range
        avg_tempo = (tempo_min + tempo_max) // 2

        # Define compatibility rules (simplified)
        compatibility_matrix = {
            'pop': {'range': (90, 140), 'optimal': (100, 130)},
            'rock': {'range': (100, 160), 'optimal': (120, 140)},
            'jazz': {'range': (80, 120), 'optimal': (90, 110)},
            'electronic': {'range': (100, 160), 'optimal': (120, 140)},
            'hip-hop': {'range': (80, 110), 'optimal': (85, 100)},
            'classical': {'range': (80, 120), 'optimal': (90, 110)},
            'dnb': {'range': (150, 180), 'optimal': (160, 175)},
            'drum-and-bass': {'range': (150, 180), 'optimal': (160, 175)}
        }

        if genre in compatibility_matrix:
            optimal_min, optimal_max = compatibility_matrix[genre]['optimal']
            range_min, range_max = compatibility_matrix[genre]['range']

            if optimal_min <= avg_tempo <= optimal_max:
                cell.setStyleSheet("""
                    QPushButton {
                        background-color: #90EE90;
                        border: 2px solid #228B22;
                        border-radius: 5px;
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: #7BC97B;
                    }
                """)
            elif range_min <= avg_tempo <= range_max:
                cell.setStyleSheet("""
                    QPushButton {
                        background-color: #FFFFE0;
                        border: 1px solid #FFD700;
                        border-radius: 5px;
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: #F0E68C;
                    }
                """)
            else:
                cell.setStyleSheet("""
                    QPushButton {
                        background-color: #FFB6C1;
                        border: 1px solid #DC143C;
                        border-radius: 5px;
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: #F08080;
                    }
                """)
        else:
            cell.setStyleSheet("""
                QPushButton {
                    background-color: #F0F0F0;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #E0E0E0;
                }
            """)

    def _on_cell_clicked(self, genre, tempo_range):
        """Handle cell click event."""
        # Determine recommended mood and density based on genre and tempo
        mood = self._get_recommended_mood(genre, tempo_range)
        density = self._get_recommended_density(genre, tempo_range)

        # Emit signal with selected parameters
        self.parameterSelected.emit(genre, tempo_range, mood, density)

        # Update cell highlighting
        self.update_highlighting({'genre': genre, 'tempo': (tempo_range[0] + tempo_range[1]) // 2, 'mood': mood, 'density': density})

    def _get_recommended_mood(self, genre, tempo_range):
        """Get recommended mood for genre-tempo combination."""
        tempo_avg = (tempo_range[0] + tempo_range[1]) // 2

        recommendations = {
            'pop': {'slow': 'calm', 'medium': 'happy', 'fast': 'energetic'},
            'rock': {'slow': 'calm', 'medium': 'energetic', 'fast': 'energetic'},
            'jazz': {'slow': 'calm', 'medium': 'happy', 'fast': 'energetic'},
            'electronic': {'slow': 'calm', 'medium': 'energetic', 'fast': 'energetic'},
            'hip-hop': {'slow': 'calm', 'medium': 'energetic', 'fast': 'energetic'},
            'classical': {'slow': 'calm', 'medium': 'sad', 'fast': 'energetic'},
            'dnb': {'slow': 'energetic', 'medium': 'energetic', 'fast': 'energetic'},
            'drum-and-bass': {'slow': 'energetic', 'medium': 'energetic', 'fast': 'energetic'}
        }

        if genre in recommendations:
            if tempo_avg < 110:
                return recommendations[genre]['slow']
            elif tempo_avg < 140:
                return recommendations[genre]['medium']
            else:
                return recommendations[genre]['fast']

        return 'happy'  # Default

    def _get_recommended_density(self, genre, tempo_range):
        """Get recommended density for genre-tempo combination."""
        tempo_avg = (tempo_range[0] + tempo_range[1]) // 2

        if tempo_avg < 100:
            return 'sparse'
        elif tempo_avg < 130:
            return 'balanced'
        else:
            if genre in ['electronic', 'rock', 'dnb', 'drum-and-bass']:
                return 'dense'
            else:
                return 'balanced'

    def update_highlighting(self, params):
        """Update matrix highlighting based on current parameters."""
        self.current_params = params

        # Update all cells
        tempo_ranges = [(80, 100), (101, 120), (121, 140), (141, 160)]

        for row, genre in enumerate(self.genres, 1):
            for col, tempo_range in enumerate(tempo_ranges, 1):
                item = self.matrix_layout.itemAtPosition(row, col)
                cell = item.widget() if item else None
                if cell:
                    # Reset to compatibility styling
                    self._style_cell(cell, genre, tempo_range)

                    # Highlight selected cell
                    if (genre == params.get('genre') and
                        tempo_range[0] <= params.get('tempo', 0) <= tempo_range[1]):
                        cell.setStyleSheet("""
                            QPushButton {
                                background-color: #4A90E2;
                                border: 3px solid #2E5C8A;
                                border-radius: 5px;
                                padding: 5px;
                                font-weight: bold;
                            }
                        """)


class MatrixCell(QPushButton):
    """
    Individual cell in the parameter matrix with click handling.
    """

    clicked = pyqtSignal(str, tuple)  # genre, tempo_range

    def __init__(self, genre, tempo_range):
        super().__init__()
        self.genre = genre
        self.tempo_range = tempo_range
        self._init_cell()

    def _init_cell(self):
        """Initialize the cell appearance and behavior."""
        tempo_min, tempo_max = self.tempo_range
        avg_tempo = (tempo_min + tempo_max) // 2

        self.setText(f"{avg_tempo}")
        self.setMinimumSize(80, 40)
        self.setMaximumSize(100, 50)

        # Connect click signal
        super().clicked.connect(lambda: self.clicked.emit(self.genre, self.tempo_range))

        # Set tooltip with recommendations
        mood = self._get_tooltip_mood()
        density = self._get_tooltip_density()
        self.setToolTip(f"Genre: {self.genre.capitalize()}\n"
                       f"Tempo: {tempo_min}-{tempo_max} BPM\n"
                       f"Recommended Mood: {mood.capitalize()}\n"
                       f"Recommended Density: {density.capitalize()}")

    def _get_tooltip_mood(self):
        """Get mood recommendation for tooltip."""
        tempo_avg = (self.tempo_range[0] + self.tempo_range[1]) // 2

        if tempo_avg < 110:
            return 'calm'
        elif tempo_avg < 140:
            return 'happy'
        else:
            return 'energetic'

    def _get_tooltip_density(self):
        """Get density recommendation for tooltip."""
        tempo_avg = (self.tempo_range[0] + self.tempo_range[1]) // 2

        if tempo_avg < 100:
            return 'sparse'
        elif tempo_avg < 130:
            return 'balanced'
        else:
            return 'dense'