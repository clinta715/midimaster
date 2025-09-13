"""
Enhanced Main GUI Window for MIDI Master

This module defines the main application window using PyQt6 with integrated
parameter controls, matrix visualization, piano roll preview, and playback.
"""

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QComboBox, QSlider, QSpinBox, QButtonGroup, QRadioButton,
    QGroupBox, QGraphicsView, QGraphicsScene, QMessageBox, QSplitter,
    QProgressBar, QStatusBar, QLineEdit, QCheckBox, QFileDialog, QDialog,
    QFormLayout, QDialogButtonBox, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

# Import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.piano_roll_view import PianoRollView
from gui.generation_worker import GenerationWorker
from gui.playback_controller import PlaybackController
from gui.config_manager import ConfigManager
from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory
from generators.density_manager import create_density_manager_from_preset
from music_theory import MusicTheory


class MainWindow(QMainWindow):
    """
    Main GUI Window for MIDI Master Application

    Provides a graphical interface for music generation parameters,
    preview, and playback controls.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDI Master v2.0")
        self.setGeometry(100, 100, 1000, 800)

        # Initialize core components
        self.config_manager = ConfigManager()
        self.music_theory = MusicTheory()

        # Initialize parameter defaults
        self.current_params = {
            'genre': 'pop',
            'mood': 'happy',
            'key': 'C',
            'mode': 'major',
            'tempo': 120,
            'bars': 16,
            'density': 'balanced',
            'output': 'output.mid'
        }

        # Setup UI components
        self.setup_ui()

        # Connect signals
        self.connect_signals()

    def setup_ui(self):
        """Set up the main user interface components"""
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create splitter for main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Parameter controls
        self.setup_parameter_controls(splitter)

        # Right panel - Preview area
        self.setup_preview_panel(splitter)

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def setup_parameter_controls(self, parent):
        """Set up parameter control panel"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Genre selection
        genre_group = QGroupBox("Genre")
        genre_layout = QHBoxLayout(genre_group)
        self.genre_combo = QComboBox()
        self.genre_combo.addItems(['pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical'])
        self.genre_combo.setCurrentText('pop')
        genre_layout.addWidget(self.genre_combo)

        # Mood selection
        mood_group = QGroupBox("Mood")
        mood_layout = QHBoxLayout(mood_group)
        self.mood_combo = QComboBox()
        self.mood_combo.addItems(['happy', 'sad', 'energetic', 'calm'])
        self.mood_combo.setCurrentText('happy')
        mood_layout.addWidget(self.mood_combo)
        # Key selection
        key_group = QGroupBox("Key")
        key_layout = QHBoxLayout(key_group)
        self.key_combo = QComboBox()
        self.key_combo.addItems(self.music_theory.get_all_roots())
        self.key_combo.setCurrentText('C')
        key_layout.addWidget(self.key_combo)

        # Mode selection
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.music_theory.get_all_modes())
        self.mode_combo.setCurrentText('major')
        mode_layout.addWidget(self.mode_combo)

        # Tempo control

        # Tempo control
        tempo_group = QGroupBox("Tempo (BPM)")
        tempo_layout = QHBoxLayout(tempo_group)
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(80, 160)
        self.tempo_spin.setValue(120)
        self.tempo_slider = QSlider(Qt.Orientation.Horizontal)
        self.tempo_slider.setRange(80, 160)
        self.tempo_slider.setValue(120)
        tempo_layout.addWidget(self.tempo_spin)
        tempo_layout.addWidget(self.tempo_slider)

        # Bars control
        bars_group = QGroupBox("Bars")
        bars_layout = QHBoxLayout(bars_group)
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(4, 32)
        self.bars_spin.setValue(16)
        self.bars_slider = QSlider(Qt.Orientation.Horizontal)
        self.bars_slider.setRange(4, 32)
        self.bars_slider.setValue(16)
        bars_layout.addWidget(self.bars_spin)
        bars_layout.addWidget(self.bars_slider)

        # Density selection
        density_group = QGroupBox("Density")
        density_layout = QVBoxLayout(density_group)
        self.density_group = QButtonGroup()
        densities = ['minimal', 'sparse', 'balanced', 'dense', 'complex']
        for density in densities:
            radio = QRadioButton(density.capitalize())
            if density == 'balanced':
                radio.setChecked(True)
            self.density_group.addButton(radio)
            density_layout.addWidget(radio)
        # Time signature controls for per-track configuration
        time_sig_group = QGroupBox("Time Signatures")
        time_sig_layout = QGridLayout(time_sig_group)

        # Melody time signature
        time_sig_layout.addWidget(QLabel("Melody:"), 0, 0)
        self.melody_time_sig_edit = QLineEdit("4/4")
        time_sig_layout.addWidget(self.melody_time_sig_edit, 0, 1)

        # Harmony time signature
        time_sig_layout.addWidget(QLabel("Harmony:"), 1, 0)
        self.harmony_time_sig_edit = QLineEdit("4/4")
        time_sig_layout.addWidget(self.harmony_time_sig_edit, 1, 1)

        # Bass time signature
        time_sig_layout.addWidget(QLabel("Bass:"), 2, 0)
        self.bass_time_sig_edit = QLineEdit("4/4")
        time_sig_layout.addWidget(self.bass_time_sig_edit, 2, 1)

        # Rhythm time signature
        time_sig_layout.addWidget(QLabel("Rhythm:"), 3, 0)
        self.rhythm_time_sig_edit = QLineEdit("4/4")
        time_sig_layout.addWidget(self.rhythm_time_sig_edit, 3, 1)

        # Output format selection
        output_format_group = QGroupBox("Output Format")
        output_format_layout = QVBoxLayout(output_format_group)
        self.output_format_group = QButtonGroup()
        self.single_file_radio = QRadioButton("Single MIDI file")
        self.separate_files_radio = QRadioButton("Separate files per instrument")
        self.single_file_radio.setChecked(True)
        self.output_format_group.addButton(self.single_file_radio)
        self.output_format_group.addButton(self.separate_files_radio)
        output_format_layout.addWidget(self.single_file_radio)
        output_format_layout.addWidget(self.separate_files_radio)

        # Output filename
        # Output filename
        output_group = QGroupBox("Output File")
        output_layout = QHBoxLayout(output_group)
        self.output_edit = QLineEdit("output.mid")
        output_layout.addWidget(self.output_edit)

        # Generate button
        self.generate_btn = QPushButton("Generate Music")
        self.generate_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")

        # Add all to layout
        control_layout.addWidget(genre_group)
        control_layout.addWidget(mood_group)
        control_layout.addWidget(key_group)
        control_layout.addWidget(mode_group)
        control_layout.addWidget(tempo_group)
        control_layout.addWidget(bars_group)
        control_layout.addWidget(time_sig_group)
        control_layout.addWidget(density_group)
        control_layout.addWidget(output_group)
        control_layout.addWidget(self.generate_btn)
        control_layout.addStretch()

        parent.addWidget(control_widget)
        parent.setSizes([300, 700])  # Set initial splitter sizes

    def setup_preview_panel(self, parent):
        """Set up preview panel with piano roll and controls"""
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        # Piano roll placeholder
        piano_roll_group = QGroupBox("Piano Roll Preview")
        piano_roll_layout = QVBoxLayout(piano_roll_group)
        self.piano_roll_view = PianoRollView()
        piano_roll_layout.addWidget(self.piano_roll_view)

        # Playback controls
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QHBoxLayout(playback_group)

        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("⏹ Stop")

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)

        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.pause_btn)
        playback_layout.addWidget(self.stop_btn)
        playback_layout.addWidget(QLabel("Volume:"))
        playback_layout.addWidget(self.volume_slider)

        # Add to layout
        preview_layout.addWidget(piano_roll_group)
        preview_layout.addWidget(playback_group)

        parent.addWidget(preview_widget)

    def connect_signals(self):
        """Connect UI signals to handlers"""
        # Connect sliders and spinboxes
        self.tempo_slider.valueChanged.connect(self.tempo_spin.setValue)
        self.tempo_spin.valueChanged.connect(self.tempo_slider.setValue)
        self.bars_slider.valueChanged.connect(self.bars_spin.setValue)
        self.bars_spin.valueChanged.connect(self.bars_slider.setValue)

        # Connect parameter changes
        self.genre_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.mood_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.key_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.mode_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.tempo_spin.valueChanged.connect(self.on_parameter_changed)
        self.bars_spin.valueChanged.connect(self.on_parameter_changed)

        # Connect buttons
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        self.play_btn.clicked.connect(self.on_play_clicked)
        self.pause_btn.clicked.connect(self.on_pause_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)

    def on_parameter_changed(self):
        """Handle parameter changes"""
        self.current_params.update({
            'genre': self.genre_combo.currentText(),
            'mood': self.mood_combo.currentText(),
            'key': self.key_combo.currentText(),
            'mode': self.mode_combo.currentText(),
            'tempo': self.tempo_spin.value(),
            'bars': self.bars_spin.value(),
            'density': self.get_selected_density(),
            'melody_time_signature': self.melody_time_sig_edit.text(),
            'harmony_time_signature': self.harmony_time_sig_edit.text(),
            'bass_time_signature': self.bass_time_sig_edit.text(),
            'rhythm_time_signature': self.rhythm_time_sig_edit.text(),
            'output': self.output_edit.text()
        })

    def get_selected_density(self):
        """Get the selected density preset"""
        for button in self.density_group.buttons():
            if button.isChecked():
                return button.text().lower()
        return 'balanced'

    def on_generate_clicked(self):
        """Handle generate button click"""
        try:
            self.status_bar.showMessage("Generating music...")

            # Collect parameters
            params = self.current_params.copy()

            # Create generation components
            genre_rules = GenreFactory.create_genre_rules(params['genre'])
            song_skeleton = SongSkeleton(params['genre'], params['tempo'], params['mood'])
            density_manager = create_density_manager_from_preset(params['density'])

            # Generate patterns
            pattern_generator = PatternGenerator(
                genre_rules,
                params['mood'],
                note_density=density_manager.note_density,
                rhythm_density=density_manager.rhythm_density,
                chord_density=density_manager.chord_density,
                bass_density=density_manager.bass_density
            )

            patterns = pattern_generator.generate_patterns(song_skeleton, params['bars'])
            song_skeleton.build_arrangement(patterns)

            # Save MIDI file
            midi_output = MidiOutput()
            midi_output.save_to_midi(song_skeleton, params['output'], genre_rules)

            # Update preview
            self.piano_roll_view.update_preview(song_skeleton)

            self.status_bar.showMessage(f"Generated: {params['output']}")

        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}")
            QMessageBox.critical(self, "Generation Error", f"Failed to generate music:\n{str(e)}")

    def on_play_clicked(self):
        """Handle play button click"""
        self.status_bar.showMessage("Playback not yet implemented")

    def on_pause_clicked(self):
        """Handle pause button click"""
        self.status_bar.showMessage("Playback paused")

    def on_stop_clicked(self):
        """Handle stop button click"""
        self.status_bar.showMessage("Playback stopped")


def run_gui():
    """
    Launch the MIDI Master GUI application

    This function creates and runs the main GUI window.
    Should be called when the --gui flag is used in CLI mode.
    """
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("MIDI Master")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("MIDI Master Team")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec())