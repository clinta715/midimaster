"""
Enhanced Main GUI Window for MIDI Master

This module defines the main application window using PyQt6 with integrated
parameter controls, matrix visualization, piano roll preview, and playback.

Phase 1 updates:
- Introduced tabbed UI structure (Basics, Advanced, Plugins/Audio, Output, Presets)
- Moved existing controls to Basics tab
- Added Advanced tab controls with minimal wiring to current_params
- Added Plugins/Audio tab with safe scan placeholders
- Added Output tab with output folder picker and temp settings save/load via ConfigManager
- Added Presets tab with basic save/load/delete of configurations
- Kept generation workflow unchanged
"""

import sys
import os
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QPushButton,
    QLabel, QComboBox, QSlider, QSpinBox, QButtonGroup, QRadioButton,
    QGroupBox, QMessageBox, QSplitter,
    QStatusBar, QLineEdit, QCheckBox, QFileDialog,
    QTabWidget, QListWidget, QListWidgetItem, QDoubleSpinBox, QSizePolicy
)
from PyQt6.QtCore import Qt

# Import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.piano_roll_view import PianoRollView
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
        # Ensure QApplication exists for QWidget creation
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        super().__init__()
        self.setWindowTitle("MIDI Master v2.0")
        # Make the window larger by default to give more space to the preview
        self.setGeometry(100, 100, 1200, 800)

        # Initialize core components
        self.config_manager = ConfigManager()
        self.music_theory = MusicTheory()
        self.genre_rules_cache = {}  # Cache for subgenre rules

        # Initialize parameter defaults (extended for Phase 1)
        self.current_params: Dict[str, Any] = {
            'genre': 'pop',
            'mood': 'happy',
            'key': 'C',
            'mode': 'major',
            'tempo': 120,
            'bars': 16,
            'density': 'balanced',
            'output': '',
            'separate_files': False,
            'subgenre': '',
            # New Phase 1 placeholders (UI only in this phase)
            'output_folder': 'output/',
            'time_signatures': {
                'melody': '4/4',
                'harmony': '4/4',
                'bass': '4/4',
                'rhythm': '4/4',
            },
            'harmonic_variance': 'medium',
            'generate_parts': {
                'melody': True,
                'harmony': True,
                'bass': True,
                'rhythm': True,
            }
        }

        # UI references used across tabs
        self.genre_combo: QComboBox
        self.subgenre_combo: QComboBox
        self.mood_combo: QComboBox
        self.key_combo: QComboBox
        self.mode_combo: QComboBox
        self.tempo_spin: QSpinBox
        self.tempo_slider: QSlider
        self.bars_spin: QSpinBox
        self.bars_slider: QSlider
        self.density_group: QButtonGroup
        self.output_edit: QLineEdit
        self.separate_files_checkbox: QCheckBox
        self.generate_btn: QPushButton

        # Advanced tab refs
        self.ts_melody_combo: QComboBox
        self.ts_harmony_combo: QComboBox
        self.ts_bass_combo: QComboBox
        self.ts_rhythm_combo: QComboBox
        self.harmonic_variance_combo: QComboBox
        self.gen_melody_checkbox: QCheckBox
        self.gen_harmony_checkbox: QCheckBox
        self.gen_bass_checkbox: QCheckBox
        self.gen_rhythm_checkbox: QCheckBox

        # Plugins/Audio refs
        self.plugin_paths_edit: QLineEdit
        self.plugin_scan_btn: QPushButton
        self.plugin_status_label: QLabel
        self.plugin_list: QListWidget
        self.render_output_edit: QLineEdit

        # Output tab refs
        self.output_folder_edit: QLineEdit
        self.temp_dir_edit: QLineEdit
        self.temp_save_btn: QPushButton
        self.temp_load_btn: QPushButton

        # Performance (Advanced) refs
        self.swing_mode_combo: QComboBox
        self.micro_ms_spin: QSpinBox
        self.grid_bias_ms_spin: QSpinBox
        self.length_var_spin: QSpinBox
        self.staccato_prob_spin: QSpinBox
        self.tenuto_prob_spin: QSpinBox
        self.marcato_prob_spin: QSpinBox
        self.vel_shape_combo: QComboBox
        self.vel_intensity_spin: QDoubleSpinBox
        self.phrase_len_spin: QSpinBox

        # Presets tab refs
        self.presets_list: QListWidget
        self.save_preset_btn: QPushButton
        self.load_preset_btn: QPushButton
        self.delete_preset_btn: QPushButton

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

        # Left panel - Tabbed parameter controls
        self.setup_tabbed_controls(splitter)

        # Right panel - Preview area
        self.setup_preview_panel(splitter)

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # -------------------------
    # Left panel - Tabbed UI
    # -------------------------

    def setup_tabbed_controls(self, parent: QSplitter):
        """Set up parameter control panel with tabs"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        self.tab_widget = QTabWidget()

        # Build tabs
        self.basics_tab = self._build_basics_tab()
        self.advanced_tab = self._build_advanced_tab()
        self.plugins_tab = self._build_plugins_audio_tab()
        self.output_tab = self._build_output_tab()
        self.presets_tab = self._build_presets_tab()

        # Add tabs
        self.tab_widget.addTab(self.basics_tab, "Basics")
        self.tab_widget.addTab(self.advanced_tab, "Advanced")
        self.tab_widget.addTab(self.plugins_tab, "Plugins/Audio")
        self.tab_widget.addTab(self.output_tab, "Output")
        self.tab_widget.addTab(self.presets_tab, "Presets")

        control_layout.addWidget(self.tab_widget)

        # Initial subgenre population is done after genre control exists
        self.populate_subgenres()

        parent.addWidget(control_widget)
        parent.setSizes([350, 650])  # Slightly wider to fit tabs
        # Prefer right-side (preview) to stretch; left stays narrower
        parent.setStretchFactor(parent.indexOf(control_widget), 0)

    def _build_basics_tab(self) -> QWidget:
        """Create the Basics tab with existing controls"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Genre selection
        genre_group = QGroupBox("Genre")
        genre_layout = QHBoxLayout(genre_group)
        self.genre_combo = QComboBox()
        available_genres = GenreFactory.get_available_genres()
        self.genre_combo.addItems(available_genres)
        if 'pop' in available_genres:
            self.genre_combo.setCurrentText('pop')
        else:
            self.genre_combo.setCurrentIndex(0)
        genre_layout.addWidget(self.genre_combo)
        self.genre_combo.currentTextChanged.connect(self.populate_subgenres)

        # Subgenre selection
        subgenre_group = QGroupBox("Subgenre")
        subgenre_layout = QHBoxLayout(subgenre_group)
        self.subgenre_combo = QComboBox()
        self.subgenre_combo.addItem('')  # Empty option
        subgenre_layout.addWidget(self.subgenre_combo)
        self.subgenre_combo.currentTextChanged.connect(self.on_parameter_changed)

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

        # Output filename
        output_group = QGroupBox("Output File")
        output_layout = QHBoxLayout(output_group)
        self.output_edit = QLineEdit("")
        self.output_edit.setPlaceholderText("Auto-generate from settings if left blank")
        output_layout.addWidget(self.output_edit)

        # Separate files option
        self.separate_files_checkbox = QCheckBox("Export separate MIDI files per instrument/track")
        self.separate_files_checkbox.setChecked(False)

        # Generate button
        self.generate_btn = QPushButton("Generate Music")
        self.generate_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")

        # Add all to Basics layout (in required order)
        layout.addWidget(genre_group)
        layout.addWidget(subgenre_group)
        layout.addWidget(mood_group)
        layout.addWidget(key_group)
        layout.addWidget(mode_group)
        layout.addWidget(tempo_group)
        layout.addWidget(bars_group)
        layout.addWidget(density_group)
        layout.addWidget(output_group)
        layout.addWidget(self.separate_files_checkbox)
        layout.addWidget(self.generate_btn)
        layout.addStretch()

        return tab

    def _build_advanced_tab(self) -> QWidget:
        """Create the Advanced tab with performance controls"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Per-track time signatures
        ts_group = QGroupBox("Per-Track Time Signatures")
        ts_layout = QHBoxLayout(ts_group)
        options = ["4/4", "3/4", "6/8"]

        def make_ts(label_text: str) -> QComboBox:
            v = QVBoxLayout()
            v.addWidget(QLabel(label_text))
            c = QComboBox()
            c.addItems(options)
            v.addWidget(c)
            h = QWidget()
            h.setLayout(v)
            ts_layout.addWidget(h)
            return c

        self.ts_melody_combo = make_ts("Melody")
        self.ts_harmony_combo = make_ts("Harmony")
        self.ts_bass_combo = make_ts("Bass")
        self.ts_rhythm_combo = make_ts("Rhythm")

        # Harmonic variance
        hv_group = QGroupBox("Harmonic Variance")
        hv_layout = QHBoxLayout(hv_group)
        self.harmonic_variance_combo = QComboBox()
        self.harmonic_variance_combo.addItems(["close", "medium", "distant"])
        self.harmonic_variance_combo.setCurrentText("medium")
        hv_layout.addWidget(QLabel("Variance:"))
        hv_layout.addWidget(self.harmonic_variance_combo)

        # Selective generation toggles
        gp_group = QGroupBox("Selective Generation")
        gp_layout = QHBoxLayout(gp_group)
        self.gen_melody_checkbox = QCheckBox("Generate Melody")
        self.gen_harmony_checkbox = QCheckBox("Generate Harmony")
        self.gen_bass_checkbox = QCheckBox("Generate Bass")
        self.gen_rhythm_checkbox = QCheckBox("Generate Rhythm")
        for cb in (self.gen_melody_checkbox, self.gen_harmony_checkbox, self.gen_bass_checkbox, self.gen_rhythm_checkbox):
            cb.setChecked(True)
            gp_layout.addWidget(cb)

        # Performance expression
        perf_group = QGroupBox("Performance Expression")
        perf_layout = QGridLayout(perf_group)

        # Swing mode
        swing_box = QVBoxLayout()
        swing_box.addWidget(QLabel("Swing Mode"))
        self.swing_mode_combo = QComboBox()
        self.swing_mode_combo.addItems(["eighth", "sixteenth", "triplet"])
        self.swing_mode_combo.setCurrentText("eighth")
        swing_box.addWidget(self.swing_mode_combo)
        swing_wrap = QWidget(); swing_wrap.setLayout(swing_box)

        # Microtiming range (ms)
        micro_box = QVBoxLayout()
        micro_box.addWidget(QLabel("Microtiming ± ms"))
        self.micro_ms_spin = QSpinBox()
        self.micro_ms_spin.setRange(0, 20)
        self.micro_ms_spin.setValue(6)
        micro_box.addWidget(self.micro_ms_spin)
        micro_wrap = QWidget(); micro_wrap.setLayout(micro_box)

        # Grid bias (ms)
        bias_box = QVBoxLayout()
        bias_box.addWidget(QLabel("Grid Bias ms (- behind, + ahead)"))
        self.grid_bias_ms_spin = QSpinBox()
        self.grid_bias_ms_spin.setRange(-20, 20)
        self.grid_bias_ms_spin.setValue(0)
        bias_box.addWidget(self.grid_bias_ms_spin)
        bias_wrap = QWidget(); bias_wrap.setLayout(bias_box)

        # Note length variance (%)
        lenvar_box = QVBoxLayout()
        lenvar_box.addWidget(QLabel("Length Variance %"))
        self.length_var_spin = QSpinBox()
        self.length_var_spin.setRange(0, 50)
        self.length_var_spin.setValue(10)
        lenvar_box.addWidget(self.length_var_spin)
        lenvar_wrap = QWidget(); lenvar_wrap.setLayout(lenvar_box)

        # Articulation probabilities (%)
        art_box = QVBoxLayout()
        art_box.addWidget(QLabel("Articulation Probabilities %"))
        row1 = QHBoxLayout()
        self.staccato_prob_spin = QSpinBox(); self.staccato_prob_spin.setRange(0, 100); self.staccato_prob_spin.setValue(10)
        self.tenuto_prob_spin = QSpinBox(); self.tenuto_prob_spin.setRange(0, 100); self.tenuto_prob_spin.setValue(20)
        self.marcato_prob_spin = QSpinBox(); self.marcato_prob_spin.setRange(0, 100); self.marcato_prob_spin.setValue(10)
        row1.addWidget(QLabel("Stacc:")); row1.addWidget(self.staccato_prob_spin)
        row1.addWidget(QLabel("Ten:")); row1.addWidget(self.tenuto_prob_spin)
        row1.addWidget(QLabel("Marc:")); row1.addWidget(self.marcato_prob_spin)
        art_box.addLayout(row1)
        art_wrap = QWidget(); art_wrap.setLayout(art_box)

        # Velocity profile
        vel_box = QVBoxLayout()
        vel_box.addWidget(QLabel("Velocity Profile"))
        self.vel_shape_combo = QComboBox()
        self.vel_shape_combo.addItems(["arch", "flat"])
        self.vel_shape_combo.setCurrentText("arch")
        vel_row = QHBoxLayout()
        vel_row.addWidget(QLabel("Intensity"))
        self.vel_intensity_spin = QDoubleSpinBox()
        self.vel_intensity_spin.setRange(0.0, 1.0)
        self.vel_intensity_spin.setSingleStep(0.05)
        self.vel_intensity_spin.setValue(0.3)
        vel_row.addWidget(self.vel_intensity_spin)
        vel_row.addWidget(QLabel("Phrase Len (beats)"))
        self.phrase_len_spin = QSpinBox()
        self.phrase_len_spin.setRange(1, 16)
        self.phrase_len_spin.setValue(4)
        vel_row.addWidget(self.phrase_len_spin)
        vel_box.addLayout(vel_row)
        vel_wrap = QWidget(); vel_wrap.setLayout(vel_box)

        # Place widgets in a compact 2-column grid to reduce overall width
        perf_layout.addWidget(swing_wrap, 0, 0)
        perf_layout.addWidget(micro_wrap, 0, 1)
        perf_layout.addWidget(bias_wrap, 1, 0)
        perf_layout.addWidget(lenvar_wrap, 1, 1)
        perf_layout.addWidget(art_wrap, 2, 0, 1, 2)
        perf_layout.addWidget(vel_wrap, 3, 0, 1, 2)

        # Balance columns
        perf_layout.setColumnStretch(0, 1)
        perf_layout.setColumnStretch(1, 1)

        # Add to tab
        layout.addWidget(ts_group)
        layout.addWidget(hv_group)
        layout.addWidget(gp_group)
        layout.addWidget(perf_group)
        layout.addStretch()

        return tab

    def _build_plugins_audio_tab(self) -> QWidget:
        """Create Plugins/Audio tab with light placeholders"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scan controls
        scan_group = QGroupBox("Plugin Scan")
        scan_layout = QHBoxLayout(scan_group)
        self.plugin_paths_edit = QLineEdit("")
        self.plugin_paths_edit.setPlaceholderText("Plugin scan paths (semicolon-separated). Leave empty for defaults")
        self.plugin_scan_btn = QPushButton("Scan")
        self.plugin_status_label = QLabel("")
        scan_layout.addWidget(self.plugin_paths_edit)
        scan_layout.addWidget(self.plugin_scan_btn)
        layout.addWidget(scan_group)
        layout.addWidget(self.plugin_status_label)

        # Plugin list
        list_group = QGroupBox("Detected Plugins")
        list_layout = QVBoxLayout(list_group)
        self.plugin_list = QListWidget()
        list_layout.addWidget(self.plugin_list)
        layout.addWidget(list_group)

        # Render controls (placeholders, disabled)
        render_group = QGroupBox("Render Controls (placeholder)")
        render_layout = QHBoxLayout(render_group)
        self.render_output_edit = QLineEdit("")
        self.render_output_edit.setPlaceholderText("Output WAV path (placeholder)")
        render_layout.addWidget(self.render_output_edit)
        render_group.setEnabled(False)
        layout.addWidget(render_group)

        layout.addStretch()
        return tab

    def _build_output_tab(self) -> QWidget:
        """Create Output tab for folder and temp settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Output folder
        out_group = QGroupBox("Output Folder")
        out_layout = QHBoxLayout(out_group)
        self.output_folder_edit = QLineEdit(self.current_params.get("output_folder", "output/"))
        out_browse_btn = QPushButton("Browse...")
        out_layout.addWidget(self.output_folder_edit)
        out_layout.addWidget(out_browse_btn)
        layout.addWidget(out_group)

        out_browse_btn.clicked.connect(self._on_browse_output_folder_clicked)

        # Temp settings
        temp_group = QGroupBox("Temporary Files Settings")
        temp_layout = QHBoxLayout(temp_group)
        self.temp_dir_edit = QLineEdit("")
        self.temp_dir_edit.setPlaceholderText("Temp directory path")
        self.temp_save_btn = QPushButton("Save")
        self.temp_load_btn = QPushButton("Load")
        temp_layout.addWidget(self.temp_dir_edit)
        temp_layout.addWidget(self.temp_save_btn)
        temp_layout.addWidget(self.temp_load_btn)
        layout.addWidget(temp_group)

        layout.addStretch()
        return tab

    def _build_presets_tab(self) -> QWidget:
        """Create Presets tab for configuration management"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.presets_list = QListWidget()
        layout.addWidget(self.presets_list)

        btn_row = QHBoxLayout()
        self.save_preset_btn = QPushButton("Save Current")
        self.load_preset_btn = QPushButton("Load Selected")
        self.delete_preset_btn = QPushButton("Delete Selected")
        btn_row.addWidget(self.save_preset_btn)
        btn_row.addWidget(self.load_preset_btn)
        btn_row.addWidget(self.delete_preset_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

        # Populate list initially
        self.refresh_presets_list()

        return tab

    # -------------------------
    # Right panel - Preview
    # -------------------------

    def setup_preview_panel(self, parent: QSplitter):
        """Set up preview panel with piano roll and controls"""
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        # Piano roll
        piano_roll_group = QGroupBox("Piano Roll Preview")
        piano_roll_layout = QVBoxLayout(piano_roll_group)

        # Zoom/Scroll controls bar
        zoom_bar = QHBoxLayout()
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setToolTip("Zoom Out (Ctrl + Mouse Wheel Down, or '-')")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setToolTip("Zoom In (Ctrl + Mouse Wheel Up, or '+/=')")
        self.zoom_reset_btn = QPushButton("100%")
        self.zoom_reset_btn.setToolTip("Reset Zoom (0 key)")
        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.setToolTip("Fit the entire scene into view")

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(25, 400)  # percent
        self.zoom_slider.setSingleStep(5)
        self.zoom_slider.setPageStep(10)
        self.zoom_slider.setValue(100)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(48)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        zoom_bar.addWidget(self.zoom_out_btn)
        zoom_bar.addWidget(self.zoom_slider, 1)
        zoom_bar.addWidget(self.zoom_in_btn)
        zoom_bar.addWidget(self.zoom_reset_btn)
        zoom_bar.addWidget(self.zoom_fit_btn)
        zoom_bar.addWidget(self.zoom_label)

        piano_roll_layout.addLayout(zoom_bar)

        # The actual piano roll view (scrollable and scalable)
        self.piano_roll_view = PianoRollView()
        piano_roll_layout.addWidget(self.piano_roll_view)

        # Basic playback controls (existing placeholders retained)
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

        # Make the piano roll area take most of the vertical space
        preview_layout.setStretch(0, 1)  # piano_roll_group
        preview_layout.setStretch(1, 0)  # playback_group

        parent.addWidget(preview_widget)
        # Ensure the preview panel stretches when resizing the window
        parent.setStretchFactor(parent.indexOf(preview_widget), 1)

    # -------------------------
    # Wiring
    # -------------------------

    def connect_signals(self):
        """Connect UI signals to handlers"""
        # Basics sliders and spinboxes
        self.tempo_slider.valueChanged.connect(self.tempo_spin.setValue)
        self.tempo_spin.valueChanged.connect(self.tempo_slider.setValue)
        self.bars_slider.valueChanged.connect(self.bars_spin.setValue)
        self.bars_spin.valueChanged.connect(self.bars_slider.setValue)

        # Basics parameter changes
        self.genre_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.mood_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.key_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.mode_combo.currentTextChanged.connect(self.on_parameter_changed)
        self.tempo_spin.valueChanged.connect(self.on_parameter_changed)
        self.bars_spin.valueChanged.connect(self.on_parameter_changed)
        self.separate_files_checkbox.toggled.connect(self.on_parameter_changed)
        # Density radios: update when any toggled
        for btn in self.density_group.buttons():
            btn.toggled.connect(self.on_parameter_changed)

        # Basics buttons
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        self.play_btn.clicked.connect(self.on_play_clicked)
        self.pause_btn.clicked.connect(self.on_pause_clicked)
        self.stop_btn.clicked.connect(self.on_stop_clicked)

        # Advanced tab handlers
        self.ts_melody_combo.currentTextChanged.connect(self._on_advanced_changed)
        self.ts_harmony_combo.currentTextChanged.connect(self._on_advanced_changed)
        self.ts_bass_combo.currentTextChanged.connect(self._on_advanced_changed)
        self.ts_rhythm_combo.currentTextChanged.connect(self._on_advanced_changed)
        self.harmonic_variance_combo.currentTextChanged.connect(self._on_advanced_changed)
        self.gen_melody_checkbox.toggled.connect(self._on_advanced_changed)
        self.gen_harmony_checkbox.toggled.connect(self._on_advanced_changed)
        self.gen_bass_checkbox.toggled.connect(self._on_advanced_changed)
        self.gen_rhythm_checkbox.toggled.connect(self._on_advanced_changed)

        # Plugins tab handlers
        self.plugin_scan_btn.clicked.connect(self._on_plugin_scan_clicked)

        # Output tab handlers
        self.output_folder_edit.textChanged.connect(self._on_output_folder_changed)
        self.temp_save_btn.clicked.connect(self._on_temp_save_clicked)
        self.temp_load_btn.clicked.connect(self._on_temp_load_clicked)

        # Presets tab handlers
        self.save_preset_btn.clicked.connect(self._on_save_preset_clicked)
        self.load_preset_btn.clicked.connect(self._on_load_preset_clicked)
        self.delete_preset_btn.clicked.connect(self._on_delete_preset_clicked)

        # Preview zoom handlers
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        self.zoom_in_btn.clicked.connect(lambda: self._nudge_zoom(1.2))
        self.zoom_out_btn.clicked.connect(lambda: self._nudge_zoom(1/1.2))
        self.zoom_reset_btn.clicked.connect(self._on_zoom_reset_clicked)
        self.zoom_fit_btn.clicked.connect(self._on_zoom_fit_clicked)

    # -------------------------
    # Basics tab behavior
    # -------------------------

    def populate_subgenres(self):
        """Populate subgenres based on selected genre."""
        genre = self.genre_combo.currentText()
        self.subgenre_combo.clear()
        self.subgenre_combo.addItem('')  # Empty option

        if not genre:
            print("No genre selected, subgenres empty")  # Log
            return

        # Get rules, cache if not present
        if genre not in self.genre_rules_cache:
            try:
                rules = GenreFactory.create_genre_rules(genre)
                self.genre_rules_cache[genre] = rules
            except ValueError:
                print(f"Failed to load rules for {genre}")  # Log
                return

        rules = self.genre_rules_cache[genre]
        subgenres = sorted(rules.get_subgenres())
        self.subgenre_combo.addItems(subgenres)
        print(f"Populated subgenres for {genre}: {subgenres}")  # Log

    def on_parameter_changed(self):
        """Handle parameter changes"""
        self.current_params.update({
            'genre': self.genre_combo.currentText(),
            'subgenre': self.subgenre_combo.currentText(),
            'mood': self.mood_combo.currentText(),
            'key': self.key_combo.currentText(),
            'mode': self.mode_combo.currentText(),
            'tempo': self.tempo_spin.value(),
            'bars': self.bars_spin.value(),
            'density': self.get_selected_density(),
            'output': self.output_edit.text(),
            'separate_files': self.separate_files_checkbox.isChecked()
        })
        # Update status for quick validation
        print("Params updated (Basics):", self.current_params)

    def get_selected_density(self):
        """Get the selected density preset"""
        for button in self.density_group.buttons():
            if button.isChecked():
                return button.text().lower()
        return "balanced"

    def on_generate_clicked(self):
        """Handle generate button click"""
        try:
            self.status_bar.showMessage("Generating music...")

            # Collect parameters and immediate UI values
            params = self.current_params.copy()
            params['subgenre'] = self.subgenre_combo.currentText()  # Ensure subgenre is included

            # Resolve output folder and effective filename
            output_folder = (self.output_folder_edit.text() or "output").strip()
            raw_output = self.output_edit.text().strip()

            def _sanitize_component(val: str) -> str:
                val = str(val or "").strip().lower().replace(" ", "_")
                invalid = '<>:"/\\|?*'
                for ch in invalid:
                    val = val.replace(ch, "_")
                safe = "".join(ch for ch in val if ch.isalnum() or ch in ("_", "-")).strip("_-")
                return safe[:100] if safe else "file"

            if not raw_output:
                # Auto-generate from settings + timestamp: {genre}_{mood}_{tempo}_{bars}_{timestamp}.mid
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                g = _sanitize_component(params.get('genre', ''))
                m = _sanitize_component(params.get('mood', ''))
                t = int(params.get('tempo', 0))
                b = int(params.get('bars', 0))
                filename_only = f"{g}_{m}_{t}_{b}_{ts}.mid"
                output_path = str(Path(output_folder) / filename_only)
            else:
                p = Path(raw_output)
                if p.suffix.lower() != ".mid":
                    p = p.with_suffix(".mid")
                # If the user provided a path (absolute or contains separators), respect it; otherwise join with output folder
                if p.is_absolute() or any(sep in str(p) for sep in ("/", "\\")):
                    output_path = str(p)
                else:
                    output_path = str(Path(output_folder) / p.name)

            params['output'] = output_path

            # Create generation components (unchanged in Phase 1)
            genre_rules = GenreFactory.create_genre_rules(params['genre'])
            song_skeleton = SongSkeleton(params['genre'], params['tempo'], params['mood'])
            density_manager = create_density_manager_from_preset(params['density'])

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

            # Save MIDI (combined or separate based on option)
            midi_output = MidiOutput()
            midi_output.save_to_midi(
                song_skeleton,
                output_path,
                genre_rules,
                separate_files=params.get('separate_files', False),
                context=getattr(pattern_generator, "context", None)
            )

            # Update preview
            self.piano_roll_view.update_preview(song_skeleton)

            # Status message
            if params.get('separate_files', False):
                base = output_path.replace('.mid', '')
                self.status_bar.showMessage(f"Generated separate instrument MIDIs based on: {base}_*.mid")
            else:
                self.status_bar.showMessage(f"Generated: {output_path}")

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

    # -------------------------
    # Advanced tab behavior (UI only in Phase 1)
    # -------------------------

    def _on_advanced_changed(self):
        """Update current_params based on Advanced tab controls"""
        self.current_params["time_signatures"] = {
            "melody": self.ts_melody_combo.currentText(),
            "harmony": self.ts_harmony_combo.currentText(),
            "bass": self.ts_bass_combo.currentText(),
            "rhythm": self.ts_rhythm_combo.currentText(),
        }
        self.current_params["harmonic_variance"] = self.harmonic_variance_combo.currentText()
        self.current_params["generate_parts"] = {
            "melody": self.gen_melody_checkbox.isChecked(),
            "harmony": self.gen_harmony_checkbox.isChecked(),
            "bass": self.gen_bass_checkbox.isChecked(),
            "rhythm": self.gen_rhythm_checkbox.isChecked(),
        }
        # Performance params
        self.current_params["performance"] = self._collect_performance_params()
        print("Advanced settings changed:", {
            "time_signatures": self.current_params["time_signatures"],
            "harmonic_variance": self.current_params["harmonic_variance"],
            "generate_parts": self.current_params["generate_parts"],
            "performance": self.current_params["performance"],
        })

    # -------------------------
    # Plugins/Audio tab behavior (visibility only)
    # -------------------------

    def _on_plugin_scan_clicked(self):
        """Handle plugin scan button click with safe fallbacks"""
        raw = self.plugin_paths_edit.text().strip()
        paths = [p for p in (raw.split(";") if raw else []) if p]
        print(f"Starting plugin scan. User paths: {paths if paths else '(defaults)'}")
        self.plugin_status_label.setText("Scanning...")
        QApplication.processEvents()

        found: List[str] = []
        try:
            # Try to use PluginHost.scan_for_plugins if lightweight
            from audio.plugin_host import PluginHost  # type: ignore
            try:
                # Instantiate may fail if pedalboard is unavailable; guard it
                host = PluginHost()
                found = host.scan_for_plugins(paths if paths else None)
                print(f"PluginHost scan succeeded. Found {len(found)} plugins.")
            except Exception as e:
                print(f"PluginHost not available or failed to initialize: {e}")
                found = self._fallback_scan(paths)
        except Exception as e:
            print(f"Could not import PluginHost, falling back. Reason: {e}")
            found = self._fallback_scan(paths)

        self.plugin_list.clear()
        for fp in found:
            item = QListWidgetItem(fp)
            self.plugin_list.addItem(item)
        msg = f"Found {len(found)} plugins."
        self.plugin_status_label.setText(msg)
        print("Plugin scan completed:", msg)

    def _fallback_scan(self, paths: List[str]) -> List[str]:
        """Simple filesystem scan for plugins if PluginHost is unavailable"""
        if not paths:
            # Minimal default paths
            paths = [
                "C:\\Program Files\\VSTPlugins",
                "C:\\Program Files\\Common Files\\VST3",
                "/Library/Audio/Plug-Ins/VST",
                "/Library/Audio/Plug-Ins/VST3",
                "/usr/local/lib/vst",
                "/usr/lib/vst"
            ]
        exts = (".dll", ".vst3", ".vst", ".component", ".clap", ".so", ".dylib")
        found: List[str] = []
        for base in paths:
            if os.path.isdir(base):
                for root, _, files in os.walk(base):
                    for f in files:
                        if f.lower().endswith(exts):
                            found.append(os.path.join(root, f))
        return found

    # -------------------------
    # Output tab behavior
    # -------------------------

    def _on_browse_output_folder_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_folder_edit.text() or ".")
        if folder:
            self.output_folder_edit.setText(folder)

    def _on_output_folder_changed(self, text: str):
        self.current_params["output_folder"] = text
        print(f"Output folder set to: {text}")

    def _on_temp_save_clicked(self):
        temp_dir = self.temp_dir_edit.text().strip()
        if not temp_dir:
            QMessageBox.warning(self, "Temp Settings", "Please provide a temp directory path before saving.")
            return
        try:
            # Defaults per Phase 1 requirements
            self.config_manager.save_temp_settings(temp_dir, auto_cleanup=True, retention_hours=24, max_size_mb=500)
            print(f"Temp settings saved to: {temp_dir}")
            QMessageBox.information(self, "Temp Settings", "Temporary settings saved.")
        except Exception as e:
            print("Error saving temp settings:", e)
            QMessageBox.critical(self, "Temp Settings", f"Failed to save temp settings:\n{e}")

    def _on_temp_load_clicked(self):
        try:
            settings = self.config_manager.load_temp_settings()
            self.temp_dir_edit.setText(settings.get("temp_directory", ""))
            print("Temp settings loaded:", settings)
            QMessageBox.information(self, "Temp Settings", "Temporary settings loaded.")
        except Exception as e:
            print("Error loading temp settings:", e)
            QMessageBox.critical(self, "Temp Settings", f"Failed to load temp settings:\n{e}")

    # -------------------------
    # Presets tab behavior
    # -------------------------

    def refresh_presets_list(self):
        self.presets_list.clear()
        try:
            configs = self.config_manager.list_configurations()
            for cfg in configs:
                label = f"{cfg.get('name','(unnamed)')}  |  {cfg.get('timestamp','')}"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, cfg.get('path', ''))
                self.presets_list.addItem(item)
            print(f"Presets list refreshed. Count={len(configs)}")
        except Exception as e:
            print("Error listing configurations:", e)

    def _get_selected_preset_path(self) -> str:
        item = self.presets_list.currentItem()
        if not item:
            return ""
        path = item.data(Qt.ItemDataRole.UserRole) or ""
        return str(path)

    def _on_save_preset_clicked(self):
        try:
            path = self.config_manager.save_configuration(self.current_params)
            print(f"Configuration saved at: {path}")
            self.refresh_presets_list()
            QMessageBox.information(self, "Presets", "Current configuration saved.")
        except Exception as e:
            print("Error saving configuration:", e)
            QMessageBox.critical(self, "Presets", f"Failed to save configuration:\n{e}")

    def _on_load_preset_clicked(self):
        path = self._get_selected_preset_path()
        if not path:
            QMessageBox.warning(self, "Presets", "Select a preset to load.")
            return
        try:
            params = self.config_manager.load_configuration(path)
            print("Loaded configuration params:", params)
            # Update Basics UI fields if present
            if 'genre' in params:
                self.genre_combo.setCurrentText(params['genre'])
                self.populate_subgenres()  # repopulate subgenres before setting
            if 'subgenre' in params:
                # Only set if exists in the newly populated list
                self.subgenre_combo.setCurrentText(params['subgenre'])
            if 'mood' in params:
                self.mood_combo.setCurrentText(params['mood'])
            if 'key' in params:
                self.key_combo.setCurrentText(params['key'])
            if 'mode' in params:
                self.mode_combo.setCurrentText(params['mode'])
            if 'tempo' in params:
                self.tempo_spin.setValue(int(params['tempo']))
            if 'bars' in params:
                self.bars_spin.setValue(int(params['bars']))
            if 'density' in params:
                for b in self.density_group.buttons():
                    if b.text().lower() == params['density']:
                        b.setChecked(True)
                        break
            if 'output' in params:
                self.output_edit.setText(params['output'])
            if 'separate_files' in params:
                self.separate_files_checkbox.setChecked(bool(params['separate_files']))
            if 'output_folder' in params:
                self.output_folder_edit.setText(params['output_folder'])
                self.current_params['output_folder'] = params['output_folder']

            # Update current params once after UI updates
            self.on_parameter_changed()

            # Optional: set advanced if present (no generation wiring yet)
            adv_ts = params.get('time_signatures', {})
            if adv_ts:
                self.ts_melody_combo.setCurrentText(adv_ts.get('melody', self.ts_melody_combo.currentText()))
                self.ts_harmony_combo.setCurrentText(adv_ts.get('harmony', self.ts_harmony_combo.currentText()))
                self.ts_bass_combo.setCurrentText(adv_ts.get('bass', self.ts_bass_combo.currentText()))
                self.ts_rhythm_combo.setCurrentText(adv_ts.get('rhythm', self.ts_rhythm_combo.currentText()))
            if 'harmonic_variance' in params:
                self.harmonic_variance_combo.setCurrentText(params['harmonic_variance'])
            gp = params.get('generate_parts', {})
            if gp:
                self.gen_melody_checkbox.setChecked(bool(gp.get('melody', True)))
                self.gen_harmony_checkbox.setChecked(bool(gp.get('harmony', True)))
                self.gen_bass_checkbox.setChecked(bool(gp.get('bass', True)))
                self.gen_rhythm_checkbox.setChecked(bool(gp.get('rhythm', True)))
                self._on_advanced_changed()

            QMessageBox.information(self, "Presets", "Configuration loaded.")
        except Exception as e:
            print("Error loading configuration:", e)
            QMessageBox.critical(self, "Presets", f"Failed to load configuration:\n{e}")

    def _on_delete_preset_clicked(self):
        path = self._get_selected_preset_path()
        if not path:
            QMessageBox.warning(self, "Presets", "Select a preset to delete.")
            return
        try:
            ok = self.config_manager.delete_configuration(path)
            if ok:
                print(f"Deleted configuration: {path}")
                self.refresh_presets_list()
                QMessageBox.information(self, "Presets", "Configuration deleted.")
            else:
                QMessageBox.warning(self, "Presets", "Failed to delete configuration.")
        except Exception as e:
            print("Error deleting configuration:", e)
            QMessageBox.critical(self, "Presets", f"Failed to delete configuration:\n{e}")


    def _collect_performance_params(self) -> dict:
        """Collect performance parameters from Advanced tab controls."""
        try:
            perf = {
                "swing_mode": self.swing_mode_combo.currentText().lower() if hasattr(self, "swing_mode_combo") else "eighth",
                "micro_timing_range_ms": float(self.micro_ms_spin.value()) if hasattr(self, "micro_ms_spin") else 6.0,
                "grid_bias_ms": float(self.grid_bias_ms_spin.value()) if hasattr(self, "grid_bias_ms_spin") else 0.0,
                "note_length_variance": float(self.length_var_spin.value()) / 100.0 if hasattr(self, "length_var_spin") else 0.10,
                "staccato_prob": float(self.staccato_prob_spin.value()) / 100.0 if hasattr(self, "staccato_prob_spin") else 0.10,
                "tenuto_prob": float(self.tenuto_prob_spin.value()) / 100.0 if hasattr(self, "tenuto_prob_spin") else 0.20,
                "marcato_prob": float(self.marcato_prob_spin.value()) / 100.0 if hasattr(self, "marcato_prob_spin") else 0.10,
                "marcato_velocity_boost": 12,
                "velocity_profile": {
                    "shape": self.vel_shape_combo.currentText().lower() if hasattr(self, "vel_shape_combo") else "arch",
                    "intensity": float(self.vel_intensity_spin.value()) if hasattr(self, "vel_intensity_spin") else 0.3,
                    "phrase_length_beats": float(self.phrase_len_spin.value()) if hasattr(self, "phrase_len_spin") else 4.0,
                }
            }
            return perf
        except Exception as e:
            print("Error collecting performance params:", e)
            return {}

    # -------------------------
    # Preview zoom handlers
    # -------------------------

    def on_zoom_slider_changed(self, value: int):
        """Update piano roll zoom from slider and label."""
        try:
            self.piano_roll_view.set_zoom_percent(float(value))
            self.zoom_label.setText(f"{int(value)}%")
        except Exception:
            pass

    def _nudge_zoom(self, factor: float):
        """Nudge zoom slider by a factor."""
        current = self.zoom_slider.value()
        new_val = int(max(25, min(400, round(current * factor))))
        if new_val != current:
            self.zoom_slider.setValue(new_val)

    def _on_zoom_reset_clicked(self):
        """Reset zoom to 100% and sync UI."""
        self.piano_roll_view.reset_zoom()
        self.zoom_slider.setValue(100)
        self.zoom_label.setText("100%")

    def _on_zoom_fit_clicked(self):
        """Fit the scene to the view (does not change slider percent)."""
        self.piano_roll_view.fit_to_scene()

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