"""
Plugins Audio Tab for MIDI Master GUI
 
This module provides the functional Plugins/Audio tab with plugin loading, parameter control, and audio rendering capabilities.
"""

import sys
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSlider, QFileDialog, QProgressBar, QGroupBox, QScrollArea, QMessageBox)
from PyQt6.QtCore import pyqtSignal, QThread, pyqtSlot, Qt
from PyQt6.QtGui import QFont
from typing import Optional, Dict, Any, List

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.plugin_host import PluginHost, ENHANCED_ENUMERATION_AVAILABLE
from audio.plugin_enumeration import ParameterCollection, ParameterEnumerator, ParameterType, ParameterCategory
from output.audio_output import AudioOutput
from structures.song_skeleton import SongSkeleton
from output.midi_output import MidiOutput

class ParameterControlWidget(QWidget):
    """
    Widget to control a single plugin parameter.
    """
    value_changed = pyqtSignal(str, Any)  # param_name, value

    def __init__(self, param_name: str, param_type: ParameterType, min_val: float, max_val: float, current_val: Any, enum_values: Optional[List[str]] = None):
        super().__init__()
        self.param_name = param_name
        self.param_type = param_type
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.enum_values = enum_values or []

        layout = QHBoxLayout(self)
        layout.addWidget(QLabel(param_name.replace('_', ' ').title() + ":"))

        if param_type == ParameterType.CONTINUOUS:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)  # Normalized 0-100
            normalized = (float(current_val) - min_val) / (max_val - min_val) if max_val != min_val else 0.0
            slider.setValue(int(normalized * 100))
            slider.valueChanged.connect(self._on_slider_changed)
            layout.addWidget(slider)
            self.slider = slider
        elif param_type == ParameterType.ENUMERATION and self.enum_values:
            combo = QComboBox()
            combo.addItems(self.enum_values)
            index = self.enum_values.index(str(current_val)) if str(current_val) in self.enum_values else 0
            combo.setCurrentIndex(index)
            combo.currentIndexChanged.connect(self._on_combo_changed)
            layout.addWidget(combo)
            self.combo = combo
        elif param_type == ParameterType.BOOLEAN:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1)
            slider.setValue(1 if current_val else 0)
            slider.valueChanged.connect(self._on_slider_changed)
            layout.addWidget(slider)
            self.slider = slider

        self.value_label = QLabel(str(current_val))
        layout.addWidget(self.value_label)

    def _on_slider_changed(self, value):
        if self.param_type == ParameterType.CONTINUOUS:
            normalized = value / 100.0
            actual_val = self.min_val + normalized * (self.max_val - self.min_val)
            self.current_val = actual_val
            self.value_label.setText(f"{actual_val:.2f}")
        else:  # BOOLEAN
            actual_val = bool(value)
            self.current_val = actual_val
            self.value_label.setText(str(actual_val))
        self.value_changed.emit(self.param_name, self.current_val)

    def _on_combo_changed(self, index):
        self.current_val = self.enum_values[index] if index < len(self.enum_values) else ""
        self.value_label.setText(str(self.current_val))
        self.value_changed.emit(self.param_name, self.current_val)

    def get_value(self):
        return self.current_val

class RenderingWorker(QThread):
    """
    Worker thread for rendering audio to avoid blocking the GUI.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, plugin_host: PluginHost, song_skeleton: SongSkeleton, output_path: str):
        super().__init__()
        self.plugin_host = plugin_host
        self.song_skeleton = song_skeleton
        self.output_path = output_path

    def run(self):
        try:
            self.progress.emit(10)
            audio_output = AudioOutput(self.plugin_host)
            success = audio_output.render_song_to_audio(self.song_skeleton, self.output_path)
            self.progress.emit(100)
            if success:
                self.finished.emit(True, f"Audio rendered successfully to {self.output_path}")
            else:
                self.finished.emit(False, "Audio rendering failed")
        except Exception as e:
            self.finished.emit(False, f"Error during rendering: {str(e)}")

class PluginsAudioTab(QWidget):
    """
    Main widget for the Plugins/Audio tab.
    """
    song_skeleton_ready = pyqtSignal(SongSkeleton)  # Signal to receive song skeleton from main window

    def __init__(self):
        super().__init__()
        self.plugin_host = PluginHost()
        self.plugin_paths = []
        self.loaded_plugin = None
        self.parameter_controls = {}
        self.song_skeleton = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Plugin Selection Group
        plugin_group = QGroupBox("Plugin Selection")
        plugin_layout = QVBoxLayout(plugin_group)

        path_layout = QHBoxLayout()
        self.plugin_combo = QComboBox()
        path_layout.addWidget(QLabel("Plugin:"))
        path_layout.addWidget(self.plugin_combo)
        self.scan_button = QPushButton("Scan for Plugins")
        self.scan_button.clicked.connect(self.scan_plugins)
        path_layout.addWidget(self.scan_button)
        self.load_button = QPushButton("Load Plugin")
        self.load_button.clicked.connect(self._load_plugin)
        path_layout.addWidget(self.load_button)
        plugin_layout.addLayout(path_layout)

        layout.addWidget(plugin_group)

        # Parameter Controls Group
        self.param_group = QGroupBox("Plugin Parameters")
        self.param_layout = QVBoxLayout(self.param_group)
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setWidget(QWidget())
        self.param_layout.addWidget(self.param_scroll)
        layout.addWidget(self.param_group, 1)

        # Rendering Group
        render_group = QGroupBox("Audio Rendering")
        render_layout = QVBoxLayout(render_group)

        file_layout = QHBoxLayout()
        self.output_file_label = QLabel("Output File: output.wav")
        file_layout.addWidget(self.output_file_label)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self._browse_output_file)
        file_layout.addWidget(self.browse_button)
        render_layout.addLayout(file_layout)

        self.render_button = QPushButton("Render Audio")
        self.render_button.clicked.connect(self._start_rendering)
        render_layout.addWidget(self.render_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        render_layout.addWidget(self.progress_bar)

        layout.addWidget(render_group)

        # Connect signal for song skeleton
        self.song_skeleton_ready.connect(self._on_song_skeleton_ready)

    @pyqtSlot(SongSkeleton)
    def _on_song_skeleton_ready(self, skeleton: SongSkeleton):
        self.song_skeleton = skeleton

    def scan_plugins(self):
        """Public method to scan plugins (for menu and button connection)."""
        self.plugin_paths = self.plugin_host.scan_for_plugins()
        self.plugin_combo.clear()
        for path in self.plugin_paths:
            self.plugin_combo.addItem(os.path.basename(path))
        if self.plugin_paths:
            self.plugin_combo.setCurrentIndex(0)

    def _load_plugin(self):
        index = self.plugin_combo.currentIndex()
        if index >= 0 and index < len(self.plugin_paths):
            path = self.plugin_paths[index]
            if self.plugin_host.load_plugin(path):
                self.loaded_plugin = self.plugin_host.loaded_plugin
                self._load_parameters()
                QMessageBox.information(self, "Success", "Plugin loaded successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to load plugin.")

    def _load_parameters(self):
        # Clear existing controls
        for widget in self.param_layout.children():
            if isinstance(widget, QWidget):
                widget.deleteLater()
        self.parameter_controls.clear()

        if not ENHANCED_ENUMERATION_AVAILABLE or not self.plugin_host.loaded_plugin:
            return

        collection = self.plugin_host.get_detailed_parameters()
        if not collection:
            return

        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)

        for param in collection.parameters.values():
            meta = param.metadata
            control = ParameterControlWidget(
                meta.name,
                meta.parameter_type,
                (meta.min_value or 0.0),
                (meta.max_value or 1.0),
                param.current_value,
                meta.enum_values if meta.parameter_type == ParameterType.ENUMERATION else None
            )
            control.value_changed.connect(self._on_parameter_changed)
            param_layout.addWidget(control)
            self.parameter_controls[meta.name] = control

        self.param_scroll.setWidget(param_widget)

    def _on_parameter_changed(self, param_name: str, value: Any):
        if self.plugin_host.loaded_plugin and hasattr(self.plugin_host.loaded_plugin, 'parameters'):
            # Update plugin parameter (assuming pedalboard supports it)
            try:
                updated = False
                for p in self.plugin_host.loaded_plugin.parameters:  # type: ignore[attr-defined]
                    if p.name == param_name:
                        p.value = value
                        updated = True
                        break
                if not updated:
                    print(f"Parameter '{param_name}' not found in plugin.")
            except Exception as e:
                print(f"Error updating parameter {param_name}: {e}")

    def _browse_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Audio File", "output.wav", "WAV Files (*.wav)")
        if file_path:
            self.output_file_label.setText(f"Output File: {os.path.basename(file_path)}")
            self.output_path = file_path

    def _start_rendering(self):
        if not self.song_skeleton:
            QMessageBox.warning(self, "Error", "No song skeleton available. Generate a song first.")
            return
        if not self.plugin_host.loaded_plugin:
            QMessageBox.warning(self, "Error", "No plugin loaded. Load a plugin first.")
            return
        if not hasattr(self, 'output_path'):
            self.output_path = "output.wav"

        self.render_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = RenderingWorker(self.plugin_host, self.song_skeleton, self.output_path)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_rendering_finished)
        self.worker.start()

    def _on_rendering_finished(self, success: bool, message: str):
        self.render_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Error", message)