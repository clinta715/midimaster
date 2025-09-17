"""
Parameter Controls Widget for MIDI Master GUI

This module provides a control panel for selecting generation parameters and emitting
updated ParameterConfig objects.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QCheckBox,
    QSpinBox,
    QLineEdit,
    QGroupBox,
    QFormLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QInputDialog,
)
from PyQt6.QtCore import pyqtSignal
from config.parameter_config import ParameterConfig, Genre, Mood, Density, HarmonicVariance
from genres.genre_factory import GenreFactory
from core.settings_preset_manager import SettingsPresetManager
from core.config_loader import load_settings_json
from gui.settings_helpers import (
    build_preview_filename,
    validate_template_str,
    validate_rhythms_path,
    persist_rhythms_default,
)

class ParameterControls(QWidget):
    """
    Widget for controlling song generation parameters.
    Emits parameters_updated signal with updated ParameterConfig.
    """
    parameters_updated = pyqtSignal(object)  # ParameterConfig

    def __init__(self):
        super().__init__()
        self._init_ui()
        self._config = self._create_default_config()
        self.parameters_updated.emit(self._config)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Genre
        genre_layout = QHBoxLayout()
        genre_layout.addWidget(QLabel("Genre:"))
        self.genre_combo = QComboBox()
        genres = GenreFactory.get_available_genres()
        self.genre_combo.addItems(genres)
        # Ensure default selection exists
        if "pop" in genres:
            self.genre_combo.setCurrentText("pop")
        self.genre_combo.currentTextChanged.connect(self._update_config)
        genre_layout.addWidget(self.genre_combo)
        layout.addLayout(genre_layout)

        # Tempo
        tempo_layout = QHBoxLayout()
        tempo_layout.addWidget(QLabel("Tempo:"))
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(60, 200)
        self.tempo_spin.setValue(120)
        self.tempo_spin.valueChanged.connect(self._update_config)
        tempo_layout.addWidget(self.tempo_spin)
        layout.addLayout(tempo_layout)

        # Mood
        mood_layout = QHBoxLayout()
        mood_layout.addWidget(QLabel("Mood:"))
        self.mood_combo = QComboBox()
        self.mood_combo.addItems(["calm", "happy", "sad", "energetic"])
        self.mood_combo.setCurrentText("happy")
        self.mood_combo.currentTextChanged.connect(self._update_config)
        mood_layout.addWidget(self.mood_combo)
        layout.addLayout(mood_layout)

        # Density
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Density:"))
        self.density_combo = QComboBox()
        self.density_combo.addItems(["minimal", "sparse", "balanced", "dense", "complex"])
        self.density_combo.setCurrentText("balanced")
        self.density_combo.currentTextChanged.connect(self._update_config)
        density_layout.addWidget(self.density_combo)
        layout.addLayout(density_layout)

        # Bars
        bars_layout = QHBoxLayout()
        bars_layout.addWidget(QLabel("Bars:"))
        self.bars_spin = QSpinBox()
        self.bars_spin.setRange(4, 64)
        self.bars_spin.setValue(32)
        self.bars_spin.valueChanged.connect(self._update_config)
        bars_layout.addWidget(self.bars_spin)
        layout.addLayout(bars_layout)

        # Harmonic Variance
        variance_layout = QHBoxLayout()
        variance_layout.addWidget(QLabel("Harmonic Variance:"))
        self.variance_combo = QComboBox()
        self.variance_combo.addItems(["close", "medium", "distant"])
        self.variance_combo.setCurrentText("medium")
        self.variance_combo.currentTextChanged.connect(self._update_config)
        variance_layout.addWidget(self.variance_combo)
        layout.addLayout(variance_layout)

        # Separate Files
        self.separate_files_check = QCheckBox("Separate Files")
        self.separate_files_check.stateChanged.connect(self._update_config)
        layout.addWidget(self.separate_files_check)

        # Advanced settings group
        adv_group = QGroupBox("Advanced Settings")
        adv_form = QFormLayout(adv_group)

        # Subgenre
        self.subgenre_edit = QLineEdit()
        self.subgenre_edit.setPlaceholderText("e.g., deep_house, drill, liquid")
        self.subgenre_edit.textChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Subgenre:"), self.subgenre_edit)

        # Output MIDI path
        self.output_edit = QLineEdit("output/generated.mid")
        self.output_edit.textChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Output MIDI Path:"), self.output_edit)

        # Render audio toggle
        self.render_audio_check = QCheckBox("Render Audio")
        self.render_audio_check.stateChanged.connect(self._update_config)
        adv_form.addRow(self.render_audio_check)

        # Plugin path
        self.plugin_path_edit = QLineEdit()
        self.plugin_path_edit.setPlaceholderText("C:/Plugins/Synth.dll")
        self.plugin_path_edit.textChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Plugin Path:"), self.plugin_path_edit)

        # Audio output
        self.audio_output_edit = QLineEdit()
        self.audio_output_edit.setPlaceholderText("output.wav")
        self.audio_output_edit.textChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Audio Output:"), self.audio_output_edit)

        # Pattern strength (discrete to match CLI)
        self.pattern_strength_combo = QComboBox()
        self.pattern_strength_combo.addItems(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        self.pattern_strength_combo.setCurrentText("1.0")
        self.pattern_strength_combo.currentTextChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Pattern Strength:"), self.pattern_strength_combo)

        # Swing percent (0.0 .. 1.0 step 0.1)
        self.swing_percent_combo = QComboBox()
        self.swing_percent_combo.addItems([f"{x/10:.1f}" for x in range(0, 11)])
        self.swing_percent_combo.setCurrentText("0.5")
        self.swing_percent_combo.currentTextChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Swing Percent:"), self.swing_percent_combo)

        # Fill frequency options
        self.fill_frequency_combo = QComboBox()
        self.fill_frequency_combo.addItems(["0.0", "0.1", "0.25", "0.33", "0.5"])
        self.fill_frequency_combo.setCurrentText("0.25")
        self.fill_frequency_combo.currentTextChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Fill Frequency:"), self.fill_frequency_combo)

        # Ghost note level options
        self.ghost_note_level_combo = QComboBox()
        self.ghost_note_level_combo.addItems(["0.0", "0.3", "0.5", "1.0", "1.5", "2.0"])
        self.ghost_note_level_combo.setCurrentText("1.0")
        self.ghost_note_level_combo.currentTextChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Ghost Note Level:"), self.ghost_note_level_combo)

        # Per-track time signatures
        self.melody_ts_edit = QLineEdit("4/4")
        self.melody_ts_edit.textChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Melody Time Sig:"), self.melody_ts_edit)

        self.harmony_ts_edit = QLineEdit("4/4")
        self.harmony_ts_edit.textChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Harmony Time Sig:"), self.harmony_ts_edit)

        self.bass_ts_edit = QLineEdit("4/4")
        self.bass_ts_edit.textChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Bass Time Sig:"), self.bass_ts_edit)

        self.rhythm_ts_edit = QLineEdit("4/4")
        self.rhythm_ts_edit.textChanged.connect(self._update_config)
        adv_form.addRow(QLabel("Rhythm Time Sig:"), self.rhythm_ts_edit)

        layout.addWidget(adv_group)

        # ---------------------------
        # New: Presets / Rhythms DB / Filename Template
        # ---------------------------
        settings_group = QGroupBox("Session Settings")
        settings_v = QVBoxLayout(settings_group)

        # Presets row
        presets_box = QGroupBox("Presets")
        presets_layout = QHBoxLayout(presets_box)
        self.preset_manager = SettingsPresetManager()
        self.presets_combo = QComboBox()
        self._refresh_presets_list()
        presets_layout.addWidget(self.presets_combo)

        self.preset_save_btn = QPushButton("Save")
        self.preset_save_btn.clicked.connect(self._on_save_preset)
        presets_layout.addWidget(self.preset_save_btn)

        self.preset_load_btn = QPushButton("Load")
        self.preset_load_btn.clicked.connect(self._on_load_preset)
        presets_layout.addWidget(self.preset_load_btn)

        self.preset_delete_btn = QPushButton("Delete")
        self.preset_delete_btn.clicked.connect(self._on_delete_preset)
        presets_layout.addWidget(self.preset_delete_btn)

        settings_v.addWidget(presets_box)

        # Rhythms DB row
        rhythms_box = QGroupBox("Rhythms Database")
        rhythms_form = QFormLayout(rhythms_box)

        self.rhythms_db_edit = QLineEdit()
        self.rhythms_db_edit.setPlaceholderText("Leave empty to use default resolver (reference_midis/)")
        rhythms_db_row = QHBoxLayout()
        rhythms_db_row.addWidget(self.rhythms_db_edit)
        self.rhythms_browse_btn = QPushButton("Browse…")
        self.rhythms_browse_btn.clicked.connect(self._on_browse_rhythms_db)
        rhythms_db_row.addWidget(self.rhythms_browse_btn)
        # Place the editable row under the "Path:" label using a container widget
        path_row_widget = QWidget()
        path_row_widget.setLayout(rhythms_db_row)
        rhythms_form.addRow(QLabel("Path:"), path_row_widget)

        self.rhythms_error_label = QLabel("")
        self.rhythms_error_label.setStyleSheet("color: #d9534f;")  # red
        rhythms_form.addRow(QLabel(""), self.rhythms_error_label)

        rhythms_apply_row = QHBoxLayout()
        self.rhythms_persist_check = QCheckBox("Set as default")
        rhythms_apply_row.addWidget(self.rhythms_persist_check)
        self.rhythms_apply_btn = QPushButton("Apply")
        self.rhythms_apply_btn.clicked.connect(self._apply_rhythms_db)
        rhythms_apply_row.addWidget(self.rhythms_apply_btn)
        apply_row_widget = QWidget()
        apply_row_widget.setLayout(rhythms_apply_row)
        rhythms_form.addRow(QLabel(""), apply_row_widget)

        self.rhythms_hint_label = QLabel("Default: reference_midis/")
        self.rhythms_hint_label.setStyleSheet("color: #9aa3b2;")
        rhythms_form.addRow(QLabel(""), self.rhythms_hint_label)

        settings_v.addWidget(rhythms_box)

        # Filename Template row
        template_box = QGroupBox("Filename Template")
        template_form = QFormLayout(template_box)

        self.template_edit = QLineEdit()
        self.template_edit.setPlaceholderText("e.g., {genre}/{mood}/{genre}_{mood}_{tempo}_{bars}")
        self.template_edit.textChanged.connect(self._on_template_changed)
        template_form.addRow(QLabel("Template:"), self.template_edit)

        self.template_error_label = QLabel("")
        self.template_error_label.setStyleSheet("color: #d9534f;")
        template_form.addRow(QLabel(""), self.template_error_label)

        self.template_preview_label = QLabel("Preview: ")
        self.template_preview_label.setStyleSheet("color: #9aa3b2;")
        template_form.addRow(QLabel(""), self.template_preview_label)

        settings_v.addWidget(template_box)

        layout.addWidget(settings_group)

        # Load optional persisted GUI settings (filename_template, rhythms_db_path)
        try:
            _cfg = load_settings_json("configs/settings.json")
            if isinstance(_cfg, dict):
                ft = _cfg.get("filename_template")
                if isinstance(ft, str):
                    self.template_edit.setText(ft)
                rdb = _cfg.get("rhythms_db_path")
                if isinstance(rdb, str):
                    self.rhythms_db_edit.setText(rdb)
        except Exception:
            pass

        # Track template validity state
        self._template_valid = True
        # Initialize preview based on starting values
        self._on_template_changed(self.template_edit.text())

        # Internal defaults not UI-controlled for simplicity are in _create_default_config()

    def _create_default_config(self):
        config = ParameterConfig()
        config.genre = Genre.POP  # Default
        config.tempo = 120
        config.mood = Mood.HAPPY
        config.density = Density.BALANCED
        config.separate_files = False
        config.subgenre = ""
        config.harmonic_variance = HarmonicVariance.MEDIUM
        config.pattern_strength = 1.0
        config.swing_percent = 0.5
        config.fill_frequency = 0.25
        config.ghost_note_level = 1.0
        config.output = "output/generated.mid"
        # Time signatures default to 4/4
        config.melody_time_signature = "4/4"
        config.harmony_time_signature = "4/4"
        config.bass_time_signature = "4/4"
        config.rhythm_time_signature = "4/4"
        config.bars = 32
        return config

    def _update_config(self):
        # Sync internal config from UI controls
        try:
            self._config.genre = Genre(self.genre_combo.currentText())
        except Exception:
            self._config.genre = Genre.POP

        try:
            self._config.mood = Mood(self.mood_combo.currentText())
        except Exception:
            self._config.mood = Mood.HAPPY

        try:
            self._config.density = Density(self.density_combo.currentText())
        except Exception:
            self._config.density = Density.BALANCED

        self._config.tempo = int(self.tempo_spin.value())
        self._config.bars = int(self.bars_spin.value())
        try:
            self._config.harmonic_variance = HarmonicVariance(self.variance_combo.currentText())
        except Exception:
            self._config.harmonic_variance = HarmonicVariance.MEDIUM
        self._config.separate_files = bool(self.separate_files_check.isChecked())

        # Advanced mappings
        self._config.subgenre = self.subgenre_edit.text().strip()
        self._config.output = self.output_edit.text().strip()
        self._config.render_audio = bool(self.render_audio_check.isChecked())
        self._config.plugin_path = self.plugin_path_edit.text().strip()
        self._config.audio_output = self.audio_output_edit.text().strip()

        try:
            self._config.pattern_strength = float(self.pattern_strength_combo.currentText())
        except Exception:
            self._config.pattern_strength = 1.0

        try:
            self._config.swing_percent = float(self.swing_percent_combo.currentText())
        except Exception:
            self._config.swing_percent = 0.5

        try:
            self._config.fill_frequency = float(self.fill_frequency_combo.currentText())
        except Exception:
            self._config.fill_frequency = 0.25

        try:
            self._config.ghost_note_level = float(self.ghost_note_level_combo.currentText())
        except Exception:
            self._config.ghost_note_level = 1.0

        # Time signatures
        self._config.melody_time_signature = self.melody_ts_edit.text().strip() or "4/4"
        self._config.harmony_time_signature = self.harmony_ts_edit.text().strip() or "4/4"
        self._config.bass_time_signature = self.bass_ts_edit.text().strip() or "4/4"
        self._config.rhythm_time_signature = self.rhythm_ts_edit.text().strip() or "4/4"

        # Emit updated config
        self.parameters_updated.emit(self._config)

    def get_config(self):
        """Return current parameters as dict for worker and CLI mapping."""
        cfg = {
            'genre': self.genre_combo.currentText(),
            'mood': self.mood_combo.currentText(),
            'tempo': int(self.tempo_spin.value()),
            'density': self.density_combo.currentText(),
            'separate_files': self.separate_files_check.isChecked(),
            'bars': int(self.bars_spin.value()),
            'harmonic_variance': self.variance_combo.currentText(),

            # Advanced/CLI-equivalent keys
            'subgenre': self.subgenre_edit.text().strip(),
            'output': self.output_edit.text().strip(),
            'render_audio': self.render_audio_check.isChecked(),
            'plugin_path': self.plugin_path_edit.text().strip(),
            'audio_output': self.audio_output_edit.text().strip(),
            'pattern_strength': float(self.pattern_strength_combo.currentText()),
            'swing_percent': float(self.swing_percent_combo.currentText()),
            'fill_frequency': float(self.fill_frequency_combo.currentText()),
            'ghost_note_level': float(self.ghost_note_level_combo.currentText()),
            'melody_time_signature': self.melody_ts_edit.text().strip() or '4/4',
            'harmony_time_signature': self.harmony_ts_edit.text().strip() or '4/4',
            'bass_time_signature': self.bass_ts_edit.text().strip() or '4/4',
            'rhythm_time_signature': self.rhythm_ts_edit.text().strip() or '4/4',

            # For GUI-only systems and ConfigManager usage
            'key': 'C',
            'mode': 'major',
            'time_signature': '4/4',
        }

        # Optional enhancements for templated filenames
        tmpl = (self.template_edit.text() or "").strip() if hasattr(self, "template_edit") else ""
        cfg['filename_template'] = tmpl or ""

        # Provide minimal template settings snapshot
        cfg['template_settings'] = {
            "genre": cfg['genre'],
            "mood": cfg['mood'],
            "tempo": cfg['tempo'],
            "bars": cfg['bars'],
        }

        # Optional rhythms DB path (None when empty to keep legacy behavior)
        if hasattr(self, "rhythms_db_edit"):
            rdb = (self.rhythms_db_edit.text() or "").strip()
            cfg['rhythms_db_path'] = rdb if rdb else None

        return cfg

    def build_cli_command(self) -> str:
        """Build a CLI command for main.py from the current control values."""
        cfg = self.get_config()

        def quote(v: str) -> str:
            v = str(v)
            return f'"{v}"' if (' ' in v or '\\' in v) else v

        args = [
            "python", "main.py",
            "--genre", cfg['genre'],
            "--tempo", str(cfg['tempo']),
            "--mood", cfg['mood'],
            "--density", cfg['density'],
            "--bars", str(cfg['bars']),
            "--harmonic-variance", cfg['harmonic_variance'],
            "--pattern-strength", f"{cfg['pattern_strength']}",
            "--swing-percent", f"{cfg['swing_percent']}",
            "--fill-frequency", f"{cfg['fill_frequency']}",
            "--ghost-note-level", f"{cfg['ghost_note_level']}",
            "--melody-time-signature", cfg['melody_time_signature'],
            "--harmony-time-signature", cfg['harmony_time_signature'],
            "--bass-time-signature", cfg['bass_time_signature'],
            "--rhythm-time-signature", cfg['rhythm_time_signature'],
        ]

        if cfg.get('separate_files'):
            args.append("--separate-files")

        if cfg.get('subgenre'):
            args += ["--subgenre", cfg['subgenre']]

        if cfg.get('output'):
            args += ["--output", quote(cfg['output'])]

        if cfg.get('render_audio'):
            args.append("--render-audio")
            if cfg.get('plugin_path'):
                args += ["--plugin-path", quote(cfg['plugin_path'])]
            if cfg.get('audio_output'):
                args += ["--audio-output", quote(cfg['audio_output'])]

        return " ".join(args)

    # ---- New: support loading external config dicts (used by MainWindow.open_file) ----
    def load_config(self, data: dict) -> None:
        """
        Load parameters from a generic dict and apply to controls if keys present.

        Expected keys (strings unless noted):
          - genre, mood, density, harmonic_variance (strings)
          - tempo, bars (ints)
          - separate_files (bool)
          - subgenre, output, plugin_path, audio_output (strings)
          - render_audio (bool)
          - pattern_strength, swing_percent, fill_frequency, ghost_note_level (floats/strings)
          - melody_time_signature, harmony_time_signature, bass_time_signature, rhythm_time_signature (strings)
        Unknown or invalid values are ignored safely.
        """
        if not isinstance(data, dict):
            return

        # Genre
        if "genre" in data:
            genre_val = str(data.get("genre", "")).lower()
            options = [self.genre_combo.itemText(i) for i in range(self.genre_combo.count())]
            if genre_val in options:
                self.genre_combo.setCurrentText(genre_val)

        # Mood
        if "mood" in data:
            mood_val = str(data.get("mood", "")).lower()
            options = [self.mood_combo.itemText(i) for i in range(self.mood_combo.count())]
            if mood_val in options:
                self.mood_combo.setCurrentText(mood_val)

        # Density
        if "density" in data:
            dens_val = str(data.get("density", "")).lower()
            options = [self.density_combo.itemText(i) for i in range(self.density_combo.count())]
            if dens_val in options:
                self.density_combo.setCurrentText(dens_val)

        # Harmonic variance
        if "harmonic_variance" in data:
            var_val = str(data.get("harmonic_variance", "")).lower()
            options = [self.variance_combo.itemText(i) for i in range(self.variance_combo.count())]
            if var_val in options:
                self.variance_combo.setCurrentText(var_val)

        # Tempo
        if "tempo" in data:
            try:
                tempo = int(data["tempo"])
                self.tempo_spin.setValue(max(self.tempo_spin.minimum(), min(self.tempo_spin.maximum(), tempo)))
            except Exception:
                pass

        # Bars
        if "bars" in data:
            try:
                bars = int(data["bars"])
                self.bars_spin.setValue(max(self.bars_spin.minimum(), min(self.bars_spin.maximum(), bars)))
            except Exception:
                pass

        # Separate files
        if "separate_files" in data:
            try:
                self.separate_files_check.setChecked(bool(data["separate_files"]))
            except Exception:
                pass

        # Subgenre
        if "subgenre" in data:
            try:
                self.subgenre_edit.setText(str(data.get("subgenre", "")).strip())
            except Exception:
                pass

        # Output path
        if "output" in data:
            try:
                self.output_edit.setText(str(data.get("output", "")).strip())
            except Exception:
                pass

        # Render audio
        if "render_audio" in data:
            try:
                self.render_audio_check.setChecked(bool(data.get("render_audio")))
            except Exception:
                pass

        # Plugin path
        if "plugin_path" in data:
            try:
                self.plugin_path_edit.setText(str(data.get("plugin_path", "")).strip())
            except Exception:
                pass

        # Audio output
        if "audio_output" in data:
            try:
                self.audio_output_edit.setText(str(data.get("audio_output", "")).strip())
            except Exception:
                pass

        # Pattern strength
        if "pattern_strength" in data and data.get("pattern_strength") is not None:
            try:
                raw = data.get("pattern_strength")
                fval = float(raw)  # type: ignore[arg-type]
                val = str(fval)  # normalized to one of ["0.0","0.2",...,"1.0"]
                options = [self.pattern_strength_combo.itemText(i) for i in range(self.pattern_strength_combo.count())]
                if val in options:
                    self.pattern_strength_combo.setCurrentText(val)
            except Exception:
                pass

        # Swing percent
        if "swing_percent" in data and data.get("swing_percent") is not None:
            try:
                raw = data.get("swing_percent")
                fval = float(raw)  # type: ignore[arg-type]
                val = f"{fval:.1f}"
                options = [self.swing_percent_combo.itemText(i) for i in range(self.swing_percent_combo.count())]
                if val in options:
                    self.swing_percent_combo.setCurrentText(val)
            except Exception:
                pass

        # Fill frequency
        if "fill_frequency" in data and data.get("fill_frequency") is not None:
            try:
                raw = data.get("fill_frequency")
                val = str(float(raw))  # type: ignore[arg-type]
                # Handle common string forms
                if val == "0.33000000000000002":
                    val = "0.33"
                options = [self.fill_frequency_combo.itemText(i) for i in range(self.fill_frequency_combo.count())]
                if val in options:
                    self.fill_frequency_combo.setCurrentText(val)
            except Exception:
                pass

        # Ghost note level
        if "ghost_note_level" in data and data.get("ghost_note_level") is not None:
            try:
                raw = data.get("ghost_note_level")
                val = str(float(raw))  # type: ignore[arg-type]
                options = [self.ghost_note_level_combo.itemText(i) for i in range(self.ghost_note_level_combo.count())]
                if val in options:
                    self.ghost_note_level_combo.setCurrentText(val)
            except Exception:
                pass

        # Per-track time signatures
        if "melody_time_signature" in data:
            try:
                self.melody_ts_edit.setText(str(data.get("melody_time_signature", "")).strip())
            except Exception:
                pass
        if "harmony_time_signature" in data:
            try:
                self.harmony_ts_edit.setText(str(data.get("harmony_time_signature", "")).strip())
            except Exception:
                pass
        if "bass_time_signature" in data:
            try:
                self.bass_ts_edit.setText(str(data.get("bass_time_signature", "")).strip())
            except Exception:
                pass
        if "rhythm_time_signature" in data:
            try:
                self.rhythm_ts_edit.setText(str(data.get("rhythm_time_signature", "")).strip())
            except Exception:
                pass

        # After applying, sync internal config and emit
        self._update_config()

    # ---------------------------
    # New helpers/slots for Session Settings
    # ---------------------------
    def _refresh_presets_list(self) -> None:
        try:
            names = self.preset_manager.list_presets()
        except Exception:
            names = []
        self.presets_combo.clear()
        if names:
            self.presets_combo.addItems(names)

    def _current_template_settings(self) -> dict:
        return {
            "genre": self.genre_combo.currentText(),
            "mood": self.mood_combo.currentText(),
            "tempo": int(self.tempo_spin.value()),
            "bars": int(self.bars_spin.value()),
        }

    def _on_save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not str(name).strip():
            return
        try:
            # Build a minimal settings dict; manager will normalize to required schema
            data = {
                "genre": self.genre_combo.currentText(),
                "mood": self.mood_combo.currentText(),
                "tempo": int(self.tempo_spin.value()),
                "bars": int(self.bars_spin.value()),
                "time_signature": "4/4",
                "complexity": self.density_combo.currentText(),  # map density -> complexity
                "filename_template": (self.template_edit.text() or "").strip() if hasattr(self, "template_edit") else "",
                "rhythms_db_path": (self.rhythms_db_edit.text() or "").strip() if hasattr(self, "rhythms_db_edit") else "",
                "version": SettingsPresetManager.SCHEMA_VERSION,
            }
            ok_saved = self.preset_manager.save_preset(str(name).strip(), data)
            if ok_saved:
                self._refresh_presets_list()
                QMessageBox.information(self, "Preset", f"Preset saved: {name}")
            else:
                QMessageBox.warning(self, "Preset", "Failed to save preset (validation error).")
        except Exception as e:
            QMessageBox.critical(self, "Preset Error", f"Failed to save preset: {e}")

    def _on_load_preset(self) -> None:
        name = self.presets_combo.currentText()
        if not name:
            return
        try:
            data = self.preset_manager.load_preset(name)
            if not isinstance(data, dict):
                QMessageBox.warning(self, "Preset", "Failed to load preset data.")
                return
            # Translate complexity -> density for GUI
            mapped = dict(data)
            if "complexity" in data and "density" not in mapped:
                mapped["density"] = str(data.get("complexity"))
            # Apply to existing controls via loader
            self.load_config(mapped)
            # Also apply filename_template and rhythms_db_path to fields
            ft = data.get("filename_template")
            if isinstance(ft, str) and hasattr(self, "template_edit"):
                self.template_edit.setText(ft)
            rdb = data.get("rhythms_db_path")
            if isinstance(rdb, str) and hasattr(self, "rhythms_db_edit"):
                self.rhythms_db_edit.setText(rdb)
            QMessageBox.information(self, "Preset", f"Preset loaded: {name}")
        except Exception as e:
            QMessageBox.critical(self, "Preset Error", f"Failed to load preset: {e}")

    def _on_delete_preset(self) -> None:
        name = self.presets_combo.currentText()
        if not name:
            return
        resp = QMessageBox.question(self, "Delete Preset", f"Delete preset '{name}'?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            if self.preset_manager.delete_preset(name):
                self._refresh_presets_list()
                QMessageBox.information(self, "Preset", f"Deleted preset: {name}")
            else:
                QMessageBox.warning(self, "Preset", "Preset could not be deleted.")
        except Exception as e:
            QMessageBox.critical(self, "Preset Error", f"Failed to delete preset: {e}")

    def _on_browse_rhythms_db(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Rhythms DB Directory", "")
        if directory:
            self.rhythms_db_edit.setText(directory)
            # Do not automatically persist; validate on Apply
            self.rhythms_error_label.setText("")

    def _apply_rhythms_db(self) -> None:
        path = (self.rhythms_db_edit.text() or "").strip()
        if not path:
            # Empty => defer to resolver default (legacy)
            self.rhythms_error_label.setText("")
            QMessageBox.information(self, "Rhythms DB", "Cleared override. Default resolver will be used.")
            return
        ok, reason = validate_rhythms_path(path)
        if not ok:
            self.rhythms_error_label.setText(reason)
            QMessageBox.warning(self, "Rhythms DB", f"Invalid path: {reason}")
            return
        self.rhythms_error_label.setText("OK")
        if self.rhythms_persist_check.isChecked():
            try:
                persist_rhythms_default(path)
                QMessageBox.information(self, "Rhythms DB", "Default path persisted to configs/settings.json")
            except Exception as e:
                QMessageBox.critical(self, "Rhythms DB", f"Failed to persist default: {e}")

    def _on_template_changed(self, text: str) -> None:
        template = (text or "").strip()
        if not template:
            self._template_valid = True
            self.template_error_label.setText("")
            self.template_preview_label.setText("Preview: (using legacy auto-naming)")
            return
        ok, reason = validate_template_str(template)
        if not ok:
            self._template_valid = False
            self.template_error_label.setText(reason)
            self.template_preview_label.setText("Preview: —")
            return
        # Valid template => build preview
        self._template_valid = True
        self.template_error_label.setText("")
        try:
            preview = build_preview_filename(self._current_template_settings(), template, base_dir=None)
            self.template_preview_label.setText(f"Preview: {preview}")
        except Exception as e:
            self.template_preview_label.setText(f"Preview error: {e}")

    def is_template_valid(self) -> bool:
        """
        True when template is empty or valid, False when invalid string present.
        """
        return bool(getattr(self, "_template_valid", True))