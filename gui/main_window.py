"""
Enhanced Main GUI Window for MIDI Master

This module defines the main application window using PyQt6 with integrated
parameter controls, matrix visualization, piano roll preview, and playback.

Key improvements in this revision:
- Auto-preview on startup using default parameters (no user interaction required)
- New Song / Generate now auto-generates contextual filenames (no Save dialog prompt)
- Persist and reuse last generated song path via ConfigManager
- Preview tab automatically displays the latest generated song
- Plugins tab is notified with the latest SongSkeleton for rendering workflows
- Fixed thread cancellation by adding GenerationWorker.cancel()
- Simplified Save/Open config via ConfigManager helpers
- Basic visual polish via a simple application stylesheet
"""

import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QMenuBar,
    QMenu,
    QToolBar,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QGuiApplication

from typing import cast
# Import local GUI components
from .parameter_controls import ParameterControls
from .parameter_matrix import ParameterMatrix
from .piano_roll_view import PianoRollView
from .playback_controller import PlaybackController
from .plugins_audio_tab import PluginsAudioTab
from .config_manager import ConfigManager
from .generation_worker import GenerationWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDI Master - Music Generation GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Styling for a cleaner look
        self._apply_styles()

        # Central widget: Tabbed interface
        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)

        # Initialize tabs with components
        self.parameter_tab = ParameterControls()
        self.central_widget.addTab(self.parameter_tab, "Parameters")

        self.matrix_tab = ParameterMatrix()
        self.matrix_tab.parameterSelected.connect(self.apply_matrix_selection)
        self.central_widget.addTab(self.matrix_tab, "Matrix")

        self.preview_tab = PianoRollView()
        self.central_widget.addTab(self.preview_tab, "Preview")

        self.playback_tab = PlaybackController()
        self.central_widget.addTab(self.playback_tab, "Playback")

        self.plugins_tab = PluginsAudioTab()
        self.central_widget.addTab(self.plugins_tab, "Plugins")

        # Analysis Tab
        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)
        self.analyze_button = QPushButton("Analyze MIDI File")
        self.analyze_button.clicked.connect(self.analyze_midi)
        analysis_layout.addWidget(self.analyze_button)
        self.analysis_text = QTextEdit()
        self.analysis_text.setPlaceholderText("Analysis results will appear here...")
        analysis_layout.addWidget(self.analysis_text)
        self.central_widget.addTab(self.analysis_tab, "Analysis")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        # File Menu
        file_menu= cast(QMenu, self.menu_bar.addMenu("File"))

        new_action = QAction("New Song", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)

        open_action = QAction("Open Config", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction("Save Config", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save Config As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_as_file)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit Menu
        edit_menu= cast(QMenu, self.menu_bar.addMenu("Edit"))

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        cut_action = QAction("Cut", self)
        cut_action.setShortcut("Ctrl+X")
        edit_menu.addAction(cut_action)

        copy_action = QAction("Copy", self)
        copy_action.setShortcut("Ctrl+C")
        edit_menu.addAction(copy_action)

        paste_action = QAction("Paste", self)
        paste_action.setShortcut("Ctrl+V")
        edit_menu.addAction(paste_action)

        # View Menu with checkable actions
        view_menu= cast(QMenu, self.menu_bar.addMenu("View"))

        toolbar_action = QAction("Toolbar", self)
        toolbar_action.setCheckable(True)
        toolbar_action.setChecked(True)
        toolbar_action.toggled.connect(self.toggle_toolbar)
        view_menu.addAction(toolbar_action)

        statusbar_action = QAction("Statusbar", self)
        statusbar_action.setCheckable(True)
        statusbar_action.setChecked(True)
        statusbar_action.toggled.connect(self.toggle_statusbar)
        view_menu.addAction(statusbar_action)

        # Generate Menu
        generate_menu= cast(QMenu, self.menu_bar.addMenu("Generate"))

        generate_action = QAction("Generate MIDI (auto filename)", self)
        generate_action.setShortcut("Ctrl+G")
        generate_action.triggered.connect(self.generate_midi)
        generate_menu.addAction(generate_action)

        stop_action = QAction("Stop Generation", self)
        stop_action.setShortcut("Ctrl+Shift+G")
        stop_action.triggered.connect(self.stop_generation)
        generate_menu.addAction(stop_action)

        preview_action = QAction("Generate Preview", self)
        preview_action.setShortcut("Ctrl+P")
        preview_action.triggered.connect(self.preview)
        generate_menu.addAction(preview_action)

        copy_cli_action = QAction("Copy CLI Command", self)
        copy_cli_action.setShortcut("Ctrl+Shift+C")
        copy_cli_action.triggered.connect(self.copy_cli_command)
        generate_menu.addAction(copy_cli_action)

        # Help Menu
        help_menu= cast(QMenu, self.menu_bar.addMenu("Help"))

        about_action = QAction("About", self)
        about_action.triggered.connect(self.about)
        help_menu.addAction(about_action)

        # Toolbar
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)
        self.toolbar.setMovable(True)

        # Add actions to toolbar (text-only for now)
        new_toolbar_action = QAction("New", self)
        new_toolbar_action.setToolTip("New Song")
        new_toolbar_action.triggered.connect(self.new_file)
        self.toolbar.addAction(new_toolbar_action)

        open_toolbar_action = QAction("Open", self)
        open_toolbar_action.setToolTip("Open Config")
        open_toolbar_action.triggered.connect(self.open_file)
        self.toolbar.addAction(open_toolbar_action)

        save_toolbar_action = QAction("Save", self)
        save_toolbar_action.setToolTip("Save Config")
        save_toolbar_action.triggered.connect(self.save_file)
        self.toolbar.addAction(save_toolbar_action)

        self.toolbar.addSeparator()

        generate_toolbar_action = QAction("Generate", self)
        generate_toolbar_action.setToolTip("Generate MIDI (auto filename)")
        generate_toolbar_action.triggered.connect(self.generate_midi)
        self.toolbar.addAction(generate_toolbar_action)

        stop_toolbar_action = QAction("Stop", self)
        stop_toolbar_action.setToolTip("Stop")
        stop_toolbar_action.triggered.connect(self.stop_generation)
        self.toolbar.addAction(stop_toolbar_action)

        copy_cli_toolbar_action = QAction("Copy CLI", self)
        copy_cli_toolbar_action.setToolTip("Copy equivalent CLI to clipboard")
        copy_cli_toolbar_action.triggered.connect(self.copy_cli_command)
        self.toolbar.addAction(copy_cli_toolbar_action)

        # Connect tabs and playback-to-preview integration
        self.central_widget.currentChanged.connect(self.on_tab_changed)
        self.playback_tab.positionUpdated.connect(self.preview_tab.update_playhead)

        # Worker reference for cleanup
        self.worker: GenerationWorker | None = None

        # Auto-generate a preview on startup (or load last if feasible)
        self._startup_autopreview()

    # -----------------------------
    # UI Behavior and Event Handlers
    # -----------------------------

    def _apply_styles(self):
        # Minimal polish: consistent font, subtle backgrounds, spacing
        self.setStyleSheet("""
            QMainWindow {
                background: #101317;
                color: #e6e6e6;
            }
            QWidget {
                color: #e6e6e6;
                font-size: 11pt;
            }
            QTabWidget::pane {
                border: 1px solid #2a2f3a;
                background: #141821;
            }
            QTabBar::tab {
                padding: 8px 12px;
                background: #1a1f29;
                border: 1px solid #2a2f3a;
                border-bottom: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #222938;
            }
            QStatusBar {
                background: #0c0f13;
            }
            QToolBar {
                background: #0c0f13;
                border: 1px solid #2a2f3a;
                spacing: 6px;
            }
            QPushButton {
                background: #243146;
                border: 1px solid #3a4a66;
                padding: 6px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #2b3a53;
            }
            QMenuBar, QMenu {
                background: #0c0f13;
            }
            QProgressBar {
                border: 1px solid #2a2f3a;
                background: #0c0f13;
                color: #e6e6e6;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
            }
        """)

    def new_file(self):
        # New Song: do not prompt for filename; auto-generate a preview immediately
        self.status_bar.showMessage("Creating new song (auto-preview)...")
        self._start_generation(preview_mode=True)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Config", "", "Config Files (*.json);;All Files (*)")
        if file_path:
            try:
                config_manager = ConfigManager()
                config = config_manager.load_config(file_path)
                if hasattr(self.parameter_tab, 'load_config'):
                    self.parameter_tab.load_config(config)
                self.status_bar.showMessage(f"Loaded config: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load config: {str(e)}")
                self.status_bar.showMessage("Load failed")

    def save_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Config", "config/session_config.json", "Config Files (*.json)")
        if file_path:
            try:
                config_manager = ConfigManager()
                if hasattr(self.parameter_tab, 'get_config'):
                    config = self.parameter_tab.get_config()
                else:
                    # Fallback to a minimal snapshot if needed
                    config = {}
                config_manager.save_config(config, file_path)
                self.status_bar.showMessage(f"Saved config: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", f"Failed to save config: {str(e)}")
                self.status_bar.showMessage("Save failed")

    def save_as_file(self):
        self.save_file()  # Same behavior; Save dialog already includes 'As'

    def toggle_toolbar(self, checked):
        self.toolbar.setVisible(checked)

    def toggle_statusbar(self, checked):
        self.status_bar.setVisible(checked)

    def _start_generation(self, preview_mode: bool):
        """
        Start a background generation with current parameters.
        Auto-filename is used; no save dialog is shown.
        """
        try:
            # Collect parameters dict from UI
            if hasattr(self.parameter_tab, 'get_config'):
                config = self.parameter_tab.get_config()
            else:
                QMessageBox.warning(self, "Parameters", "Parameter panel not ready.")
                return

            # Ensure time_signature is present
            if "time_signature" not in config:
                config["time_signature"] = "4/4"

            # Validate basic ranges early to avoid thread spin-up if invalid
            if not self._validate_config(config):
                return

            # Spawn worker
            worker = GenerationWorker(params=config, preview_mode=preview_mode)
            worker.progress.connect(self.progress_bar.setValue)
            worker.finished.connect(self._on_worker_finished)
            worker.error.connect(self._on_worker_error)
            self.worker = worker
            worker.start()
            self.progress_bar.show()
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("Generating (auto filename)...")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Generation Error", f"Failed to start generation: {str(e)}")

    def generate_midi(self):
        """
        Generate MIDI to an auto-generated contextual filename (no prompt).
        """
        self._start_generation(preview_mode=False)

    def _validate_config(self, config):
        # Basic validation
        required = ['genre', 'mood', 'tempo', 'bars', 'density']
        for r in required:
            if r not in config:
                QMessageBox.warning(self, "Validation Error", f"Missing parameter: {r}")
                return False
        if not (60 <= int(config['tempo']) <= 200):
            QMessageBox.warning(self, "Validation Error", "Tempo must be between 60-200 BPM")
            return False
        if not (4 <= int(config['bars']) <= 64):
            QMessageBox.warning(self, "Validation Error", "Bars must be between 4-64")
            return False

        # Prevent generation when filename template is invalid (empty is OK)
        try:
            if hasattr(self.parameter_tab, "is_template_valid") and not self.parameter_tab.is_template_valid():
                QMessageBox.warning(self, "Validation Error", "Filename template is invalid. Clear or correct it.")
                return False
        except Exception:
            pass

        return True

    def stop_generation(self):
        if self.worker and self.worker.isRunning():
            try:
                self.worker.cancel()
            except Exception:
                pass
            self.worker.quit()
            self.worker.wait(3000)  # Wait up to 3 seconds
            self.progress_bar.hide()
            self.status_bar.showMessage("Generation stopped")
            self.worker = None
        else:
            QMessageBox.information(self, "Stop", "No active generation to stop.")

    def _on_worker_finished(self, success: bool, message: str, output_path: str):
        # Cleanup progress UI
        self.progress_bar.hide()

        if not success:
            self.status_bar.showMessage(f"Generation failed: {message}")
            QMessageBox.critical(self, "Generation Error", message)
            self.worker = None
            return

        # Access the generated SongSkeleton from the worker for preview/playback
        skeleton = getattr(self.worker, "skeleton", None) if self.worker else None
        if skeleton is not None:
            # Update preview tab
            self.preview_tab.update_preview(skeleton)

            # Update playback controller
            self.playback_tab.set_song_skeleton(skeleton)

            # Notify plugins tab (provide skeleton for offline rendering)
            try:
                self.plugins_tab.song_skeleton_ready.emit(skeleton)
            except Exception:
                pass

        # Persist last song path
        try:
            cfg = ConfigManager()
            cfg.set_last_song_path(output_path or "")
        except Exception:
            pass

        # Switch to preview tab so the user sees something immediately
        self.central_widget.setCurrentWidget(self.preview_tab)

        self.status_bar.showMessage(f"Generation complete: {output_path or '(in-memory)'}")
        self.worker = None

    def _on_worker_error(self, error_msg: str):
        self.progress_bar.hide()
        self.status_bar.showMessage(f"Generation error: {error_msg}")
        QMessageBox.critical(self, "Generation Error", error_msg)
        self.worker = None

    def preview(self):
        # Generate a quick preview and show the Preview tab
        self._start_generation(preview_mode=True)
        self.central_widget.setCurrentWidget(self.preview_tab)

    def apply_matrix_selection(self, genre: str, tempo_range: tuple, mood: str, density: str):
        """
        Apply selection from Matrix tab to Parameters controls.
        """
        try:
            tempo_avg = int((tempo_range[0] + tempo_range[1]) // 2)
        except Exception:
            tempo_avg = 120
        # Delegate to ParameterControls loader to handle validation/clamping
        if hasattr(self.parameter_tab, 'load_config'):
            self.parameter_tab.load_config({
                "genre": genre,
                "tempo": tempo_avg,
                "mood": mood,
                "density": density
            })
        # Switch user to Parameters tab for visibility
        self.central_widget.setCurrentWidget(self.parameter_tab)
        self.status_bar.showMessage(f"Applied from Matrix: {genre} @ {tempo_avg} BPM, {mood}, {density}")

    def copy_cli_command(self):
        """
        Build the equivalent CLI command from current Parameters and copy to clipboard.
        """
        try:
            if hasattr(self.parameter_tab, 'build_cli_command'):
                cmd = self.parameter_tab.build_cli_command()
                # Use QGuiApplication clipboard for PyQt6
                cb = QGuiApplication.clipboard()
                if cb is not None:
                    cb.setText(cmd)
                    self.status_bar.showMessage("CLI command copied to clipboard")
                else:
                    # Fallback: show command in a dialog if clipboard unavailable
                    QMessageBox.information(self, "CLI Command", cmd)
                    self.status_bar.showMessage("Clipboard unavailable; showed CLI command in dialog")
            else:
                QMessageBox.information(self, "Copy CLI", "Parameter panel does not support CLI export.")
        except Exception as e:
            QMessageBox.critical(self, "Copy CLI Error", f"Failed to build/copy CLI: {str(e)}")

    def analyze_midi(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open MIDI for Analysis", "", "MIDI Files (*.mid)")
        if file_path:
            try:
                from fixed_midi_analysis import analyze_midi_file  # Assume function name
                results = analyze_midi_file(file_path)
                self.analysis_text.setText(str(results))
                self.status_bar.showMessage("Analysis completed")
            except ImportError:
                QMessageBox.critical(self, "Analysis Error", "fixed_midi_analysis.py not found or import error")
            except Exception as e:
                QMessageBox.critical(self, "Analysis Error", f"Failed to analyze: {str(e)}")

    def about(self):
        QMessageBox.about(self, "About MIDI Master", "MIDI Master v1.0 - AI Music Generation")

    def on_tab_changed(self, index):
        tab_names = ["Parameters", "Matrix", "Preview", "Playback", "Plugins", "Analysis"]
        if 0 <= index < len(tab_names):
            self.status_bar.showMessage(f"Switched to {tab_names[index]} tab")

    def _startup_autopreview(self):
        """
        On startup: if possible, use last song path (for future extensions),
        but always ensure the Preview tab is populated by auto-generating a preview.
        """
        try:
            cfg = ConfigManager()
            last_path = cfg.get_last_song_path()
            # Future: if we add MIDI->Skeleton import, we could load last_path here.
            # For now, ensure there's a fresh preview so the UI isn't empty.
        except Exception:
            pass

        # Kick off a preview generation immediately
        self._start_generation(preview_mode=True)


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())