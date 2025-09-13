"""
Generation Worker for MIDI Master GUI

This module provides a QThread-based worker class for running music generation
in the background without freezing the GUI.
"""

import sys
import os
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.pattern_orchestrator import PatternOrchestrator as PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from genres.genre_factory import GenreFactory
from generators.density_manager import create_density_manager_from_preset


class GenerationWorker(QThread):
    """
    Background worker thread for music generation.
    """

    # Signals
    progress = pyqtSignal(int)  # Progress percentage (0-100)
    finished = pyqtSignal()     # Generation completed successfully
    error = pyqtSignal(str)     # Error message
    songGenerated = pyqtSignal(object)  # SongSkeleton object
    previewReady = pyqtSignal(str)     # Preview file path
    overwriteConfirmationRequested = pyqtSignal(str, str)  # temp_path, final_path - requests overwrite confirmation

    def __init__(self, params, preview_mode=False, temp_dir=None):
        """
        Initialize the generation worker.

        Args:
            params (dict): Generation parameters
            preview_mode (bool): Whether to generate in preview mode
            temp_dir (str): Temporary directory for preview files
        """
        super().__init__()
        self.params = params
        self.preview_mode = preview_mode
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.temp_file_path = None
        self.is_cancelled = False

    def run(self):
        """Execute the generation process in the background thread."""
        try:
            self.progress.emit(0)

            # Validate parameters
            if not self._validate_params():
                self.error.emit("Invalid generation parameters")
                return

            self.progress.emit(10)

            # Create genre rules
            genre_rules = GenreFactory.create_genre_rules(self.params['genre'])
            self.progress.emit(20)

            # Create song skeleton
            song_skeleton = SongSkeleton(
                self.params['genre'],
                self.params['tempo'],
                self.params['mood']
            )
            self.progress.emit(30)

            # Create density manager
            density_manager = create_density_manager_from_preset(self.params['density'])

            # In preview mode, reduce complexity
            if self.preview_mode:
                density_manager = self._reduce_complexity_for_preview(density_manager)

            self.progress.emit(40)

            # Create generator context and set user key/mode if specified
            from generators.generator_context import GeneratorContext
            context = GeneratorContext(
                genre_rules=genre_rules,
                mood=self.params['mood'],
                note_density=density_manager.note_density,
                rhythm_density=density_manager.rhythm_density,
                chord_density=density_manager.chord_density,
                bass_density=density_manager.bass_density
            )
            if self.params.get('user_key') and self.params.get('user_mode'):
                context.set_user_key_mode(self.params['user_key'], self.params['user_mode'])

            # Initialize pattern generator with context
            pattern_generator = PatternGenerator(
                genre_rules,
                self.params['mood'],
                note_density=density_manager.note_density,
                rhythm_density=density_manager.rhythm_density,
                chord_density=density_manager.chord_density,
                bass_density=density_manager.bass_density,
                context=context
            )
            self.progress.emit(50)

            # Check for cancellation
            if self.is_cancelled:
                self.error.emit("Generation cancelled")
                return

            # Generate patterns (use fewer bars for preview)
            bars_to_generate = min(4, self.params['bars']) if self.preview_mode else self.params['bars']
            patterns = pattern_generator.generate_patterns(song_skeleton, bars_to_generate)
            self.progress.emit(70)

            # Check for cancellation
            if self.is_cancelled:
                self.error.emit("Generation cancelled")
                return

            # Build song arrangement
            song_skeleton.build_arrangement(patterns)
            self.progress.emit(85)

            # Check for cancellation
            if self.is_cancelled:
                self.error.emit("Generation cancelled")
                return

            # Save MIDI file
            midi_output = MidiOutput()

            if self.preview_mode:
                # Generate temp file path for preview
                if self.temp_dir:
                    self.temp_file_path = midi_output.generate_temp_filename(
                        prefix="preview",
                        genre=self.params['genre'],
                        mood=self.params['mood'],
                        temp_dir=self.temp_dir
                    )
                else:
                    self.temp_file_path = midi_output.generate_temp_filename(
                        prefix="preview",
                        genre=self.params['genre'],
                        mood=self.params['mood']
                    )

                midi_output.save_to_midi(
                    song_skeleton,
                    str(self.temp_file_path),
                    genre_rules,
                    separate_files=self.params.get('separate_files', False)
                )
            else:
                # Normal generation to specified output path
                midi_output.save_to_midi(
                    song_skeleton,
                    self.params['output'],
                    genre_rules,
                    separate_files=self.params.get('separate_files', False)
                )

            self.progress.emit(95)

            # Check for cancellation
            if self.is_cancelled:
                self.error.emit("Generation cancelled")
                return

            # Emit success signals
            self.progress.emit(100)

            if self.preview_mode:
                self.previewReady.emit(str(self.temp_file_path))
            else:
                self.songGenerated.emit(song_skeleton)
                self.finished.emit()

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)

    def cancel(self):
        """Cancel the ongoing generation process."""
        self.is_cancelled = True

    def _validate_params(self):
        """Validate the generation parameters."""
        required_params = ['genre', 'mood', 'tempo', 'bars', 'density', 'output']

        for param in required_params:
            if param not in self.params:
                print(f"Missing required parameter: {param}")
                return False

        # Validate optional user key/mode if provided
        if self.params.get('user_key') and self.params.get('user_mode'):
            from music_theory import MusicTheory
            music_theory = MusicTheory()
            if not music_theory.validate_key_mode(self.params['user_key'], self.params['user_mode']):
                print(f"Invalid user key/mode combination: {self.params['user_key']} {self.params['user_mode']}")
                return False

        # Validate ranges
        if not (80 <= self.params['tempo'] <= 160):
            print(f"Invalid tempo: {self.params['tempo']}")
            return False

        if not (4 <= self.params['bars'] <= 32):
            print(f"Invalid bars: {self.params['bars']}")
            return False

        # Validate choices
        if self.params['genre'] not in ['pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical']:
            print(f"Invalid genre: {self.params['genre']}")
            return False

        if self.params['mood'] not in ['happy', 'sad', 'energetic', 'calm']:
            print(f"Invalid mood: {self.params['mood']}")
            return False

        if self.params['density'] not in ['minimal', 'sparse', 'balanced', 'dense', 'complex']:
            print(f"Invalid density: {self.params['density']}")
            return False

        return True

    def _reduce_complexity_for_preview(self, density_manager):
        """
        Reduce complexity settings for preview generation to speed up processing.

        Args:
            density_manager: Original density manager

        Returns:
            Modified density manager with reduced complexity
        """
        # Create a copy and reduce densities for preview
        import copy
        preview_density = copy.deepcopy(density_manager)

        # Reduce all densities by 30-50% for faster preview generation
        preview_density.note_density = max(0.1, density_manager.note_density * 0.6)
        preview_density.rhythm_density = max(0.1, density_manager.rhythm_density * 0.7)
        preview_density.chord_density = max(0.1, density_manager.chord_density * 0.6)
        preview_density.bass_density = max(0.1, density_manager.bass_density * 0.7)

        return preview_density

    def get_temp_file_path(self):
        """
        Get the path of the temporary file generated in preview mode.

        Returns:
            Path to temporary file, or None if not in preview mode
        """
        return self.temp_file_path

    def render_to_final_file(self, temp_path, final_path):
        """
        Copy a temporary file to its final destination.

        Args:
            temp_path (str): Path to temporary file
            final_path (str): Path to final destination

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            midi_output = MidiOutput()

            # Check if target file exists
            if midi_output.check_file_exists(final_path):
                # File exists, request overwrite confirmation
                self.overwriteConfirmationRequested.emit(temp_path, final_path)
                return False

            # File doesn't exist, proceed with copy
            success = midi_output.copy_to_final_location(temp_path, final_path)
            return success
        except Exception as e:
            print(f"Failed to render to final file: {e}")
            return False