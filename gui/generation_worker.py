"""
Background generation worker for the MIDI Master GUI.

Provides parameter validation and (optionally) executes music generation
in a separate thread to keep the UI responsive.
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
from typing import Dict, Any
import os

# Keep imports lightweight at module load; heavy imports happen lazily in run()
from genres.genre_factory import GenreFactory


class GenerationWorker(QThread):
    """
    QThread worker that validates parameters and (optionally) generates a MIDI.

    Signals:
      - progress(int): Percentage progress updates (0-100)
      - finished(bool, str, str): (success, message, output_path)
      - error(str): Error message on failure
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str, str)
    error = pyqtSignal(str)

    def __init__(self, params: Dict[str, Any], preview_mode: bool = True):
        super().__init__()
        self.params = dict(params or {})
        self.preview_mode = bool(preview_mode)
        self._cancelled = False
        self.skeleton = None  # Will be set after generation

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        self._cancelled = True

    # Public validation used by tests (test_gui_verification.py)
    def _validate_params(self) -> bool:
        """
        Validate GUI parameter dict without raising.

        Expected keys (strings unless noted):
          - genre: one of GenreFactory.get_available_genres()
          - mood: one of {'calm','happy','sad','energetic'}
          - tempo: int, recommended 60..200 (GUI range)
          - bars: int, recommended 4..64 (GUI range)
          - density: one of {'minimal','sparse','balanced','dense','complex'}
          - separate_files: bool (optional)
          - harmonic_variance: one of {'close','medium','distant'} (optional)
          - key: musical key string (optional)
          - mode: musical mode string (optional)
          - time_signature: like '4/4' (optional)
          - output: optional filepath ending with .mid (not required; auto-naming if absent)
        """
        try:
            # Genre
            genres = set(GenreFactory.get_available_genres())
            genre = str(self.params.get("genre", "")).lower()
            if genre not in genres:
                return False

            # Mood
            mood = str(self.params.get("mood", "")).lower()
            if mood not in {"calm", "happy", "sad", "energetic"}:
                return False

            # Tempo (GUI sensible range; main ParameterConfig clamps broader ranges)
            tempo = int(self.params.get("tempo", 120))
            if not (60 <= tempo <= 200):
                return False

            # Bars (GUI sensible range)
            bars = int(self.params.get("bars", 16))
            if not (4 <= bars <= 64):
                return False

            # Density
            density = str(self.params.get("density", "balanced")).lower()
            if density not in {"minimal", "sparse", "balanced", "dense", "complex"}:
                return False

            # Optional: time signature format 'num/den'
            ts = str(self.params.get("time_signature", "4/4"))
            if "/" in ts:
                try:
                    num_s, den_s = ts.split("/", 1)
                    num, den = int(num_s), int(den_s)
                    if num < 1 or num > 16 or den not in {1, 2, 4, 8, 16}:
                        return False
                except Exception:
                    return False
            else:
                return False

            # If output provided, must end with .mid
            output = self.params.get("output")
            if output:
                if not str(output).lower().endswith(".mid"):
                    return False

            return True
        except Exception:
            return False

    def run(self) -> None:
        """
        Execute generation (optional) with progress updates.
        For verification tests, only validation is essential.
        """
        try:
            self.progress.emit(1)
            if not self._validate_params():
                msg = "Parameter validation failed"
                self.error.emit(msg)
                self.finished.emit(False, msg, "")
                return

            if self._cancelled:
                self.finished.emit(False, "Cancelled", "")
                return

            # Thread optional resolver override via environment (only when provided)
            try:
                rdb_path = self.params.get("rhythms_db_path")
                if isinstance(rdb_path, str) and rdb_path.strip():
                    os.environ["MIDIMASTER_RHYTHMS_DB"] = rdb_path.strip()
            except Exception:
                # Non-fatal; fall back to default resolver behavior
                pass

            # Lazy imports to avoid heavy load until needed
            from output.midi_output import MidiOutput
            from structures.song_skeleton import SongSkeleton
            from generators.density_manager import create_density_manager_from_preset
            from generators.pattern_orchestrator import PatternOrchestrator

            # Collect parameters
            genre: str = str(self.params["genre"]).lower()
            mood: str = str(self.params["mood"]).lower()
            tempo: int = int(self.params["tempo"])
            bars: int = int(self.params["bars"])
            density: str = str(self.params["density"]).lower()
            separate_files: bool = bool(self.params.get("separate_files", False))
            time_signature: str = str(self.params.get("time_signature", "4/4"))

            self.progress.emit(5)
            if self._cancelled:
                self.finished.emit(False, "Cancelled", "")
                return

            # Create rules/context objects
            genre_rules = GenreFactory.create_genre_rules(genre)
            skeleton = SongSkeleton(genre, tempo, mood)
            density_manager = create_density_manager_from_preset(density)

            self.progress.emit(15)
            if self._cancelled:
                self.finished.emit(False, "Cancelled", "")
                return

            # Generate patterns (reduced complexity if preview may be applied higher up)
            generator = PatternOrchestrator(
                genre_rules,
                mood,
                note_density=density_manager.note_density,
                rhythm_density=density_manager.rhythm_density,
                chord_density=density_manager.chord_density,
                bass_density=density_manager.bass_density,
            )

            # Optionally reduce bars for preview to speed up
            gen_bars = bars if not self.preview_mode else max(4, min(bars, 16))
            patterns = generator.generate_patterns(skeleton, gen_bars)
            skeleton.build_arrangement(patterns)

            self.progress.emit(60)
            if self._cancelled:
                self.finished.emit(False, "Cancelled", "")
                return

            # Save to MIDI (auto-generate filename if output omitted)
            midi_out = MidiOutput()
            output_path = self.params.get("output")
            if not output_path:
                # Auto-generate contextual filename in output/ folder
                output_path = midi_out.generate_output_filename(
                    genre, mood, tempo, time_signature, output_folder="output"
                )
            # Ensure uniqueness
            output_path = midi_out.get_unique_filename(str(output_path))

            # Always produce a file so the preview/playback can use it
            # Optional templating inputs (preserve legacy behavior when not set)
            filename_template = None
            try:
                raw_tmpl = self.params.get("filename_template", "")
                if isinstance(raw_tmpl, str) and raw_tmpl.strip():
                    filename_template = raw_tmpl.strip()
            except Exception:
                filename_template = None

            template_settings = None
            try:
                ts = self.params.get("template_settings")
                if isinstance(ts, dict):
                    template_settings = dict(ts)
            except Exception:
                template_settings = None

            base_output_dir = None
            try:
                bod = self.params.get("base_output_dir")
                if isinstance(bod, str) and bod.strip():
                    base_output_dir = bod.strip()
            except Exception:
                base_output_dir = None

            midi_out.save_to_midi(
                skeleton,
                output_path=output_path,
                genre_rules=genre_rules,
                separate_files=separate_files,
                genre=genre,
                mood=mood,
                tempo=tempo,
                time_signature=time_signature,
                filename_template=filename_template,
                template_settings=template_settings,
                template_context=None,
                base_output_dir=base_output_dir,
            )

            # Expose skeleton to the UI thread
            self.skeleton = skeleton

            self.progress.emit(100)
            self.finished.emit(True, "Generation complete", str(output_path))
        except Exception as e:
            msg = f"Generation failed: {e}"
            self.error.emit(msg)
            self.finished.emit(False, msg, "")