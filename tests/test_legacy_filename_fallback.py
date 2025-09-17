import re
from pathlib import Path

import pytest

from output.midi_output import MidiOutput
from structures.data_structures import PatternType


class DummyNote:
    def __init__(self, pitch=60, duration=0.5, velocity=64, start_time=0.0):
        self.pitch = pitch
        self.duration = duration
        self.velocity = velocity
        self.start_time = start_time


class DummyPattern:
    def __init__(self, pattern_type=PatternType.MELODY):
        self.pattern_type = pattern_type
        self.notes = [DummyNote()]
        self.chords = []


class FakeSongSkeleton:
    def __init__(self, tempo=120):
        self.tempo = tempo
        # Provide one melody pattern so separate export writes a file
        self.patterns = [DummyPattern(PatternType.MELODY)]
        # No sections needed for this legacy naming check
        self.sections = []

    def get_time_signature(self, _ptype):
        # Default legacy behavior aligns to 4/4 unless specified; keep 4/4 for test
        return (4, 4)


def test_generate_output_filename_legacy_format_no_template(tmp_path: Path, monkeypatch):
    """
    When no template is provided, ensure legacy naming remains EXACT:
    {genre}_{mood}_{tempo}_{time_sig}_{timestamp}.mid under 'output/' (or provided folder).
    We won't actually save; just check the format function result.
    """
    m = MidiOutput()
    # Use output folder override to ensure we don't write to repo root
    legacy_path = m.generate_output_filename("pop", "happy", 120, "4/4", output_folder=str(tmp_path))
    # Expect pop_happy_120_4-4_YYYYMMDD_HHMMSS.mid
    name = Path(legacy_path).name
    assert name.startswith("pop_happy_120_4-4_")
    assert name.endswith(".mid")
    # Timestamp pattern
    assert re.search(r"\d{8}_\d{6}\.mid$", name), f"unexpected legacy filename: {name}"


def test_separate_files_legacy_suffixes_no_template(tmp_path: Path):
    """
    When saving separate files without a template, ensure files are named:
    {base}_{stem}.mid (melody, harmony, bass, rhythm) based on the legacy behavior.
    """
    song = FakeSongSkeleton()
    m = MidiOutput()

    base = tmp_path / "legacy_test"
    # Call separate save directly to avoid generating combined first
    m.save_to_separate_midi_files(song, str(base))

    # Verify melody exists (we only created a melody pattern); other stems may not be created if empty
    melody = tmp_path / "legacy_test_melody.mid"
    assert melody.exists(), "Legacy melody filename was not created"

    # Ensure names are exactly legacy style (no templating involved)
    assert melody.name == "legacy_test_melody.mid"