import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
import mido
from unittest.mock import Mock
from structures.song_skeleton import SongSkeleton
from generators.rhythm_generator import RhythmGenerator
from generators.melody_generator import MelodyGenerator
from generators.harmony_generator import HarmonyGenerator
from generators.pattern_orchestrator import PatternOrchestrator
from music_theory import MusicTheory
import tempfile
import os

@pytest.fixture
def mock_song_skeleton():
    """Mock SongSkeleton with basic attributes."""
    skeleton = Mock(spec=SongSkeleton)
    skeleton.genre = "pop"
    skeleton.tempo = 120
    skeleton.mood = "happy"
    skeleton.bars = 16
    skeleton.key = "C"
    skeleton.scale = "major"
    return skeleton

@pytest.fixture
def mock_music_theory():
    """Mock MusicTheory instance."""
    theory = Mock(spec=MusicTheory)
    theory.get_scale_pitches_from_string.return_value = [60, 62, 64, 65, 67, 69, 71, 72]
    return theory

@pytest.fixture
def mock_rhythm_generator(mock_song_skeleton, mock_music_theory):
    """Mock RhythmGenerator."""
    gen = Mock(spec=RhythmGenerator)
    gen.generate.return_value = Mock()  # Mock pattern
    return gen

@pytest.fixture
def mock_melody_generator(mock_song_skeleton, mock_music_theory):
    """Mock MelodyGenerator."""
    gen = Mock(spec=MelodyGenerator)
    gen.generate.return_value = Mock()  # Mock pattern
    return gen

@pytest.fixture
def mock_harmony_generator(mock_song_skeleton, mock_music_theory):
    """Mock HarmonyGenerator."""
    gen = Mock(spec=HarmonyGenerator)
    gen.generate.return_value = Mock()  # Mock pattern
    return gen

@pytest.fixture
def mock_pattern_orchestrator(mock_song_skeleton):
    """Mock PatternOrchestrator."""
    orch = Mock(spec=PatternOrchestrator)
    orch.generate.return_value = Mock()  # Mock full generation
    return orch

@pytest.fixture
def tmp_midi_path(tmp_path):
    """Temporary path for MIDI files."""
    midi_path = tmp_path / "test.mid"
    return midi_path

@pytest.fixture
def valid_midi_file(tmp_midi_path):
    """Create a valid minimal MIDI file."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.save(tmp_midi_path)
    return tmp_midi_path

@pytest.fixture
def invalid_midi_file(tmp_path):
    """Create an invalid MIDI file (empty binary)."""
    invalid_path = tmp_path / "invalid.mid"
    invalid_path.write_bytes(b"")
    return invalid_path

def validate_midi_file(file_path):
    """Utility to validate if a file is a valid MIDI."""
    try:
        mido.MidiFile(file_path)
        return True
    except Exception:
        return False

@pytest.fixture
def validate_midi():
    """Fixture for MIDI validation utility."""
    return validate_midi_file