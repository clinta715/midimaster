import mido
from unittest.mock import Mock, patch
from structures.data_structures import Pattern, Note, Chord, PatternType
from structures.song_skeleton import SongSkeleton
from music_theory import MusicTheory
from generators.rhythm_generator import RhythmGenerator
from generators.melody_generator import MelodyGenerator
from generators.harmony_generator import HarmonyGenerator
import random
import tempfile
import os

def create_mock_song_skeleton(genre="pop", tempo=120, mood="happy", bars=16, key="C", scale="major"):
    """Create a standardized mock SongSkeleton."""
    skeleton = Mock(spec=SongSkeleton)
    skeleton.genre = genre
    skeleton.tempo = tempo
    skeleton.mood = mood
    skeleton.bars = bars
    skeleton.key = key
    skeleton.scale = scale
    return skeleton

def create_mock_pattern(pattern_type=PatternType.MELODY, num_notes=4):
    """Create a mock Pattern with sample notes."""
    notes = [Note(pitch=random.randint(60, 84), duration=1.0, velocity=80, start_time=i) for i in range(num_notes)]
    chords = [Chord(notes=[notes[0]]) for _ in range(num_notes // 2)]
    pattern = Mock(spec=Pattern)
    pattern.type = pattern_type
    pattern.notes = notes
    pattern.chords = chords
    pattern.length = num_notes
    return pattern

def create_mock_music_theory():
    """Create a mock MusicTheory with common scale."""
    theory = Mock(spec=MusicTheory)
    theory.get_scale_pitches_from_string.return_value = list(range(60, 72))  # C major
    theory.get_chord.return_value = [60, 64, 67]  # C major chord
    return theory

def generate_test_midi_data(num_tracks=1, num_notes_per_track=10):
    """Generate a simple MIDI file with test data."""
    mid = mido.MidiFile()
    for track_num in range(num_tracks):
        track = mido.MidiTrack()
        mid.tracks.append(track)
        for i in range(num_notes_per_track):
            track.append(mido.Message('note_on', note=60 + i % 12, velocity=64, time=0, channel=track_num))
            track.append(mido.Message('note_off', note=60 + i % 12, velocity=0, time=480, channel=track_num))
    return mid

def create_test_midi_file(mid=None, filename="test.mid"):
    """Create a temporary MIDI file from MIDI data."""
    if mid is None:
        mid = generate_test_midi_data()
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
        mid.save(tmp.name)
        return tmp.name

def validate_pattern(pattern):
    """Validate a pattern has required attributes and valid data."""
    if not isinstance(pattern, Pattern):
        return False
    if len(pattern.notes) == 0:
        return False
    for note in pattern.notes:
        if note.pitch < 0 or note.velocity < 0 or note.duration <= 0:
            return False
    return True

def validate_midi_file(file_path):
    """Validate MIDI file integrity."""
    try:
        mid = mido.MidiFile(file_path)
        if len(mid.tracks) == 0:
            return False
        return True
    except Exception:
        return False

def mock_generator_class(generator_class, generate_return_value=None):
    """Patch a generator class and set generate return value."""
    mock_gen = Mock(spec=generator_class)
    if generate_return_value is None:
        generate_return_value = create_mock_pattern()
    mock_gen.generate.return_value = generate_return_value
    return patch.object(generator_class, '__init__', return_value=mock_gen)

def generate_random_scale_pitches(num_octaves=2):
    """Generate random scale pitches for testing."""
    base_notes = [60, 62, 64, 65, 67, 69, 71]  # C major
    pitches = []
    for octave in range(num_octaves):
        pitches.extend([note + octave * 12 for note in base_notes])
    return pitches

def assert_pattern_properties(pattern, expected_type=None, min_notes=1, max_notes=20):
    """Assert pattern has expected properties."""
    assert hasattr(pattern, 'type'), "Pattern missing type"
    if expected_type:
        assert pattern.type == expected_type
    assert len(pattern.notes) >= min_notes, f"Too few notes: {len(pattern.notes)}"
    assert len(pattern.notes) <= max_notes, f"Too many notes: {len(pattern.notes)}"
    for note in pattern.notes:
        assert note.pitch > 0
        assert note.velocity > 0
        assert note.duration > 0