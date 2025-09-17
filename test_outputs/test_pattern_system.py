import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Comprehensive testing for the pattern-based generation system.

This module provides unit, integration, and validation tests for all components
of the pattern generation system using pytest. Run with:
$ pytest test_outputs/test_pattern_system.py -v --html=reports/test_report.html --self-contained-html

Fixtures provide synthetic data for controlled testing.
Tests cover edge cases, errors, and different genres/moods including drum-and-bass.
Enhanced with cross-genre, multi-component, performance, and quality tests.
"""

import pytest
import time
import threading
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity  # For diversity

# Core imports - adjust paths as needed
from structures.data_structures import Note, Pattern, PatternType, Chord
from analyzers.midi_pattern_extractor import NoteEvent, TempoEvent, TimeSignatureEvent, MidiPatternData, extract_from_file, extract_from_directory
from analyzers.reference_pattern_library import PatternMetadata, ReferencePatternLibrary
from generators.rhythm_generator import RhythmGenerator
from generators.harmony_generator import HarmonyGenerator
from generators.melody_generator import MelodyGenerator
from generators.pattern_variations import PatternVariationEngine, VariationType
from generators.adaptive_selector import AdaptivePatternSelector
from generators.pattern_blender import PatternBlender
from generators.adaptation_engine import AdaptationEngine, AdaptationSuggestion, AdaptationTrigger, PerformanceMetrics, PlayerSkill
from generators.generator_context import GeneratorContext  # Assume exists or mock
from genres.genre_factory import GenreFactory  # For moods/genres
from ml_insights.feature_extraction import extract_features  # Mock ML integration

# Mock dependencies
from enum import Enum

class MockChordType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DIMINISHED = "diminished"
    AUGMENTED = "augmented"
    SEVENTH = "seventh"

class MockNote(Enum):
    C4 = 60
    D4 = 62
    E4 = 64
    def get_chord_inversion(self, chord_pitches, inversion):
        """Mock chord inversion method."""
        if inversion == 0:
            return chord_pitches
        elif inversion == 1:
            return chord_pitches[1:] + [chord_pitches[0] + 12]
        elif inversion == 2:
            return chord_pitches[2:] + [chord_pitches[0] + 24, chord_pitches[1] + 12]
        else:
            return chord_pitches  # No inversion
    F4 = 65
    G4 = 67
    A4 = 69
    B4 = 71
    C5 = 72

class MockMusicTheory:
    def get_chord_pitches_from_roman(self, roman, key_scale):
        return [60, 64, 67]

    def filter_progressions_by_distance(self, progressions, key_scale, variance):
        return progressions  # Mock implementation

    def get_scale_pitches_from_string(self, scale):
        return [60, 62, 64, 65, 67, 69, 71]

    def build_chord(self, root, chord_type):
        if isinstance(chord_type, MockChordType):
            if chord_type == MockChordType.MAJOR:
                return [root, root + 4, root + 7]
            elif chord_type == MockChordType.MINOR:
                return [root, root + 3, root + 7]
            elif chord_type == MockChordType.DIMINISHED:
                return [root, root + 3, root + 6]
            elif chord_type == MockChordType.AUGMENTED:
                return [root, root + 4, root + 8]
            elif chord_type == MockChordType.SEVENTH:
                return [root, root + 4, root + 7, root + 10]
        # Fallback for string or other
        if isinstance(chord_type, str):
            if chord_type.lower() == 'major':
                return [root, root + 4, root + 7]
            elif chord_type.lower() == 'minor':
                return [root, root + 3, root + 7]
        return [root, root + 4, root + 7]  # Default major

class MockGenreRules:
    def get_scales(self):
        return ["C major"]

    def get_chord_progressions(self):
        return [["I", "IV", "V"]]

    def get_rhythm_patterns(self):
        return [{"pattern": [0.25, 0.25, 0.25, 0.25]}]

    def get_beat_characteristics(self):
        return {"tempo_range": [120, 140], "emphasis_patterns": [1, 3]}

    def get_melody_style(self, subgenre):
        return {"contour_weights": {"rising": 0.5}, "interval_weights": {1: 0.5}}

class GeneratorContextFactory:
    @staticmethod
    def create(scale_pitches=None, current_key=None, current_scale=None, mood=None, genre_rules=None, music_theory=None, density_manager=None, user_key=None, user_mode=None, subgenre=None):
        context = Mock(spec=GeneratorContext)
        # Initialize attributes
        context.scale_pitches = scale_pitches or [60, 62, 64, 65, 67, 69, 71]
        context.current_key = current_key or "C major"
        context.current_scale = current_scale or "major"
        context.mood = mood or "energetic"
        context.genre_rules = genre_rules or MockGenreRules()
        context.music_theory = music_theory or MockMusicTheory()
        context.density_manager = density_manager or MockDensityManager()
        context.user_key = user_key
        context.user_mode = user_mode
        context.subgenre = subgenre
        # Validation
        if not context.scale_pitches or not isinstance(context.scale_pitches, list):
            raise ValueError("scale_pitches must be a non-empty list")
        if not isinstance(context.mood, str):
            raise ValueError("mood must be a string")
        if context.genre_rules is None:
            raise ValueError("genre_rules is required")
        return context

    def _mood_mapping_score(self, features, mood):
        """Mock mood mapping score for tests."""
        if mood == "energetic":
            return features.get('density', 0.5) + features.get('avg_velocity', 64) / 127.0
        elif mood == "calm":
            return 1.0 - (features.get('density', 0.5) + features.get('avg_velocity', 64) / 127.0) / 2
        else:
            return 0.5

    def get_transposed_chord_progressions(self, key, mode):
        return [["C", "F", "G"]]

    # Genre-specific enhancements
    def get_genre_characteristics(self, genre):
        characteristics = {
            "drum-and-bass": {"tempo_min": 160, "syncopation_level": 0.8, "swing_factor": 0.5, "breakbeat": True},
            "jazz": {"tempo_min": 120, "syncopation_level": 0.6, "swing_factor": 0.66, "bebop": True},
            "pop": {"tempo_min": 90, "syncopation_level": 0.3, "swing_factor": 0.55, "straight": True},
            "rock": {"tempo_min": 100, "syncopation_level": 0.2, "swing_factor": 0.5, "power_chord": True},
            "electronic": {"tempo_min": 120, "syncopation_level": 0.4, "swing_factor": 0.5, "four_on_floor": True}
        }
        return characteristics.get(genre, {"tempo_min": 120, "syncopation_level": 0.5, "swing_factor": 0.5})

class MockDensityManager:
    def get_available_durations(self, density):
        return [0.25, 0.5, 1.0]

    def calculate_note_probability(self):
        return 0.5

    def should_place_note(self, time, total):
        return True

    def get_chord_voicing_size(self, size):
        return min(3, size)

    def get_rhythm_pattern_complexity(self):
        return 0.5

    def get_bass_note_count(self, bars):
        return 4

    def get_density_settings(self):
        return {"note_density": 0.5}

# Fixtures for synthetic data
@pytest.fixture
def mock_note():
    return Note(pitch=60, duration=1.0, velocity=64, start_time=0.0, channel=1)

@pytest.fixture
def mock_pattern(mock_note):
    notes = [mock_note, Note(pitch=62, duration=0.5, velocity=70, start_time=1.0, channel=1)]
    return Pattern(PatternType.MELODY, notes, [])

@pytest.fixture
def mock_chord(mock_note):
    notes = [mock_note, Note(pitch=64, duration=1.0, velocity=64, start_time=0.0, channel=1)]
    return Chord(notes, start_time=0.0)

@pytest.fixture
def mock_midi_pattern_data():
    notes = [NoteEvent(start_time=0.0, note=60, velocity=64, duration=480.0, channel=0, instrument="Piano")]
    tempos = [TempoEvent(time=0.0, bpm=120.0)]
    time_sigs = [TimeSignatureEvent(time=0.0, numerator=4, denominator=4)]
    return MidiPatternData(
        file_path="test.mid",
        ticks_per_beat=480,
        tracks=1,
        length_ticks=1920,
        tempos=tempos,
        time_signatures=time_sigs,
        notes=notes,
        track_info={}
    )

@pytest.fixture
def mock_pattern_metadata():
    return PatternMetadata(
        bpm=120.0,
        time_signature=(4, 4),
        complexity=0.5,
        genre="pop",
        instrument_type="melody",
        key="C major",
        pattern_category="melodic",
        chord_types=["major"],
        melodic_intervals=[2, -1]
    )

@pytest.fixture
def mock_library(mock_midi_pattern_data, mock_pattern_metadata):
    lib = ReferencePatternLibrary()
    lib.add_pattern(mock_midi_pattern_data, "pop", "melody")
    lib.metadata[mock_midi_pattern_data.file_path] = mock_pattern_metadata()
    # Add references for all genres
    genres = ["drum-and-bass", "jazz", "pop", "rock", "electronic"]
    for genre in genres:
        lib.add_pattern(mock_midi_pattern_data, genre, "rhythm")
    return lib

@pytest.fixture
def mock_generator_context():
    context = Mock(spec=GeneratorContext)
    context.scale_pitches = [60, 62, 64, 65, 67, 69, 71]
    context.current_key = "C major"
    context.current_scale = "major"
    context.mood = "energetic"
    context.genre_rules = MockGenreRules()
    context.music_theory = MockMusicTheory()
    context.density_manager = MockDensityManager()
    context.user_key = None
    context.user_mode = None
    context.subgenre = None
    return context

@pytest.fixture
def mock_genres_moods():
    return [
        ("drum-and-bass", "energetic"),
        ("jazz", "calm"),
        ("pop", "happy"),
        ("rock", "energetic"),
        ("electronic", "sad")
    ]

# Timing fixture for performance tests
@pytest.fixture
def mock_random():
    with patch('random.choice') as mock_choice, \
         patch('random.random') as mock_random, \
         patch('random.uniform') as mock_uniform, \
         patch('numpy.random.choice') as mock_np_choice, \
         patch('numpy.random.random') as mock_np_random:
        # Set deterministic values
        mock_choice.side_effect = lambda x: x[0]  # Always first choice
        mock_random.return_value = 0.5
        mock_uniform.return_value = 0.5
        mock_np_choice.side_effect = lambda choices: choices[0] if choices else None
        mock_np_random.return_value = 0.5
        yield mock_choice, mock_random, mock_np_choice

@pytest.fixture
def timing_context():
    start = time.time()
    yield
    end = time.time()
    print(f"Test execution time: {end - start:.2f}s")

# Large library mock for scalability
@pytest.fixture
def large_library():
    lib = ReferencePatternLibrary()
    for i in range(1000):  # Simulate large library
        data = MidiPatternData(f"mock_{i}.mid", 480, 1, 1920, [], [], [], {})
        lib.add_pattern(data, "pop", "melody")
    return lib

# Unit Tests for MIDI Pattern Extraction Utilities
class TestMidiPatternExtractor:
    def test_note_event_creation(self):
        event = NoteEvent(start_time=0.0, note=60, velocity=64, duration=480.0, channel=0)
        assert event.note == 60
        assert event.velocity == 64
        assert event.duration == 480.0

    def test_tempo_event_creation(self):
        event = TempoEvent(time=0.0, bpm=120.0)
        assert event.bpm == 120.0

    def test_time_signature_event_creation(self):
        event = TimeSignatureEvent(time=0.0, numerator=4, denominator=4)
        assert event.numerator == 4
        assert event.denominator == 4

    @patch('analyzers.midi_pattern_extractor.MidiFile')
    def test_extract_from_file(self, mock_midi):
        # Mock MIDI file structure
        mock_track = MagicMock()
        mock_track.name = "Test Track"
        mock_midi.return_value.tracks = [mock_track]
        mock_midi.return_value.ticks_per_beat = 480

        # Mock messages
        mock_msg_on = MagicMock()
        mock_msg_on.type = 'note_on'
        mock_msg_on.velocity = 64
        mock_msg_on.note = 60
        mock_msg_on.channel = 0
        mock_msg_on.time = 0

        mock_msg_off = MagicMock()
        mock_msg_off.type = 'note_off'
        mock_msg_off.note = 60
        mock_msg_off.channel = 0
        mock_msg_off.time = 480

        mock_track.__iter__.return_value = [mock_msg_on, mock_msg_off]

        data = extract_from_file("test.mid")
        assert isinstance(data, MidiPatternData)
        assert len(data.notes) == 1
        assert data.notes[0].note == 60

    def test_extract_from_directory(self, tmp_path):
        # Create temporary MIDI files
        mid_file = tmp_path / "test.mid"
        mid_file.write_bytes(b'')  # Empty MIDI for test

        with patch('analyzers.midi_pattern_extractor.extract_from_file') as mock_extract:
            mock_extract.return_value = MidiPatternData("test.mid", 480, 1, 1920, [], [], [], {})
            patterns = extract_from_directory(str(tmp_path))
            assert isinstance(patterns, list)
            assert len(patterns) == 1

    def test_invalid_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            extract_from_file("nonexistent.mid")

# Unit Tests for ReferencePatternLibrary Management
class TestReferencePatternLibrary:
    def test_pattern_metadata_creation(self):
        meta = PatternMetadata(bpm=120, time_signature=(4,4), complexity=0.5, genre="pop", instrument_type="drums")
        assert meta.bpm == 120
        assert meta.genre == "pop"

    def test_classify_genre(self):
        lib = ReferencePatternLibrary()
        genre = lib.classify_genre("reference_midis/midi6/test.mid")
        assert isinstance(genre, str)
        assert genre == "trap_808"  # From mapping

    def test_classify_instrument_type(self, mock_midi_pattern_data):
        lib = ReferencePatternLibrary()
        instr = lib.classify_instrument_type(mock_midi_pattern_data)
        assert instr == "melody"  # Default

    def test_detect_chords(self, mock_midi_pattern_data):
        lib = ReferencePatternLibrary()
        chords = lib.detect_chords(mock_midi_pattern_data)
        assert isinstance(chords, list)
        assert len(chords) == 0  # Single note

    def test_extract_melodic_contour(self, mock_midi_pattern_data):
        lib = ReferencePatternLibrary()
        intervals = lib.extract_melodic_contour(mock_midi_pattern_data)
        assert isinstance(intervals, list)
        assert len(intervals) == 0  # Single note

    def test_add_pattern(self, mock_library, mock_midi_pattern_data):
        key = ("pop", "melody")
        assert key not in mock_library.patterns
        mock_library.add_pattern(mock_midi_pattern_data, "pop", "melody")
        assert key in mock_library.patterns
        assert len(mock_library.patterns[key]) == 1

    def test_load_from_directory(self, tmp_path):
        mid_file = tmp_path / "test.mid"
        mid_file.write_bytes(b'')
        lib = ReferencePatternLibrary()
        with patch('analyzers.midi_pattern_extractor.extract_from_directory') as mock_extract:
            mock_extract.return_value = [MidiPatternData("test.mid", 480, 1, 1920, [], [], [], {})]
            lib.load_from_directory(str(tmp_path))
            assert len(lib.patterns) > 0

    def test_get_patterns(self, mock_library):
        patterns = mock_library.get_patterns(genre="pop", instrument="melody")
        assert isinstance(patterns, list)
        assert len(patterns) >= 1

    def test_get_harmonic_patterns(self, mock_library):
        patterns = mock_library.get_harmonic_patterns(genre="pop")
        assert isinstance(patterns, list)

    def test_get_melodic_patterns(self, mock_library):
        patterns = mock_library.get_melodic_patterns(genre="pop")
        assert isinstance(patterns, list)
        assert len(patterns) >= 1

    def test_compute_similarity(self, mock_midi_pattern_data):
        lib = ReferencePatternLibrary()
        sim = lib.compute_similarity(mock_midi_pattern_data, mock_midi_pattern_data)
        assert 0 <= sim <= 1
        assert sim > 0.9  # Same pattern

    def test_find_similar(self, mock_midi_pattern_data):
        lib = ReferencePatternLibrary()
        lib.patterns = {("pop", "melody"): [mock_midi_pattern_data, mock_midi_pattern_data]}
        lib.metadata[mock_midi_pattern_data.file_path] = mock_pattern_metadata()
        similar = lib.find_similar(mock_midi_pattern_data, threshold=0.5)
        assert isinstance(similar, list)
        assert len(similar) >= 1

    def test_save_load_disk(self, tmp_path):
        lib = ReferencePatternLibrary()
        file_path = tmp_path / "lib.pkl"
        lib.save_to_disk(str(file_path))
        loaded = ReferencePatternLibrary()
        loaded.load_from_disk(str(file_path))
        assert isinstance(loaded, ReferencePatternLibrary)

    def test_get_stats(self, mock_library):
        stats = mock_library.get_stats()
        assert "total_patterns" in stats
        assert "genres" in stats
        assert stats["total_patterns"] >= 1

# Unit Tests for RhythmGenerator
class TestRhythmGenerator:
    def test_rhythm_generator_init(self, mock_generator_context):
        gen = RhythmGenerator(mock_generator_context)
        assert isinstance(gen, RhythmGenerator)

    def test_generate_basic(self, mock_generator_context):
        gen = RhythmGenerator(mock_generator_context)
        pattern = gen.generate(num_bars=4, beat_complexity=0.5)
        assert isinstance(pattern, Pattern)
        assert pattern.pattern_type == PatternType.RHYTHM
        assert len(pattern.notes) > 0

    def test_generate_with_templates(self, mock_generator_context, mock_library):
        mock_generator_context.pattern_library = mock_library
        gen = RhythmGenerator(mock_generator_context, mock_library)
        pattern = gen.generate_with_templates(num_bars=4, beat_complexity=0.5, variation_level=0.2)
        assert isinstance(pattern, Pattern)

    def test_invalid_beat_complexity_raises_error(self, mock_generator_context):
        gen = RhythmGenerator(mock_generator_context)
        with pytest.raises(ValueError):
            gen.generate(num_bars=4, beat_complexity=1.5)

    def test_generate_no_scale_fallback(self, mock_generator_context):
        mock_generator_context.scale_pitches = []
        gen = RhythmGenerator(mock_generator_context)
        pattern = gen.generate(num_bars=1, beat_complexity=0.5)
        assert len(pattern.notes) > 0  # Should fallback

# Unit Tests for HarmonyGenerator
class TestHarmonyGenerator:
    def test_harmony_generator_init(self, mock_generator_context):
        gen = HarmonyGenerator(mock_generator_context)
        assert isinstance(gen, HarmonyGenerator)

    def test_generate_basic(self, mock_generator_context):
        gen = HarmonyGenerator(mock_generator_context)
        pattern = gen.generate_harmony_pattern(genre="pop", mood=mock_generator_context.mood, bars=4)
        assert isinstance(pattern, Pattern)
        assert pattern.pattern_type == PatternType.HARMONY
        assert len(pattern.chords) > 0

    def test_invalid_complexity_raises_error(self, mock_generator_context):
        gen = HarmonyGenerator(mock_generator_context)
        with pytest.raises(ValueError):
            gen.generate_harmony_pattern("invalid", "energetic", 4)

    def test_invalid_variance_raises_error(self, mock_generator_context):
        gen = HarmonyGenerator(mock_generator_context)
        with pytest.raises(ValueError):
            gen.generate_harmony_pattern("invalid", "energetic", 4)

    def test_parse_chord_symbol(self, mock_generator_context):
        gen = HarmonyGenerator(mock_generator_context)
        pitches = gen.music_theory.get_chord_pitches_from_roman("I", "C major")
        assert len(pitches) == 3  # Major chord
        assert pitches[0] == 60

    def test_generate_user_key(self, mock_generator_context):
        mock_generator_context.user_key = "C"
        mock_generator_context.user_mode = "major"
        gen = HarmonyGenerator(mock_generator_context)
        pattern = gen.generate_harmony_pattern(genre="pop", mood=mock_generator_context.mood, bars=1)
        assert len(pattern.chords) > 0

# Unit Tests for MelodyGenerator
class TestMelodyGenerator:
    def test_melody_generator_init(self, mock_generator_context):
        gen = MelodyGenerator(mock_generator_context)
        assert isinstance(gen, MelodyGenerator)

    def test_generate_basic(self, mock_generator_context):
        gen = MelodyGenerator(mock_generator_context)
        pattern = gen.generate(num_bars=4)
        assert isinstance(pattern, Pattern)
        assert pattern.pattern_type == PatternType.MELODY
        assert len(pattern.notes) > 0

    def test_choose_next_pitch(self, mock_generator_context):
        gen = MelodyGenerator(mock_generator_context)
        pitch = gen._choose_next_pitch("rising", {1: 0.5, 2: 0.5}, None, None)
        assert pitch in mock_generator_context.scale_pitches

    def test_generate_no_scale_fallback(self, mock_generator_context):
        mock_generator_context.scale_pitches = []
        gen = MelodyGenerator(mock_generator_context)
        pattern = gen.generate(num_bars=1)
        assert len(pattern.notes) > 0

# Unit Tests for Pattern Variation Algorithms
class TestPatternVariations:
    def test_variation_engine_init(self):
        engine = PatternVariationEngine()
        assert isinstance(engine, PatternVariationEngine)

    def test_time_stretch(self, mock_pattern):
        engine = PatternVariationEngine()
        stretched = engine.time_stretch(mock_pattern, factor=1.5, intensity=1.0)
        assert stretched.notes[0].start_time == 0.0
        assert stretched.notes[0].duration == 1.5

    def test_velocity_randomization(self, mock_pattern):
        engine = PatternVariationEngine()
        randomized = engine.velocity_randomization(mock_pattern, std_dev=10, intensity=1.0)
        assert 0 <= randomized.notes[0].velocity <= 127

    def test_pitch_transpose(self, mock_pattern):
        engine = PatternVariationEngine()
        transposed = engine.pitch_transpose(mock_pattern, semitones=2, intensity=1.0)
        assert transposed.notes[0].pitch == 62

    def test_rhythmic_density_adjust(self, mock_pattern):
        engine = PatternVariationEngine()
        adjusted = engine.rhythmic_density_adjust(mock_pattern, density_factor=0.5, intensity=1.0)
        assert len(adjusted.notes) <= len(mock_pattern.notes)

    def test_add_ornamentation(self, mock_pattern):
        engine = PatternVariationEngine()
        ornamented = engine.add_ornamentation(mock_pattern, "ghost", intensity=0.5)
        assert len(ornamented.notes) >= len(mock_pattern.notes)

    def test_apply_variation(self, mock_pattern):
        engine = PatternVariationEngine()
        varied = engine.apply_variation(mock_pattern, VariationType.PITCH_TRANSPOSITION, {"semitones": 2}, 1.0)
        assert isinstance(varied, Pattern)

    def test_invalid_intensity_raises_error(self, mock_pattern):
        engine = PatternVariationEngine()
        with pytest.raises(ValueError):
            engine.time_stretch(mock_pattern, 1.0, intensity=1.5)

# Unit Tests for Adaptive Selection System
class TestAdaptiveSelector:
    def test_selector_init(self, mock_library):
        selector = AdaptivePatternSelector(mock_library)
        assert isinstance(selector, AdaptivePatternSelector)

    def test_extract_features(self, mock_midi_pattern_data):
        selector = AdaptivePatternSelector(ReferencePatternLibrary())
        features = selector._extract_pattern_features(mock_midi_pattern_data)
        assert "density" in features
        assert "bpm" in features

    def test_mood_mapping_score(self):
        selector = AdaptivePatternSelector(ReferencePatternLibrary())
        features = {"density": 0.8, "avg_velocity": 0.9, "note_speed": 0.7, "contour_bias": 0.5}
        score = selector._mood_mapping_score(features, "energetic")
        assert 0 <= score <= 1

    def test_genre_match_score(self, mock_midi_pattern_data):
        selector = AdaptivePatternSelector(ReferencePatternLibrary())
        mock_midi_pattern_data.genre = "pop"  # Mock
        score = selector._genre_match_score(mock_midi_pattern_data, "pop")
        assert score == 1.0

    def test_bpm_match_score(self):
        selector = AdaptivePatternSelector(ReferencePatternLibrary())
        score = selector._bpm_match_score(130, 120, 140)
        assert score == 1.0

    def test_complexity_match_score(self):
        selector = AdaptivePatternSelector(ReferencePatternLibrary())
        score = selector._complexity_match_score(0.6, 0.5)
        assert score > 0.8

    def test_instrument_match_score(self, mock_midi_pattern_data):
        selector = AdaptivePatternSelector(ReferencePatternLibrary())
        score = selector._instrument_match_score(mock_midi_pattern_data, "melody")
        assert score == 1.0  # Default

    def test_score_pattern(self, mock_library, mock_midi_pattern_data):
        selector = AdaptivePatternSelector(mock_library)
        score = selector.score_pattern(mock_midi_pattern_data, "energetic", "pop", 0.5, (120, 140), "melody")
        assert 0 <= score <= 1

    def test_get_best_patterns(self, mock_library):
        selector = AdaptivePatternSelector(mock_library)
        best = selector.get_best_patterns("energetic", "pop", 0.5, 120, 140, "melody", n=1, threshold=0.0)
        assert isinstance(best, list)

    def test_get_exact_matches(self, mock_library):
        selector = AdaptivePatternSelector(mock_library)
        matches = selector.get_exact_matches("pop", "melody", (120, 140), (0.4, 0.6))
        assert isinstance(matches, list)

    def test_recommend_blends(self, mock_library, mock_midi_pattern_data):
        selector = AdaptivePatternSelector(mock_library)
        blends = selector.recommend_blends(mock_midi_pattern_data, "energetic", "pop", "melody")
        assert isinstance(blends, list)

    def test_get_pattern_clusters(self, mock_library):
        selector = AdaptivePatternSelector(mock_library)
        clusters = selector.get_pattern_clusters("pop", "melody")
        assert isinstance(clusters, dict)

# Unit Tests for ML Analysis Capabilities (AdaptationEngine as proxy)
class TestAdaptationEngine:
    def test_engine_init(self):
        engine = AdaptationEngine()
        assert isinstance(engine, AdaptationEngine)

    def test_performance_metrics(self):
        metrics = PerformanceMetrics()
        metrics.add_measurement("tempo_stability", 0.8)
        avg = metrics.get_average("tempo_stability")
        assert avg == 0.8

    def test_adapt_to_feedback(self):
        engine = AdaptationEngine()
        suggestions = engine.adapt_to_performance_feedback("tempo_stability", 0.2)
        assert isinstance(suggestions, list)
        if suggestions:
            assert isinstance(suggestions[0], AdaptationSuggestion)

    def test_apply_adaptation(self):
        engine = AdaptationEngine()
        suggestion = AdaptationSuggestion(
            trigger=AdaptationTrigger.TEMPO_DRIFT,
            parameter="tempo",
            current_value=120.0,
            suggested_value=100.0,
            confidence=0.8,
            reasoning="Test",
            priority=5
        )
        success = engine.apply_adaptation(suggestion)
        assert success is True

    def test_get_context_aware_suggestions(self):
        engine = AdaptationEngine()
        suggestions = engine.get_context_aware_suggestions()
        assert isinstance(suggestions, list)

    def test_update_player_skill(self):
        engine = AdaptationEngine()
        engine.update_player_skill(PlayerSkill.BEGINNER)
        assert engine.context.player_skill == PlayerSkill.BEGINNER

    def test_analyze_performance_trends(self):
        engine = AdaptationEngine()
        trends = engine.analyze_performance_trends()
        assert "overall_performance_trend" in trends

# Unit Tests for Pattern Blending Capabilities
class TestPatternBlender:
    def test_blender_init(self):
        blender = PatternBlender()
        assert isinstance(blender, PatternBlender)

    def test_blend_rhythms(self, mock_pattern):
        mock_pattern.pattern_type = PatternType.RHYTHM
        blender = PatternBlender()
        patterns = [mock_pattern, mock_pattern]
        blended = blender.blend_rhythms(patterns)
        assert isinstance(blended, Pattern)
        assert blended.pattern_type == PatternType.RHYTHM

    def test_blend_harmonies(self, mock_pattern):
        mock_pattern.pattern_type = PatternType.HARMONY
        blender = PatternBlender()
        patterns = [mock_pattern, mock_pattern]
        blended = blender.blend_harmonies(patterns)
        assert isinstance(blended, Pattern)
        assert blended.pattern_type == PatternType.HARMONY

    def test_blend_melodies(self, mock_pattern):
        blender = PatternBlender()
        patterns = [mock_pattern, mock_pattern]
        blended = blender.blend_melodies(patterns)
        assert isinstance(blended, Pattern)
        assert blended.pattern_type == PatternType.MELODY

    def test_cross_fade(self, mock_pattern):
        blender = PatternBlender()
        faded = blender.cross_fade(mock_pattern, mock_pattern, fade_length=2.0)
        assert isinstance(faded, Pattern)

    def test_layer_patterns(self, mock_pattern):
        blender = PatternBlender()
        layered = blender.layer_patterns([mock_pattern, mock_pattern])
        assert isinstance(layered, Pattern)

    def test_invalid_blend_type_raises_error(self, mock_pattern):
        blender = PatternBlender()
        with pytest.raises(ValueError):
            blender.blend_rhythms([mock_pattern], blend_type="invalid")

# Enhanced Validation Metrics
def calculate_pattern_quality(pattern: Pattern) -> float:
    """Simple quality metric: notes / duration."""
    if not pattern.notes:
        return 0.0
    duration = max((n.start_time + n.duration for n in pattern.notes), default=1.0)
    return len(pattern.notes) / duration

def calculate_authenticity(reference: Pattern, generated: Pattern, library: ReferencePatternLibrary) -> float:
    """Enhanced authenticity: similarity in features vs. reference library."""
    ref_features = extract_features(reference) if hasattr(reference, 'notes') else {'density': 0.5}
    gen_features = extract_features(generated)
    features_list = [extract_features(p) for p in library.get_patterns('pop', 'melody')]
    keys = ['density', 'avg_velocity', 'num_notes', 'mean_pitch']
    ref_lib_features = np.mean([[f.get(k, 0.0) for k in keys] for f in features_list], axis=0)
    sim = float(cosine_similarity(np.array([list(ref_features.values())]), np.array([list(gen_features.values())]))[0][0])
    lib_sim = float(cosine_similarity(np.array([list(gen_features.values())]), np.array([list(ref_lib_features)]))[0][0])
    return (sim + lib_sim) / 2

def calculate_diversity(patterns: List[Pattern]) -> float:
    """Enhanced diversity: cosine similarity variance on features."""
    if len(patterns) < 2:
        return 0.0
    features_list = [list(extract_features(p).values()) for p in patterns]
    features_array = np.array(features_list)
    sim_matrix = cosine_similarity(features_array)
    np.fill_diagonal(sim_matrix, 0)
    diversity = 1 - float(np.mean(sim_matrix))
    return diversity

def measure_performance(pattern: Pattern) -> float:
    """Performance benchmark: mock generation time."""
    return len(pattern.notes) * 0.001  # ms per note

def check_harmonic_consistency(pattern: Pattern) -> bool:
    """Check if chords follow valid progressions."""
    if pattern.pattern_type != PatternType.HARMONY or not pattern.chords:
        return True
    # Mock: check if all chords are major/minor
    return all(len(c.notes) >= 3 for c in pattern.chords)

def check_rhythmic_consistency(pattern: Pattern, genre_char: Dict) -> bool:
    """Check syncopation and swing match genre."""
    if pattern.pattern_type != PatternType.RHYTHM:
        return True
    sync_level = len([n for n in pattern.notes if n.start_time % 1 != 0]) / len(pattern.notes) if pattern.notes else 0
    sync_key = 'syncopation_level' if 'syncopation_level' in genre_char else 'syncopation'
    return abs(sync_level - genre_char.get(sync_key, 0.0)) < 0.2


class TestValidationMetrics:
    def test_pattern_quality(self, mock_pattern):
        quality = calculate_pattern_quality(mock_pattern)
        assert 0 <= quality <= 10

    def test_pattern_quality_empty(self):
        empty = Pattern(PatternType.MELODY, [], [])
        quality = calculate_pattern_quality(empty)
        assert quality == 0.0

    def test_authenticity_identical(self, mock_pattern, mock_library):
        authenticity = calculate_authenticity(mock_pattern, mock_pattern, mock_library)
        assert authenticity >= 0.8

    def test_authenticity_different(self, mock_pattern, mock_library):
        diff = Pattern(PatternType.MELODY, [Note(72, 1.0, 64, 0.0)], [])
        authenticity = calculate_authenticity(mock_pattern, diff, mock_library)
        assert 0 < authenticity < 1

    def test_diversity_identical(self, mock_pattern):
        diversity = calculate_diversity([mock_pattern, mock_pattern])
        assert diversity < 0.1

    def test_diversity_varied(self, mock_pattern):
        varied = Pattern(PatternType.MELODY, [Note(72, 1.0, 64, 0.0)], [])
        diversity = calculate_diversity([mock_pattern, varied])
        assert diversity > 0.5

    def test_performance(self, mock_pattern):
        perf = measure_performance(mock_pattern)
        assert perf > 0

    def test_harmonic_consistency(self, mock_pattern):
        mock_pattern.pattern_type = PatternType.HARMONY
        mock_pattern.chords = [mock_chord(mock_note())]
        assert check_harmonic_consistency(mock_pattern)

    def test_rhythmic_consistency(self, mock_pattern, mock_generator_context):
        mock_pattern.pattern_type = PatternType.RHYTHM
        genre_char = mock_generator_context.genre_rules.get_genre_characteristics("pop")
        assert check_rhythmic_consistency(mock_pattern, genre_char)

# Integration Tests for Cross-Genre Pattern Generation
class TestCrossGenrePatterns:
    @pytest.mark.parametrize("genre", ["drum-and-bass", "jazz", "pop", "rock", "electronic"])
    def test_pattern_generation_for_genre(self, genre, mock_generator_context, mock_library):
        """Test pattern generation for all supported genres."""
        mock_generator_context.mood = "energetic"
        rules = GenreFactory.create_genre_rules(genre)
        mock_generator_context.genre_rules = rules
        rhythm_gen = RhythmGenerator(mock_generator_context)
        pattern = rhythm_gen.generate(4)
        assert isinstance(pattern, Pattern)
        assert len(pattern.notes) > 0
        genre_char = rules.get_genre_characteristics(genre)
        assert pattern.tempo >= genre_char["tempo_min"] - 20  # Allow some variance

    @pytest.mark.parametrize("genre", ["drum-and-bass", "jazz", "pop", "rock", "electronic"])
    def test_genre_specific_characteristics(self, genre, mock_generator_context):
        """Validate genre-specific pattern characteristics."""
        rules = GenreFactory.create_genre_rules(genre)
        mock_generator_context.genre_rules = rules
        rhythm_gen = RhythmGenerator(mock_generator_context)
        pattern = rhythm_gen.generate(4)
        genre_char = rules.get_genre_characteristics(genre)
        assert check_rhythmic_consistency(pattern, genre_char)
        # Specific checks
        sync_level = len([n for n in pattern.notes if n.start_time % 1 != 0]) / len(pattern.notes) if pattern.notes else 0
        swing_factor = genre_char.get('swing_factor', 0.5)
        if genre == "drum-and-bass":
            assert any(n.start_time % 0.25 == 0 for n in pattern.notes)  # Breakbeat quarters
        elif genre == "jazz":
            assert swing_factor >= 0.6  # Swing
        elif genre == "pop":
            assert sync_level <= 0.4  # Straight
        elif genre == "rock":
            assert len([n for n in pattern.notes if n.start_time in [0, 2]]) > 0  # 1 and 3 emphasis
        elif genre == "electronic":
            assert len([n for n in pattern.notes if n.start_time % 1 == 0]) >= 4  # Four-on-floor

    @pytest.mark.parametrize("genre,mood", [("drum-and-bass", "energetic"), ("jazz", "calm"), ("pop", "happy"), ("rock", "sad"), ("electronic", "energetic")])
    def test_mood_based_pattern_selection(self, genre, mood, mock_generator_context, mock_library):
        """Test mood-based pattern selection across genres."""
        mock_generator_context.mood = mood
        rules = GenreFactory.create_genre_rules(genre)
        mock_generator_context.genre_rules = rules
        selector = AdaptivePatternSelector(mock_library)
        best_patterns = selector.get_best_patterns(mood, genre, 0.5, 120, 160, "rhythm", n=3, threshold=0.3)
        assert len(best_patterns) > 0
        # Mood influences selection score
        features = extract_features(best_patterns[0])

    @pytest.mark.parametrize("genre", ["drum-and-bass", "jazz", "pop", "rock", "electronic"])
    def test_pattern_authenticity(self, genre, mock_generator_context, mock_library):
        """Verify pattern authenticity for each genre."""
        rules = GenreFactory.create_genre_rules(genre)
        mock_generator_context.genre_rules = rules
        rhythm_gen = RhythmGenerator(mock_generator_context)
        pattern = rhythm_gen.generate(4)
        ref_pattern = mock_library.get_patterns(genre, "rhythm")[0]
        authenticity = calculate_authenticity(ref_pattern, pattern, mock_library)
        assert authenticity > 0.7  # Threshold for authenticity

# Integration Tests for Multi-Component Workflow
class TestMultiComponentWorkflow:
    def test_complete_generation_workflow(self, mock_generator_context, mock_library):
        """Test complete generation workflows combining rhythm, harmony, and melody."""
        mock_generator_context.pattern_library = mock_library
        rhythm_gen = RhythmGenerator(mock_generator_context)
        harmony_gen = HarmonyGenerator(mock_generator_context)
        melody_gen = MelodyGenerator(mock_generator_context)
        blender = PatternBlender()
        variation_engine = PatternVariationEngine()

        rhythm = rhythm_gen.generate(4)
        harmony = harmony_gen.generate_harmony_pattern(genre="pop", mood=mock_generator_context.mood, bars=4)
        melody = melody_gen.generate(4)

        # Blend components
        blended_rhythm = variation_engine.apply_variation(rhythm, VariationType.TIME_STRETCH, {"factor": 1.1}, 0.8)
        blended_harmony = blender.blend_harmonies([harmony])
        blended_melody = blender.blend_melodies([melody])

        # Verify workflow
        assert all(isinstance(p, Pattern) for p in [blended_rhythm, blended_harmony, blended_melody])
        assert check_harmonic_consistency(blended_harmony)
        assert len(blended_melody.notes) > 0

    def test_pattern_blending_between_components(self, mock_generator_context):
        """Validate pattern blending between different components."""
        rhythm_gen = RhythmGenerator(mock_generator_context)
        harmony_gen = HarmonyGenerator(mock_generator_context)
        rhythm = rhythm_gen.generate(2)
        harmony = harmony_gen.generate_harmony_pattern(genre="pop", mood=mock_generator_context.mood, bars=1)
        blender = PatternBlender()
        # Cross-blend (mock: rhythm influences harmony density)
        blended = blender.layer_patterns([rhythm, harmony])
        assert len(blended.notes) > len(rhythm.notes)
        assert len(blended.chords) > 0

    def test_adaptive_selection_across_components(self, mock_generator_context, mock_library):
        """Test adaptive selection across all components."""
        mock_generator_context.pattern_library = mock_library
        selector = AdaptivePatternSelector(mock_library)
        rhythm_patterns, _ = selector.get_best_patterns("energetic", "pop", 0.5, 120, 140, "rhythm", n=1)
        harmony_patterns, _ = selector.get_best_patterns("energetic", "pop", 0.5, 120, 140, "harmony", n=1)
        assert len(rhythm_patterns) > 0
        assert len(rhythm_patterns[0].notes) > 0
        assert len(harmony_patterns) > 0
        assert len(harmony_patterns[0].notes) > 0

    def test_ml_analysis_integration(self, mock_generator_context, mock_library):
        """Verify ML analysis integration in workflows."""
        mock_generator_context.pattern_library = mock_library
        rhythm_gen = RhythmGenerator(mock_generator_context)
        pattern = rhythm_gen.generate(4)
        features = extract_features(pattern)
        assert "density" in features
        assert "intervals" in features
        # Mock ML feedback
        engine = AdaptationEngine()
        suggestions = engine.adapt_to_performance_feedback("ml_similarity", features.get("density", 0))
        assert isinstance(suggestions, list)

# Performance and Scalability Tests
class TestPerformanceScalability:
    @pytest.mark.performance
    def test_system_performance_large_library(self, mock_generator_context, large_library, timing_context):
        """Test system performance with large pattern libraries."""
        mock_generator_context.pattern_library = large_library
        gen = RhythmGenerator(mock_generator_context, large_library)
        start = time.time()
        pattern = gen.generate(8)
        end = time.time()
        assert end - start < 2.0  # <2s for scalability
        assert len(pattern.notes) > 0

    @pytest.mark.performance
    def test_memory_management_under_load(self, large_library):
        """Validate memory management under load."""
        lib = large_library
        selector = AdaptivePatternSelector(lib)
        patterns = selector.get_best_patterns("energetic", "pop", 0.5, 120, 140, "melody", n=100)
        assert len(patterns) == 100
        # Mock memory check: no crash

    @pytest.mark.performance
    def test_concurrent_pattern_generation(self, mock_generator_context, mock_library):
        """Test concurrent pattern generation."""
        mock_generator_context.pattern_library = mock_library
        gen = RhythmGenerator(mock_generator_context)

        def generate_pattern():
            return gen.generate(4)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_pattern) for _ in range(5)]
            patterns = [f.result() for f in concurrent.futures.as_completed(futures)]
        assert all(isinstance(p, Pattern) for p in patterns)
        assert all(len(p.notes) > 0 for p in patterns)

    @pytest.mark.parametrize("genre", ["drum-and-bass", "jazz", "pop", "rock", "electronic"])
    def test_benchmark_generation_speed(self, genre, mock_generator_context, timing_context):
        """Benchmark generation speed across genres."""
        rules = GenreFactory.create_genre_rules(genre)
        mock_generator_context.genre_rules = rules
        gen = RhythmGenerator(mock_generator_context)
        start = time.time()
        pattern = gen.generate(16)  # Longer for benchmark
        end = time.time()
        assert end - start < 1.0  # <1s per genre

# Quality and Consistency Tests
class TestQualityConsistency:
    def test_pattern_quality_metrics_across_genres(self, mock_genres_moods, mock_generator_context):
        """Validate pattern quality metrics across genres."""
        patterns = []
        for genre, mood in mock_genres_moods:
            mock_generator_context.mood = mood
            rules = GenreFactory.create_genre_rules(genre)
            mock_generator_context.genre_rules = rules
            gen = RhythmGenerator(mock_generator_context)
            pattern = gen.generate(4)
            quality = calculate_pattern_quality(pattern)
            assert quality > 0.5  # Minimum quality
            patterns.append(pattern)
        avg_quality = float(np.mean([calculate_pattern_quality(p) for p in patterns]))
        assert avg_quality > 0.7

    def test_pattern_diversity_avoid_repetition(self, mock_generator_context):
        """Test pattern diversity and avoid repetition."""
        rules = GenreFactory.create_genre_rules("pop")
        mock_generator_context.genre_rules = rules
        gen = RhythmGenerator(mock_generator_context)
        patterns = [gen.generate(4) for _ in range(10)]
        diversity = calculate_diversity(patterns)
        assert diversity > 0.3  # Avoid high repetition
        # Check no identical consecutive
        for i in range(len(patterns) - 1):
            assert len(patterns[i].notes) != len(patterns[i+1].notes) or any(n.pitch != patterns[i+1].notes[j].pitch for j, n in enumerate(patterns[i].notes))

    def test_harmonic_rhythmic_consistency(self, mock_generator_context):
        """Verify harmonic and rhythmic consistency."""
        harmony_gen = HarmonyGenerator(mock_generator_context)
        rhythm_gen = RhythmGenerator(mock_generator_context)
        harmony = harmony_gen.generate_harmony_pattern(genre="pop", mood=mock_generator_context.mood, bars=4)
        rhythm = rhythm_gen.generate(4)
        assert check_harmonic_consistency(harmony)
        genre_char = mock_generator_context.genre_rules.get_genre_characteristics("pop")
        assert check_rhythmic_consistency(rhythm, genre_char)
        # Cross-consistency: rhythm aligns with harmony beats
        rhythm_beats = {n.start_time // 1 for n in rhythm.notes}
        harmony_beats = {c.start_time // 1 for c in harmony.chords}
        overlap = len(rhythm_beats & harmony_beats) / max(len(rhythm_beats), len(harmony_beats), 1)
        assert overlap > 0.5

    def test_error_handling_complex_workflows(self, mock_generator_context, mock_library):
        """Test error handling in complex workflows."""
        mock_generator_context.pattern_library = mock_library
        # Invalid genre
        with pytest.raises(ValueError):
            GenreFactory.create_genre_rules("invalid")
        # Empty library selection
        empty_lib = ReferencePatternLibrary()
        selector = AdaptivePatternSelector(empty_lib)
        patterns = selector.get_best_patterns("energetic", "pop", 0.5, 120, 140, "melody")
        assert len(patterns) == 0  # Graceful handling
        # Workflow with error recovery
        try:
            gen = RhythmGenerator(mock_generator_context)
            gen.generate(num_bars=-1)  # Invalid
        except ValueError:
            pass  # Handled

# Integration Tests for End-to-End Workflows
class TestIntegration:
    def test_full_pattern_generation(self, mock_generator_context):
        rhythm_gen = RhythmGenerator(mock_generator_context)
        harmony_gen = HarmonyGenerator(mock_generator_context)
        melody_gen = MelodyGenerator(mock_generator_context)

        rhythm = rhythm_gen.generate(4)
        harmony = harmony_gen.generate_harmony_pattern(genre="pop", mood=mock_generator_context.mood, bars=4)
        melody = melody_gen.generate(4)

        assert all(isinstance(p, Pattern) for p in [rhythm, harmony, melody])
        assert len(rhythm.notes) > 0
        assert len(harmony.chords) > 0
        assert len(melody.notes) > 0

    def test_blending_with_variation(self, mock_pattern):
        blender = PatternBlender()
        engine = PatternVariationEngine()
        blended = blender.blend_melodies([mock_pattern, mock_pattern])
        varied = engine.time_stretch(blended, 1.1, 1.0)
        assert isinstance(varied, Pattern)
        assert len(varied.notes) == len(blended.notes)

    def test_adaptive_selection_with_library(self, mock_library):
        selector = AdaptivePatternSelector(mock_library)
        best = selector.get_best_patterns("energetic", "pop", 0.5, 120, 140, "melody", n=1, threshold=0.0)
        if best:
            engine = AdaptationEngine()
            engine.adapt_to_performance_feedback("rhythm_accuracy", 0.8)
            assert True  # Integrated without error

    def test_end_to_end_with_genres(self, mock_genres_moods, mock_generator_context):
        for genre, mood in mock_genres_moods:
            mock_generator_context.mood = mood
            rules = GenreFactory.create_genre_rules(genre)
            mock_generator_context.genre_rules = rules
            rhythm_gen = RhythmGenerator(mock_generator_context)
            pattern = rhythm_gen.generate(2)
            assert len(pattern.notes) > 0

# Cross-Component Compatibility Tests
class TestCompatibility:
    def test_variation_with_blender(self, mock_pattern):
        blender = PatternBlender()
        engine = PatternVariationEngine()
        layered = blender.layer_patterns([mock_pattern, mock_pattern])
        varied = engine.pitch_transpose(layered, 2, 1.0)
        assert len(varied.notes) == len(layered.notes)
        assert varied.notes[0].pitch == 62

    def test_library_selector_compatibility(self, mock_library):
        selector = AdaptivePatternSelector(mock_library)
        patterns = selector.get_best_patterns("energetic", "pop", 0.5, 120, 140, "melody", threshold=0.0)
        assert isinstance(patterns, list)

    def test_generator_library_integration(self, mock_generator_context, mock_library):
        mock_generator_context.pattern_library = mock_library
        gen = RhythmGenerator(mock_generator_context, mock_library)
        pattern = gen.generate_with_templates(4, variation_level=0.2)
        assert isinstance(pattern, Pattern)
        assert len(pattern.notes) > 0

    def test_blender_with_adaptive(self, mock_library, mock_pattern):
        selector = AdaptivePatternSelector(mock_library)
        blender = PatternBlender()
        # Mock blend with selected
        blended = blender.blend_melodies([mock_pattern])
        assert isinstance(blended, Pattern)

# Error Handling and Edge Case Tests
class TestErrorCases:
    def test_empty_pattern_quality(self):
        empty = Pattern(PatternType.MELODY, [], [])
        quality = calculate_pattern_quality(empty)
        assert quality == 0.0

    def test_invalid_complexity_rhythm(self, mock_generator_context):
        gen = RhythmGenerator(mock_generator_context)
        with pytest.raises(ValueError):
            gen.generate(4, beat_complexity=-0.1)

    def test_no_patterns_in_library(self):
        empty_lib = ReferencePatternLibrary()
        selector = AdaptivePatternSelector(empty_lib)
        best = selector.get_best_patterns("energetic", "unknown", 0.5, 120, 140, "melody")
        assert len(best) == 0

    def test_blend_empty_patterns(self):
        blender = PatternBlender()
        empty = Pattern(PatternType.RHYTHM, [], [])
        with pytest.raises(ValueError):
            blender.blend_rhythms([empty])

    def test_adaptation_cooldown(self):
        engine = AdaptationEngine()
        suggestion = AdaptationSuggestion(AdaptationTrigger.TEMPO_DRIFT, "tempo", 120, 100, 0.8, "test", 5)
        engine.apply_adaptation(suggestion)
        # Simulate short time
        engine.context.last_adaptation_time = time.time() - 1
        success = engine.apply_adaptation(suggestion)
        assert success is False  # Within cooldown

    def test_invalid_midi_file(self):
        with pytest.raises(FileNotFoundError):
            extract_from_file("invalid.mid")

    def test_library_add_invalid_pattern(self, mock_library):
        invalid_data = Mock()  # Not MidiPatternData
        with pytest.raises(AttributeError):
            mock_library.add_pattern(invalid_data, "pop", "melody")

    # Enhanced edge cases
    def test_zero_density_workflow(self, mock_generator_context):
        mock_generator_context.density_manager.get_density_settings.return_value = {"note_density": 0}
        gen = RhythmGenerator(mock_generator_context)
        pattern = gen.generate(4)
        assert len(pattern.notes) == 0  # Handles zero density

    def test_large_bar_count(self, mock_generator_context):
        gen = RhythmGenerator(mock_generator_context)
        pattern = gen.generate(100)  # Large
        assert len(pattern.notes) > 0  # No crash

# Test Data Generation for Genres and Moods
class TestDataGeneration:
    @pytest.mark.parametrize("genre,mood", [("drum-and-bass", "energetic"), ("jazz", "calm"), ("pop", "happy"), ("rock", "energetic"), ("electronic", "sad")])
    def test_parametrized_genres_moods(self, genre, mood, mock_generator_context):
        mock_generator_context.mood = mood
        rules = GenreFactory.create_genre_rules(genre)
        mock_generator_context.genre_rules = rules
        rhythm_gen = RhythmGenerator(mock_generator_context)
        pattern = rhythm_gen.generate(2)
        assert len(pattern.notes) > 0
        assert pattern.pattern_type == PatternType.RHYTHM

    def test_synthetic_genre_data_variety(self):
        genres = ["drum-and-bass", "jazz", "pop", "rock", "electronic"]
        patterns = []
        for genre in genres:
            rules = GenreFactory.create_genre_rules(genre)
            context = Mock()
            context.genre_rules = rules
            context.mood = "energetic"
            context.scale_pitches = [60, 62, 64, 65, 67, 69, 71]
            context.density_manager = MockDensityManager()
            gen = RhythmGenerator(context)
            pattern = gen.generate(2)
            patterns.append(pattern)
        # Check diversity
        diversity = calculate_diversity(patterns)
        assert diversity > 0.2  # Varied by genre

# Manual Validation Tests (pytest.mark.manual)
class TestManualValidation:
    @pytest.mark.manual
    def test_dnb_breakbeat_authenticity(self):
        """Manual: Listen to DnB output and verify breakbeat feel."""
        pass  # Run manually: python main.py --genre drum-and-bass --mood energetic

    @pytest.mark.manual
    def test_jazz_swing_timing(self):
        """Manual: Verify jazz swing in playback."""
        pass

    @pytest.mark.manual
    def test_overall_system_integration(self):
        """Manual: Generate full song across genres and check coherence."""
        pass

# Pytest Organization and Reports
def test_all_imports():
    """Test that all imports work for organization."""
    from generators.pattern_orchestrator import PatternOrchestrator
    from music_theory import MusicTheory
    assert True

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--html=reports/test_report.html", "--self-contained-html"])