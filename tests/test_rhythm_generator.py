import pytest
from unittest.mock import Mock, patch
from generators.rhythm_generator import RhythmGenerator
from structures.data_structures import Pattern, PatternType, Note
from generators.generator_context import GeneratorContext
from tests.test_helpers import (
    create_mock_song_skeleton,
    create_mock_music_theory,
    validate_pattern,
    assert_pattern_properties,
    generate_random_scale_pitches
)

# Mock PopRules for testing
class MockPopRules:
    def get_rhythm_patterns(self):
        return [{'pattern': [0.25, 0.25, 0.25, 0.25]}]  # Simple eighth notes

    def get_genre_characteristics(self, genre):
        return {
            'syncopation_level': 0.3,
            'swing_factor': 0.5,
            'tempo_min': 120
        }

    def get_beat_characteristics(self, subgenre=None):
        return {
            'swing_factor': 0.5,
            'syncopation_level': 0.3,
            'emphasis_patterns': [1, 3]
        }

class TestRhythmGenerator:
    """Test suite for RhythmGenerator edge cases and functionality."""

    @pytest.fixture
    def context(self):
        """Basic GeneratorContext for testing."""
        context = Mock(spec=GeneratorContext)
        context.genre_rules = MockPopRules()
        context.mood = "happy"
        context.tempo = 120
        context.density_manager = Mock()
        context.density_manager.get_rhythm_pattern_complexity.return_value = 0.5
        context.density_manager.calculate_note_probability.return_value = 0.8
        context.scale_pitches = generate_random_scale_pitches()
        return context

    @pytest.fixture
    def rhythm_generator(self, context):
        """RhythmGenerator instance."""
        return RhythmGenerator(context)

    def test_generate_basic_rhythm(self, rhythm_generator, context):
        """Test basic rhythm generation."""
        pattern = rhythm_generator.generate(num_bars=4)
        assert validate_pattern(pattern)
        assert pattern.type == PatternType.RHYTHM
        assert len(pattern.notes) > 0
        assert pattern.tempo == 120.0
        assert_pattern_properties(pattern, PatternType.RHYTHM, min_notes=4)

    def test_generate_with_invalid_complexity(self, rhythm_generator):
        """Test generation with invalid beat_complexity (edge case)."""
        with pytest.raises(ValueError, match="beat_complexity must be between 0.0 and 1.0"):
            rhythm_generator.generate(num_bars=4, beat_complexity=-0.1)

    def test_generate_with_high_complexity(self, rhythm_generator, context):
        """Test generation with high beat complexity."""
        with patch.object(context.density_manager, 'get_rhythm_pattern_complexity', return_value=0.9):
            pattern = rhythm_generator.generate(num_bars=4, beat_complexity=0.9)
            assert len(pattern.notes) >= 8  # Expect more notes for high complexity
            assert_pattern_properties(pattern, min_notes=8)

    def test_generate_with_zero_bars(self, rhythm_generator):
        """Test generation with zero bars (edge case)."""
        pattern = rhythm_generator.generate(num_bars=0)
        assert isinstance(pattern, Pattern)
        assert len(pattern.notes) == 0  # Should generate empty pattern

    def test_generate_with_no_genre_rules(self, rhythm_generator):
        """Test generation without genre rules (edge case)."""
        rhythm_generator.context.genre_rules = None
        with pytest.raises(ValueError, match="No genre rules provided"):
            rhythm_generator.generate(num_bars=4)

    def test_generate_with_empty_rhythm_patterns(self, rhythm_generator, context):
        """Test fallback to straight eighths when no patterns available."""
        with patch.object(context.genre_rules, 'get_rhythm_patterns', return_value=[]):
            pattern = rhythm_generator.generate(num_bars=4)
            assert len(pattern.notes) > 0
            # Check if fallback pattern used (eighth notes)
            durations = [note.duration for note in pattern.notes]
            assert all(d == 0.25 for d in durations[:16])  # First 4 bars eighth notes

    def test_generate_with_templates_no_library(self, rhythm_generator):
        """Test template generation fallback without library."""
        pattern = rhythm_generator.generate_with_templates(num_bars=4)
        assert isinstance(pattern, Pattern)
        # Should fallback to standard generate
        assert len(pattern.notes) > 0

    def test_generate_with_templates_invalid_variation(self, rhythm_generator):
        """Test template generation with invalid variation_level (edge case)."""
        with pytest.raises(ValueError, match="variation_level must be between 0.0 and 1.0"):
            rhythm_generator.generate_with_templates(num_bars=4, variation_level=1.5)

    def test_adjust_syncopation_reduces_off_beats(self, rhythm_generator):
        """Test _adjust_syncopation reduces off-beat notes for low syncopation."""
        # Create notes with high syncopation (many off-beats)
        notes = [
            Note(60, 0.25, 80, 0.0),  # On beat
            Note(60, 0.25, 80, 0.125),  # Off beat
            Note(60, 0.25, 80, 0.25),  # On beat
            Note(60, 0.25, 80, 0.375),  # Off beat
            Note(60, 0.25, 80, 0.5),  # On beat
            Note(60, 0.25, 80, 0.625),  # Off beat
        ]
        adjusted = rhythm_generator._adjust_syncopation(notes, target_sync=0.2)
        off_beats_after = sum(1 for n in adjusted if n.start_time % 1 != 0)
        total_notes_after = len(adjusted)
        sync_ratio_after = off_beats_after / total_notes_after if total_notes_after > 0 else 0
        assert sync_ratio_after <= 0.4  # Within tolerance of target 0.2

    def test_is_compatible_pattern(self, rhythm_generator, context):
        """Test pattern compatibility checking."""
        # Compatible pattern (low syncopation for pop)
        compatible_pattern = [1.0, 1.0, 1.0, 1.0]  # All on beats
        assert rhythm_generator._is_compatible_pattern(compatible_pattern, context.genre_rules.get_genre_characteristics('pop'))

        # Incompatible pattern (high syncopation)
        incompatible_pattern = [0.125] * 16  # All off beats
        assert not rhythm_generator._is_compatible_pattern(incompatible_pattern, context.genre_rules.get_genre_characteristics('pop'))

    def test_generate_with_mock_library(self, rhythm_generator, context):
        """Test template generation with mocked library."""
        mock_library = Mock()
        mock_pattern = Mock()
        mock_pattern.notes = [Mock(time=0, duration=480, note=35, velocity=80) for _ in range(4)]
        mock_pattern.ticks_per_beat = 480
        mock_pattern.length_ticks = 1920  # 4 bars
        mock_library.get_patterns.return_value = [mock_pattern]
        mock_library.get_metadata.return_value = {'bpm': 120}
        rhythm_generator.pattern_library = mock_library

        pattern = rhythm_generator.generate_with_templates(num_bars=4, variation_level=0.0)
        assert len(pattern.notes) == 4
        assert all(note.pitch == 35 for note in pattern.notes)  # From mock

    @pytest.mark.parametrize("num_bars", [1, 8, 16])
    def test_generate_different_lengths(self, rhythm_generator, num_bars):
        """Test generation for different bar lengths."""
        pattern = rhythm_generator.generate(num_bars=num_bars)
        expected_min_notes = num_bars * 2  # At least 2 notes per bar
        assert len(pattern.notes) >= expected_min_notes
        total_duration = sum(note.duration for note in pattern.notes)
        expected_duration = num_bars * 4  # 4 beats per bar
        assert abs(total_duration - expected_duration) < 1.0  # Tolerance for variations