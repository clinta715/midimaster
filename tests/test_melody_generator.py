import pytest
from generators.melody_generator import MelodyGenerator
from structures.data_structures import Pattern, PatternType, Note
from generators.generator_context import GeneratorContext
from genres.genre_rules import PopRules
from tests.test_helpers import (
    create_mock_song_skeleton,
    create_mock_music_theory,
    validate_pattern,
    assert_pattern_properties,
    generate_random_scale_pitches
)
from unittest.mock import Mock, patch

class TestMelodyGenerator:
    """Test suite for MelodyGenerator edge cases and functionality."""

    @pytest.fixture
    def context(self):
        """Basic GeneratorContext for testing."""
        context = Mock(spec=GeneratorContext)
        context.genre_rules = PopRules()
        context.mood = "happy"
        context.tempo = 120
        context.density_manager = Mock()
        context.density_manager.get_available_durations.return_value = [0.25, 0.5, 1.0]
        context.density_manager.should_place_note.return_value = True
        context.scale_pitches = generate_random_scale_pitches()
        return context

    @pytest.fixture
    def melody_generator(self, context):
        """MelodyGenerator instance."""
        return MelodyGenerator(context)

    def test_generate_basic_melody(self, melody_generator, context):
        """Test basic melody generation."""
        pattern = melody_generator.generate(num_bars=4)
        assert validate_pattern(pattern)
        assert pattern.type == PatternType.MELODY
        assert len(pattern.notes) > 0
        for note in pattern.notes:
            assert note.pitch in context.scale_pitches
        assert_pattern_properties(pattern, PatternType.MELODY, min_notes=4)

    def test_generate_with_empty_scale(self, melody_generator):
        """Test generation with empty scale (edge case)."""
        melody_generator.context.scale_pitches = []
        pattern = melody_generator.generate(num_bars=4)
        assert len(pattern.notes) > 0
        # Should fallback to C major
        fallback_scale = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83]
        for note in pattern.notes:
            assert note.pitch in fallback_scale

    def test_generate_zero_bars(self, melody_generator):
        """Test generation with zero bars (edge case)."""
        pattern = melody_generator.generate(num_bars=0)
        assert isinstance(pattern, Pattern)
        assert len(pattern.notes) == 0

    def test_choose_next_pitch_rising_contour(self, melody_generator):
        """Test pitch selection with rising contour."""
        melody_generator.last_pitch = 60
        pitch = melody_generator._choose_next_pitch('rising', {1: 0.4, 2: 0.3}, None, None)
        assert pitch >= 60  # Should prefer up or same

    def test_choose_next_pitch_falling_contour(self, melody_generator):
        """Test pitch selection with falling contour."""
        melody_generator.last_pitch = 72
        pitch = melody_generator._choose_next_pitch('falling', {1: 0.4, 2: 0.3}, None, None)
        assert pitch <= 72  # Should prefer down or same

    def test_choose_next_pitch_register_bounds(self, melody_generator):
        """Test pitch selection with register bounds."""
        melody_generator.last_pitch = 60
        pitch = melody_generator._choose_next_pitch('rising', {1: 0.4}, 60, 72)
        assert 60 <= pitch <= 72

    def test_generate_with_low_density(self, melody_generator, context):
        """Test generation with low density."""
        context.density_manager.should_place_note.return_value = False
        pattern = melody_generator.generate(num_bars=4)
        assert len(pattern.notes) == 0  # No notes placed

    def test_generate_with_no_genre_rules(self, melody_generator):
        """Test generation without genre rules (edge case)."""
        melody_generator.context.genre_rules = None
        with patch('builtins.print') as mock_print:
            pattern = melody_generator.generate(num_bars=4)
        mock_print.assert_called_with("Warning: Missing melody style configuration. Using default interval weights.")
        assert len(pattern.notes) > 0

    def test_generate_different_lengths(self, melody_generator):
        """Test generation for different bar lengths."""
        for num_bars in [1, 8, 16]:
            pattern = melody_generator.generate(num_bars=num_bars)
            expected_min_notes = num_bars * 1  # At least 1 note per bar
            assert len(pattern.notes) >= expected_min_notes

    @pytest.mark.parametrize("contour", ['rising', 'falling', 'arc', 'valley'])
    def test_invalid_contour_fallback(self, melody_generator, contour):
        """Test fallback for invalid contour."""
        with patch.object(melody_generator.context.genre_rules, 'get_melody_style', return_value={'contour_weights': {contour: 1.0}}):
            melody_generator.generate(num_bars=1)
            # No exception, uses the contour