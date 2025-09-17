"""
Test Advanced Rhythm Engine

This module tests the new AdvancedRhythmEngine and its components:
- PolyrhythmEngine for cross-rhythmic patterns
- SyncopationEngine for sophisticated syncopation
- Groove-specific timing variations
- Complex rhythmic tension and resolution systems
"""

import unittest
from unittest.mock import Mock

from generators.advanced_rhythm_engine import (
    AdvancedRhythmEngine,
    PolyrhythmEngine,
    SyncopationEngine,
    RhythmPattern
)
from structures.data_structures import Note, Pattern, PatternType


class TestPolyrhythmEngine(unittest.TestCase):
    """Test the PolyrhythmEngine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = PolyrhythmEngine()

    def test_polyrhythm_templates_loaded(self):
        """Test that polyrhythm templates are properly loaded."""
        templates = self.engine.polyrhythm_templates
        self.assertIsInstance(templates, dict)
        self.assertGreater(len(templates), 0)

        # Check for expected polyrhythm types
        expected_types = ['3_over_4', '5_over_4', '7_over_8', '2_over_3']
        for polyrhythm_type in expected_types:
            self.assertIn(polyrhythm_type, templates)

    def test_apply_polyrhythm_simple(self):
        """Test applying polyrhythm to a simple pattern."""
        # Create a simple base pattern
        base_notes = [
            Note(36, 0.25, 100, 0.0),    # Kick
            Note(38, 0.25, 90, 1.0),     # Snare
            Note(36, 0.25, 100, 2.0),    # Kick
            Note(38, 0.25, 90, 3.0),     # Snare
        ]
        base_pattern = RhythmPattern(
            notes=base_notes,
            tempo=120.0,
            time_signature=(4, 4),
            complexity=0.3,
            groove_intensity=0.5
        )

        # Apply polyrhythm
        result = self.engine.apply_polyrhythm(base_pattern, 'jazz', 'simple')

        # Check that result is a RhythmPattern
        self.assertIsInstance(result, RhythmPattern)
        self.assertGreater(len(result.notes), len(base_notes))  # Should have added polyrhythmic notes

    def test_cross_rhythm_generation(self):
        """Test cross-rhythm pattern generation."""
        positions = self.engine.generate_cross_rhythm(4, 3, 4.0)

        # Should generate 3 positions over 4 beats
        self.assertEqual(len(positions), 3)

        # All positions should be within the duration
        for pos in positions:
            self.assertGreaterEqual(pos, 0.0)
            self.assertLess(pos, 4.0)


class TestSyncopationEngine(unittest.TestCase):
    """Test the SyncopationEngine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = SyncopationEngine()

    def test_syncopation_patterns_loaded(self):
        """Test that syncopation patterns are properly loaded."""
        patterns = self.engine.syncopation_patterns
        self.assertIsInstance(patterns, dict)
        self.assertGreater(len(patterns), 0)

    def test_apply_syncopation(self):
        """Test applying syncopation to a pattern."""
        # Create a simple on-beat pattern
        base_notes = [
            Note(36, 0.25, 100, 0.0),    # Kick on beat 1
            Note(38, 0.25, 90, 1.0),     # Snare on beat 2
            Note(36, 0.25, 100, 2.0),    # Kick on beat 3
            Note(38, 0.25, 90, 3.0),     # Snare on beat 4
        ]
        base_pattern = RhythmPattern(
            notes=base_notes,
            tempo=120.0,
            time_signature=(4, 4),
            complexity=0.3,
            groove_intensity=0.5
        )

        # Apply syncopation
        result = self.engine.apply_syncopation(base_pattern, 'funk', 0.6)

        # Check that result is enhanced
        self.assertIsInstance(result, RhythmPattern)
        self.assertGreaterEqual(result.complexity, base_pattern.complexity)

    def test_generate_swing_feel(self):
        """Test swing feel generation."""
        base_notes = [Note(38, 0.25, 90, 0.5)]  # Off-beat snare
        base_pattern = RhythmPattern(
            notes=base_notes,
            tempo=120.0,
            time_signature=(4, 4),
            complexity=0.5,
            groove_intensity=0.5
        )

        # Apply swing
        result = self.engine.generate_swing_feel(base_pattern, 0.7)

        # Check that groove intensity increased
        self.assertGreater(result.groove_intensity, base_pattern.groove_intensity)


class TestAdvancedRhythmEngine(unittest.TestCase):
    """Test the AdvancedRhythmEngine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AdvancedRhythmEngine()

    def test_generate_complex_rhythm_jazz(self):
        """Test generating complex jazz rhythm."""
        result = self.engine.generate_complex_rhythm(
            genre='jazz',
            complexity='complex',
            time_signature=(4, 4),
            tempo=120.0
        )

        self.assertIsInstance(result, RhythmPattern)
        self.assertEqual(result.time_signature, (4, 4))
        self.assertEqual(result.tempo, 120.0)
        self.assertGreater(len(result.notes), 0)

    def test_generate_complex_rhythm_funk(self):
        """Test generating complex funk rhythm."""
        result = self.engine.generate_complex_rhythm(
            genre='funk',
            complexity='dense',
            time_signature=(4, 4),
            tempo=100.0
        )

        self.assertIsInstance(result, RhythmPattern)
        self.assertGreater(result.complexity, 0.5)  # Should be complex

    def test_generate_complex_rhythm_latin(self):
        """Test generating complex latin rhythm."""
        result = self.engine.generate_complex_rhythm(
            genre='latin',
            complexity='complex',
            time_signature=(4, 4),
            tempo=110.0
        )

        self.assertIsInstance(result, RhythmPattern)
        self.assertGreater(len(result.notes), 8)  # Latin should have good density

    def test_apply_rhythmic_tension_resolution(self):
        """Test applying tension-resolution systems."""
        base_notes = [
            Note(36, 0.25, 100, 0.0),
            Note(38, 0.25, 90, 1.0),
            Note(36, 0.25, 100, 2.0),
            Note(38, 0.25, 90, 3.0),
        ]
        base_pattern = RhythmPattern(
            notes=base_notes,
            tempo=120.0,
            time_signature=(4, 4),
            complexity=0.5,
            groove_intensity=0.5
        )

        # Apply tension-resolution
        result = self.engine.apply_rhythmic_tension_resolution(base_pattern, 'build_release')

        self.assertIsInstance(result, RhythmPattern)
        self.assertGreaterEqual(result.complexity, base_pattern.complexity)

    def test_groove_variations(self):
        """Test groove-specific timing variations."""
        base_notes = [Note(36, 0.25, 100, i * 0.25) for i in range(16)]  # 16th notes
        base_pattern = RhythmPattern(
            notes=base_notes,
            tempo=120.0,
            time_signature=(4, 4),
            complexity=0.5,
            groove_intensity=0.5
        )

        # Test jazz groove
        jazz_result = self.engine._apply_groove_variations(
            base_pattern, 'jazz', 'complex'
        )
        self.assertGreater(jazz_result.groove_intensity, base_pattern.groove_intensity)

        # Test funk groove
        funk_result = self.engine._apply_groove_variations(
            base_pattern, 'funk', 'dense'
        )
        self.assertGreater(funk_result.groove_intensity, base_pattern.groove_intensity)


class TestRhythmIntegration(unittest.TestCase):
    """Test integration between rhythm components."""

    def test_full_rhythm_generation_pipeline(self):
        """Test the complete rhythm generation pipeline."""
        engine = AdvancedRhythmEngine()

        # Generate a complex rhythm
        pattern = engine.generate_complex_rhythm(
            genre='jazz',
            complexity='complex',
            time_signature=(4, 4),
            tempo=140.0
        )

        # Apply tension-resolution
        enhanced_pattern = engine.apply_rhythmic_tension_resolution(pattern, 'wave_pattern')

        # Verify the result
        self.assertIsInstance(enhanced_pattern, RhythmPattern)
        self.assertGreater(len(enhanced_pattern.notes), 0)
        self.assertEqual(enhanced_pattern.time_signature, (4, 4))
        self.assertEqual(enhanced_pattern.tempo, 140.0)

    def test_polyrhythm_and_syncopation_combination(self):
        """Test combining polyrhythm and syncopation."""
        polyrhythm_engine = PolyrhythmEngine()
        syncopation_engine = SyncopationEngine()

        # Create base pattern
        base_notes = [
            Note(36, 0.25, 100, 0.0),
            Note(38, 0.25, 90, 1.0),
            Note(42, 0.25, 70, 2.0),
            Note(38, 0.25, 90, 3.0),
        ]
        base_pattern = RhythmPattern(
            notes=base_notes,
            tempo=120.0,
            time_signature=(4, 4),
            complexity=0.3,
            groove_intensity=0.5
        )

        # Apply polyrhythm first
        polyrhythmic = polyrhythm_engine.apply_polyrhythm(base_pattern, 'latin', 'complex')

        # Then apply syncopation
        final = syncopation_engine.apply_syncopation(polyrhythmic, 'latin', 0.7)

        # Verify enhancement
        self.assertGreater(len(final.notes), len(base_notes))
        self.assertGreater(final.complexity, base_pattern.complexity)
        self.assertGreater(final.groove_intensity, base_pattern.groove_intensity)


if __name__ == '__main__':
    unittest.main()