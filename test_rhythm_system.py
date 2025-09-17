#!/usr/bin/env python3
"""
Unit and Integration Tests for Rhythm Generation System

Tests cover:
- Drum pattern template loading and validation
- Per-voice drum generation (kick, snare, ghost, hats)
- Rhythm variation engine functionality
- Beat analyzer validation
- Genre-specific rhythm characteristics

Run with: python -m pytest test_rhythm_system.py -v
"""

import pytest
import os
import tempfile
from typing import Dict, List, Any

# Import the modules we need to test
from generators.rhythm_generator import RhythmGenerator
from generators.generator_context import GeneratorContext
from genres.genre_factory import GenreFactory
from analyzers.beat_audit import BeatAuditor
from rhythm_generator_variations import RhythmVariationEngine

class TestRhythmSystem:
    """Test suite for rhythm generation system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = None

    def create_context(self, genre: str, subgenre: str = None, mood: str = "energetic"):
        """Create a test generator context."""
        genre_rules = GenreFactory.create_genre_rules(genre)
        self.context = GeneratorContext(genre_rules, mood, subgenre=subgenre)
        return self.context

    def test_drum_patterns_exist(self):
        """Test that all genres have drum pattern methods."""
        genres = ['pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical']

        for genre in genres:
            genre_rules = GenreFactory.create_genre_rules(genre)

            # All genres should have get_drum_patterns method
            assert hasattr(genre_rules, 'get_drum_patterns'), f"{genre} missing get_drum_patterns"

            # Call should not raise exception
            patterns = genre_rules.get_drum_patterns()
            assert isinstance(patterns, list), f"{genre} get_drum_patterns should return list"

    def test_hip_hop_drum_patterns(self):
        """Test Hip-Hop specific drum patterns."""
        context = self.create_context('hip-hop')
        rhythm_gen = RhythmGenerator(context)

        # Test trap patterns
        patterns = context.genre_rules.get_drum_patterns('trap')
        assert len(patterns) > 0, "Trap should have drum patterns"

        # Check pattern structure
        for pattern in patterns:
            assert 'name' in pattern, "Pattern missing name"
            assert 'steps_per_bar' in pattern, "Pattern missing steps_per_bar"
            assert 'voices' in pattern, "Pattern missing voices"
            assert isinstance(pattern['voices'], dict), "Voices should be dict"

            # Check for essential trap elements
            voices = pattern['voices']
            assert 'kick' in voices, "Trap pattern missing kick"
            assert 'snare' in voices, "Trap pattern missing snare"
            assert 'ch' in voices, "Trap pattern missing closed hats"

    def test_reggaeton_patterns(self):
        """Test reggaeton/reggaeton drum patterns."""
        context = self.create_context('hip-hop', 'reggaeton')
        rhythm_gen = RhythmGenerator(context)

        patterns = context.genre_rules.get_drum_patterns('reggaeton')
        assert len(patterns) > 0, "Reggaeton should have drum patterns"

        # Reggaeton should have shakers and specific rhythm
        found_reggaeton = False
        for pattern in patterns:
            voices = pattern['voices']
            if 'shaker' in voices:
                found_reggaeton = True
                # Check dem bow pattern (kick on 1, 2.5, 4)
                kick_steps = voices['kick']
                assert 0 in kick_steps, "Reggaeton missing kick on beat 1"
                break

        assert found_reggaeton, "Should find reggaeton pattern with shaker"

    def test_dnb_patterns_structure(self):
        """Test DnB pattern structure and A/B variations."""
        context = self.create_context('electronic', 'jungle')
        rhythm_gen = RhythmGenerator(context)

        patterns = context.genre_rules.get_drum_patterns('jungle')
        assert len(patterns) > 0, "DnB jungle should have patterns"

        # Check for A/B pattern variations
        pattern_names = [p['name'] for p in patterns]
        assert len(pattern_names) > 1, "DnB should have multiple pattern variations"

        # Check amen break characteristics
        for pattern in patterns:
            voices = pattern['voices']
            steps_per_bar = pattern['steps_per_bar']
            assert steps_per_bar == 16, "DnB should use 16 steps per bar"

            # Should have ghost snares for DnB feel
            assert 'ghost_snare' in voices, "DnB missing ghost snares"

    def test_variation_engine_initialization(self):
        """Test rhythm variation engine setup."""
        engine = RhythmVariationEngine(
            pattern_strength=0.8,
            swing_percent=0.6,
            fill_frequency=0.25,
            ghost_note_level=1.2
        )

        assert engine.pattern_strength == 0.8
        assert engine.swing_percent == 0.6
        assert engine.fill_frequency == 0.25
        assert engine.ghost_note_level == 1.2

    def test_variation_engine_pattern_switching(self):
        """Test pattern switching logic."""
        engine = RhythmVariationEngine(pattern_strength=0.5)  # Allow variation

        # Should switch patterns occasionally with low pattern_strength
        switches = []
        for bar in range(20):
            should_switch = engine.should_switch_pattern(bar, f"pattern_{bar%2}")
            switches.append(should_switch)

        # Should have some variation (not all False)
        assert any(switches), "Should have some pattern switching with low pattern_strength"

    def test_variation_engine_fill_logic(self):
        """Test fill generation logic."""
        engine = RhythmVariationEngine(fill_frequency=0.5)  # Fills every 2 bars

        fills = []
        for bar in range(8):
            has_fill = engine.should_add_fill(bar)
            fills.append(has_fill)

        # Should have fills roughly every 2 bars
        fill_count = sum(fills)
        assert fill_count > 2, f"Should have multiple fills, got {fill_count}"

    def test_variation_engine_micro_drop(self):
        """Test micro-drop logic."""
        engine = RhythmVariationEngine()

        # Micro-drops every 8 bars
        drops = []
        for bar in range(24):
            has_drop = engine.should_micro_drop(bar)
            drops.append(has_drop)

        drop_count = sum(drops)
        assert drop_count >= 2, f"Should have micro-drops every 8 bars, got {drop_count}"

    def test_variation_engine_ghost_modification(self):
        """Test ghost note probability modification."""
        engine = RhythmVariationEngine(ghost_note_level=2.0)

        # High ghost level should increase probability
        base_prob = 0.5
        modified_prob = engine.modify_ghost_probability('ghost_snare', False, base_prob)
        assert modified_prob > base_prob, "High ghost level should increase probability"

        # Fill bar should increase ghost probability further
        fill_prob = engine.modify_ghost_probability('ghost_snare', True, base_prob)
        assert fill_prob > modified_prob, "Fill bars should further increase ghost probability"

    def test_variation_engine_hat_thinning(self):
        """Test hat density reduction during micro-drops."""
        engine = RhythmVariationEngine()

        base_prob = 0.8

        # Normal bar
        normal_prob = engine.modify_hat_density('ch', False, base_prob)
        assert normal_prob == base_prob, "Normal bars should not modify hat density"

        # Micro-drop bar
        drop_prob = engine.modify_hat_density('ch', True, base_prob)
        assert drop_prob < base_prob, "Micro-drops should reduce hat density"

    def test_variation_engine_swing_timing(self):
        """Test swing timing application."""
        engine = RhythmVariationEngine(swing_percent=0.7)

        start_time = 1.0
        step = 1  # Odd step should be delayed in swing

        swing_time = engine.apply_swing_timing(start_time, 'snare', step)
        assert swing_time > start_time, "Swing should delay odd steps"

    def test_rhythm_generation_with_variations(self):
        """Integration test: generate rhythm with variation engine."""
        context = self.create_context('hip-hop', 'trap')
        rhythm_gen = RhythmGenerator(context)

        # Generate with variation parameters
        pattern = rhythm_gen.generate(
            num_bars=8,
            beat_complexity=0.7,
            pattern_strength=0.8,
            swing_percent=0.6,
            fill_frequency=0.25,
            ghost_note_level=1.2
        )

        # Should generate a valid pattern
        assert pattern is not None
        assert len(pattern.notes) > 0, "Should generate notes"

        # Should have drum notes in valid ranges
        for note in pattern.notes:
            assert 35 <= note.pitch <= 81, f"Drum pitch {note.pitch} out of valid range"
            assert 0 <= note.start_time, "Note start time should be non-negative"

    def test_beat_audit_analysis(self):
        """Test beat analyzer with generated rhythms."""
        context = self.create_context('electronic', 'techno')
        rhythm_gen = RhythmGenerator(context)

        # Generate a pattern
        pattern = rhythm_gen.generate(num_bars=4, beat_complexity=0.8)

        # Create temporary MIDI file for analysis
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Convert pattern to MIDI and analyze
            from output.midi_output import MidiOutput
            midi_output = MidiOutput()

            # Create minimal song skeleton for testing
            from structures.song_skeleton import SongSkeleton
            skeleton = SongSkeleton('electronic', 130, 'energetic')
            skeleton.set_rhythm_pattern(pattern)

            midi_output.save_to_midi(skeleton, tmp_path, context.genre_rules, context=context)

            # Analyze the generated file
            auditor = BeatAuditor()
            results = auditor.analyze_midi_file(tmp_path)

            # Should have analysis results
            assert 'overall_alignment' in results
            assert 'signature_diversity' in results
            assert 'duration_stats' in results

            # Electronic/techno should have reasonably good grid alignment
            alignment = results['overall_alignment']
            assert alignment > 0.5, f"Poor grid alignment: {alignment}"

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_genre_specific_characteristics(self):
        """Test that different genres produce distinct rhythm characteristics."""
        test_cases = [
            ('hip-hop', 'trap', {'tempo_range': (70, 90)}),
            ('electronic', 'techno', {'tempo_range': (130, 150)}),
            ('electronic', 'jungle', {'tempo_range': (155, 165)}),
            ('pop', None, {'tempo_range': (90, 140)})
        ]

        for genre, subgenre, expected in test_cases:
            context = self.create_context(genre, subgenre)
            characteristics = context.genre_rules.get_beat_characteristics(subgenre)

            tempo_range = characteristics['tempo_range']
            assert tempo_range[0] >= expected['tempo_range'][0] - 5, \
                f"{genre}/{subgenre} tempo too low: {tempo_range}"
            assert tempo_range[1] <= expected['tempo_range'][1] + 5, \
                f"{genre}/{subgenre} tempo too high: {tempo_range}"


if __name__ == "__main__":
    # Run tests directly
    test_suite = TestRhythmSystem()

    print("Running rhythm system tests...")

    # Run individual test methods
    test_methods = [
        test_suite.test_drum_patterns_exist,
        test_suite.test_hip_hop_drum_patterns,
        test_suite.test_reggaeton_patterns,
        test_suite.test_dnb_patterns_structure,
        test_suite.test_variation_engine_initialization,
        test_suite.test_variation_engine_pattern_switching,
        test_suite.test_variation_engine_fill_logic,
        test_suite.test_variation_engine_micro_drop,
        test_suite.test_variation_engine_ghost_modification,
        test_suite.test_variation_engine_hat_thinning,
        test_suite.test_variation_engine_swing_timing,
        test_suite.test_rhythm_generation_with_variations,
        test_suite.test_genre_specific_characteristics
    ]

    passed = 0
    failed = 0

    for test_method in test_methods:
        try:
            if hasattr(test_suite, 'setup_method'):
                test_suite.setup_method()

            test_method()
            print(f"✓ {test_method.__name__}")
            passed += 1

        except Exception as e:
            print(f"✗ {test_method.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")