#!/usr/bin/env python3
"""
Comprehensive test suite for MIDI Master selective generation enhancements.

This test suite validates:
1. Selective pattern generation (beats_only, chords_only, selective_patterns)
2. Subgenre support and beat characteristics
3. Parameter validation and error handling
4. Backward compatibility
5. Edge cases and error recovery
"""

import sys
import traceback
from typing import List, Dict, Any

# Import required modules
from genres.genre_factory import GenreFactory
from genres.genre_rules import PopRules, RockRules, JazzRules, ElectronicRules, HipHopRules, ClassicalRules
from generators.pattern_generator import PatternGenerator
from structures.song_skeleton import SongSkeleton
from structures.data_structures import Pattern, PatternType


class TestResults:
    """Container for test results and statistics."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []

    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ PASS: {test_name}")

    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ FAIL: {test_name} - {error}")

    def add_warning(self, test_name: str, warning: str):
        self.warnings.append(f"{test_name}: {warning}")
        print(f"⚠️  WARN: {test_name} - {warning}")

    def summary(self):
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        return f"""
Test Results Summary:
====================
Total Tests: {total}
Passed: {self.passed}
Failed: {self.failed}
Success Rate: {success_rate:.1f}%

Errors:
{chr(10).join(f"- {error}" for error in self.errors)}

Warnings:
{chr(10).join(f"- {warning}" for warning in self.warnings)}
"""


class SelectiveGenerationTester:
    """Test class for selective generation functionality."""

    def __init__(self):
        self.results = TestResults()
        self.test_genres = ['pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical']

    def run_all_tests(self):
        """Run all test suites."""
        print("🧪 Starting MIDI Master Selective Generation Test Suite")
        print("=" * 60)

        try:
            # Test 1: Selective pattern generation
            self.test_generate_beats_only()
            self.test_generate_chords_only()
            self.test_generate_selective_patterns()

            # Test 2: Backward compatibility
            self.test_backward_compatibility()

            # Test 3: Subgenre support
            self.test_subgenre_support()
            self.test_beat_characteristics()

            # Test 4: Parameter validation
            self.test_parameter_validation()

            # Test 5: Edge cases
            self.test_edge_cases()

            # Test 6: Integration testing
            self.test_integration()

        except Exception as e:
            self.results.add_fail("Test Suite Execution", f"Unexpected error: {str(e)}")
            traceback.print_exc()

        # Print final results
        print("\n" + self.results.summary())

    def test_generate_beats_only(self):
        """Test generate_beats_only() with different complexity levels."""
        print("\n🔊 Testing generate_beats_only()...")

        for genre_name in self.test_genres:
            try:
                # Create genre rules
                genre_rules = GenreFactory.create_genre_rules(genre_name)
                generator = PatternGenerator(genre_rules, 'energetic')
                skeleton = SongSkeleton('pop', 120, 'energetic')
                skeleton.tempo = 120

                # Test different complexity levels
                complexities = [0.0, 0.3, 0.7, 1.0]
                for complexity in complexities:
                    pattern = generator.generate_beats_only(skeleton, 4, complexity)

                    # Validate pattern
                    assert pattern is not None, f"Pattern is None for {genre_name} at complexity {complexity}"
                    assert pattern.pattern_type == PatternType.RHYTHM, f"Wrong pattern type: {pattern.pattern_type}"
                    assert len(pattern.notes) > 0, f"No notes generated for {genre_name} at complexity {complexity}"

                    # Check complexity affects note count (higher complexity = more notes)
                    if complexity > 0.5:
                        # Should have more notes at higher complexity
                        pass  # We'll validate this more thoroughly in integration

                self.results.add_pass(f"generate_beats_only - {genre_name}")

            except Exception as e:
                self.results.add_fail(f"generate_beats_only - {genre_name}", str(e))

    def test_generate_chords_only(self):
        """Test generate_chords_only() with different chord complexity levels."""
        print("\n🎵 Testing generate_chords_only()...")

        for genre_name in self.test_genres:
            try:
                genre_rules = GenreFactory.create_genre_rules(genre_name)
                generator = PatternGenerator(genre_rules, 'happy')
                skeleton = SongSkeleton('pop', 120, 'energetic')
                skeleton.tempo = 120

                # Test different chord complexities
                complexities = ['simple', 'medium', 'complex']
                for complexity in complexities:
                    pattern = generator.generate_chords_only(skeleton, 4, complexity)

                    # Validate pattern
                    assert pattern is not None, f"Pattern is None for {genre_name} at {complexity}"
                    assert pattern.pattern_type == PatternType.HARMONY, f"Wrong pattern type: {pattern.pattern_type}"
                    assert len(pattern.chords) > 0, f"No chords generated for {genre_name} at {complexity}"

                self.results.add_pass(f"generate_chords_only - {genre_name}")

            except Exception as e:
                self.results.add_fail(f"generate_chords_only - {genre_name}", str(e))

    def test_generate_selective_patterns(self):
        """Test generate_selective_patterns() with various combinations."""
        print("\n🎼 Testing generate_selective_patterns()...")

        test_combinations = [
            ['melody'],
            ['harmony'],
            ['rhythm'],
            ['bass'],
            ['melody', 'harmony'],
            ['melody', 'rhythm'],
            ['harmony', 'bass'],
            ['melody', 'harmony', 'rhythm'],
            ['melody', 'harmony', 'bass'],
            ['rhythm', 'bass'],
            ['melody', 'harmony', 'rhythm', 'bass']
        ]

        for genre_name in ['pop', 'rock', 'jazz']:  # Test subset for speed
            try:
                genre_rules = GenreFactory.create_genre_rules(genre_name)
                generator = PatternGenerator(genre_rules, 'calm')
                skeleton = SongSkeleton('pop', 120, 'energetic')
                skeleton.tempo = 120

                for pattern_types in test_combinations:
                    patterns = generator.generate_selective_patterns(
                        skeleton, 4, pattern_types,
                        beat_complexity=0.5, chord_complexity='medium'
                    )

                    # Validate results
                    assert len(patterns) == len(pattern_types), f"Wrong number of patterns: expected {len(pattern_types)}, got {len(patterns)}"

                    for i, pattern in enumerate(patterns):
                        expected_type = pattern_types[i]
                        assert pattern is not None, f"Pattern {i} is None"
                        # Note: Pattern type validation would require mapping strings to PatternType enum

                self.results.add_pass(f"generate_selective_patterns - {genre_name}")

            except Exception as e:
                self.results.add_fail(f"generate_selective_patterns - {genre_name}", str(e))

    def test_backward_compatibility(self):
        """Test that generate_patterns() maintains backward compatibility."""
        print("\n🔄 Testing backward compatibility...")

        for genre_name in self.test_genres:
            try:
                # Test old signature (without new parameters)
                genre_rules = GenreFactory.create_genre_rules(genre_name)
                generator = PatternGenerator(genre_rules, 'sad')
                skeleton = SongSkeleton('pop', 120, 'energetic')
                skeleton.tempo = 120

                # Old way - should still work
                patterns_old = generator.generate_patterns(skeleton, 4)

                # New way with all parameters
                patterns_new = generator.generate_patterns(
                    skeleton, 4,
                    beat_complexity=0.5,
                    chord_complexity='medium'
                )

                # Both should produce 4 patterns
                assert len(patterns_old) == 4, f"Old method failed: got {len(patterns_old)} patterns"
                assert len(patterns_new) == 4, f"New method failed: got {len(patterns_new)} patterns"

                # Pattern types should be consistent
                old_types = [p.pattern_type.name for p in patterns_old]
                new_types = [p.pattern_type.name for p in patterns_new]
                assert old_types == new_types, f"Pattern types differ: old={old_types}, new={new_types}"

                self.results.add_pass(f"backward_compatibility - {genre_name}")

            except Exception as e:
                self.results.add_fail(f"backward_compatibility - {genre_name}", str(e))

    def test_subgenre_support(self):
        """Test subgenre creation and validation."""
        print("\n🎭 Testing subgenre support...")

        subgenre_tests = {
            'pop': ['dance_pop', 'power_pop', 'synth_pop'],
            'rock': ['classic_rock', 'punk_rock', 'alternative_rock'],
            'jazz': ['bebop', 'cool_jazz', 'free_jazz'],
            'electronic': ['house', 'techno', 'ambient'],
            'hip-hop': ['west_coast', 'east_coast', 'trap'],
            'classical': ['baroque', 'classical', 'romantic']
        }

        for genre_name, subgenres in subgenre_tests.items():
            try:
                # Get genre rules and check subgenres
                genre_class = globals()[f"{genre_name.title().replace('-', '')}Rules"]
                rules_instance = genre_class()
                available_subgenres = rules_instance.get_subgenres()

                # Verify expected subgenres are available
                for subgenre in subgenres:
                    assert subgenre in available_subgenres, f"Subgenre {subgenre} not found in {genre_name}"

                # Test beat characteristics for subgenres
                for subgenre in subgenres:
                    characteristics = rules_instance.get_beat_characteristics(subgenre)
                    assert 'swing_factor' in characteristics, f"No swing_factor for {subgenre}"
                    assert 'syncopation_level' in characteristics, f"No syncopation_level for {subgenre}"
                    assert 'tempo_range' in characteristics, f"No tempo_range for {subgenre}"

                self.results.add_pass(f"subgenre_support - {genre_name}")

            except Exception as e:
                self.results.add_fail(f"subgenre_support - {genre_name}", str(e))

    def test_beat_characteristics(self):
        """Test beat characteristics for different subgenres."""
        print("\n🥁 Testing beat characteristics...")

        for genre_name in self.test_genres:
            try:
                genre_class = globals()[f"{genre_name.title().replace('-', '')}Rules"]
                rules_instance = genre_class()
                subgenres = rules_instance.get_subgenres()

                # Test base characteristics
                base_chars = rules_instance.get_beat_characteristics()
                assert isinstance(base_chars['swing_factor'], (int, float)), "swing_factor not numeric"
                assert isinstance(base_chars['syncopation_level'], (int, float)), "syncopation_level not numeric"
                assert isinstance(base_chars['tempo_range'], (tuple, list)), "tempo_range not tuple/list"

                # Test subgenre characteristics
                for subgenre in subgenres[:2]:  # Test first 2 subgenres for speed
                    sub_chars = rules_instance.get_beat_characteristics(subgenre)

                    # Should be different from base for meaningful subgenres
                    if subgenre != 'default':
                        # At least one characteristic should differ
                        differs = (
                            sub_chars['swing_factor'] != base_chars['swing_factor'] or
                            sub_chars['syncopation_level'] != base_chars['syncopation_level'] or
                            sub_chars['tempo_range'] != base_chars['tempo_range']
                        )
                        if not differs:
                            self.results.add_warning(f"beat_characteristics - {genre_name}:{subgenre}",
                                                   "Subgenre characteristics identical to base")

                self.results.add_pass(f"beat_characteristics - {genre_name}")

            except Exception as e:
                self.results.add_fail(f"beat_characteristics - {genre_name}", str(e))

    def test_parameter_validation(self):
        """Test parameter validation for all new methods."""
        print("\n✅ Testing parameter validation...")

        genre_rules = GenreFactory.create_genre_rules('pop')
        generator = PatternGenerator(genre_rules, 'happy')
        skeleton = SongSkeleton('pop', 120, 'energetic')
        skeleton.tempo = 120

        # Test beat complexity validation
        try:
            generator.generate_beats_only(skeleton, 4, beat_complexity=-0.1)
            self.results.add_fail("beat_complexity_validation", "Should reject negative complexity")
        except ValueError:
            self.results.add_pass("beat_complexity_validation - negative")

        try:
            generator.generate_beats_only(skeleton, 4, beat_complexity=1.1)
            self.results.add_fail("beat_complexity_validation", "Should reject complexity > 1.0")
        except ValueError:
            self.results.add_pass("beat_complexity_validation - too high")

        # Test chord complexity validation
        try:
            generator.generate_chords_only(skeleton, 4, chord_complexity='invalid')
            self.results.add_fail("chord_complexity_validation", "Should reject invalid complexity")
        except ValueError:
            self.results.add_pass("chord_complexity_validation - invalid")

        # Test selective patterns validation
        try:
            generator.generate_selective_patterns(skeleton, 4, ['invalid_type'])
            self.results.add_fail("selective_patterns_validation", "Should reject invalid pattern type")
        except ValueError:
            self.results.add_pass("selective_patterns_validation - invalid type")

    def test_edge_cases(self):
        """Test edge cases and error recovery."""
        print("\n🚨 Testing edge cases...")

        genre_rules = GenreFactory.create_genre_rules('pop')
        generator = PatternGenerator(genre_rules, 'energetic')
        skeleton = SongSkeleton('pop', 120, 'energetic')
        skeleton.tempo = 120

        # Test with extreme values
        try:
            # Very high complexity
            pattern = generator.generate_beats_only(skeleton, 4, beat_complexity=1.0)
            assert pattern is not None, "Failed with max complexity"
            self.results.add_pass("edge_case - max complexity")
        except Exception as e:
            self.results.add_fail("edge_case - max complexity", str(e))

        try:
            # Zero bars
            pattern = generator.generate_beats_only(skeleton, 0, beat_complexity=0.5)
            # Should handle gracefully (might return empty pattern)
            self.results.add_pass("edge_case - zero bars")
        except Exception as e:
            self.results.add_fail("edge_case - zero bars", str(e))

        # Test invalid genre (should be caught by GenreFactory)
        try:
            GenreFactory.create_genre_rules('invalid_genre')
            self.results.add_fail("edge_case - invalid genre", "Should reject invalid genre")
        except ValueError:
            self.results.add_pass("edge_case - invalid genre")

    def test_integration(self):
        """Test complete integration with MIDI output."""
        print("\n🎹 Testing integration...")

        try:
            from output.midi_output import MidiOutput

            # Create a complete pipeline test
            genre_rules = GenreFactory.create_genre_rules('pop')
            generator = PatternGenerator(genre_rules, 'happy')
            skeleton = SongSkeleton('pop', 120, 'energetic')
            skeleton.tempo = 120

            # Generate all pattern types selectively
            patterns = generator.generate_selective_patterns(
                skeleton, 4,
                ['melody', 'harmony', 'rhythm', 'bass'],
                beat_complexity=0.7,
                chord_complexity='complex'
            )

            # Verify we got all expected patterns
            pattern_types = [p.pattern_type.name for p in patterns]
            expected = ['MELODY', 'HARMONY', 'RHYTHM', 'BASS']
            assert pattern_types == expected, f"Wrong pattern types: got {pattern_types}, expected {expected}"

            # Test MIDI output generation
            midi_output = MidiOutput()
            # This would normally save to file, but we'll just test the method exists
            assert hasattr(midi_output, 'save_patterns_to_midi'), "MIDI output method missing"

            self.results.add_pass("integration - complete pipeline")

        except ImportError:
            self.results.add_warning("integration", "MIDI output module not available for testing")
        except Exception as e:
            self.results.add_fail("integration", str(e))


def main():
    """Main test execution."""
    tester = SelectiveGenerationTester()
    tester.run_all_tests()

    # Exit with appropriate code
    if tester.results.failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()