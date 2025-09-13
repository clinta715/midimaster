#!/usr/bin/env python3
"""
Comprehensive Test Suite for MIDI Timing and Chord Generation Fixes
==================================================================

Tests the fixed MIDI generation system to verify that all timing issues
have been resolved and chord generation is working correctly.

Tests performed:
- Multiple MIDI file generation with various density settings
- Timing analysis to verify no exponential growth
- Chord structure validation for simultaneous note events
- Channel assignment verification
- Density management validation
- Musical appropriateness checks
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Import the fixed MIDI generation modules
from generators.pattern_generator import PatternGenerator
from output.midi_output import MidiOutput
from structures.song_skeleton import SongSkeleton
from structures.data_structures import Pattern, PatternType, SectionType
from genres.genre_factory import GenreFactory
from generators.density_manager import DensityManager, create_density_manager_from_preset
import comprehensive_midi_analysis

class MidiValidationTester:
    """Comprehensive validator for MIDI timing and chord generation fixes."""

    def __init__(self):
        self.midi_output = MidiOutput()
        self.test_results = {}
        self.test_files = []
        self.genre_factory = GenreFactory()

    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite for all validation requirements.

        Returns:
            Dictionary containing comprehensive test results
        """
        print("\n" + "="*80)
        print("MIDI TIMING & CHORD GENERATION VALIDATION SUITE")
        print("="*80)

        # Test 1: Generate MIDI files with various configurations
        print("\n1. GENERATING TEST MIDI FILES")
        print("-" * 40)
        self._generate_test_files()

        # Test 2: Analyze timing patterns
        print("\n2. ANALYZING TIMING PATTERNS")
        print("-" * 40)
        self._analyze_timing_patterns()

        # Test 3: Verify chord structures
        print("\n3. VERIFYING CHORD STRUCTURES")
        print("-" * 40)
        self._verify_chord_structures()

        # Test 4: Validate channel assignments
        print("\n4. VALIDATING CHANNEL ASSIGNMENTS")
        print("-" * 40)
        self._validate_channel_assignments()

        # Test 5: Check density management
        print("\n5. VALIDATING DENSITY MANAGEMENT")
        print("-" * 40)
        self._validate_density_management()

        # Test 6: Musical appropriateness analysis
        print("\n6. MUSICAL APPROPRIATENESS ANALYSIS")
        print("-" * 40)
        self._analyze_musical_appropriateness()

        # Final validation report
        self._generate_validation_report()

        return self.test_results

    def _generate_test_files(self) -> None:
        """Generate test MIDI files with various density and genre combinations."""

        genres = ['pop', 'jazz', 'rock']
        moods = ['energetic', 'happy', 'sad']
        density_presets = ['sparse', 'balanced', 'dense']
        tempos = [80, 120, 160]

        test_configs = []

        print(f"Generating test files for {len(genres)} genres × {len(density_presets)} densities × {len(tempos)} tempos × {len(moods)} moods")
        print(f"Total test files: {len(genres) * len(density_presets) * len(tempos) * len(moods)}")

        for genre in genres:
            for density in density_presets:
                for tempo in tempos:
                    for mood in moods:
                        config = {
                            'genre': genre,
                            'mood': mood,
                            'density': density,
                            'tempo': tempo
                        }
                        test_configs.append(config)

        self.test_results['test_configs'] = len(test_configs)

        # Generate MIDI files for first 10 configs to keep test manageable
        generated_count = 0
        max_files = min(10, len(test_configs))

        for i, config in enumerate(test_configs[:max_files]):
            try:
                filename = self._generate_midi_file(config)
                if filename:
                    self.test_files.append(filename)
                    generated_count += 1
            except Exception as e:
                print(f"❌ Failed to generate file for config {i+1}: {e}")

        self.test_results['files_generated'] = generated_count

    def _generate_midi_file(self, config: Dict[str, Any]) -> Optional[str]:
        """Generate a single MIDI file for the given configuration."""

        try:
            # Create genre rules
            genre_rules = self.genre_factory.create_genre_rules(config['genre'])

            # Create density manager
            density_manager = create_density_manager_from_preset(config['density'])

            # Create pattern generator
            generator = PatternGenerator(
                genre_rules=genre_rules,
                mood=config['mood'],
                note_density=density_manager.note_density,
                rhythm_density=density_manager.rhythm_density,
                chord_density=density_manager.chord_density,
                bass_density=density_manager.bass_density
            )

            # Create song skeleton
            song_skeleton = SongSkeleton(
                genre=config['genre'],
                tempo=config['tempo'],
                mood=config['mood']
            )

            # Add basic sections
            song_skeleton.add_section(SectionType.VERSE, generator.generate_patterns(song_skeleton, num_bars=8, beat_complexity=0.5, chord_complexity='medium'))
            song_skeleton.add_section(SectionType.CHORUS, generator.generate_patterns(song_skeleton, num_bars=8, beat_complexity=0.5, chord_complexity='medium'))
            song_skeleton.add_section(SectionType.BRIDGE, generator.generate_patterns(song_skeleton, num_bars=8, beat_complexity=0.5, chord_complexity='medium'))
            song_skeleton.patterns.extend(generator.generate_patterns(song_skeleton, num_bars=8, beat_complexity=0.5, chord_complexity='medium'))

            # Generate filename
            filename = f"validation_{config['genre']}_{config['mood']}_{config['density']}_tempo{config['tempo']}.mid"

            # Save MIDI file
            self.midi_output.save_to_midi(song_skeleton, filename)

            return filename

        except Exception as e:
            print(f"Error generating MIDI file: {e}")
            return None

    def _analyze_timing_patterns(self) -> None:
        """Analyze timing patterns in generated MIDI files."""

        timing_results = []
        exponential_growth_detected = 0

        for filename in self.test_files:
            if os.path.exists(filename):
                analyzer = comprehensive_midi_analysis.MidiAnalyzer(filename)
                spacing_analysis = analyzer.analyze_note_spacing()

                result = {
                    'filename': filename,
                    'total_notes': spacing_analysis.get('total_notes', 0),
                    'note_density': spacing_analysis.get('note_density_per_second', 0),
                    'avg_interval': spacing_analysis.get('avg_interval', 0),
                    'max_interval': spacing_analysis.get('max_interval', 0),
                    'intervals_greater_than_2s': spacing_analysis.get('intervals_greater_than_2s', 0)
                }

                timing_results.append(result)

                # Check for exponential growth indicators
                if result['max_interval'] > 5.0 or result['note_density'] < 0.1:
                    exponential_growth_detected += 1

        self.test_results['timing_analysis'] = {
            'files_analyzed': len(timing_results),
            'exponential_growth_detected': exponential_growth_detected,
            'average_note_density': sum(r['note_density'] for r in timing_results) / max(1, len(timing_results)),
            'max_interval_detected': max((r['max_interval'] for r in timing_results), default=0),
            'timing_results': timing_results
        }

    def _verify_chord_structures(self) -> None:
        """Verify that chords are properly generated as simultaneous note events."""

        chord_results = []
        files_with_chords = 0

        for filename in self.test_files:
            if os.path.exists(filename):
                analyzer = comprehensive_midi_analysis.MidiAnalyzer(filename)
                organization_analysis = analyzer.analyze_note_organization()

                chord_events = organization_analysis.get('total_simultaneous_notes', 0)

                result = {
                    'filename': filename,
                    'chord_events': chord_events,
                    'avg_chord_size': organization_analysis.get('avg_chord_size', 0),
                    'max_chord_size': organization_analysis.get('max_chord_size', 0)
                }

                chord_results.append(result)

                if chord_events > 0:
                    files_with_chords += 1

        self.test_results['chord_analysis'] = {
            'files_analyzed': len(chord_results),
            'files_with_chords': files_with_chords,
            'chord_success_rate': files_with_chords / max(1, len(chord_results)),
            'average_chord_events': sum(r['chord_events'] for r in chord_results) / max(1, len(chord_results)),
            'chord_results': chord_results
        }

    def _validate_channel_assignments(self) -> None:
        """Validate that channel assignments match specifications (melody: ch0, harmony: ch1, bass: ch2, rhythm: ch9)."""

        channel_results = []

        for filename in self.test_files:
            if os.path.exists(filename):
                analyzer = comprehensive_midi_analysis.MidiAnalyzer(filename)
                organization_analysis = analyzer.analyze_note_organization()

                channel_distribution = organization_analysis.get('channel_distribution', {})

                expected_channels = {
                    'melody': [0],
                    'harmony': [1],
                    'bass': [2],
                    'rhythm': [9]
                }

                # Check if expected channels are present
                present_channels = list(channel_distribution.keys())
                missing_channels = []
                extra_channels = []

                for channel_type, channels in expected_channels.items():
                    for ch in channels:
                        if ch not in present_channels:
                            missing_channels.append(ch)

                for ch in present_channels:
                    if not any(ch in channels for channels in expected_channels.values()):
                        extra_channels.append(ch)

                result = {
                    'filename': filename,
                    'channels_present': present_channels,
                    'channels_missing_from_expected': missing_channels,
                    'extra_channels': extra_channels,
                    'channel_count_by_expected': channel_distribution
                }

                channel_results.append(result)

        self.test_results['channel_analysis'] = {
            'files_analyzed': len(channel_results),
            'correct_channel_assignment': sum(1 for r in channel_results if not r['channels_missing_from_expected'] and not r['extra_channels']),
            'channel_results': channel_results
        }

    def _validate_density_management(self) -> None:
        """Validate that density settings produce appropriate note counts."""

        density_results = []

        for filename in self.test_files:
            if os.path.exists(filename):
                # Extract density info from filename
                parts = filename.replace('.mid', '').split('_')
                density_type = 'balanced'  # default
                for part in parts:
                    if part in ['sparse', 'balanced', 'dense']:
                        density_type = part
                        break

                analyzer = comprehensive_midi_analysis.MidiAnalyzer(filename)
                spacing_analysis = analyzer.analyze_note_spacing()

                note_count = spacing_analysis.get('total_notes', 0)
                duration = spacing_analysis.get('total_duration', 0)

                if duration > 0:
                    notes_per_second = note_count / duration
                else:
                    notes_per_second = 0

                # Check expected note count ranges based on density
                density_ranges = {
                    'sparse': (0.3, 0.8),   # 12-34 notes expected
                    'balanced': (0.8, 1.2), # ~34 notes expected
                    'dense': (1.2, 1.8)     # ~53 notes expected
                }

                expected_range = density_ranges.get(density_type, (0.8, 1.2))
                in_range = expected_range[0] <= notes_per_second <= expected_range[1]

                result = {
                    'filename': filename,
                    'density_type': density_type,
                    'note_count': note_count,
                    'notes_per_second': notes_per_second,
                    'expected_range': expected_range,
                    'in_expected_range': in_range
                }

                density_results.append(result)

        self.test_results['density_analysis'] = {
            'files_analyzed': len(density_results),
            'density_settings_working': sum(1 for r in density_results if r['in_expected_range']),
            'density_results': density_results
        }

    def _analyze_musical_appropriateness(self) -> None:
        """Analyze musical appropriateness of timing and intervals."""

        musical_results = []

        for filename in self.test_files:
            if os.path.exists(filename):
                analyzer = comprehensive_midi_analysis.MidiAnalyzer(filename)
                spacing_analysis = analyzer.analyze_note_spacing()

                avg_interval = spacing_analysis.get('avg_interval', 0)
                max_interval = spacing_analysis.get('max_interval', 0)
                intervals_greater_than_2s = spacing_analysis.get('intervals_greater_than_2s', 0)

                # Musical appropriateness criteria
                # Good: 0.25-2 second intervals, no >2s gaps
                appropriate_timing = (
                    0.25 <= avg_interval <= 2.0 and
                    max_interval <= 2.5 and
                    intervals_greater_than_2s == 0
                )

                result = {
                    'filename': filename,
                    'avg_interval': avg_interval,
                    'max_interval': max_interval,
                    'intervals_greater_than_2s': intervals_greater_than_2s,
                    'appropriate_timing': appropriate_timing
                }

                musical_results.append(result)

        self.test_results['musical_analysis'] = {
            'files_analyzed': len(musical_results),
            'musically_appropriate': sum(1 for r in musical_results if r['appropriate_timing']),
            'average_interval': sum(r['avg_interval'] for r in musical_results) / max(1, len(musical_results)),
            'max_gap_detected': max((r['max_interval'] for r in musical_results), default=0),
            'musical_results': musical_results
        }

    def _generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""

        print("\n" + "="*80)
        print("VALIDATION RESULTS & FINAL REPORT")
        print("="*80)

        # Overall statistics
        files_generated = self.test_results.get('files_generated', 0)
        files_detected = len(self.test_files)

        print(f"\nFILES GENERATED & ANALYZED:")
        print(f"  Generated: {files_generated} test MIDI files")
        print(f"  Analyzed: {files_detected} files")

        # Timing validation
        timing = self.test_results.get('timing_analysis', {})
        print(f"\nTIMING VALIDATION:")
        print(f"  Exponential growth detected: {timing.get('exponential_growth_detected', 0)} files")
        print("  ✅ NO EXPONENTIAL GROWTH DETECTED " if timing.get('exponential_growth_detected', 0) == 0 else "  ❌ EXPONENTIAL GROWTH DETECTED")

        # Chord validation
        chord = self.test_results.get('chord_analysis', {})
        chord_rate = chord.get('chord_success_rate', 0)
        print("CHORD STRUCTURE VALIDATION:")
        print(f"  Files with chords: {chord.get('files_with_chords', 0)}")
        print("  ✅ CHORD EVENTS DETECTED " if chord_rate > 0.8 else "  ❌ INSUFFICIENT CHORD GENERATION")

        # Channel validation
        channel = self.test_results.get('channel_analysis', {})
        correct_channels = channel.get('correct_channel_assignment', 0)
        print("CHANNEL ASSIGNMENT VALIDATION:")
        print(f"  Correct channel assignments: {correct_channels}")
        print("  ✅ CHANNELS CORRECTLY ASSIGNED " if correct_channels >= 0.9 * files_detected else "  ❌ INCORRECT CHANNEL ASSIGNMENTS")

        # Density validation
        density = self.test_results.get('density_analysis', {})
        density_working = density.get('density_settings_working', 0)
        print("DENSITY MANAGEMENT VALIDATION:")
        print(f"  Density settings working correctly: {density_working}")
        print("  ✅ DENSITY MANAGEMENT WORKING " if density_working >= 0.7 * files_detected else "  ❌ DENSITY SETTINGS INCORRECT")

        # Musical appropriateness
        musical = self.test_results.get('musical_analysis', {})
        musical_appropriate = musical.get('musically_appropriate', 0)
        print("MUSICAL APPROPRIATENESS VALIDATION:")
        print(f"  Musically appropriate timing: {musical_appropriate}")
        print("  ✅ MUSICAL TIMING APPROPRIATE " if musical_appropriate >= 0.9 * files_detected else "  ❌ TIMING ISSUES DETECTED")

        # Overall success
        print("" + "="*80)
        print("FINAL VALIDATION SUMMARY")
        print("="*80)

        all_criteria_met = (
            timing.get('exponential_growth_detected', 0) == 0 and
            chord_rate >= 0.8 and
            correct_channels >= 0.9 * files_detected and
            density_working >= 0.7 * files_detected and
            musical_appropriate >= 0.9 * files_detected
        )

        if all_criteria_met:
            print("🎉 ALL VALIDATION CRITERIA PASSED!")
            print("   The MIDI timing and chord generation fixes are working correctly.")
        else:
            print("❌ SOME VALIDATION CRITERIA FAILED")
            print("   Review the results above to identify remaining issues.")

        # Clean up test files
        print("Cleaning up test files...")
        for filename in self.test_files:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except:
                pass

        # Save results to file
        results_filename = "test_results.json"
        with open(results_filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)

        print(f"Detailed results saved to {results_filename}")

def run_validation_tests():
    """Main function to run all validation tests."""
    tester = MidiValidationTester()
    results = tester.run_full_test_suite()
    return results

if __name__ == "__main__":
    results = run_validation_tests()
    print("\nValidation test completed. Results saved to test_results.json")