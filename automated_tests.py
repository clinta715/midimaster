#!/usr/bin/env python3
"""
Automated Test Suite for MIDI Master

This module provides comprehensive testing of the MIDI Master system by generating
music with various parameter combinations and analyzing the results. It helps
validate the system's behavior across different genres, moods, densities, and
tempo settings.

Usage:
    python automated_tests.py

Output:
    - Individual MIDI files for each test combination
    - Analysis reports for each generated file
    - Summary comparison report
    - Statistical insights about parameter effects
"""

import subprocess
import os
import sys
import json
import csv
import time
from pathlib import Path
from collections import defaultdict
import argparse

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detailed_analysis import analyze_midi_file
import mido

# Test parameter combinations
TEST_PARAMETERS = {
    'genres': ['pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical'],
    'moods': ['happy', 'sad', 'energetic', 'calm'],
    'densities': ['minimal', 'sparse', 'balanced', 'dense', 'complex'],
    'tempos': [80, 100, 120, 140, 160],
    'bars': [8, 16, 32],
    'repetitions': 3  # How many times to repeat each test for consistency
}

class MidiTestSuite:
    """Comprehensive test suite for MIDI Master."""

    def __init__(self, output_dir="test_outputs", summary_file="test_results.json"):
        self.output_dir = Path(output_dir)
        self.summary_file = Path(summary_file)
        self.test_results = {}
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.midi_dir = self.output_dir / "midi_files"
        self.analysis_dir = self.output_dir / "analysis_results"
        self.midi_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)

    def generate_test_cases(self):
        """Generate all test parameter combinations."""
        test_cases = []

        # For performance, use a subset for extensive testing
        if '--lite' in sys.argv:
            genres = ['pop', 'rock', 'jazz']
            moods = ['happy', 'energetic']
            densities = ['sparse', 'balanced', 'dense']
            tempos = [80, 120, 160]
            bars = [16]
            repetitions = 1
        else:
            genres = TEST_PARAMETERS['genres']
            moods = TEST_PARAMETERS['moods']
            densities = TEST_PARAMETERS['densities']
            tempos = TEST_PARAMETERS['tempos']
            bars = TEST_PARAMETERS['bars']
            repetitions = TEST_PARAMETERS['repetitions']

        for i in range(repetitions):
            for genre in genres:
                for mood in moods:
                    for density in densities:
                        for tempo in tempos:
                            for bar_count in bars:
                                test_id = f"{genre}_{mood}_{density}_tempo{tempo}_bars{bar_count}_run{i+1}"
                                test_cases.append({
                                    'id': test_id,
                                    'genre': genre,
                                    'mood': mood,
                                    'density': density,
                                    'tempo': tempo,
                                    'bars': bar_count,
                                    'run': i + 1
                                })

        return test_cases

    def run_test_case(self, test_case):
        """Run a single test case."""
        test_id = test_case['id']
        output_file = self.midi_dir / f"{test_id}.mid"
        analysis_file = self.analysis_dir / f"{test_id}_analysis.json"

        print(f"Running test: {test_id}")

        try:
            # Build command for main.py
            cmd = [
                sys.executable, 'main.py',
                '--genre', test_case['genre'],
                '--mood', test_case['mood'],
                '--density', test_case['density'],
                '--tempo', str(test_case['tempo']),
                '--bars', str(test_case['bars']),
                '--output', str(output_file)
            ]

            # Run the generation
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            generation_time = time.time() - start_time

            if result.returncode == 0:
                status = 'success'

                # Analyze the generated MIDI file
                midi_analysis = self.analyze_generated_midi(output_file, test_id)

                # Store results
                test_result = {
                    'test_case': test_case,
                    'status': status,
                    'generation_time': round(generation_time, 2),
                    'stdout': result.stdout.strip(),
                    'stderr': result.stderr.strip(),
                    'analysis': midi_analysis,
                    'timestamp': time.time()
                }
            else:
                status = 'failed'
                test_result = {
                    'test_case': test_case,
                    'status': status,
                    'generation_time': round(generation_time, 2),
                    'stdout': result.stdout.strip(),
                    'stderr': result.stderr.strip(),
                    'error': result.stderr.strip(),
                    'timestamp': time.time()
                }

        except subprocess.TimeoutExpired:
            status = 'timeout'
            test_result = {
                'test_case': test_case,
                'status': status,
                'error': 'Test timed out after 60 seconds',
                'timestamp': time.time()
            }
        except Exception as e:
            status = 'error'
            test_result = {
                'test_case': test_case,
                'status': status,
                'error': str(e),
                'timestamp': time.time()
            }

        self.test_results[test_id] = test_result

        if status == 'success':
            color = '\033[92m'  # Green
        elif status == 'failed':
            color = '\033[91m'  # Red
        else:
            color = '\033[93m'  # Yellow

        print(f"{color}Result: {status}\033[0m ({round(generation_time, 2)}s)")
        print()

        return test_result

    def analyze_generated_midi(self, midi_file, test_id):
        """Analyze a generated MIDI file."""
        analysis = {}

        try:
            midi_file_obj = mido.MidiFile(str(midi_file))
            analysis['format'] = midi_file_obj.type
            analysis['ticks_per_beat'] = midi_file_obj.ticks_per_beat
            analysis['track_count'] = len(midi_file_obj.tracks)

            # Analyze all tracks
            total_notes = 0
            pitch_range = []
            note_counts = defaultdict(int)
            duration_stats = {'count': 0, 'sum': 0, 'min': float('inf'), 'max': 0}
            velocity_stats = {'count': 0, 'sum': 0, 'min': float('inf'), 'max': 0}

            for track in midi_file_obj.tracks:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        total_notes += 1
                        pitch = msg.note
                        # Note: We'll use velocity instead of duration for now
                        velocity = msg.velocity
                        pitch_range.append(pitch)
                        note_counts[pitch] += 1
                        velocity_stats['count'] += 1
                        velocity_stats['sum'] += velocity
                        if velocity < velocity_stats['min']:
                            velocity_stats['min'] = velocity
                        if velocity > velocity_stats['max']:
                            velocity_stats['max'] = velocity

            analysis['total_notes'] = total_notes
            if pitch_range:
                analysis['pitch_range'] = {
                    'min': min(pitch_range),
                    'max': max(pitch_range),
                    'span': max(pitch_range) - min(pitch_range)
                }
                analysis['unique_pitches'] = len(set(pitch_range))
                analysis['pitch_distribution'] = dict(note_counts)

            if velocity_stats['count'] > 0:
                analysis['average_velocity'] = velocity_stats['sum'] / velocity_stats['count']
                analysis['velocity_range'] = {
                    'min': velocity_stats['min'],
                    'max': velocity_stats['max']
                }

            # Calculate note density (notes per bar)
            # Assume 4 beats per bar, 16 bars default
            bars_generated = 16  # This could be made dynamic
            analysis['note_density'] = total_notes / bars_generated if bars_generated > 0 else 0

        except Exception as e:
            analysis['error'] = f"Analysis failed: {str(e)}"

        return analysis

    def save_results_to_csv(self):
        """Save test results to CSV for analysis."""
        csv_file = self.output_dir / "test_results.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'test_id', 'genre', 'mood', 'density', 'tempo', 'bars', 'run',
                'status', 'generation_time', 'total_notes', 'unique_pitches',
                'pitch_span', 'average_velocity', 'note_density'
            ])
            writer.writeheader()

            for test_id, result in self.test_results.items():
                row = {
                    'test_id': test_id,
                    'genre': result['test_case']['genre'],
                    'mood': result['test_case']['mood'],
                    'density': result['test_case']['density'],
                    'tempo': result['test_case']['tempo'],
                    'bars': result['test_case']['bars'],
                    'run': result['test_case']['run'],
                    'status': result['status'],
                    'generation_time': result['generation_time'],
                    'total_notes': result.get('analysis', {}).get('total_notes', 0),
                    'unique_pitches': result.get('analysis', {}).get('unique_pitches', 0),
                    'pitch_span': result.get('analysis', {}).get('pitch_range', {}).get('span', 0),
                    'average_velocity': result.get('analysis', {}).get('average_velocity', 0),
                    'note_density': result.get('analysis', {}).get('note_density', 0)
                }
                writer.writerow(row)

        print(f"Results saved to: {csv_file}")

    def generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        report = self.output_dir / "comparison_report.md"

        with open(report, 'w') as f:
            f.write("# MIDI Master Test Results Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall statistics
            successful_tests = [r for r in self.test_results.values() if r['status'] == 'success']
            failed_tests = [r for r in self.test_results.values() if r['status'] != 'success']

            f.write("## Overall Statistics\n\n")
            f.write(f"- **Total Tests**: {len(self.test_results)}\n")
            f.write(f"- **Successful**: {len(successful_tests)}\n")
            f.write(f"- **Failed**: {len(failed_tests)}\n")
            f.write(".1f")
            f.write(f"- **Average Generation Time**: {sum(r.get('generation_time', 0) for r in successful_tests) / len(successful_tests):.2f}s\n\n")

            # Genre Analysis
            genre_stats = defaultdict(list)
            for result in successful_tests:
                genre = result['test_case']['genre']
                genre_stats[genre].append(result)

            f.write("## Genre Analysis\n\n")
            for genre, results in genre_stats.items():
                generation_times = [r['generation_time'] for r in results]
                note_counts = [r['analysis'].get('total_notes', 0) for r in results]
                f.write(f"### {genre.title()}\n")
                f.write(f"- Tests: {len(results)}\n")
                f.write(".2f")
                f.write(f"- Avg Notes: {sum(note_counts) / len(note_counts):.1f}\n")
                f.write(f"- Success Rate: {(len(results) / len([r for r in self.test_results.values() if r['test_case']['genre'] == genre])) * 100:.1f}%\n\n")

            # Density Analysis
            f.write("## Density Presets Analysis\n\n")
            density_stats = defaultdict(list)
            for result in successful_tests:
                density = result['test_case']['density']
                density_stats[density].append(result)

            for density, results in density_stats.items():
                note_counts = [r['analysis'].get('total_notes', 0) for r in results]
                f.write(f"### {density.title()} Density\n")
                f.write(f"- Tests: {len(results)}\n")
                f.write(f"- Avg Notes: {sum(note_counts) / len(note_counts):.1f}\n")
                if note_counts:
                    f.write(f"- Note Range: {min(note_counts)} - {max(note_counts)}\n\n")

            # Mood Analysis
            f.write("## Mood Analysis\n\n")
            mood_stats = defaultdict(list)
            for result in successful_tests:
                mood = result['test_case']['mood']
                mood_stats[mood].append(result)

            for mood, results in mood_stats.items():
                velocities = [r['analysis'].get('average_velocity', 0) for r in results]
                f.write(f"### {mood.title()} Mood\n")
                f.write(f"- Tests: {len(results)}\n")
                f.write(f"- Avg Velocity: {sum(velocities) / len(velocities):.1f}\n\n")

        print(f"Comparison report saved to: {report}")

    def run_complete_test_suite(self):
        """Run the complete test suite."""
        print("🧪 Starting MIDI Master Test Suite\n")
        print("=" * 50)

        test_cases = self.generate_test_cases()
        print(f"Total test cases: {len(test_cases)}\n")

        successful = 0
        total = len(test_cases)
        start_time = time.time()

        for i, test_case in enumerate(test_cases, 1):
            print(f"Progress: {i}/{total} ({i*100/total:.1f}%)")
            result = self.run_test_case(test_case)
            if result['status'] == 'success':
                successful += 1

        total_time = time.time() - start_time

        print("\n" + "=" * 50)
        print("🎯 Test Suite Complete!")
        print(f"📊 Results: {successful}/{total} successful")
        success_rate = (successful / total) * 100 if total > 0 else 0
        end_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"✅ Success Rate: {success_rate:.2f}%")
        print(f"⏱️  Total Duration: {total_time:.2f}s")
        print(f"📅 Performed: {end_time}")

        # Save results
        self.save_results_to_csv()
        self.generate_comparison_report()

        with open(self.summary_file, 'w') as f:
            json.dump(dict(self.test_results), f, indent=2)

        print(f"\n📁 All results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run MIDI Master automated tests")
    parser.add_argument("--lite", action="store_true", help="Run with fewer test cases for faster execution")
    args = parser.parse_args()

    # Create test suite and run
    test_suite = MidiTestSuite()
    test_suite.run_complete_test_suite()


if __name__ == "__main__":
    main()