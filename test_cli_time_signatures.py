#!/usr/bin/env python3
"""
Test script for command-line per-track time signature functionality.

This script tests the CLI interface for per-track time signatures by:
1. Running the main.py script with various time signature options
2. Verifying that the generated separate MIDI files have correct time signatures
3. Testing error handling for invalid time signature formats
"""

import sys
import os
import subprocess
import tempfile
import mido


def run_cli_command(args):
    """Run main.py with given arguments and return the result."""
    cmd = [sys.executable, 'main.py'] + args
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                          capture_output=True, text=True)
    return result


def test_cli_time_signatures():
    """Test CLI time signature functionality."""
    print("Testing CLI per-track time signatures...")

    # Test 1: Basic functionality with custom time signatures
    with tempfile.TemporaryDirectory() as temp_dir:
        output_base = os.path.join(temp_dir, "test_cli")

        args = [
            '--genre', 'pop',
            '--tempo', '120',
            '--mood', 'happy',
            '--bars', '4',
            '--separate-files',
            '--melody-time-signature', '4/4',
            '--harmony-time-signature', '3/4',
            '--bass-time-signature', '6/8',
            '--rhythm-time-signature', '4/4',
            '--output', f'{output_base}.mid'
        ]

        result = run_cli_command(args)

        if result.returncode != 0:
            print(f"❌ CLI command failed: {result.stderr}")
            return False

        print("✅ CLI command executed successfully")

        # Check that separate files were created with correct time signatures
        expected_files = {
            f'{output_base}_melody.mid': (4, 4),
            f'{output_base}_harmony.mid': (3, 4),
            f'{output_base}_bass.mid': (6, 8),
            f'{output_base}_rhythm.mid': (4, 4)
        }

        all_correct = True
        for filename, (expected_num, expected_den) in expected_files.items():
            if os.path.exists(filename):
                midi_file = mido.MidiFile(filename)
                time_sig_found = False
                for track in midi_file.tracks:
                    for msg in track:
                        if msg.type == 'time_signature':
                            if msg.numerator == expected_num and msg.denominator == expected_den:
                                print(f"✅ {os.path.basename(filename)}: Correct time signature {msg.numerator}/{msg.denominator}")
                                time_sig_found = True
                            else:
                                print(f"❌ {os.path.basename(filename)}: Wrong time signature {msg.numerator}/{msg.denominator}, expected {expected_num}/{expected_den}")
                                all_correct = False
                            break
                    if time_sig_found:
                        break
                if not time_sig_found:
                    print(f"❌ {os.path.basename(filename)}: No time signature found")
                    all_correct = False
            else:
                print(f"❌ {os.path.basename(filename)}: File not found")
                all_correct = False

    # Test 2: Invalid time signature format
    args_invalid = [
        '--genre', 'pop',
        '--tempo', '120',
        '--mood', 'happy',
        '--bars', '4',
        '--separate-files',
        '--melody-time-signature', 'invalid',
        '--output', 'test_invalid.mid'
    ]

    result_invalid = run_cli_command(args_invalid)

    # Should still succeed but use default time signatures
    if result_invalid.returncode != 0:
        print(f"❌ CLI command with invalid time signature failed: {result_invalid.stderr}")
        return False

    print("✅ CLI handles invalid time signature gracefully")

    if all_correct:
        print("\n🎉 All CLI time signature tests passed!")
        return True
    else:
        print("\n❌ Some CLI time signature tests failed.")
        return False


if __name__ == "__main__":
    success = test_cli_time_signatures()
    sys.exit(0 if success else 1)