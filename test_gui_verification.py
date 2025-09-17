#!/usr/bin/env python3
"""
GUI Verification Test Script
Tests GUI components programmatically without GUI interaction.
"""

import sys
import os
from pathlib import Path

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_parameter_controls():
    """Test parameter controls get_config"""
    try:
        from gui.parameter_controls import ParameterControls
        from PyQt6.QtWidgets import QApplication

        app = QApplication([])  # Need QApplication for Qt widgets
        controls = ParameterControls()
        config = controls.get_config()
        print("✓ Parameter controls get_config works")
        print(f"  Config keys: {list(config.keys())}")
        print(f"  Sample values: genre={config['genre']}, tempo={config['tempo']}, bars={config['bars']}")
        return True
    except Exception as e:
        print(f"✗ Parameter controls test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_worker():
    """Test generation worker with sample params"""
    try:
        from gui.generation_worker import GenerationWorker
        from PyQt6.QtWidgets import QApplication

        # Create minimal app if needed
        app = QApplication([]) if not QApplication.instance() else QApplication.instance()

        # Test params
        test_params = {
            'genre': 'pop',
            'mood': 'happy',
            'tempo': 120,
            'density': 'balanced',
            'bars': 4,  # Small for testing
            'harmonic_variance': 'medium',
            'separate_files': False,
            'key': 'C',
            'mode': 'major',
            'time_signature': '4/4',
            'output': 'output/test_gui_verify.mid'
        }

        print("Testing generation worker...")
        worker = GenerationWorker(test_params, preview_mode=False)

        # Note: In real GUI, worker would be started in thread
        # For testing, we can call run() directly but it would block
        # Instead, just test parameter validation
        if worker._validate_params():
            print("✓ Worker parameter validation passed")
            return True
        else:
            print("✗ Worker parameter validation failed")
            return False

    except Exception as e:
        print(f"✗ Generation worker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_midi_analysis():
    """Test MIDI analysis functionality"""
    try:
        from fixed_midi_analysis import analyze_midi_file

        print("Testing MIDI analysis...")
        if os.path.exists('test_gui.mid'):
            results = analyze_midi_file('test_gui.mid')
            print("✓ MIDI analysis works")
            print(f"  Analysis keys: {list(results.keys())}")
            return True
        else:
            print("✗ test_gui.mid not found")
            return False
    except Exception as e:
        print(f"✗ MIDI analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_ranges():
    """Test parameter validation ranges"""
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.main_window import MainWindow

        app = QApplication([]) if not QApplication.instance() else QApplication.instance()

        # Create main window
        window = MainWindow()

        # Test valid config
        valid_config = {
            'genre': 'pop',
            'mood': 'happy',
            'tempo': 120,
            'bars': 16,
            'density': 'balanced'
        }

        if window._validate_config(valid_config):
            print("✓ Valid config passes validation")
        else:
            print("✗ Valid config fails validation")
            return False

        # Test invalid tempo
        invalid_config = valid_config.copy()
        invalid_config['tempo'] = 50  # Too low

        if not window._validate_config(invalid_config):
            print("✓ Invalid tempo correctly rejected")
        else:
            print("✗ Invalid tempo incorrectly accepted")
            return False

        # Test invalid bars
        invalid_config2 = valid_config.copy()
        invalid_config2['bars'] = 100  # Too high

        if not window._validate_config(invalid_config2):
            print("✓ Invalid bars correctly rejected")
        else:
            print("✗ Invalid bars incorrectly accepted")
            return False

        return True

    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests"""
    print("="*60)
    print("GUI VERIFICATION TESTS")
    print("="*60)

    tests = [
        ("Parameter Controls", test_parameter_controls),
        ("Generation Worker", test_generation_worker),
        ("MIDI Analysis", test_midi_analysis),
        ("Validation Ranges", test_validation_ranges),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n--- Testing {name} ---")
        if test_func():
            passed += 1
        else:
            print(f"FAILED: {name}")

    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("✓ All GUI components verified successfully!")
    else:
        print(f"✗ {total - passed} tests failed")
    print("="*60)

if __name__ == "__main__":
    main()