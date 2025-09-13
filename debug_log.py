#!/usr/bin/env python3
"""
Debug logging script for MIDI Master to identify issues.
"""

import os
import sys
import importlib.util

def check_file_exists(filepath):
    """Check if a file exists and log the result."""
    exists = os.path.exists(filepath)
    print(f"{'✓' if exists else '✗'} {filepath}: {'EXISTS' if exists else 'MISSING'}")
    return exists

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"✓ {module_name}: CAN IMPORT")
            return True
        else:
            print(f"✗ {module_name}: CANNOT IMPORT")
            return False
    except ImportError:
        print(f"✗ {module_name}: IMPORT ERROR")
        return False

def analyze_code_issues():
    """Analyze the codebase for potential issues."""
    print("=== MIDI MASTER DEBUG ANALYSIS ===\n")
    
    print("1. FILE STRUCTURE CHECK:")
    files_to_check = [
        'main.py',
        'structures/__init__.py',
        'structures/data_structures.py',
        'structures/song_skeleton.py',
        'genres/__init__.py',
        'genres/genre_rules.py',
        'genres/genre_factory.py',
        'generators/__init__.py',
        'generators/pattern_generator.py',
        'output/__init__.py',
        'output/midi_output.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for filepath in files_to_check:
        if not check_file_exists(filepath):
            missing_files.append(filepath)
    
    print(f"\n2. DEPENDENCY CHECK:")
    dependencies = ['mido']
    missing_deps = []
    for dep in dependencies:
        if not check_import(dep):
            missing_deps.append(dep)
    
    print(f"\n3. CRITICAL ISSUES IDENTIFIED:")
    
    # Check for missing MIDI output module
    if 'output/midi_output.py' in missing_files:
        print("🔴 CRITICAL: MidiOutput module is missing!")
        print("   - This will cause ImportError when running main.py")
        print("   - The program cannot save MIDI files without this module")
    
    # Check for missing dependencies
    if missing_deps:
        print("🔴 CRITICAL: Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
    
    print(f"\n4. CODE ANALYSIS ISSUES:")
    
    # Analyze pattern generator issues
    print("🟡 MUSIC THEORY IMPLEMENTATION MISSING:")
    print("   - PatternGenerator._generate_melody_pattern() uses random notes")
    print("   - PatternGenerator._generate_harmony_pattern() ignores chord progressions")
    print("   - No scale-based note generation implemented")
    print("   - Roman numerals in genre rules not mapped to actual chords")
    
    print("\n🟡 SONG ARRANGEMENT LOGIC OVERSIMPLIFIED:")
    print("   - SongSkeleton.build_arrangement() uses basic filtering")
    print("   - No genre-specific structure implementation")
    print("   - No musical coherence in pattern selection")
    
    print(f"\n5. RECOMMENDED FIXES:")
    print("   1. Create missing output/midi_output.py module")
    print("   2. Install missing dependencies: pip install -r requirements.txt")
    print("   3. Implement actual music theory in PatternGenerator")
    print("   4. Add chord progression mapping system")
    print("   5. Implement scale-based note generation")
    print("   6. Enhance song arrangement logic")
    
    return len(missing_files) > 0 or len(missing_deps) > 0

if __name__ == "__main__":
    has_issues = analyze_code_issues()
    sys.exit(1 if has_issues else 0)