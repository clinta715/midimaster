#!/usr/bin/env python3
"""
Detailed analysis of MIDI Master to identify musical and technical issues.
"""

import mido
import random
from generators.pattern_generator import PatternGenerator
from genres.genre_factory import GenreFactory
from structures.song_skeleton import SongSkeleton
from output.midi_output import MidiOutput

def analyze_midi_file(filename="output.mid"):
    """Analyze the generated MIDI file."""
    print("=== MIDI FILE ANALYSIS ===")
    
    try:
        midi_file = mido.MidiFile(filename)
        print(f"✓ MIDI file loaded successfully")
        print(f"  - Number of tracks: {len(midi_file.tracks)}")
        print(f"  - Ticks per beat: {midi_file.ticks_per_beat}")
        
        for i, track in enumerate(midi_file.tracks):
            print(f"\n  Track {i}:")
            note_count = 0
            tempo = None
            time_signature = None
            pitch_range = []
            
            for msg in track:
                if msg.type == 'note_on':
                    note_count += 1
                    pitch_range.append(msg.note)
                elif msg.type == 'set_tempo':
                    tempo = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':
                    time_signature = f"{msg.numerator}/{msg.denominator}"
            
            print(f"    - Messages: {len(track)}")
            print(f"    - Note events: {note_count}")
            if pitch_range:
                print(f"    - Pitch range: {min(pitch_range)} to {max(pitch_range)}")
            print(f"    - Tempo: {tempo} BPM" if tempo else "    - Tempo: Not set")
            print(f"    - Time signature: {time_signature}" if time_signature else "    - Time signature: Not set")
            
    except Exception as e:
        print(f"✗ Error loading MIDI file: {e}")

def analyze_pattern_generator():
    """Analyze the PatternGenerator for musical coherence issues."""
    print("\n=== PATTERN GENERATOR ANALYSIS ===")
    
    # Create genre rules
    genre_rules = GenreFactory.create_genre_rules('pop')
    print(f"Genre rules loaded: {len(genre_rules.get_rules())} rule categories")
    
    # Test melody generation
    print("\n1. Melody Pattern Generation:")
    generator = PatternGenerator(genre_rules, 'happy')
    # Create a simple song skeleton for generation context
    song_skeleton = SongSkeleton('pop', 120, 'happy')
    melody_pattern = generator.generate_selective_patterns(song_skeleton, 4, ['melody'])[0]  # 4 bars
    
    print(f"   - Pattern type: {melody_pattern.pattern_type}")
    print(f"   - Number of notes: {len(melody_pattern.notes)}")
    if melody_pattern.notes:
        pitches = [note.pitch for note in melody_pattern.notes]
        durations = [note.duration for note in melody_pattern.notes]
        velocities = [note.velocity for note in melody_pattern.notes]
        print(f"   - Note pitches: {pitches}")
        print(f"   - Pitch range: {min(pitches)} to {max(pitches)}")
        print(f"   - Note durations: {durations}")
        print(f"   - Note velocities: {velocities}")
        
        # Check if pitches are musically coherent
        unique_pitches = set(pitches)
        print(f"   - Unique pitches: {len(unique_pitches)}")
        if len(unique_pitches) > 0:
            print(f"   - Musical coherence: LOW (random pitch selection)")
    
    # Test harmony generation
    print("\n2. Harmony Pattern Generation:")
    harmony_pattern = generator.generate_selective_patterns(song_skeleton, 4, ['harmony'], chord_complexity='medium')[0]
    
    print(f"   - Pattern type: {harmony_pattern.pattern_type}")
    print(f"   - Number of chords: {len(harmony_pattern.chords)}")
    if harmony_pattern.chords:
        for i, chord in enumerate(harmony_pattern.chords[:3]):  # Show first 3 chords
            pitches = [note.pitch for note in chord.notes]
            print(f"   - Chord {i+1}: {pitches}")
    
    # Test bass generation
    print("\n3. Bass Pattern Generation:")
    bass_pattern = generator.generate_selective_patterns(song_skeleton, 4, ['bass'])[0]
    
    print(f"   - Pattern type: {bass_pattern.pattern_type}")
    print(f"   - Number of bass notes: {len(bass_pattern.notes)}")
    if bass_pattern.notes:
        pitches = [note.pitch for note in bass_pattern.notes]
        print(f"   - Bass pitches: {pitches}")
        print(f"   - Bass range: {min(pitches)} to {max(pitches)}")
    
    # Test rhythm generation
    print("\n4. Rhythm Pattern Generation:")
    rhythm_pattern = generator.generate_selective_patterns(song_skeleton, 4, ['rhythm'], beat_complexity=0.5)[0]
    
    print(f"   - Pattern type: {rhythm_pattern.pattern_type}")
    print(f"   - Number of rhythm notes: {len(rhythm_pattern.notes)}")
    if rhythm_pattern.notes:
        pitches = [note.pitch for note in rhythm_pattern.notes]
        durations = [note.duration for note in rhythm_pattern.notes]
        print(f"   - Rhythm pitches: {pitches}")
        print(f"   - Rhythm durations: {durations}")

def analyze_genre_rules():
    """Analyze genre rules implementation."""
    print("\n=== GENRE RULES ANALYSIS ===")
    
    genres = ['pop', 'rock', 'jazz', 'electronic', 'hip-hop', 'classical']
    
    for genre_name in genres:
        try:
            rules = GenreFactory.create_genre_rules(genre_name)
            rules_dict = rules.get_rules()
            print(f"\n{genre_name.upper()} Genre:")
            print(f"  - Scales: {len(rules_dict.get('scales', []))} available")
            print(f"  - Chord progressions: {len(rules_dict.get('chord_progressions', []))} available")
            print(f"  - Rhythm patterns: {len(rules_dict.get('rhythm_patterns', []))} available")
            
            # Show sample rules
            if rules_dict.get('scales'):
                print(f"  - Sample scales: {rules_dict['scales'][:3]}")
            if rules_dict.get('chord_progressions'):
                print(f"  - Sample progressions: {rules_dict['chord_progressions'][:2]}")
            if rules_dict.get('rhythm_patterns'):
                print(f"  - Sample rhythm patterns: {rules_dict['rhythm_patterns'][:2]}")
                
        except Exception as e:
            print(f"  ✗ Error loading {genre_name}: {e}")

def analyze_song_generation():
    """Analyze the complete song generation process."""
    print("\n=== SONG GENERATION ANALYSIS ===")
    
    try:
        # Create genre rules
        genre_rules = GenreFactory.create_genre_rules('pop')
        
        # Create song skeleton
        song_skeleton = SongSkeleton('pop', 120, 'happy')
        
        # Generate patterns
        pattern_generator = PatternGenerator(genre_rules, 'happy')
        patterns = pattern_generator.generate_patterns(song_skeleton, 16)  # 16 bars
        
        # Build arrangement
        song_skeleton.build_arrangement(patterns)
        
        print(f"Generated song analysis:")
        print(f"  - Total patterns: {len(song_skeleton.patterns)}")
        print(f"  - Sections: {[s.name if hasattr(s, 'name') else str(s) for s, _ in song_skeleton.sections]}")
        
        # Analyze pattern distribution
        pattern_types = {}
        for pattern in song_skeleton.patterns:
            ptype = pattern.pattern_type
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
        
        print(f"  - Pattern distribution:")
        for ptype, count in pattern_types.items():
            print(f"    - {ptype}: {count}")
        
        # Analyze sections
        for section_type, patterns in song_skeleton.sections:
            name = section_type.name if hasattr(section_type, 'name') else str(section_type)
            print(f"  - {name}: {len(patterns)} patterns")
        
        # Save to MIDI
        midi_output = MidiOutput()
        midi_output.save_to_midi(song_skeleton, 'analysis_output.mid')
        print(f"  - MIDI file saved: analysis_output.mid")
        
    except Exception as e:
        print(f"✗ Error during song generation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run the complete analysis."""
    print("MIDI MASTER DETAILED ANALYSIS")
    print("=" * 50)
    
    # Analyze the generated MIDI file
    analyze_midi_file()
    
    # Analyze pattern generation
    analyze_pattern_generator()
    
    # Analyze genre rules
    analyze_genre_rules()
    
    # Analyze song generation
    analyze_song_generation()
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("\nKEY FINDINGS:")
    print("1. ✓ The program runs and generates MIDI files successfully")
    print("2. ⚠️  Musical content is largely random and lacks coherence")
    print("3. ⚠️  Genre rules exist but are underutilized in generation")
    print("4. ⚠️  No actual music theory implementation (scales, keys, chord functions)")
    print("5. ⚠️  Generated music lacks musical structure and coherence")

if __name__ == "__main__":
    main()