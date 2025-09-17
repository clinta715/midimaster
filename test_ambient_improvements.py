#!/usr/bin/env python3
"""
Test script for ambient improvements validation.
Tests MIDI classification on midi5 files and generates sample ambient compositions.
"""

import os
import glob
from ml_insights.genre_classifier import GenreClassifier
from generators.generator_context import GeneratorContext
from generators.ambient_rhythm_engine import AmbientRhythmEngine
from generators.atmosphere_generator import AtmosphereGenerator
from generators.ambient_patterns import AmbientPatternTemplates
from generators.dynamic_rhythm_adaptor import DynamicRhythmAdaptor, AdaptationMetrics
import mido
from music_theory import MusicTheory
from structures.data_structures import Note
from genres.genre_factory import GenreFactory

# Directory with midi5 files
MIDI5_DIR = "reference_midis/midi5"

def test_classification():
    """Test enhanced classification on midi5 files."""
    print("=== Testing Enhanced MIDI Classification on midi5 files ===")
    
    # Find all midi files
    midi_files = glob.glob(os.path.join(MIDI5_DIR, "*.mid"))
    if not midi_files:
        print(f"No MIDI files found in {MIDI5_DIR}")
        return
    
    classifier = GenreClassifier()
    
    results = []
    for midi_path in midi_files:
        print(f"\nAnalyzing: {os.path.basename(midi_path)}")
        result = classifier.classify_genre(midi_path, top_k=3)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            continue
        
        primary = result['primary_genre']
        confidence = result['confidence_score']
        genres = [g['genre'] for g in result['genres']]
        is_ambient = 'ambient' in genres or primary == 'ambient'
        is_electronic = 'electronic' in genres or primary == 'electronic'
        
        print(f"Primary Genre: {primary} (confidence: {confidence:.2f})")
        print(f"Top Genres: {genres}")
        print(f"Is Ambient/Electronic: {is_ambient or is_electronic}")
        
        if 'feature_analysis' in result:
            print(f"Feature Analysis: {result['feature_analysis']}")
        
        results.append({
            'file': os.path.basename(midi_path),
            'primary': primary,
            'confidence': confidence,
            'is_ambient': is_ambient,
            'is_electronic': is_electronic,
            'top_genres': genres
        })
    
    # Summary
    print("\n=== Classification Summary ===")
    ambient_count = sum(1 for r in results if r['is_ambient'])
    electronic_count = sum(1 for r in results if r['is_electronic'])
    total = len(results)
    print(f"Total files: {total}")
    print(f"Detected as Ambient: {ambient_count}/{total} ({ambient_count/total*100:.1f}%)")
    print(f"Detected as Electronic: {electronic_count}/{total} ({electronic_count/total*100:.1f}%)")
    
    # All results
    for res in results:
        print(f"{res['file']}: {res['primary']} ({res['confidence']:.2f}) - Ambient: {res['is_ambient']}, Electronic: {res['is_electronic']}")

def test_generation():
    """Test generation with new components."""
    print("\n=== Testing Generation Components ===")
    
    # Create context
    genre_rules = GenreFactory.create_genre_rules('ambient')
    context = GeneratorContext(
        genre_rules=genre_rules,  # Use default
        mood="calm"
    )
    context.set_user_key_mode("C", "minor")
    context.scale_pitches = MusicTheory.get_scale_pitches_from_string("C minor", octave_range=3)
    
    # Test AmbientRhythmEngine
    engine = AmbientRhythmEngine("C", "minor")
    rhythm_notes = engine.generate_ambient_rhythm(duration_beats=16, sparsity_level=0.7, sustain_probability=0.8)
    print(f"Generated {len(rhythm_notes)} rhythm notes")
    
    # Test AtmosphereGenerator
    atm_gen = AtmosphereGenerator(context)
    atm_notes = atm_gen.create_atmospheric_layer(duration_beats=16, complexity=0.6, texture_type="pad")
    print(f"Generated {len(atm_notes)} atmospheric notes")
    
    # Test DynamicRhythmAdaptor
    adaptor = DynamicRhythmAdaptor(context)
    metrics = AdaptationMetrics(tempo=80.0, mood="calm", sparsity=0.7, complexity=0.6)
    adapted_notes = adaptor.adapt_rhythm_to_ambient_style(rhythm_notes, {
        'tempo': 80.0,
        'mood': 0.5,  # calm factor
        'sparsity': 0.7,
        'complexity': 0.6
    })
    print(f"Adapted {len(adapted_notes)} notes")
    
    # Test AmbientPatternTemplates
    templates = AmbientPatternTemplates(context)
    pattern_notes = templates.generate_full_ambient_pattern(duration_beats=16, subgenre="ethereal", pattern_type="all")
    print(f"Generated {len(pattern_notes)} pattern notes")
    
    # Save sample to MIDI (simple mido export)
    def notes_to_midi(notes, output_path):
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Sort notes
        notes = sorted(notes, key=lambda n: n.start_time)
        
        for note in notes:
            track.append(mido.Message('note_on', note=note.pitch, velocity=note.velocity, time=0, channel=(note.channel-1) % 16))
            # Approximate duration in ticks (assume 480 ticks/beat, duration in beats)
            duration_ticks = int(note.duration * 480)
            track.append(mido.Message('note_off', note=note.pitch, velocity=0, time=duration_ticks, channel=(note.channel-1) % 16))
        
        mid.save(output_path)
        print(f"Saved generated notes to {output_path}")
    
    # Save samples
    notes_to_midi(atm_notes, "output/test_atmosphere.mid")
    notes_to_midi(pattern_notes, "output/test_ambient_pattern.mid")

if __name__ == "__main__":
    test_classification()
    test_generation()
    print("\nTesting complete. Check output files and console results.")