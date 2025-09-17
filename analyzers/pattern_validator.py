import os
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict
from analyzers.midi_pattern_extractor import MidiPatternData, extract_from_file
from analyzers.reference_pattern_library import PatternMetadata

class PatternValidator:
    def __init__(self):
        self.genre_rules = {
            'drum-and-bass': {'min_bpm': 140, 'max_bpm': 200, 'preferred_instr': ['drums', 'bass', 'melody']},
            'pop': {'min_bpm': 80, 'max_bpm': 140, 'melody_intervals_avg': 5, 'melody_intervals_std': 3},
            'jazz': {'min_bpm': 60, 'max_bpm': 220, 'complexity_min': 1.0},
            'electronic': {'min_bpm': 100, 'max_bpm': 180},
            'unknown': {}
        }

    def classify_instrument_type(self, pattern: MidiPatternData) -> str:
        if not pattern.notes:
            return 'unknown'
        
        instruments = set(n.instrument for n in pattern.notes if n.instrument)
        channels = set(n.channel for n in pattern.notes)
        
        if 9 in channels or any('drum' in i.lower() or 'perc' in i.lower() for i in instruments):
            return 'drums'
        
        avg_pitch = np.mean([n.note for n in pattern.notes])
        if avg_pitch < 60 or any('bass' in i.lower() for i in instruments):
            return 'bass'
        
        harmony_instruments = ['piano chords', 'strings', 'brass', 'pad']
        if any(any(instr in i.lower() for instr in harmony_instruments) for i in instruments):
            return 'harmony'
        
        time_to_notes = defaultdict(list)
        for n in pattern.notes:
            time_to_notes[n.start_time].append(n.note)
        max_simultaneous = max(len(notes) for notes in time_to_notes.values())
        if max_simultaneous > 1:
            return 'harmony'
        
        melody_instruments = ['synth', 'piano', 'violin', 'lead', 'pluck', 'bells']
        if any(any(instr in i.lower() for instr in melody_instruments) for i in instruments):
            return 'melody'
        
        return 'melody'

    def validate_pattern(self, pattern: MidiPatternData, metadata: PatternMetadata) -> Dict[str, Any]:
        report = {'errors': [], 'warnings': [], 'genre_specific': {}, 'consistency': []}

        # Metadata validation
        meta_val = metadata.validate()
        report['errors'].extend(meta_val['errors'])
        report['warnings'].extend(meta_val['warnings'])

        # Genre specific
        genre = metadata.genre.lower()
        if genre in self.genre_rules:
            rules = self.genre_rules[genre]
            min_bpm = rules.get('min_bpm', 0)
            max_bpm = rules.get('max_bpm', 250)
            if not min_bpm <= metadata.bpm <= max_bpm:
                report['errors'].append(f"BPM {metadata.bpm} out of genre range {min_bpm}-{max_bpm} for {genre}")

            if genre == 'pop' and metadata.pattern_category == 'melodic' and metadata.melodic_intervals:
                avg_interval = np.mean(metadata.melodic_intervals)
                std_interval = np.std(metadata.melodic_intervals)
                if abs(avg_interval) > rules.get('melody_intervals_avg', 5):
                    report['warnings'].append("Melody intervals average too large for pop")
                if std_interval > rules.get('melody_intervals_std', 3):
                    report['warnings'].append("Melody intervals too variable for pop")

            if genre == 'jazz' and metadata.complexity < rules.get('complexity_min', 1.0):
                report['warnings'].append("Low complexity for jazz")

        # Consistency checks
        # BPM consistency
        if pattern.tempos:
            avg_pattern_bpm = np.mean([t.bpm for t in pattern.tempos])
            if abs(avg_pattern_bpm - metadata.bpm) > 5:
                report['warnings'].append(f"BPM mismatch: metadata {metadata.bpm:.1f} vs pattern avg {avg_pattern_bpm:.1f}")

        # Instrument consistency
        classified_instr = self.classify_instrument_type(pattern)
        if classified_instr != metadata.instrument_type:
            report['warnings'].append(f"Instrument mismatch: classified {classified_instr} vs metadata {metadata.instrument_type}")

        # Pattern specific
        if not pattern.notes:
            report['errors'].append("Pattern has no notes")
        if pattern.length_ticks == 0:
            report['errors'].append("Pattern has zero length")
        if len(pattern.notes) > 10000:
            report['warnings'].append("Pattern has unusually high note count")

        # Time signature consistency if present
        if pattern.time_signatures:
            ts = pattern.time_signatures[0]
            if (ts.numerator, ts.denominator) != metadata.time_signature:
                report['warnings'].append(f"Time signature mismatch: metadata {metadata.time_signature} vs pattern {(ts.numerator, ts.denominator)}")

        return report

    def validate_all_patterns(self, library):
        """Validate all patterns in the library."""
        report = {'total': 0, 'valid': 0, 'errors': 0, 'warnings': 0, 'details': {}}
        for file_path, meta in library.metadata.items():
            if file_path in library.pattern_cache:
                pat = library.pattern_cache[file_path]
            else:
                try:
                    pat = extract_from_file(file_path)
                except Exception as e:
                    print(f"Could not load pattern for validation: {e}")
                    pat = MidiPatternData(file_path=file_path, ticks_per_beat=480, tracks=0, length_ticks=0, tempos=[], time_signatures=[], notes=[], track_info={})
                    
            val_report = self.validate_pattern(pat, meta)
            report['total'] += 1
            if not val_report['errors']:
                report['valid'] += 1
            report['errors'] += len(val_report['errors'])
            report['warnings'] += len(val_report['warnings'])
            report['details'][file_path] = val_report
        return report