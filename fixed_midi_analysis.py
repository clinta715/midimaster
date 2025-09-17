#!/usr/bin/env python3
"""
Comprehensive MIDI Analysis for Note Spacing, Timing, and Musical Structure
"""

import mido
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
import statistics
import os

class MidiAnalyzer:
    """Comprehensive MIDI file analyzer for debugging note spacing and timing issues."""

    def __init__(self, midi_file: str):
        self.midi_file = midi_file
        self.notes = []
        self.chords = []
        self.rhythms = []
        self.track_info = {}
        self.load_midi()

    def load_midi(self):
        """Load and parse MIDI file into structured data."""
        try:
            midi = mido.MidiFile(self.midi_file)
            print(f"Loading: {self.midi_file}")
            print(f"- Tracks: {len(midi.tracks)}")
            print(f"- Ticks per beat: {midi.ticks_per_beat}")
            print(f"- Type: {midi.type}")

            active_notes = {}  # (pitch, channel) -> (start_time, velocity)
            current_time = 0.0
            tempo = 500000  # Default 120 BPM in microseconds per quarter note

            # Process all tracks
            for track_idx, track in enumerate(midi.tracks):
                track_time = 0.0
                track_notes = []

                for msg in track:
                    track_time += msg.time * mido.tick2second(msg.time, midi.ticks_per_beat, tempo)

                    if msg.type == 'set_tempo':
                        tempo = msg.tempo

                    elif msg.type == 'note_on' and msg.velocity > 0:
                        # Note start
                        key = (msg.note, msg.channel)
                        active_notes[key] = (track_time, msg.velocity)

                    elif (msg.type == 'note_off' or
                          (msg.type == 'note_on' and msg.velocity == 0)):
                        # Note end
                        key = (msg.note, msg.channel)
                        if key in active_notes:
                            start_time, velocity = active_notes[key]
                            duration = track_time - start_time
                            if duration > 0:
                                note_info = {
                                    'pitch': msg.note,
                                    'velocity': velocity,
                                    'start_time': start_time,
                                    'duration': duration,
                                    'channel': msg.channel,
                                    'track': track_idx,
                                    'end_time': start_time + duration
                                }
                                track_notes.append(note_info)
                                self.notes.append(note_info)
                            del active_notes[key]

                # Analyze simultaneous notes (chords) in this track
                track_chords = self._find_chords(track_notes)
                if track_chords:
                    self.chords.append({
                        'track': track_idx,
                        'chords': track_chords
                    })

                self.track_info[track_idx] = {
                    'notes_count': len(track_notes),
                    'chord_count': len(track_chords),
                    'start_times': [n['start_time'] for n in track_notes],
                    'durations': [n['duration'] for n in track_notes],
                    'pitches': [n['pitch'] for n in track_notes]
                }

                # Analyze rhythmic patterns
                if track_notes:
                    self.rhythms.append({
                        'track': track_idx,
                        'intervals': self._calculate_note_intervals(track_notes),
                        'start_times': [n['start_time'] for n in track_notes]
                    })

        except Exception as e:
            print(f"Error loading MIDI: {e}")
            sys.exit(1)

    def _find_chords(self, notes: List[Dict]) -> List[Dict]:
        """Find groups of simultaneous notes (chords)."""
        chords = []
        if not notes:
            return chords

        # Group notes by start time (within 0.1 second tolerance)
        time_groups = defaultdict(list)
        for note in sorted(notes, key=lambda x: x['start_time']):
            time_key = round(note['start_time'] / 0.1) * 0.1
            time_groups[time_key].append(note)

        # Find chords (2+ notes at same time)
        for time_key, group_notes in time_groups.items():
            if len(group_notes) >= 2:
                pitches = sorted([n['pitch'] for n in group_notes])
                chords.append({
                    'time': time_key,
                    'pitches': pitches,
                    'size': len(pitches),
                    'duration': min(n['duration'] for n in group_notes)
                })

        return chords

    def _calculate_note_intervals(self, notes: List[Dict]) -> List[float]:
        """Calculate time intervals between consecutive notes."""
        if len(notes) < 2:
            return []

        intervals = []
        sorted_notes = sorted(notes, key=lambda x: x['start_time'])

        for i in range(1, len(sorted_notes)):
            interval = sorted_notes[i]['start_time'] - sorted_notes[i-1]['start_time']
            if interval > 0.001:  # Skip very small intervals (same time)
                intervals.append(interval)

        return intervals

    def analyze_note_spacing(self) -> Dict[str, Any]:
        """Analyze note spacing and timing patterns."""
        print("\n" + "="*60)
        print("NOTE SPACING & TIMING ANALYSIS")
        print("="*60)

        results: Dict[str, Any] = {
            'total_notes': len(self.notes),
            'tracks_with_notes': len([t for t in self.track_info.values() if t['notes_count'] > 0])
        }

        if not self.notes:
            print("No notes found in MIDI file")
            return results

        # Sort notes by start time
        sorted_notes = sorted(self.notes, key=lambda x: x['start_time'])

        # Calculate timing statistics
        start_times = [n['start_time'] for n in sorted_notes]
        durations = [n['duration'] for n in sorted_notes]

        total_duration = sorted_notes[-1]['end_time'] - sorted_notes[0]['start_time']

        results.update({
            'total_duration': total_duration,
            'avg_note_duration': statistics.mean(durations),
            'median_note_duration': statistics.median(durations),
            'note_density_per_second': len(self.notes) / (max(start_times) - min(start_times) + 0.001)
        })

        print(f"Total notes: {len(self.notes)}")
        print(f"Total duration: {results['total_duration']:.2f}s")
        print(f"Avg note duration: {results['avg_note_duration']:.2f}s")
        print(f"Median note duration: {results['median_note_duration']:.3f}s")
        print(f"Note density: {results['note_density_per_second']:.2f} notes/second")

        # Analyze intervals between notes
        intervals = []

        for i in range(1, len(sorted_notes)):
            curr_time = sorted_notes[i]['start_time']
            prev_time = sorted_notes[i-1]['start_time']
            interval = curr_time - prev_time

            if interval > 0.001:  # Only count meaningful gaps
                intervals.append(interval)
                if interval > 2.0:  # Large gaps (>2 seconds)
                    print(f"Large gap detected: {interval:.2f}s at {prev_time:.2f}s")

        if intervals:
            results.update({
                'avg_interval': statistics.mean(intervals),
                'median_interval': statistics.median(intervals),
                'min_interval': min(intervals),
                'max_interval': max(intervals),
                'intervals_greater_than_1s': len([i for i in intervals if i > 1.0]),
                'intervals_greater_than_2s': len([i for i in intervals if i > 2.0])
            })

            print("\nInterval Analysis:")
            print(f"Average interval: {results['avg_interval']:.3f}s")
            print(f"Median interval: {results['median_interval']:.3f}s")
            print(f"Interval range: {results['min_interval']:.3f}s - {results['max_interval']:.3f}s")
            print(f"Gaps > 1s: {results['intervals_greater_than_1s']} ({(results['intervals_greater_than_1s']/len(intervals))*100:.1f}%)")
            print(f"Gaps > 2s: {results['intervals_greater_than_2s']} ({(results['intervals_greater_than_2s']/len(intervals))*100:.1f}%)")

            # Show distribution of interval lengths
            print("\nInterval Distribution:")
            buckets = [(0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float('inf'))]
            for min_val, max_val in buckets:
                if max_val == float('inf'):
                    count = len([i for i in intervals if i >= min_val])
                    label = f">= {min_val}s"
                else:
                    count = len([i for i in intervals if min_val <= i < max_val])
                    label = f"{min_val}-{max_val}s"

                if count > 0:
                    print(f"  {label}: {count} ({count/len(intervals)*100:.1f}%)")

        return results

    def analyze_note_organization(self) -> Dict[str, Any]:
        """Analyze how notes are organized by channel, track, and simultaneity."""
        print("\n" + "="*60)
        print("NOTE ORGANIZATION ANALYSIS")
        print("="*60)

        results: Dict[str, Any] = {
            'total_notes': len(self.notes),
            'tracks_used': len([t for t in self.track_info.keys() if self.track_info[t]['notes_count'] > 0])
        }

        if not self.notes:
            return results

        # Channel analysis
        channels_used = set(n['channel'] for n in self.notes)
        channel_counts = Counter(n['channel'] for n in self.notes)

        print(f"Channels used: {sorted(channels_used)}")
        for ch, count in sorted(channel_counts.items()):
            print(f"  Channel {ch}: {count} notes ({count/len(self.notes)*100:.1f}%)")

        results['channels_used'] = len(channels_used)
        results['channel_distribution'] = dict(channel_counts)
        results['channels_used'] = len(channels_used)
        results['channel_distribution'] = dict(channel_counts)

        # Calculate total duration for density calculations
        sorted_notes = sorted(self.notes, key=lambda x: x['start_time'])
        total_duration = sorted_notes[-1]['end_time'] - sorted_notes[0]['start_time'] if self.notes else 0
        results['total_duration'] = total_duration

        # Simultaneous notes analysis (potential chords)

        # Simultaneous notes analysis (potential chords)
        all_chords = []
        for track_data in self.chords:
            all_chords.extend(track_data['chords'])

        if all_chords:
            chord_sizes = [c['size'] for c in all_chords]
            results.update({
                'total_simultaneous_notes': len(all_chords),
                'avg_chord_size': statistics.mean(chord_sizes),
                'max_chord_size': max(chord_sizes),
                'chord_intervals': [c['time'] for c in all_chords]
            })

            print("\nSimultaneous Notes (Chords):")
            print(f"  Total chord events: {len(all_chords)}")
            print(f"  Average chord size: {statistics.mean(chord_sizes):.1f}")
            print(f"  Largest chord: {max(chord_sizes)} notes")
            print(f"  Chord density: {len(all_chords)/(results['total_duration'] + 0.001):.2f} chords/second")
        else:
            print("\nNO CHORDS DETECTED")
            results['chords_detected'] = 0

        return results

    def analyze_rhythmic_patterns(self) -> Dict[str, Any]:
        """Analyze rhythmic patterns across tracks."""
        print("\n" + "="*60)
        print("RHYTHMIC PATTERN ANALYSIS")
        print("="*60)

        results: Dict[str, Any] = {'patterns_found': 0}

        if not self.notes:
            return results

        # Calculate tempo consistency
        tempo_changes = []

        try:
            midi = mido.MidiFile(self.midi_file)
            for track in midi.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo_changes.append(mido.tempo2bpm(msg.tempo))
        except:
            pass

        if tempo_changes:
            results['tempo_changes'] = len(tempo_changes)
            results['avg_tempo'] = statistics.mean(tempo_changes)
            results['tempo_variability'] = statistics.stdev(tempo_changes) if len(tempo_changes) > 1 else 0
            print(f"Average tempo: {results['avg_tempo']:.1f} BPM")
            print(f"Tempo variability: {results['tempo_variability']:.1f}")

        # Analyze regularity of note timing
        sorted_notes = sorted(self.notes, key=lambda x: x['start_time'])
        start_times = [n['start_time'] for n in sorted_notes]

        if len(start_times) >= 3:  # Need enough notes for pattern analysis
            # Check for consistent timing patterns (e.g., every 0.5, 1.0, etc.)
            possible_intervals = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
            regularity_scores = {}

            for interval in possible_intervals:
                matches = 0
                total_checks = len(start_times) - 1

                for i in range(1, len(start_times)):
                    diff = start_times[i] - start_times[i-1]
                    # Check if timing is close to interval (within 10%)
                    tolerance = interval * 0.1
                    if abs(diff - interval) < tolerance:
                        matches += 1

                regularity_score = matches / total_checks if total_checks > 0 else 0
                regularity_scores[interval] = regularity_score
                print(f"Regularity at {interval}s intervals: {regularity_score:.1f}")

            if regularity_scores:
                best_interval = max(regularity_scores, key=lambda k: regularity_scores[k])
                if regularity_scores[best_interval] > 0.3:
                    print(f"Most regular rhythm: Every {best_interval}s ({regularity_scores[best_interval]*100:.1f}% regularity)")
                    results['best_rhythmic_interval'] = best_interval
                    results['rhythmic_regularity'] = regularity_scores[best_interval]
                    results['patterns_found'] = 1

        return results

    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MIDI ANALYSIS REPORT")
        print("="*80)

        spacing_analysis = self.analyze_note_spacing()
        organization_analysis = self.analyze_note_organization()
        rhythmic_analysis = self.analyze_rhythmic_patterns()

        full_report = {
            'file_info': {
                'filename': self.midi_file,
                'format': self.midi_file.split('.')[-1],
                'exists': os.path.exists(self.midi_file)
            },
            'spacing_analysis': spacing_analysis,
            'organization_analysis': organization_analysis,
            'rhythmic_analysis': rhythmic_analysis
        }

        print("\n" + "="*80)
        print("SUMMARY OF KEY ISSUES")
        print("="*80)

        # Spacing issues
        if 'intervals_greater_than_2s' in spacing_analysis:
            large_gaps_pct = spacing_analysis['intervals_greater_than_2s'] / len(self._calculate_note_intervals(self.notes)) * 100 if self.notes else 0
            if large_gaps_pct > 20:
                print(f"⚠️  Large gaps (>2s): {large_gaps_pct:.1f}% of intervals")

        # Chord issues
        if organization_analysis.get('total_simultaneous_notes', 0) == 0:
            print("⚠️  NO CHORDS DETECTED - All notes are monophonic!")

        # Rhythm issues
        if 'avg_interval' in spacing_analysis:
            avg_interval = spacing_analysis['avg_interval']
            if avg_interval > 2.0:
                print(f"⚠️  Very sparse timing: Average {avg_interval:.2f}s between notes")
            elif avg_interval > 1.0:
                print(f"⚠️  Sparse timing: Average {avg_interval:.2f}s between notes")
            elif avg_interval > 0.5:
                print("✓ Note spacing is moderate but could be more rhythmic")
            else:
                print("✓ Notes are closely spaced - good rhythmic density")

        return full_report

def analyze_midi_file(filename: str):
    """Main function to analyze a MIDI file."""
    analyzer = MidiAnalyzer(filename)
    return analyzer.generate_report()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fixed_midi_analysis.py <midi_file>")
        sys.exit(1)

    analyze_midi_file(sys.argv[1])