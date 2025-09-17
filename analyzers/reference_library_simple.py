#!/usr/bin/env python3
"""
Simple Reference Library Integration System

Purpose:
- Integrate with reference MIDI libraries
- Implement style matching algorithms
- Provide trend analysis capabilities
"""

import argparse
import json
import os
import statistics
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import mido


class SimpleReferenceLibrary:
    """Simple reference library for style matching and trend analysis."""

    def __init__(self, library_path: str = "reference_midis"):
        self.library_path = library_path
        self.reference_data = []
        self._load_library()

    def _load_library(self):
        """Load reference tracks from library."""
        if not os.path.exists(self.library_path):
            print(f"Reference library path {self.library_path} does not exist")
            return

        track_count = 0
        for root, dirs, files in os.walk(self.library_path):
            for file in files:
                # Skip non-MIDI and hidden/system artifact files like "._*"
                if not file.endswith('.mid'):
                    continue
                if file.startswith('._') or file.startswith('.'):
                    continue
                file_path = os.path.join(root, file)
                track_data = self._analyze_track(file_path)
                if track_data:
                    self.reference_data.append(track_data)
                    track_count += 1

        print(f"Loaded {track_count} reference tracks")

    def _analyze_track(self, file_path: str) -> Optional[Dict]:
        """Analyze a single reference track."""
        try:
            midi_file = mido.MidiFile(file_path)

            # Extract basic information
            tempo = self._extract_tempo(midi_file)
            notes = self._extract_notes(midi_file)

            if not notes:
                return None

            # Calculate features
            note_count = len(notes)
            avg_velocity = statistics.mean([n['velocity'] for n in notes])
            pitches = [n['pitch'] for n in notes]
            pitch_range = (min(pitches), max(pitches)) if pitches else (60, 60)

            # Genre detection
            genre = self._detect_genre_from_path(file_path)

            return {
                'file_path': file_path,
                'genre': genre,
                'tempo': tempo,
                'note_count': note_count,
                'avg_velocity': avg_velocity,
                'pitch_range': pitch_range,
                'notes': notes
            }

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def _extract_tempo(self, midi_file: mido.MidiFile) -> float:
        """Extract tempo from MIDI file."""
        for track in midi_file.tracks:
            for msg in track:
                if hasattr(msg, 'type') and msg.type == 'set_tempo':
                    return 60000000 / msg.tempo
        return 120.0

    def _extract_notes(self, midi_file: mido.MidiFile) -> List[Dict]:
        """Extract notes from MIDI file."""
        notes = []
        active_notes = {}

        for track_idx, track in enumerate(midi_file.tracks):
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if hasattr(msg, 'type'):
                    if msg.type == 'note_on' and msg.velocity > 0:
                        key = (msg.note, msg.channel)
                        active_notes[key] = {
                            'pitch': msg.note,
                            'velocity': msg.velocity,
                            'start_time': abs_time,
                            'channel': msg.channel,
                            'track': track_idx
                        }
                    elif ((msg.type == 'note_off') or
                          (msg.type == 'note_on' and msg.velocity == 0)):
                        key = (msg.note, msg.channel)
                        if key in active_notes:
                            note = active_notes[key]
                            note['end_time'] = abs_time
                            note['duration'] = abs_time - note['start_time']
                            notes.append(note)
                            del active_notes[key]

        return notes

    def _detect_genre_from_path(self, file_path: str) -> str:
        """Detect genre from file path."""
        path_lower = file_path.lower()

        genre_hints = {
            'rock': 'rock',
            'jazz': 'jazz',
            'pop': 'pop',
            'electronic': 'electronic',
            'hip-hop': 'hip-hop',
            'hiphop': 'hip-hop',
            'classical': 'classical',
            'dnb': 'electronic',
            'drum-and-bass': 'electronic'
        }

        for hint, genre in genre_hints.items():
            if hint in path_lower:
                return genre

        return 'unknown'

    def find_similar_tracks(self, query_file: str, target_genre: Optional[str] = None,
                           top_n: int = 5) -> List[Dict]:
        """Find tracks similar to the query."""
        try:
            # Analyze query track
            query_data = self._analyze_track(query_file)
            if not query_data:
                return []

            similarities = []

            for ref_data in self.reference_data:
                if target_genre and ref_data['genre'] != target_genre:
                    continue

                similarity = self._calculate_similarity(query_data, ref_data)
                similarities.append({
                    'reference_file': ref_data['file_path'],
                    'similarity_score': similarity,
                    'genre': ref_data['genre'],
                    'tempo_diff': abs(query_data['tempo'] - ref_data['tempo'])
                })

            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:top_n]

        except Exception as e:
            print(f"Error finding similar tracks: {e}")
            return []

    def _calculate_similarity(self, query: Dict, reference: Dict) -> float:
        """Calculate similarity between two tracks."""
        scores = []

        # Tempo similarity
        tempo_diff = abs(query['tempo'] - reference['tempo'])
        tempo_score = max(0.0, 1.0 - tempo_diff / 30.0)
        scores.append(tempo_score)

        # Note density similarity
        density_diff = abs(query['note_count'] - reference['note_count'])
        max_density = max(query['note_count'], reference['note_count'])
        density_score = 1.0 - (density_diff / max_density) if max_density > 0 else 0.5
        scores.append(density_score)

        # Velocity similarity
        velocity_diff = abs(query['avg_velocity'] - reference['avg_velocity'])
        velocity_score = max(0.0, 1.0 - velocity_diff / 50.0)
        scores.append(velocity_score)

        # Pitch range similarity
        range1 = query['pitch_range']
        range2 = reference['pitch_range']
        range_overlap = self._calculate_range_overlap(range1, range2)
        scores.append(range_overlap)

        return statistics.mean(scores)

    def _calculate_range_overlap(self, range1: Tuple[int, int],
                                range2: Tuple[int, int]) -> float:
        """Calculate overlap between two pitch ranges."""
        start1, end1 = range1
        start2, end2 = range2

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_size = overlap_end - overlap_start
        union_size = max(end1, end2) - min(start1, start2)

        return overlap_size / union_size if union_size > 0 else 0.0

    def analyze_trends(self) -> Dict:
        """Analyze trends in the reference library."""
        if not self.reference_data:
            return {}

        # Group by genre
        genre_groups = defaultdict(list)
        for track in self.reference_data:
            genre_groups[track['genre']].append(track)

        trends = {}

        for genre, tracks in genre_groups.items():
            tempos = [t['tempo'] for t in tracks]
            note_counts = [t['note_count'] for t in tracks]
            velocities = [t['avg_velocity'] for t in tracks]

            trends[genre] = {
                'track_count': len(tracks),
                'avg_tempo': statistics.mean(tempos),
                'tempo_range': (min(tempos), max(tempos)),
                'avg_note_count': statistics.mean(note_counts),
                'avg_velocity': statistics.mean(velocities)
            }

        return trends


def main():
    parser = argparse.ArgumentParser(description="Simple reference library demo.")
    parser.add_argument("--action", choices=["match", "trends"], default="match",
                       help="Action to perform.")
    parser.add_argument("--input", help="Input MIDI file for matching.")
    parser.add_argument("--genre", default="pop", help="Target genre.")
    parser.add_argument("--output", default="test_outputs",
                       help="Output directory.")

    args = parser.parse_args()

    # Initialize library
    library = SimpleReferenceLibrary()

    if args.action == "match" and args.input:
        print(f"Finding similar tracks for {args.input}...")

        similar_tracks = library.find_similar_tracks(args.input, args.genre)

        print("\nTop similar tracks:")
        for i, track in enumerate(similar_tracks, 1):
            print(f"{i}. {os.path.basename(track['reference_file'])}")
            print(f"   Similarity score: {track['similarity_score']:.2f}")
            print(f"   Tempo difference: {track['tempo_diff']:.1f} BPM")

        # Save results
        os.makedirs(args.output, exist_ok=True)
        results = {
            "query_file": args.input,
            "target_genre": args.genre,
            "similar_tracks": similar_tracks
        }

        with open(os.path.join(args.output, "reference_matches.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {args.output}/reference_matches.json")

    elif args.action == "trends":
        print("Analyzing trends in reference library...")

        trends = library.analyze_trends()

        print("\nGenre Trends:")
        for genre, data in trends.items():
            print(f"\n{genre.upper()}:")
            print(f"  Tracks: {data['track_count']}")
            print(f"  Avg tempo: {data['avg_tempo']:.1f} BPM")
            print(f"  Tempo range: {data['tempo_range'][0]:.1f} - {data['tempo_range'][1]:.1f} BPM")
            print(f"  Avg note count: {data['avg_note_count']:.0f}")
            print(f"  Avg velocity: {data['avg_velocity']:.1f}")

        # Save trends
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "reference_trends.json"), "w") as f:
            json.dump(trends, f, indent=2)

        print(f"\nTrends saved to {args.output}/reference_trends.json")


if __name__ == "__main__":
    main()