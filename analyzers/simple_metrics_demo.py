#!/usr/bin/env python3
"""
Simple Metrics Dashboard Demo

Purpose:
- Demonstrate basic genre consistency scoring
- Production quality metrics
- Comprehensive analysis capabilities

This is a simplified working version of the advanced metrics dashboard.
"""

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from typing import List, Optional

import mido


@dataclass
class AnalysisResult:
    """Basic analysis result."""
    file_path: str
    genre_consistency_score: float
    production_quality_score: float
    overall_score: float
    recommendations: List[str]


class SimpleAnalyzer:
    """Simple analyzer for demonstration."""

    def analyze_file(self, file_path: str, target_genre: str = "pop") -> Optional[AnalysisResult]:
        """Analyze a MIDI file."""
        try:
            midi_file = mido.MidiFile(file_path)

            # Extract basic metrics
            tempo = self._extract_tempo(midi_file)
            note_count = self._count_notes(midi_file)

            # Calculate scores
            genre_score = self._calculate_genre_score(tempo, target_genre)
            quality_score = self._calculate_quality_score(note_count, tempo)
            overall_score = (genre_score + quality_score) / 2.0

            # Generate recommendations
            recommendations = []
            if genre_score < 0.7:
                recommendations.append(f"Adjust tempo for better {target_genre} fit")
            if quality_score < 0.6:
                recommendations.append("Improve note density and timing")

            return AnalysisResult(
                file_path=file_path,
                genre_consistency_score=genre_score,
                production_quality_score=quality_score,
                overall_score=overall_score,
                recommendations=recommendations
            )

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

    def _count_notes(self, midi_file: mido.MidiFile) -> int:
        """Count notes in MIDI file."""
        count = 0
        for track in midi_file.tracks:
            for msg in track:
                if hasattr(msg, 'type') and msg.type == 'note_on' and msg.velocity > 0:
                    count += 1
        return count

    def _calculate_genre_score(self, tempo: float, genre: str) -> float:
        """Calculate genre consistency score."""
        genre_ranges = {
            "pop": (90, 140),
            "rock": (100, 160),
            "jazz": (120, 200),
            "electronic": (120, 140)
        }

        if genre in genre_ranges:
            min_tempo, max_tempo = genre_ranges[genre]
            if min_tempo <= tempo <= max_tempo:
                return 1.0
            else:
                # Partial score based on distance
                distance = min(abs(tempo - min_tempo), abs(tempo - max_tempo))
                return max(0.0, 1.0 - distance / 30.0)

        return 0.5

    def _calculate_quality_score(self, note_count: int, tempo: float) -> float:
        """Calculate production quality score."""
        # Simple scoring based on note density
        duration_seconds = 30  # Assume 30 second piece
        density = note_count / duration_seconds

        if 2 <= density <= 8:
            return 1.0
        elif 1 <= density <= 12:
            return 0.8
        else:
            return 0.5


def main():
    parser = argparse.ArgumentParser(description="Simple metrics dashboard demo.")
    parser.add_argument("--inputs", nargs="+", required=True,
                       help="Input MIDI file paths.")
    parser.add_argument("--genre", default="pop",
                       help="Target genre for analysis.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    analyzer = SimpleAnalyzer()
    results = []

    for file_path in args.inputs:
        if args.verbose:
            print(f"Analyzing {file_path}...")

        result = analyzer.analyze_file(file_path, args.genre)
        if result:
            results.append(result)
            print(f"  Genre score: {result.genre_consistency_score:.2f}")
            print(f"  Quality score: {result.production_quality_score:.2f}")
            print(f"  Overall score: {result.overall_score:.2f}")
            if result.recommendations:
                print("  Recommendations:")
                for rec in result.recommendations:
                    print(f"    - {rec}")

    if results:
        avg_score = statistics.mean([r.overall_score for r in results])
        print(f"Average overall score: {avg_score:.2f}")

        # Save results to JSON
        os.makedirs("test_outputs", exist_ok=True)
        output_data = {
            "results": [
                {
                    "file": r.file_path,
                    "genre_score": r.genre_consistency_score,
                    "quality_score": r.production_quality_score,
                    "overall_score": r.overall_score,
                    "recommendations": r.recommendations
                } for r in results
            ],
            "summary": {
                "total_files": len(results),
                "average_score": avg_score
            }
        }

        with open("test_outputs/simple_metrics_results.json", "w") as f:
            json.dump(output_data, f, indent=2)

        print("Results saved to test_outputs/simple_metrics_results.json")


if __name__ == "__main__":
    main()