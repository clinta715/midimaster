#!/usr/bin/env python3
"""
Mix Readiness Indicators

Purpose:
- Evaluate tracks for mixing/mastering readiness
- Provide specific recommendations for mix improvements
- Assess overall production quality and completeness
- Generate actionable mixing checklists

Features:
- Frequency balance analysis
- Dynamic range assessment
- Stereo field evaluation
- Mix clarity indicators
- Production readiness scoring
"""

import argparse
import json
import os
import statistics
from collections import Counter
from typing import Dict, List, Optional, Tuple

import mido


class MixReadinessAnalyzer:
    """Analyze tracks for mix readiness and provide recommendations."""

    def __init__(self):
        self.mix_criteria = {
            'frequency_balance': {
                'weight': 0.25,
                'thresholds': {'good': 0.8, 'acceptable': 0.6}
            },
            'dynamic_range': {
                'weight': 0.20,
                'thresholds': {'good': 0.7, 'acceptable': 0.5}
            },
            'stereo_field': {
                'weight': 0.15,
                'thresholds': {'good': 0.8, 'acceptable': 0.6}
            },
            'mix_clarity': {
                'weight': 0.20,
                'thresholds': {'good': 0.75, 'acceptable': 0.55}
            },
            'production_completeness': {
                'weight': 0.20,
                'thresholds': {'good': 0.8, 'acceptable': 0.6}
            }
        }

    def analyze_mix_readiness(self, midi_file: str) -> Dict:
        """
        Perform comprehensive mix readiness analysis.

        Args:
            midi_file: Path to MIDI file to analyze

        Returns:
            Dictionary containing analysis results and recommendations
        """

        try:
            midi_data = mido.MidiFile(midi_file)
            notes = self._extract_notes(midi_data)

            if not notes:
                return self._create_empty_analysis(midi_file)

            # Perform individual analyses
            frequency_balance = self._analyze_frequency_balance(notes)
            dynamic_range = self._analyze_dynamic_range(notes)
            stereo_field = self._analyze_stereo_field(notes, midi_data)
            mix_clarity = self._analyze_mix_clarity(notes)
            production_completeness = self._analyze_production_completeness(notes, midi_data)

            # Calculate overall readiness score
            overall_score = self._calculate_overall_score({
                'frequency_balance': frequency_balance['score'],
                'dynamic_range': dynamic_range['score'],
                'stereo_field': stereo_field['score'],
                'mix_clarity': mix_clarity['score'],
                'production_completeness': production_completeness['score']
            })

            # Generate recommendations
            recommendations = self._generate_recommendations({
                'frequency_balance': frequency_balance,
                'dynamic_range': dynamic_range,
                'stereo_field': stereo_field,
                'mix_clarity': mix_clarity,
                'production_completeness': production_completeness
            })

            # Create mixing checklist
            mixing_checklist = self._create_mixing_checklist(overall_score)

            return {
                'file_path': midi_file,
                'overall_readiness_score': overall_score,
                'readiness_rating': self._score_to_rating(overall_score),
                'detailed_scores': {
                    'frequency_balance': frequency_balance,
                    'dynamic_range': dynamic_range,
                    'stereo_field': stereo_field,
                    'mix_clarity': mix_clarity,
                    'production_completeness': production_completeness
                },
                'recommendations': recommendations,
                'mixing_checklist': mixing_checklist,
                'estimated_mix_time': self._estimate_mix_time(overall_score)
            }

        except Exception as e:
            print(f"Error analyzing mix readiness: {e}")
            return self._create_empty_analysis(midi_file)

    def _extract_notes(self, midi_data: mido.MidiFile) -> List[Dict]:
        """Extract note information from MIDI file."""
        notes = []
        active_notes = {}

        for track_idx, track in enumerate(midi_data.tracks):
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

    def _analyze_frequency_balance(self, notes: List[Dict]) -> Dict:
        """Analyze frequency balance across the mix."""
        if not notes:
            return {'score': 0.0, 'assessment': 'No notes found', 'issues': []}

        pitches = [n['pitch'] for n in notes]

        # Define frequency ranges (MIDI note numbers)
        ranges = {
            'sub_bass': (0, 23),      # C0-B1
            'bass': (24, 35),         # C2-B2
            'low_mid': (36, 47),      # C3-B3
            'mid': (48, 59),          # C4-B4
            'high_mid': (60, 71),     # C5-B5
            'high': (72, 83),         # C6-B6
            'super_high': (84, 127)   # C7-G9
        }

        # Count notes in each range
        range_counts = {}
        total_notes = len(pitches)

        for range_name, (low, high) in ranges.items():
            count = sum(1 for p in pitches if low <= p <= high)
            range_counts[range_name] = count / total_notes if total_notes > 0 else 0

        # Evaluate balance
        issues = []
        score = 1.0

        # Check for missing frequency ranges
        essential_ranges = ['bass', 'mid', 'high']
        for range_name in essential_ranges:
            if range_counts[range_name] < 0.05:  # Less than 5%
                issues.append(f"Very low presence in {range_name.replace('_', ' ')} range")
                score -= 0.2

        # Check for over-represented ranges
        for range_name, percentage in range_counts.items():
            if percentage > 0.4:  # More than 40%
                issues.append(f"Over-represented {range_name.replace('_', ' ')} range")
                score = min(score, 0.8)

        score = max(0.0, score)

        assessment = "Good frequency balance"
        if score < 0.6:
            assessment = "Frequency balance needs attention"
        elif score < 0.8:
            assessment = "Frequency balance could be improved"

        return {
            'score': score,
            'assessment': assessment,
            'issues': issues,
            'range_distribution': range_counts
        }

    def _analyze_dynamic_range(self, notes: List[Dict]) -> Dict:
        """Analyze dynamic range."""
        if not notes:
            return {'score': 0.0, 'assessment': 'No notes found', 'issues': []}

        velocities = [n['velocity'] for n in notes]

        if len(velocities) < 2:
            return {'score': 0.5, 'assessment': 'Limited dynamic range', 'issues': ['Minimal velocity variation']}

        min_vel = min(velocities)
        max_vel = max(velocities)
        range_size = max_vel - min_vel

        # Calculate coefficient of variation
        mean_vel = statistics.mean(velocities)
        std_vel = statistics.stdev(velocities)
        cv = std_vel / mean_vel if mean_vel > 0 else 0

        score = 0.0
        issues = []

        # Range assessment
        if range_size < 20:
            issues.append("Limited dynamic range")
            score -= 0.3
        elif range_size > 80:
            issues.append("Very wide dynamic range")

        # Variation assessment
        if cv < 0.1:
            issues.append("Low velocity variation")
            score -= 0.2
        elif cv > 0.5:
            issues.append("High velocity variation")

        score = max(0.0, 0.7 + score)  # Base score of 0.7, adjusted by issues

        assessment = "Good dynamic range"
        if score < 0.5:
            assessment = "Dynamic range needs improvement"
        elif score < 0.7:
            assessment = "Dynamic range could be optimized"

        return {
            'score': score,
            'assessment': assessment,
            'issues': issues,
            'velocity_range': (min_vel, max_vel),
            'coefficient_of_variation': cv
        }

    def _analyze_stereo_field(self, notes: List[Dict], midi_data: mido.MidiFile) -> Dict:
        """Analyze stereo field usage."""
        # For MIDI, stereo field is determined by channel assignments
        channels = [n['channel'] for n in notes]
        channel_counts = Counter(channels)

        score = 0.5  # Default neutral score
        issues = []

        # Check channel distribution
        if len(channel_counts) == 1:
            issues.append("Mono - consider using stereo channels")
            score = 0.3
        elif len(channel_counts) >= 3:
            issues.append("Multiple channels detected")
            score = 0.8

        # Check for percussion on channel 9 (which is typically centered)
        if 9 in channel_counts:
            percussion_ratio = channel_counts[9] / len(notes)
            if percussion_ratio > 0.3:
                issues.append("High percussion content (consider stereo percussion)")

        assessment = "Stereo field usage"
        if score < 0.5:
            assessment = "Limited stereo field usage"
        elif score >= 0.7:
            assessment = "Good stereo field usage"

        return {
            'score': score,
            'assessment': assessment,
            'issues': issues,
            'channel_distribution': dict(channel_counts)
        }

    def _analyze_mix_clarity(self, notes: List[Dict]) -> Dict:
        """Analyze mix clarity."""
        if not notes:
            return {'score': 0.0, 'assessment': 'No notes found', 'issues': []}

        # Analyze note density and overlap
        total_duration = max((n['end_time'] for n in notes), default=1)

        # Calculate note density (notes per second)
        density = len(notes) / (total_duration / 1000)

        # Check for simultaneous notes (potential masking)
        time_points = {}
        for note in notes:
            start_time = note['start_time']
            if start_time not in time_points:
                time_points[start_time] = []
            time_points[start_time].append(note)

        simultaneous_notes = [len(notes_at_time) for notes_at_time in time_points.values()]
        avg_simultaneous = statistics.mean(simultaneous_notes) if simultaneous_notes else 1

        score = 1.0
        issues = []

        if density > 20:  # Very dense
            issues.append("Very high note density may cause clutter")
            score -= 0.3
        elif density > 10:  # Moderately dense
            issues.append("Moderate note density")
            score -= 0.1

        if avg_simultaneous > 5:
            issues.append("High polyphony may cause masking")
            score -= 0.2

        score = max(0.0, score)

        assessment = "Good mix clarity"
        if score < 0.6:
            assessment = "Mix clarity needs attention"
        elif score < 0.8:
            assessment = "Mix clarity could be improved"

        return {
            'score': score,
            'assessment': assessment,
            'issues': issues,
            'note_density': density,
            'avg_simultaneous_notes': avg_simultaneous
        }

    def _analyze_production_completeness(self, notes: List[Dict], midi_data: mido.MidiFile) -> Dict:
        """Analyze production completeness."""
        score = 0.5
        issues = []

        # Check for basic elements
        has_notes = len(notes) > 0
        has_velocity_variation = len(set(n['velocity'] for n in notes)) > 1
        has_multiple_channels = len(set(n['channel'] for n in notes)) > 1
        has_tempo_info = self._has_tempo_info(midi_data)

        if has_notes:
            score += 0.2
        else:
            issues.append("No notes detected")

        if has_velocity_variation:
            score += 0.2
        else:
            issues.append("No velocity variation")

        if has_multiple_channels:
            score += 0.2
        else:
            issues.append("Mono channel usage only")

        if has_tempo_info:
            score += 0.2
        else:
            issues.append("No tempo information")

        # Clamp to [0, 1] to keep category score in a normalized range
        score = max(0.0, min(1.0, score))

        assessment = "Production completeness"
        if score < 0.6:
            assessment = "Incomplete production elements"
        elif score >= 0.8:
            assessment = "Good production completeness"

        return {
            'score': score,
            'assessment': assessment,
            'issues': issues,
            'elements_present': {
                'notes': has_notes,
                'velocity_variation': has_velocity_variation,
                'multiple_channels': has_multiple_channels,
                'tempo_info': has_tempo_info
            }
        }

    def _has_tempo_info(self, midi_data: mido.MidiFile) -> bool:
        """Check if MIDI file has tempo information."""
        for track in midi_data.tracks:
            for msg in track:
                if hasattr(msg, 'type') and msg.type == 'set_tempo':
                    return True
        return False

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall readiness score."""
        total_weight = sum(criteria['weight'] for criteria in self.mix_criteria.values())
        overall = 0.0

        for criterion_name, score in scores.items():
            if criterion_name in self.mix_criteria:
                weight = self.mix_criteria[criterion_name]['weight']
                # Ensure each sub-score is in [0,1]
                s = max(0.0, min(1.0, score))
                overall += s * weight

        result = overall / total_weight if total_weight > 0 else 0.0
        # Extra safety: clamp final result to [0,1]
        return max(0.0, min(1.0, result))

    def _generate_recommendations(self, analyses: Dict) -> List[str]:
        """Generate specific recommendations based on analysis."""
        recommendations = []

        # Frequency balance recommendations
        freq_analysis = analyses['frequency_balance']
        if freq_analysis['score'] < 0.7:
            recommendations.extend([
                "Balance frequency ranges more evenly",
                "Ensure presence across bass, mid, and high frequencies",
                "Consider EQ adjustments for frequency compensation"
            ])

        # Dynamic range recommendations
        dyn_analysis = analyses['dynamic_range']
        if dyn_analysis['score'] < 0.7:
            recommendations.extend([
                "Add more dynamic variation",
                "Use velocity automation for expression",
                "Consider compression if dynamics are too wide"
            ])

        # Mix clarity recommendations
        clarity_analysis = analyses['mix_clarity']
        if clarity_analysis['score'] < 0.7:
            recommendations.extend([
                "Reduce note density if mix sounds cluttered",
                "Consider panning to separate elements",
                "Use reverb strategically to create space"
            ])

        return recommendations if recommendations else ["Mix is ready for mastering"]

    def _create_mixing_checklist(self, overall_score: float) -> List[Dict]:
        """Create a mixing checklist based on readiness score."""
        checklist = [
            {
                "task": "Level balancing",
                "completed": overall_score > 0.6,
                "priority": "high" if overall_score < 0.7 else "medium"
            },
            {
                "task": "EQ adjustments",
                "completed": overall_score > 0.7,
                "priority": "high" if overall_score < 0.8 else "medium"
            },
            {
                "task": "Compression",
                "completed": overall_score > 0.5,
                "priority": "medium"
            },
            {
                "task": "Stereo imaging",
                "completed": overall_score > 0.6,
                "priority": "medium"
            },
            {
                "task": "Reverb and effects",
                "completed": overall_score > 0.8,
                "priority": "low"
            },
            {
                "task": "Mastering preparation",
                "completed": overall_score > 0.9,
                "priority": "low"
            }
        ]

        return checklist

    def _estimate_mix_time(self, overall_score: float) -> str:
        """Estimate mixing time based on readiness score."""
        if overall_score > 0.9:
            return "1-2 hours (polish only)"
        elif overall_score > 0.8:
            return "2-4 hours (basic mixing)"
        elif overall_score > 0.7:
            return "4-6 hours (moderate work needed)"
        elif overall_score > 0.6:
            return "6-8 hours (significant work needed)"
        else:
            return "8+ hours (major mixing required)"

    def _score_to_rating(self, score: float) -> str:
        """Convert numeric score to descriptive rating."""
        if score >= 0.9:
            return "Master Ready"
        elif score >= 0.8:
            return "Mix Ready"
        elif score >= 0.7:
            return "Mostly Ready"
        elif score >= 0.6:
            return "Needs Work"
        elif score >= 0.4:
            return "Major Work Needed"
        else:
            return "Not Mix Ready"

    def _create_empty_analysis(self, file_path: str) -> Dict:
        """Create empty analysis for files with no analyzable content."""
        return {
            'file_path': file_path,
            'overall_readiness_score': 0.0,
            'readiness_rating': 'Cannot Analyze',
            'detailed_scores': {},
            'recommendations': ['File contains no analyzable musical content'],
            'mixing_checklist': [],
            'estimated_mix_time': 'N/A'
        }


def main():
    parser = argparse.ArgumentParser(description="Mix readiness analysis.")
    parser.add_argument("--input", required=True, help="Input MIDI file.")
    parser.add_argument("--output", default="test_outputs",
                       help="Output directory for results.")

    args = parser.parse_args()

    analyzer = MixReadinessAnalyzer()
    results = analyzer.analyze_mix_readiness(args.input)

    print(f"Mix Readiness Analysis for {os.path.basename(args.input)}")
    print(f"Overall Readiness Score: {results['overall_readiness_score']:.2f}")
    print(f"Readiness Rating: {results['readiness_rating']}")
    print(f"Estimated Mix Time: {results['estimated_mix_time']}")

    print("\nDetailed Scores:")
    for category, analysis in results['detailed_scores'].items():
        print(f"  {category.replace('_', ' ').title()}: {analysis['score']:.2f}")
        print(f"    Assessment: {analysis['assessment']}")
        if analysis.get('issues'):
            print(f"    Issues: {', '.join(analysis['issues'])}")

    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  - {rec}")

    print("\nMixing Checklist:")
    for item in results['mixing_checklist']:
        status = "✓" if item['completed'] else "○"
        print(f"  {status} {item['task']} ({item['priority']} priority)")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "mix_readiness_analysis.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()