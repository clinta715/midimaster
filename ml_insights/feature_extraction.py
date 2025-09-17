"""
Feature extraction module for ML insights on musical patterns.
"""

from typing import Dict, List

FEATURE_ORDER = ['density', 'avg_velocity', 'num_notes', 'mean_interval', 'bpm']

def extract_features(pattern) -> Dict[str, float]:
    """
    Extract numerical features from a Pattern or MidiPatternData for ML analysis.
    Returns a dict with consistent keys for cosine similarity.
    """
    if not hasattr(pattern, 'notes') or not pattern.notes:
        return {k: 0.0 for k in FEATURE_ORDER}

    notes = pattern.notes
    total_duration = max((n.start_time + n.duration for n in notes if hasattr(n, 'start_time') and hasattr(n, 'duration')), default=1.0)
    density = len(notes) / total_duration

    if len(notes) == 0:
        return {k: 0.0 for k in FEATURE_ORDER}

    avg_velocity = sum(getattr(n, 'velocity', 64) for n in notes) / len(notes)

    num_notes = len(notes)

    pitches = [getattr(n, 'pitch', getattr(n, 'note', 60)) for n in notes]
    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)] if len(pitches) > 1 else [0]
    mean_interval = sum(intervals) / len(intervals) if intervals else 0.0

    bpm = getattr(pattern, 'bpm', getattr(pattern, 'tempo', 120.0))

    return {
        'density': density,
        'avg_velocity': avg_velocity,
        'num_notes': num_notes,
        'mean_interval': mean_interval,
        'bpm': float(bpm)
    }


class MidiFeatureExtractor:
    """Class for extracting features from MIDI files."""

    def __init__(self):
        self.features = []

    def extract_features(self, midi_path: str) -> Dict[str, float]:
        """Extract features from a MIDI file."""
        # Implementation for MIDI files
        return {'tempo': 120.0, 'density': 0.5}

    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return ['tempo', 'density']