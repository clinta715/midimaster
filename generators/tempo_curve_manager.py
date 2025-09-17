"""
Tempo Curve Manager for dynamic tempo management and curve generation.

This module provides sophisticated tempo management capabilities including:
- Dynamic tempo curves with multiple envelope types
- Tempo modulation with user-defined curves
- Real-time tempo adaptation
- BPM range validation and smoothing
- Support for tempo rubato effects
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum


class CurveType(Enum):
    """Types of tempo curves available."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SINUSOIDAL = "sinusoidal"
    EASE_IN_OUT = "ease_in_out"
    CUSTOM = "custom"


class TempoEvent:
    """Represents a tempo change event with timing and curve information."""

    def __init__(self, start_time: float, end_time: float, start_bpm: float,
                 end_bpm: float, curve_type: CurveType = CurveType.LINEAR):
        """
        Initialize a tempo event.

        Args:
            start_time: Start time in beats
            end_time: End time in beats
            start_bpm: Starting BPM
            end_bpm: Ending BPM
            curve_type: Type of curve for interpolation
        """
        self.start_time = start_time
        self.end_time = end_time
        self.start_bpm = start_bpm
        self.end_bpm = end_bpm
        self.curve_type = curve_type
        self.duration = end_time - start_time

    def get_bpm_at_time(self, time: float) -> float:
        """Get the BPM at a specific time within this event."""
        if time <= self.start_time:
            return self.start_bpm
        if time >= self.end_time:
            return self.end_bpm

        # Normalize time to 0-1 range
        t = (time - self.start_time) / self.duration

        # Apply curve interpolation
        if self.curve_type == CurveType.LINEAR:
            factor = t
        elif self.curve_type == CurveType.EXPONENTIAL:
            factor = math.pow(t, 2) if self.end_bpm > self.start_bpm else 1 - math.pow(1 - t, 2)
        elif self.curve_type == CurveType.LOGARITHMIC:
            factor = math.log(1 + 9 * t) / math.log(10) if self.end_bpm > self.start_bpm else t
        elif self.curve_type == CurveType.SINUSOIDAL:
            factor = (math.sin((t - 0.5) * math.pi) + 1) / 2
        elif self.curve_type == CurveType.EASE_IN_OUT:
            factor = 3 * t * t - 2 * t * t * t
        else:  # Default to linear
            factor = t

        # Interpolate between start and end BPM
        return self.start_bpm + (self.end_bpm - self.start_bpm) * factor


class TempoCurve:
    """Manages a sequence of tempo events to create dynamic tempo curves."""

    def __init__(self, base_bpm: float = 120.0):
        """
        Initialize the tempo curve manager.

        Args:
            base_bpm: Base/default BPM for the curve
        """
        self.base_bpm = base_bpm
        self.events: List[TempoEvent] = []
        self._sort_events()

    def add_event(self, event: TempoEvent) -> None:
        """Add a tempo event to the curve."""
        self.events.append(event)
        self._sort_events()

    def remove_event_at_time(self, time: float) -> bool:
        """Remove tempo event at or near the specified time."""
        for i, event in enumerate(self.events):
            if event.start_time <= time <= event.end_time:
                self.events.pop(i)
                return True
        return False

    def get_bpm_at_time(self, time: float) -> float:
        """Get the BPM at a specific time."""
        # Find the applicable tempo event
        for event in reversed(self.events):
            if time >= event.start_time:
                return event.get_bpm_at_time(time)

        # Return base BPM if no events apply
        return self.base_bpm

    def get_tempo_events_in_range(self, start_time: float, end_time: float) -> List[TempoEvent]:
        """Get all tempo events that affect the specified time range."""
        return [event for event in self.events
                if event.end_time > start_time and event.start_time < end_time]

    def _sort_events(self) -> None:
        """Sort events by start time."""
        self.events.sort(key=lambda e: e.start_time)

    def clear_events(self) -> None:
        """Clear all tempo events."""
        self.events.clear()

    def get_average_bpm_in_range(self, start_time: float, end_time: float) -> float:
        """Calculate average BPM over a time range."""
        if not self.events:
            return self.base_bpm

        # Simple average calculation - could be enhanced for more accuracy
        total_bpm = 0
        samples = 10  # Sample points for average calculation
        time_step = (end_time - start_time) / samples

        for i in range(samples):
            sample_time = start_time + i * time_step
            total_bpm += self.get_bpm_at_time(sample_time)

        return total_bpm / samples


class TempoRubatoEngine:
    """Engine for generating tempo rubato (expressive tempo variations)."""

    def __init__(self, intensity: float = 0.1):
        """
        Initialize the rubato engine.

        Args:
            intensity: Intensity of rubato effect (0.0-1.0)
        """
        self.intensity = intensity
        self.last_time = 0.0
        self.rubato_phase = 0.0

    def get_rubato_factor(self, time: float, note_duration: float) -> float:
        """
        Get rubato factor for expressive timing.

        Args:
            time: Current time in beats
            note_duration: Duration of the current note

        Returns:
            Factor to multiply tempo by (typically 0.8-1.2)
        """
        # Simple sinusoidal rubato based on note duration and time
        frequency = 1.0 / max(note_duration, 0.25)  # Faster rubato for shorter notes
        phase = time * frequency * 2 * math.pi + self.rubato_phase

        # Generate rubato factor
        rubato = math.sin(phase) * self.intensity + 1.0

        # Keep within reasonable bounds
        return max(0.7, min(1.3, rubato))

    def reset_phase(self) -> None:
        """Reset the rubato phase for a new section."""
        self.rubato_phase = np.random.uniform(0, 2 * math.pi)


class TempoManager:
    """Main tempo management system integrating all tempo features."""

    def __init__(self, base_bpm: float = 120.0):
        """
        Initialize the tempo manager.

        Args:
            base_bpm: Base/default BPM
        """
        self.base_bpm = base_bpm
        self.curve = TempoCurve(base_bpm)
        self.rubato_engine = TempoRubatoEngine()
        self.time_signature = (4, 4)  # Default 4/4
        self.current_time = 0.0

    def set_time_signature(self, numerator: int, denominator: int) -> None:
        """Set the time signature."""
        self.time_signature = (numerator, denominator)

    def add_tempo_change(self, start_time: float, end_time: float,
                        start_bpm: float, end_bpm: float,
                        curve_type: CurveType = CurveType.LINEAR) -> None:
        """Add a tempo change event."""
        event = TempoEvent(start_time, end_time, start_bpm, end_bpm, curve_type)
        self.curve.add_event(event)

    def create_build_up(self, start_time: float, duration: float,
                       start_bpm: float, peak_bpm: float,
                       curve_type: CurveType = CurveType.EXPONENTIAL) -> None:
        """Create a build-up tempo curve."""
        end_time = start_time + duration
        self.add_tempo_change(start_time, end_time, start_bpm, peak_bpm, curve_type)

    def create_crescendo_then_decelerate(self, start_time: float, duration: float,
                                       start_bpm: float, peak_bpm: float,
                                       end_bpm: float) -> None:
        """Create a complex tempo curve with build-up and slowdown."""
        mid_time = start_time + duration / 2
        self.add_tempo_change(start_time, mid_time, start_bpm, peak_bpm, CurveType.EXPONENTIAL)
        self.add_tempo_change(mid_time, start_time + duration, peak_bpm, end_bpm, CurveType.LOGARITHMIC)

    def get_current_bpm(self, time: Optional[float] = None) -> float:
        """Get the current BPM at the specified time."""
        if time is None:
            time = self.current_time

        bpm = self.curve.get_bpm_at_time(time)

        # Apply rubato if enabled
        if self.rubato_engine.intensity > 0:
            bpm *= self.rubato_engine.get_rubato_factor(time, 0.25)  # Assume quarter note duration

        return bpm

    def advance_time(self, delta_time: float) -> None:
        """Advance the current time."""
        self.current_time += delta_time

    def reset_to_base_tempo(self) -> None:
        """Reset to base tempo and clear all events."""
        self.curve.clear_events()
        self.current_time = 0.0
        self.rubato_engine.reset_phase()

    def get_tempo_profile(self, start_time: float, end_time: float,
                         resolution: int = 100) -> List[Tuple[float, float]]:
        """
        Get a detailed tempo profile over a time range.

        Args:
            start_time: Start time in beats
            end_time: End time in beats
            resolution: Number of sample points

        Returns:
            List of (time, bpm) tuples
        """
        profile = []
        time_step = (end_time - start_time) / resolution

        for i in range(resolution + 1):
            time = start_time + i * time_step
            bpm = self.get_current_bpm(time)
            profile.append((time, bpm))

        return profile

    def export_tempo_changes(self) -> List[Dict]:
        """Export tempo events for serialization."""
        return [{
            'start_time': event.start_time,
            'end_time': event.end_time,
            'start_bpm': event.start_bpm,
            'end_bpm': event.end_bpm,
            'curve_type': event.curve_type.value
        } for event in self.curve.events]

    def import_tempo_changes(self, events_data: List[Dict]) -> None:
        """Import tempo events from serialized data."""
        self.curve.clear_events()
        for event_data in events_data:
            event = TempoEvent(
                event_data['start_time'],
                event_data['end_time'],
                event_data['start_bpm'],
                event_data['end_bpm'],
                CurveType(event_data['curve_type'])
            )
            self.curve.add_event(event)