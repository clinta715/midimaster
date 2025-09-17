#!/usr/bin/env python3
"""
Unit tests for the Advanced Musical Structure Insights Analyzer
"""

import unittest
import tempfile
import os
from typing import List

import mido

from analyzers.structure_insights import (
    HarmonyAnalyzer, MelodyAnalyzer, FormAnalyzer,
    ComplexityAnalyzer, VoiceLeadingAnalyzer, StructureInsightsAnalyzer,
    MidiNote, Chord, KeyDetection, Motif, Section
)


class TestHarmonyAnalyzer(unittest.TestCase):
    """Test cases for harmony analysis engine."""

    def setUp(self):
        self.analyzer = HarmonyAnalyzer()

    def test_chord_templates_built(self):
        """Test that chord templates are properly built."""
        self.assertGreater(len(self.analyzer.chord_templates), 0)
        # Should have templates for all root notes and chord types
        self.assertIn("C_MAJOR_major", self.analyzer.chord_templates)

    def test_detect_chords_empty_input(self):
        """Test chord detection with empty input."""
        chords = self.analyzer.detect_chords([])
        self.assertEqual(len(chords), 0)

    def test_detect_chords_single_note(self):
        """Test chord detection with single note (should not detect chord)."""
        notes = [MidiNote(60, 64, 0, 100, 0.0, 0.5, 0, 0)]
        chords = self.analyzer.detect_chords(notes)
        self.assertEqual(len(chords), 0)

    def test_detect_chords_major_triad(self):
        """Test chord detection with major triad."""
        # C major triad: C, E, G
        notes = [
            MidiNote(60, 64, 0, 100, 0.0, 0.5, 0, 0),  # C
            MidiNote(64, 64, 0, 100, 0.0, 0.5, 0, 0),  # E
            MidiNote(67, 64, 0, 100, 0.0, 0.5, 0, 0),  # G
        ]
        chords = self.analyzer.detect_chords(notes, window_size=1.0)
        self.assertGreater(len(chords), 0)
        # Should detect some chord
        self.assertIsInstance(chords[0], Chord)

    def test_detect_key_empty_input(self):
        """Test key detection with empty input."""
        keys = self.analyzer.detect_key([])
        self.assertEqual(len(keys), 0)

    def test_detect_key_c_major_notes(self):
        """Test key detection with C major scale notes."""
        # C major scale notes
        c_major_notes = [60, 62, 64, 65, 67, 69, 71]  # C D E F G A B
        notes = []
        for i, pitch in enumerate(c_major_notes):
            notes.append(MidiNote(pitch, 64, i*100, (i+1)*100, i*0.5, (i+1)*0.5, 0, 0))

        keys = self.analyzer.detect_key(notes)
        self.assertGreater(len(keys), 0)
        # Should detect C major as top key
        top_key = keys[0]
        self.assertEqual(self.analyzer._pc_to_note_name(top_key.root_note % 12), 'C')
        self.assertIn('major', top_key.scale_type)


class TestMelodyAnalyzer(unittest.TestCase):
    """Test cases for melody analysis engine."""

    def setUp(self):
        self.analyzer = MelodyAnalyzer()

    def test_detect_motifs_empty_input(self):
        """Test motif detection with empty input."""
        motifs = self.analyzer.detect_motifs([])
        self.assertEqual(len(motifs), 0)

    def test_detect_motifs_short_sequence(self):
        """Test motif detection with short sequence."""
        notes = [MidiNote(60, 64, 0, 100, 0.0, 0.5, 0, 0)]
        motifs = self.analyzer.detect_motifs(notes)
        self.assertEqual(len(motifs), 0)

    def test_calculate_intervals(self):
        """Test interval calculation."""
        pitches = [60, 62, 64, 65]  # C D E F
        intervals = self.analyzer._calculate_intervals(pitches)
        self.assertEqual(intervals, [2, 2, 1])  # Major second, major second, minor second

    def test_analyze_contour_empty_input(self):
        """Test contour analysis with empty input."""
        contour = self.analyzer.analyze_contour([])
        self.assertEqual(len(contour), 0)

    def test_analyze_contour_simple_sequence(self):
        """Test contour analysis with simple note sequence."""
        notes = [
            MidiNote(60, 64, 0, 100, 0.0, 0.5, 0, 0),    # C
            MidiNote(64, 64, 100, 200, 0.5, 1.0, 0, 0),   # E (up)
            MidiNote(62, 64, 200, 300, 1.0, 1.5, 0, 0),   # D (down)
            MidiNote(67, 64, 300, 400, 1.5, 2.0, 0, 0),   # G (up)
        ]
        contour = self.analyzer.analyze_contour(notes, window_size=2.0)
        self.assertGreater(len(contour), 0)

    def test_get_pitch_range_empty_input(self):
        """Test pitch range with empty input."""
        pitch_range = self.analyzer.get_pitch_range([])
        self.assertEqual(pitch_range, (60, 60))  # Default values

    def test_get_pitch_range_normal_input(self):
        """Test pitch range with normal input."""
        notes = [
            MidiNote(60, 64, 0, 100, 0.0, 0.5, 0, 0),  # C
            MidiNote(72, 64, 0, 100, 0.0, 0.5, 0, 0),  # C (octave higher)
        ]
        pitch_range = self.analyzer.get_pitch_range(notes)
        self.assertEqual(pitch_range, (60, 72))


class TestFormAnalyzer(unittest.TestCase):
    """Test cases for form analysis engine."""

    def setUp(self):
        self.analyzer = FormAnalyzer()

    def test_detect_sections_empty_input(self):
        """Test section detection with empty input."""
        sections = self.analyzer.detect_sections([], [])
        self.assertEqual(len(sections), 0)

    def test_detect_sections_with_chords(self):
        """Test section detection with chord changes."""
        chords = [
            Chord([60, 64, 67], 0.0, 2.0, root=60, quality="major"),      # C major
            Chord([60, 64, 67], 2.0, 2.0, root=60, quality="major"),      # C major (same)
            Chord([62, 66, 69], 4.0, 2.0, root=62, quality="minor"),      # D minor (change)
            Chord([69, 72, 76], 6.0, 2.0, root=69, quality="major"),      # A major (change)
        ]
        notes = []  # Empty notes for this test

        sections = self.analyzer.detect_sections(notes, chords)
        self.assertGreater(len(sections), 0)

    def test_detect_repetition_empty_input(self):
        """Test repetition detection with empty input."""
        patterns = self.analyzer.detect_repetition([])
        self.assertEqual(len(patterns), 0)


class TestComplexityAnalyzer(unittest.TestCase):
    """Test cases for complexity analysis engine."""

    def setUp(self):
        self.analyzer = ComplexityAnalyzer()

    def test_calculate_complexity_score_empty_input(self):
        """Test complexity score with empty input."""
        score = self.analyzer.calculate_complexity_score([], [])
        self.assertEqual(score, 0.0)

    def test_calculate_complexity_score_with_chords(self):
        """Test complexity score with chord input."""
        chords = [
            Chord([60, 64, 67], 0.0, 1.0, root=60, quality="major"),
            Chord([62, 66, 69], 1.0, 1.0, root=62, quality="minor"),
            Chord([64, 68, 71], 2.0, 1.0, root=64, quality="major"),
        ]
        key_detections = [KeyDetection(60, "major", 0.8, 3.0)]

        score = self.analyzer.calculate_complexity_score(chords, key_detections)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_analyze_tension_profile_empty_input(self):
        """Test tension profile with empty input."""
        profile = self.analyzer.analyze_tension_profile([], "C major")
        self.assertEqual(len(profile), 0)

    def test_analyze_dissonance_profile_empty_input(self):
        """Test dissonance profile with empty input."""
        profile = self.analyzer.analyze_dissonance_profile([])
        self.assertEqual(len(profile), 0)


class TestVoiceLeadingAnalyzer(unittest.TestCase):
    """Test cases for voice leading analysis engine."""

    def setUp(self):
        self.analyzer = VoiceLeadingAnalyzer()

    def test_analyze_voice_leading_empty_input(self):
        """Test voice leading with empty input."""
        movements = self.analyzer.analyze_voice_leading([])
        self.assertEqual(len(movements), 0)

    def test_analyze_voice_leading_single_chord(self):
        """Test voice leading with single chord."""
        chords = [Chord([60, 64, 67], 0.0, 1.0)]
        movements = self.analyzer.analyze_voice_leading(chords)
        self.assertEqual(len(movements), 0)  # Need at least 2 chords

    def test_analyze_voice_leading_two_chords(self):
        """Test voice leading with two chords."""
        chords = [
            Chord([60, 64, 67], 0.0, 1.0),  # C major
            Chord([62, 66, 69], 1.0, 1.0),  # D minor
        ]
        movements = self.analyzer.analyze_voice_leading(chords)
        self.assertEqual(len(movements), 1)
        self.assertIn("movements", movements[0])

    def test_calculate_motion_scores_empty_input(self):
        """Test motion scores with empty input."""
        parallel, contrary = self.analyzer.calculate_motion_scores([])
        self.assertEqual(parallel, 0.0)
        self.assertEqual(contrary, 0.0)


class TestStructureInsightsAnalyzer(unittest.TestCase):
    """Test cases for main structure insights analyzer."""

    def setUp(self):
        self.analyzer = StructureInsightsAnalyzer()

    def test_analyze_file_missing_file(self):
        """Test analysis of missing file."""
        result = self.analyzer.analyze_file("nonexistent.mid")
        self.assertIsNone(result)

    def test_analyze_file_empty_midi(self):
        """Test analysis of empty MIDI file."""
        # Create a temporary empty MIDI file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            temp_path = f.name
            # Create minimal MIDI file
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            mid.save(temp_path)

        try:
            result = self.analyzer.analyze_file(temp_path)
            self.assertIsNone(result)  # Should return None for file with no notes
        finally:
            os.unlink(temp_path)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""

    def setUp(self):
        self.analyzer = StructureInsightsAnalyzer()

    def test_full_analysis_workflow(self):
        """Test the complete analysis workflow."""
        # Create a simple MIDI file with some notes
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            temp_path = f.name

            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)

            # Add tempo meta-message (120 BPM)
            track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

            # Add some notes (C major chord)
            track.append(mido.Message('note_on', note=60, velocity=64, time=0))    # C
            track.append(mido.Message('note_on', note=64, velocity=64, time=0))    # E
            track.append(mido.Message('note_on', note=67, velocity=64, time=0))    # G
            track.append(mido.Message('note_off', note=60, velocity=0, time=480))  # C off
            track.append(mido.Message('note_off', note=64, velocity=0, time=0))    # E off
            track.append(mido.Message('note_off', note=67, velocity=0, time=0))    # G off
            track.append(mido.MetaMessage('end_of_track', time=0))

            mid.save(temp_path)

        try:
            result = self.analyzer.analyze_file(temp_path)
            self.assertIsNotNone(result)

            # Check that we have some basic results
            self.assertGreaterEqual(len(result.detected_chords), 0)  # type: ignore
            self.assertGreaterEqual(len(result.key_detections), 0)   # type: ignore
            self.assertIsInstance(result.harmonic_complexity_score, float)  # type: ignore

        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()