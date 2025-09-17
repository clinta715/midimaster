"""
Professional MIDI Metadata Preservation System for MIDI Master.

This module provides comprehensive metadata handling for MIDI files, including
SMPTE synchronization, track naming conventions, DAW compatibility features,
and metadata validation.

Key Features:
- SMPTE timecode synchronization for video/audio alignment
- Professional track naming with DAW compatibility
- Copyright and licensing metadata
- Enhanced markers and text events for DAW navigation
- Metadata validation and error handling
- Configuration persistence for metadata preferences
"""

import os
import json
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido library not available. MIDI metadata will not work.")


class DAWType(Enum):
    """Supported DAW types for compatibility features."""
    GENERIC = "generic"
    ABLETON_LIVE = "ableton_live"
    LOGIC_PRO = "logic_pro"
    PRO_TOOLS = "pro_tools"
    CUBASE = "cubase"
    REAPER = "reaper"


class TrackType(Enum):
    """Standard track types with naming conventions."""
    MELODY = "melody"
    HARMONY = "harmony"
    BASS = "bass"
    RHYTHM = "rhythm"
    DRUMS = "drums"
    PERCUSSION = "percussion"
    LEAD = "lead"
    PAD = "pad"
    FX = "fx"
    VOCALS = "vocals"


@dataclass
class SMPTEConfig:
    """SMPTE timecode configuration for synchronization."""
    frame_rate: int = 30  # 24, 25, 29.97, 30
    start_hour: int = 0
    start_minute: int = 0
    start_second: int = 0
    start_frame: int = 0
    drop_frame: bool = False  # For 29.97/59.94 fps

    def to_midi_smpte_offset(self) -> bytes:
        """Convert to MIDI SMPTE offset format (5 bytes)."""
        # MIDI SMPTE offset: hour|rate, minute, second, frame, subframe
        hour_byte = self.start_hour & 0x1F  # 5 bits for hour
        if self.drop_frame:
            hour_byte |= 0x40  # Set drop frame bit

        # Frame rate encoding in hour byte bits 5-6
        rate_encoding = {
            24: 0x00,
            25: 0x20,
            29.97: 0x40,
            30: 0x60
        }.get(self.frame_rate, 0x60)  # Default to 30 fps

        hour_byte |= rate_encoding

        return bytes([
            hour_byte,
            self.start_minute & 0x3F,  # 6 bits
            self.start_second & 0x3F,  # 6 bits
            self.start_frame & 0x1F,   # 5 bits
            0  # subframe (centiframes)
        ])


@dataclass
class CopyrightInfo:
    """Copyright and licensing metadata."""
    copyright_text: str = ""
    license_type: str = "All Rights Reserved"
    author: str = ""
    composer: str = ""
    arranger: str = ""
    publisher: str = ""
    isrc: str = ""  # International Standard Recording Code
    upc: str = ""   # Universal Product Code


@dataclass
class TrackMetadata:
    """Comprehensive metadata for a single track."""
    name: str
    type: TrackType
    channel: int
    program: int = 0
    bank: int = 0  # For bank select MSB/LSB
    bank_lsb: int = 0
    volume: int = 100  # CC7
    pan: int = 64     # CC10, 64 = center
    mute: bool = False
    solo: bool = False
    comments: List[str] = field(default_factory=list)


@dataclass
class ProjectMetadata:
    """Complete project metadata configuration."""
    project_name: str = ""
    genre: str = ""
    mood: str = ""
    tempo: int = 120
    time_signature: Tuple[int, int] = (4, 4)
    key: str = "C"
    scale: str = "major"
    daw_target: DAWType = DAWType.GENERIC
    smpte: SMPTEConfig = field(default_factory=SMPTEConfig)
    copyright: CopyrightInfo = field(default_factory=CopyrightInfo)
    tracks: Dict[str, TrackMetadata] = field(default_factory=dict)
    markers: List[Tuple[float, str]] = field(default_factory=list)  # (time_beats, text)
    text_events: List[Tuple[float, str]] = field(default_factory=list)  # (time_beats, text)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


class MetadataValidator:
    """Validates MIDI metadata for correctness and compatibility."""

    @staticmethod
    def validate_smpte_config(config: SMPTEConfig) -> List[str]:
        """Validate SMPTE configuration."""
        errors = []

        if config.frame_rate not in [24, 25, 29.97, 30]:
            errors.append(f"Invalid frame rate: {config.frame_rate}. Must be 24, 25, 29.97, or 30")

        for field, value, max_val in [
            ("start_hour", config.start_hour, 23),
            ("start_minute", config.start_minute, 59),
            ("start_second", config.start_second, 59),
            ("start_frame", config.start_frame, config.frame_rate - 1 if config.frame_rate != 29.97 else 29)
        ]:
            if not (0 <= value <= max_val):
                errors.append(f"{field} must be between 0 and {max_val}, got {value}")

        return errors

    @staticmethod
    def validate_track_metadata(track: TrackMetadata) -> List[str]:
        """Validate track metadata."""
        errors = []

        if not track.name.strip():
            errors.append("Track name cannot be empty")

        if len(track.name.encode('utf-8')) > 255:
            errors.append("Track name too long (max 255 bytes)")

        if not (0 <= track.channel <= 15):
            errors.append(f"MIDI channel must be 0-15, got {track.channel}")

        if not (0 <= track.program <= 127):
            errors.append(f"Program number must be 0-127, got {track.program}")

        for cc, name in [
            (track.volume, "volume"),
            (track.pan, "pan")
        ]:
            if not (0 <= cc <= 127):
                errors.append(f"{name} must be 0-127, got {cc}")

        return errors

    @staticmethod
    def validate_project_metadata(metadata: ProjectMetadata) -> List[str]:
        """Validate complete project metadata."""
        errors = []

        # Basic validations
        if metadata.tempo < 20 or metadata.tempo > 300:
            errors.append(f"Tempo must be 20-300 BPM, got {metadata.tempo}")

        if metadata.time_signature[0] < 1 or metadata.time_signature[0] > 64:
            errors.append(f"Time signature numerator must be 1-64, got {metadata.time_signature[0]}")

        if metadata.time_signature[1] not in [1, 2, 4, 8, 16, 32, 64]:
            errors.append(f"Time signature denominator must be power of 2 (1-64), got {metadata.time_signature[1]}")

        # SMPTE validation
        errors.extend(MetadataValidator.validate_smpte_config(metadata.smpte))

        # Track validation
        for track_name, track in metadata.tracks.items():
            track_errors = MetadataValidator.validate_track_metadata(track)
            for error in track_errors:
                errors.append(f"Track '{track_name}': {error}")

        return errors


class MetadataManager:
    """Professional MIDI metadata management system."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the metadata manager."""
        self.config_path = Path(config_path) if config_path else Path("config/metadata_config.json")
        self.default_metadata = self._create_default_metadata()
        self.current_metadata = self.load_config()

    def _create_default_metadata(self) -> ProjectMetadata:
        """Create default project metadata."""
        return ProjectMetadata(
            project_name="MIDI Master Project",
            genre="electronic",
            mood="energetic",
            tempo=120,
            time_signature=(4, 4),
            key="C",
            scale="major",
            tracks={
                "melody": TrackMetadata(
                    name="Melody",
                    type=TrackType.MELODY,
                    channel=0,
                    program=0,  # Grand Piano
                    volume=100,
                    pan=64
                ),
                "harmony": TrackMetadata(
                    name="Harmony",
                    type=TrackType.HARMONY,
                    channel=1,
                    program=0,  # Grand Piano
                    volume=90,
                    pan=64
                ),
                "bass": TrackMetadata(
                    name="Bass",
                    type=TrackType.BASS,
                    channel=2,
                    program=32,  # Acoustic Bass
                    volume=95,
                    pan=64
                ),
                "rhythm": TrackMetadata(
                    name="Rhythm",
                    type=TrackType.RHYTHM,
                    channel=9,  # Drum channel
                    program=0,
                    volume=100,
                    pan=64
                )
            }
        )

    def load_config(self) -> ProjectMetadata:
        """Load metadata configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert dict back to ProjectMetadata
                metadata = ProjectMetadata(**data)

                # Reconstruct nested objects
                metadata.smpte = SMPTEConfig(**data.get('smpte', {}))
                metadata.copyright = CopyrightInfo(**data.get('copyright', {}))

                # Reconstruct tracks
                tracks = {}
                for name, track_data in data.get('tracks', {}).items():
                    tracks[name] = TrackMetadata(**track_data)
                metadata.tracks = tracks

                return metadata
            except Exception as e:
                print(f"Warning: Failed to load metadata config: {e}")
                return copy.deepcopy(self.default_metadata)
        else:
            return copy.deepcopy(self.default_metadata)

    def save_config(self) -> bool:
        """Save current metadata configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict for JSON serialization
            data = {
                'project_name': self.current_metadata.project_name,
                'genre': self.current_metadata.genre,
                'mood': self.current_metadata.mood,
                'tempo': self.current_metadata.tempo,
                'time_signature': list(self.current_metadata.time_signature),
                'key': self.current_metadata.key,
                'scale': self.current_metadata.scale,
                'daw_target': self.current_metadata.daw_target.value,
                'smpte': {
                    'frame_rate': self.current_metadata.smpte.frame_rate,
                    'start_hour': self.current_metadata.smpte.start_hour,
                    'start_minute': self.current_metadata.smpte.start_minute,
                    'start_second': self.current_metadata.smpte.start_second,
                    'start_frame': self.current_metadata.smpte.start_frame,
                    'drop_frame': self.current_metadata.smpte.drop_frame
                },
                'copyright': {
                    'copyright_text': self.current_metadata.copyright.copyright_text,
                    'license_type': self.current_metadata.copyright.license_type,
                    'author': self.current_metadata.copyright.author,
                    'composer': self.current_metadata.copyright.composer,
                    'arranger': self.current_metadata.copyright.arranger,
                    'publisher': self.current_metadata.copyright.publisher,
                    'isrc': self.current_metadata.copyright.isrc,
                    'upc': self.current_metadata.copyright.upc
                },
                'tracks': {
                    name: {
                        'name': track.name,
                        'type': track.type.value,
                        'channel': track.channel,
                        'program': track.program,
                        'bank': track.bank,
                        'bank_lsb': track.bank_lsb,
                        'volume': track.volume,
                        'pan': track.pan,
                        'mute': track.mute,
                        'solo': track.solo,
                        'comments': track.comments
                    }
                    for name, track in self.current_metadata.tracks.items()
                },
                'markers': self.current_metadata.markers,
                'text_events': self.current_metadata.text_events,
                'custom_metadata': self.current_metadata.custom_metadata
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error saving metadata config: {e}")
            return False

    def apply_metadata_to_midi(self, midi_file: 'mido.MidiFile', metadata: ProjectMetadata) -> 'mido.MidiFile':
        """Apply comprehensive metadata to a MIDI file.

        Args:
            midi_file: The MIDI file to enhance with metadata
            metadata: Project metadata to apply

        Returns:
            Enhanced MIDI file with metadata
        """
        if not MIDO_AVAILABLE:
            return midi_file

        # Validate metadata first
        validation_errors = MetadataValidator.validate_project_metadata(metadata)
        if validation_errors:
            print("Metadata validation errors:")
            for error in validation_errors:
                print(f"  - {error}")
            # Continue with best effort, but log warnings

        # Apply SMPTE offset
        self._apply_smpte_offset(midi_file, metadata.smpte)

        # Apply copyright and licensing
        self._apply_copyright_info(midi_file, metadata.copyright)

        # Apply track metadata
        self._apply_track_metadata(midi_file, metadata.tracks, metadata.daw_target)

        # Apply markers and text events
        self._apply_markers_and_text(midi_file, metadata.markers, metadata.text_events)

        # Apply DAW-specific metadata
        self._apply_daw_specific_metadata(midi_file, metadata)

        return midi_file

    def _apply_smpte_offset(self, midi_file: 'mido.MidiFile', smpte: SMPTEConfig) -> None:
        """Apply SMPTE timecode offset to MIDI file."""
        if midi_file.tracks:
            # Add SMPTE offset to the first track
            track = midi_file.tracks[0]
            smpte_data = smpte.to_midi_smpte_offset()

            # Insert at the beginning, after any existing meta messages
            insert_pos = 0
            for i, msg in enumerate(track):
                if msg.type != 'meta':
                    insert_pos = i
                    break
                insert_pos = i + 1

            track.insert(insert_pos, mido.MetaMessage('smpte_offset', data=smpte_data, time=0))

    def _apply_copyright_info(self, midi_file: 'mido.MidiFile', copyright: CopyrightInfo) -> None:
        """Apply copyright and licensing information."""
        if midi_file.tracks:
            track = midi_file.tracks[0]

            # Add copyright notice
            if copyright.copyright_text:
                track.insert(0, mido.MetaMessage('copyright', text=copyright.copyright_text, time=0))

            # Add other metadata as text events
            metadata_events = []
            if copyright.author:
                metadata_events.append(f"Author: {copyright.author}")
            if copyright.composer:
                metadata_events.append(f"Composer: {copyright.composer}")
            if copyright.arranger:
                metadata_events.append(f"Arranger: {copyright.arranger}")
            if copyright.publisher:
                metadata_events.append(f"Publisher: {copyright.publisher}")
            if copyright.isrc:
                metadata_events.append(f"ISRC: {copyright.isrc}")
            if copyright.upc:
                metadata_events.append(f"UPC: {copyright.upc}")
            if copyright.license_type:
                metadata_events.append(f"License: {copyright.license_type}")

            # Add metadata events early in the track
            for i, text in enumerate(metadata_events):
                track.insert(i + 1, mido.MetaMessage('text', text=text, time=0))

    def _apply_track_metadata(self, midi_file: 'mido.MidiFile', tracks: Dict[str, TrackMetadata], daw_target: DAWType) -> None:
        """Apply track-specific metadata to MIDI tracks."""
        for track in midi_file.tracks:
            # Try to identify track by name or channel
            track_meta = self._identify_track_metadata(track, tracks)
            if track_meta:
                self._apply_individual_track_metadata(track, track_meta, daw_target)

    def _identify_track_metadata(self, track: 'mido.MidiTrack', tracks: Dict[str, TrackMetadata]) -> Optional[TrackMetadata]:
        """Identify which metadata applies to a given track."""
        # Look for track name meta message
        track_name = None
        for msg in track:
            if msg.type == 'track_name':
                track_name = msg.text
                break

        if track_name:
            # Try to match by name
            for meta in tracks.values():
                if meta.name.lower() == track_name.lower():
                    return meta

        # Fallback: match by channel
        channel = None
        for msg in track:
            if hasattr(msg, 'channel'):
                channel = msg.channel
                break

        if channel is not None:
            for meta in tracks.values():
                if meta.channel == channel:
                    return meta

        return None

    def _apply_individual_track_metadata(self, track: 'mido.MidiTrack', metadata: TrackMetadata, daw_target: DAWType) -> None:
        """Apply metadata to an individual track."""
        # Add/update track name
        name_added = False
        for i, msg in enumerate(track):
            if msg.type == 'track_name':
                track[i] = mido.MetaMessage('track_name', text=metadata.name, time=msg.time)
                name_added = True
                break

        if not name_added:
            # Add track name at the beginning
            track.insert(0, mido.MetaMessage('track_name', text=metadata.name, time=0))

        # Apply DAW-specific track metadata
        if daw_target == DAWType.ABLETON_LIVE:
            self._apply_ableton_track_metadata(track, metadata)
        elif daw_target == DAWType.LOGIC_PRO:
            self._apply_logic_track_metadata(track, metadata)
        elif daw_target == DAWType.PRO_TOOLS:
            self._apply_protools_track_metadata(track, metadata)

        # Add track comments as text events
        for comment in metadata.comments:
            track.append(mido.MetaMessage('text', text=f"Comment: {comment}", time=0))

    def _apply_daw_specific_metadata(self, midi_file: 'mido.MidiFile', metadata: ProjectMetadata) -> None:
        """Apply DAW-specific metadata to the MIDI file."""
        if metadata.daw_target == DAWType.ABLETON_LIVE:
            self._apply_ableton_file_metadata(midi_file, metadata)
        elif metadata.daw_target == DAWType.LOGIC_PRO:
            self._apply_logic_file_metadata(midi_file, metadata)

    def _apply_ableton_track_metadata(self, track: 'mido.MidiTrack', metadata: TrackMetadata) -> None:
        """Apply Ableton Live specific track metadata."""
        # Ableton uses specific text events for track metadata
        ableton_meta = [
            f"Ableton:TrackType={metadata.type.value}",
            f"Ableton:Volume={metadata.volume}",
            f"Ableton:Pan={metadata.pan}",
            f"Ableton:Mute={1 if metadata.mute else 0}",
            f"Ableton:Solo={1 if metadata.solo else 0}"
        ]

        for meta in ableton_meta:
            track.append(mido.MetaMessage('text', text=meta, time=0))

    def _apply_logic_track_metadata(self, track: 'mido.MidiTrack', metadata: TrackMetadata) -> None:
        """Apply Logic Pro specific track metadata."""
        # Logic uses specific meta events
        logic_meta = [
            f"Logic:TrackType={metadata.type.value}",
            f"Logic:Volume={metadata.volume}",
            f"Logic:Pan={metadata.pan}"
        ]

        for meta in logic_meta:
            track.append(mido.MetaMessage('text', text=meta, time=0))

    def _apply_protools_track_metadata(self, track: 'mido.MidiTrack', metadata: TrackMetadata) -> None:
        """Apply Pro Tools specific track metadata."""
        # Pro Tools uses specific text events
        pt_meta = [
            f"ProTools:TrackType={metadata.type.value}",
            f"ProTools:Volume={metadata.volume}",
            f"ProTools:Pan={metadata.pan}"
        ]

        for meta in pt_meta:
            track.append(mido.MetaMessage('text', text=meta, time=0))

    def _apply_ableton_file_metadata(self, midi_file: 'mido.MidiFile', metadata: ProjectMetadata) -> None:
        """Apply Ableton Live specific file metadata."""
        if midi_file.tracks:
            track = midi_file.tracks[0]
            ableton_file_meta = [
                f"Ableton:ProjectName={metadata.project_name}",
                f"Ableton:Tempo={metadata.tempo}",
                f"Ableton:TimeSignature={metadata.time_signature[0]}/{metadata.time_signature[1]}",
                f"Ableton:Key={metadata.key} {metadata.scale}"
            ]

            for meta in ableton_file_meta:
                track.append(mido.MetaMessage('text', text=meta, time=0))

    def _apply_logic_file_metadata(self, midi_file: 'mido.MidiFile', metadata: ProjectMetadata) -> None:
        """Apply Logic Pro specific file metadata."""
        if midi_file.tracks:
            track = midi_file.tracks[0]
            logic_file_meta = [
                f"Logic:Project={metadata.project_name}",
                f"Logic:Tempo={metadata.tempo}",
                f"Logic:Signature={metadata.time_signature[0]}/{metadata.time_signature[1]}"
            ]

            for meta in logic_file_meta:
                track.append(mido.MetaMessage('text', text=meta, time=0))

    def _apply_markers_and_text(self, midi_file: 'mido.MidiFile', markers: List[Tuple[float, str]], text_events: List[Tuple[float, str]]) -> None:
        """Apply markers and text events to the MIDI file."""
        if not midi_file.tracks:
            return

        track = midi_file.tracks[0]  # Add to first track
        ticks_per_beat = midi_file.ticks_per_beat or 480

        # Add markers
        for time_beats, text in markers:
            time_ticks = int(time_beats * ticks_per_beat)
            # Find insertion point
            insert_pos = len(track)
            for i, msg in enumerate(track):
                if hasattr(msg, 'time') and msg.time > time_ticks:
                    insert_pos = i
                    break

            track.insert(insert_pos, mido.MetaMessage('marker', text=text, time=time_ticks))

        # Add text events
        for time_beats, text in text_events:
            time_ticks = int(time_beats * ticks_per_beat)
            # Find insertion point
            insert_pos = len(track)
            for i, msg in enumerate(track):
                if hasattr(msg, 'time') and msg.time > time_ticks:
                    insert_pos = i
                    break

            track.insert(insert_pos, mido.MetaMessage('text', text=text, time=time_ticks))

    def get_standard_track_name(self, track_type: TrackType, daw_target: DAWType = DAWType.GENERIC) -> str:
        """Get standardized track name for a given type and DAW."""
        base_names = {
            TrackType.MELODY: "Melody",
            TrackType.HARMONY: "Harmony",
            TrackType.BASS: "Bass",
            TrackType.RHYTHM: "Rhythm",
            TrackType.DRUMS: "Drums",
            TrackType.PERCUSSION: "Percussion",
            TrackType.LEAD: "Lead",
            TrackType.PAD: "Pad",
            TrackType.FX: "FX",
            TrackType.VOCALS: "Vocals"
        }

        name = base_names.get(track_type, str(track_type.value).title())

        # DAW-specific naming conventions
        if daw_target == DAWType.ABLETON_LIVE:
            return name
        elif daw_target == DAWType.LOGIC_PRO:
            return name
        elif daw_target == DAWType.PRO_TOOLS:
            return name.upper()
        elif daw_target == DAWType.CUBASE:
            return name

        return name

    def create_project_metadata_from_generation(
        self,
        genre: str,
        mood: str,
        tempo: int,
        time_signature: Tuple[int, int] = (4, 4),
        key: str = "C",
        scale: str = "major",
        daw_target: DAWType = DAWType.GENERIC
    ) -> ProjectMetadata:
        """Create project metadata from generation parameters."""
        metadata = self.current_metadata

        # Update basic project info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata.project_name = f"{genre.title()}_{mood.title()}_{tempo}bpm_{timestamp}"
        metadata.genre = genre
        metadata.mood = mood
        metadata.tempo = tempo
        metadata.time_signature = time_signature
        metadata.key = key
        metadata.scale = scale
        metadata.daw_target = daw_target

        # Update track names for DAW compatibility
        for track_key, track_meta in metadata.tracks.items():
            track_meta.name = self.get_standard_track_name(track_meta.type, daw_target)

        return metadata

    def validate_metadata(self, metadata: ProjectMetadata) -> Tuple[bool, List[str]]:
        """Validate metadata and return validation results."""
        errors = MetadataValidator.validate_project_metadata(metadata)
        return len(errors) == 0, errors

    def export_metadata_template(self, path: str) -> bool:
        """Export a metadata template file for customization."""
        try:
            template = {
                "template_version": "1.0",
                "description": "MIDI Master Metadata Template",
                "example_project": {
                    "project_name": "My Awesome Track",
                    "genre": "electronic",
                    "mood": "energetic",
                    "tempo": 128,
                    "time_signature": [4, 4],
                    "key": "C",
                    "scale": "minor",
                    "daw_target": "ableton_live",
                    "copyright": {
                        "copyright_text": "© 2024 Your Name",
                        "author": "Your Name",
                        "composer": "Your Name"
                    },
                    "tracks": {
                        "custom_melody": {
                            "name": "Custom Melody",
                            "type": "melody",
                            "channel": 0,
                            "program": 81,
                            "volume": 100,
                            "pan": 64
                        }
                    }
                }
            }

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error exporting metadata template: {e}")
            return False