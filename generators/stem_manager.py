"""
Enhanced Stem Manager for Multi-Stem MIDI Generation
 
This module provides a comprehensive stem management system that supports:
- 8-12 individual stems per song
- Genre-specific instrument role assignments
- Intelligent track routing and mixing
- Performance optimization features
- Integration with existing generator architecture
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import mido
from core.filename_templater import format_filename as templ_format

from structures.data_structures import Pattern, PatternType
from structures.song_skeleton import SongSkeleton
from generators.pattern_orchestrator import PatternOrchestrator
from genres.genre_rules import GenreRules


class StemRole(Enum):
    """Defines the functional role of each stem in the mix."""
    DRUMS_KICK = "drums_kick"
    DRUMS_SNARE = "drums_snare"
    DRUMS_HATS = "drums_hats"
    DRUMS_PERCUSSION = "drums_percussion"
    DRUMS_808 = "drums_808"
    BASS_SYNTH = "bass_synth"
    BASS_ACOUSTIC = "bass_acoustic"
    LEAD_MELODY = "lead_melody"
    HARMONY_PAD = "harmony_pad"
    HARMONY_PLUCK = "harmony_pluck"
    ATMOSPHERE = "atmosphere"
    FX_PERCUSSION = "fx_percussion"


class StemPriority(Enum):
    """Priority levels for stem processing and mixing."""
    CRITICAL = 3  # Drums, bass - foundation elements
    PRIMARY = 2   # Lead melody, main harmony
    SECONDARY = 1 # Pads, atmospheres, additional percussion
    OPTIONAL = 0  # Effects, additional layers


@dataclass
class StemConfig:
    """Configuration for an individual stem."""
    role: StemRole
    priority: StemPriority
    midi_channel: int
    instrument_name: str
    volume: float = 0.8
    pan: float = 0.0
    enabled: bool = True
    genre_specific: bool = False
    dependencies: Set[StemRole] = field(default_factory=set)
    max_polyphony: int = 16
    velocity_range: Tuple[int, int] = (40, 127)


@dataclass
class StemData:
    """Container for generated stem MIDI data."""
    config: StemConfig
    midi_messages: List[mido.Message] = field(default_factory=list)
    pattern: Optional[Pattern] = None
    processing_time: float = 0.0
    memory_usage: int = 0
    validation_errors: List[str] = field(default_factory=list)


class GenreStemTemplates:
    """Templates for stem configurations by genre."""

    @staticmethod
    def get_electronic_template() -> Dict[StemRole, StemConfig]:
        """Electronic music stem template (8-12 stems)."""
        return {
            StemRole.DRUMS_KICK: StemConfig(
                role=StemRole.DRUMS_KICK,
                priority=StemPriority.CRITICAL,
                midi_channel=1,
                instrument_name="Kick Drum",
                volume=0.9,
                pan=0.0
            ),
            StemRole.DRUMS_SNARE: StemConfig(
                role=StemRole.DRUMS_SNARE,
                priority=StemPriority.CRITICAL,
                midi_channel=2,
                instrument_name="Snare Drum",
                volume=0.85,
                pan=0.0
            ),
            StemRole.DRUMS_HATS: StemConfig(
                role=StemRole.DRUMS_HATS,
                priority=StemPriority.PRIMARY,
                midi_channel=3,
                instrument_name="Hi-Hat",
                volume=0.7,
                pan=0.1
            ),
            StemRole.DRUMS_PERCUSSION: StemConfig(
                role=StemRole.DRUMS_PERCUSSION,
                priority=StemPriority.SECONDARY,
                midi_channel=4,
                instrument_name="Percussion",
                volume=0.6,
                pan=-0.1
            ),
            StemRole.BASS_SYNTH: StemConfig(
                role=StemRole.BASS_SYNTH,
                priority=StemPriority.CRITICAL,
                midi_channel=5,
                instrument_name="Synth Bass",
                volume=0.8,
                pan=0.0
            ),
            StemRole.LEAD_MELODY: StemConfig(
                role=StemRole.LEAD_MELODY,
                priority=StemPriority.PRIMARY,
                midi_channel=6,
                instrument_name="Lead Synth",
                volume=0.75,
                pan=0.2,
                dependencies={StemRole.HARMONY_PAD}
            ),
            StemRole.HARMONY_PAD: StemConfig(
                role=StemRole.HARMONY_PAD,
                priority=StemPriority.SECONDARY,
                midi_channel=7,
                instrument_name="Pad Synth",
                volume=0.6,
                pan=-0.2
            ),
            StemRole.ATMOSPHERE: StemConfig(
                role=StemRole.ATMOSPHERE,
                priority=StemPriority.OPTIONAL,
                midi_channel=8,
                instrument_name="Atmosphere",
                volume=0.4,
                pan=0.0
            )
        }

    @staticmethod
    def get_hip_hop_template() -> Dict[StemRole, StemConfig]:
        """Hip-hop stem template."""
        template = GenreStemTemplates.get_electronic_template()
        # Add 808 bass as separate stem
        template[StemRole.DRUMS_808] = StemConfig(
            role=StemRole.DRUMS_808,
            priority=StemPriority.PRIMARY,
            midi_channel=9,
            instrument_name="808 Bass",
            volume=0.8,
            pan=0.0
        )
        return template

    @staticmethod
    def get_rock_template() -> Dict[StemRole, StemConfig]:
        """Rock music stem template."""
        return {
            StemRole.DRUMS_KICK: StemConfig(
                role=StemRole.DRUMS_KICK,
                priority=StemPriority.CRITICAL,
                midi_channel=1,
                instrument_name="Kick Drum",
                volume=0.9
            ),
            StemRole.DRUMS_SNARE: StemConfig(
                role=StemRole.DRUMS_SNARE,
                priority=StemPriority.CRITICAL,
                midi_channel=2,
                instrument_name="Snare Drum",
                volume=0.85
            ),
            StemRole.DRUMS_HATS: StemConfig(
                role=StemRole.DRUMS_HATS,
                priority=StemPriority.PRIMARY,
                midi_channel=3,
                instrument_name="Cymbal",
                volume=0.75
            ),
            StemRole.BASS_ACOUSTIC: StemConfig(
                role=StemRole.BASS_ACOUSTIC,
                priority=StemPriority.CRITICAL,
                midi_channel=4,
                instrument_name="Bass Guitar",
                volume=0.8
            ),
            StemRole.LEAD_MELODY: StemConfig(
                role=StemRole.LEAD_MELODY,
                priority=StemPriority.PRIMARY,
                midi_channel=5,
                instrument_name="Electric Guitar Lead",
                volume=0.75
            ),
            StemRole.HARMONY_PLUCK: StemConfig(
                role=StemRole.HARMONY_PLUCK,
                priority=StemPriority.SECONDARY,
                midi_channel=6,
                instrument_name="Rhythm Guitar",
                volume=0.7
            ),
            StemRole.ATMOSPHERE: StemConfig(
                role=StemRole.ATMOSPHERE,
                priority=StemPriority.OPTIONAL,
                midi_channel=7,
                instrument_name="Reverb Send",
                volume=0.3
            )
        }


class StemManager:
    """
    Enhanced Stem Manager for multi-stem MIDI generation.
 
    Features:
    - Support for 8-12 stems with intelligent management
    - Genre-specific instrument role assignments
    - Intelligent track routing and mixing
    - Performance optimization with parallel processing
    - Memory management and resource optimization
    - Comprehensive validation and error handling
    """

    def __init__(self,
                 genre_rules: GenreRules,
                 mood: str = 'energetic',
                 max_stems: int = 10,
                 enable_parallel: bool = True,
                 memory_limit_mb: int = 512,
                 pattern_strength: float = 1.0):
        """
        Initialize the StemManager.
 
        Args:
            genre_rules: Genre-specific rules and configurations
            mood: Overall mood for the composition
            max_stems: Maximum number of stems to generate (8-12)
            enable_parallel: Enable parallel stem generation
            memory_limit_mb: Memory limit in MB for stem processing
            pattern_strength: Velocity scaling strength for patterns (0.0-1.0)
        """
        self.genre_rules = genre_rules
        self.mood = mood
        self.max_stems = max(8, min(12, max_stems))  # Clamp between 8-12
        self.enable_parallel = enable_parallel
        self.memory_limit_mb = memory_limit_mb
        self.pattern_strength = pattern_strength

        # Initialize stem configurations based on genre
        self.stem_configs = self._initialize_stem_configs()
        self.active_stems: Dict[StemRole, StemData] = {}

        # Performance tracking
        self.generation_stats = {
            'total_processing_time': 0.0,
            'peak_memory_usage': 0,
            'stems_generated': 0,
            'errors_encountered': 0
        }

        # Threading and resource management
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4) if enable_parallel else None

    def _initialize_stem_configs(self) -> Dict[StemRole, StemConfig]:
        """Initialize stem configurations based on genre."""
        genre_name = getattr(self.genre_rules, 'genre_name', 'electronic').lower()

        if 'hip' in genre_name or 'rap' in genre_name:
            template = GenreStemTemplates.get_hip_hop_template()
        elif 'rock' in genre_name:
            template = GenreStemTemplates.get_rock_template()
        else:
            template = GenreStemTemplates.get_electronic_template()

        # Limit to max_stems by priority
        sorted_stems = sorted(template.items(),
                            key=lambda x: x[1].priority.value,
                            reverse=True)
        return dict(sorted_stems[:self.max_stems])

    def generate_stems(self,
                      song_skeleton: SongSkeleton,
                      num_bars: int,
                      stem_roles: Optional[List[StemRole]] = None) -> Dict[StemRole, StemData]:
        """
        Generate multiple stems for the composition.
 
        Args:
            song_skeleton: Song structure definition
            num_bars: Number of bars to generate
            stem_roles: Specific stem roles to generate (None for all configured)
 
        Returns:
            Dictionary mapping stem roles to their generated data
        """
        start_time = time.time()

        # Determine which stems to generate
        if stem_roles is None:
            target_stems = list(self.stem_configs.keys())
        else:
            target_stems = [role for role in stem_roles if role in self.stem_configs]

        # Resolve dependencies
        target_stems = self._resolve_dependencies(target_stems)

        # Generate stems
        if self.enable_parallel and len(target_stems) > 1:
            generated_stems = self._generate_parallel(target_stems, song_skeleton, num_bars)
        else:
            generated_stems = self._generate_sequential(target_stems, song_skeleton, num_bars)

        # Update performance stats
        self.generation_stats['total_processing_time'] = time.time() - start_time
        self.generation_stats['stems_generated'] = len(generated_stems)

        # Store active stems
        with self._lock:
            self.active_stems.update(generated_stems)

        return generated_stems

    def _resolve_dependencies(self, stem_roles: List[StemRole]) -> List[StemRole]:
        """Resolve stem dependencies and return ordered list."""
        resolved = set()
        to_process = set(stem_roles)

        while to_process:
            # Find stems with no unresolved dependencies
            ready = []
            for role in to_process:
                config = self.stem_configs[role]
                if not config.dependencies or config.dependencies.issubset(resolved):
                    ready.append(role)

            if not ready:
                # Circular dependency or missing dependency
                break

            # Add ready stems to resolved
            resolved.update(ready)
            to_process -= set(ready)

        # Return in dependency order
        result = [role for role in stem_roles if role in resolved]
        result.extend(to_process)  # Add any remaining (problematic) stems

        return result

    def _generate_parallel(self,
                          stem_roles: List[StemRole],
                          song_skeleton: SongSkeleton,
                          num_bars: int) -> Dict[StemRole, StemData]:
        """Generate stems in parallel using thread pool."""
        futures = {}
        results = {}

        # Submit generation tasks
        for role in stem_roles:
            if self._executor is not None:
                future = self._executor.submit(
                    self._generate_single_stem,
                    role, song_skeleton, num_bars
                )
                futures[future] = role
            else:
                # Fallback to sequential if executor is None
                try:
                    stem_data = self._generate_single_stem(role, song_skeleton, num_bars)
                    results[role] = stem_data
                except Exception as e:
                    self.generation_stats['errors_encountered'] += 1
                    results[role] = StemData(
                        config=self.stem_configs[role],
                        validation_errors=[f"Generation failed: {str(e)}"]
                    )

        # Collect results
        for future in as_completed(futures):
            role = futures[future]
            try:
                stem_data = future.result()
                results[role] = stem_data
            except Exception as e:
                self.generation_stats['errors_encountered'] += 1
                # Create error stem data
                results[role] = StemData(
                    config=self.stem_configs[role],
                    validation_errors=[f"Generation failed: {str(e)}"]
                )

        return results

    def _generate_sequential(self,
                            stem_roles: List[StemRole],
                            song_skeleton: SongSkeleton,
                            num_bars: int) -> Dict[StemRole, StemData]:
        """Generate stems sequentially."""
        results = {}

        for role in stem_roles:
            try:
                stem_data = self._generate_single_stem(role, song_skeleton, num_bars)
                results[role] = stem_data
            except Exception as e:
                self.generation_stats['errors_encountered'] += 1
                results[role] = StemData(
                    config=self.stem_configs[role],
                    validation_errors=[f"Generation failed: {str(e)}"]
                )

        return results

    def _generate_single_stem(self,
                            role: StemRole,
                            song_skeleton: SongSkeleton,
                            num_bars: int) -> StemData:
        """Generate a single stem."""
        start_time = time.time()
        config = self.stem_configs[role]

        # Create stem data container
        stem_data = StemData(config=config)

        try:
            # Generate pattern based on stem role
            pattern = self._generate_stem_pattern(role, song_skeleton, num_bars)

            # Convert pattern to MIDI messages
            midi_messages = self._pattern_to_midi(pattern, config)

            # Apply routing and mixing
            midi_messages = self._apply_routing(midi_messages, config)

            # Store results
            stem_data.pattern = pattern
            stem_data.midi_messages = midi_messages
            stem_data.processing_time = time.time() - start_time

            # Validate stem
            self._validate_stem(stem_data)

        except Exception as e:
            stem_data.validation_errors.append(f"Stem generation error: {str(e)}")
            stem_data.processing_time = time.time() - start_time

        return stem_data

    def _generate_stem_pattern(self,
                              role: StemRole,
                              song_skeleton: SongSkeleton,
                              num_bars: int) -> Pattern:
        """Generate pattern for specific stem role."""
        # Create temporary orchestrator for this stem
        orchestrator = PatternOrchestrator(
            genre_rules=self.genre_rules,
            mood=self.mood,
            pattern_strength=self.pattern_strength
        )

        # Map stem role to pattern generation
        if role in [StemRole.DRUMS_KICK, StemRole.DRUMS_SNARE,
                   StemRole.DRUMS_HATS, StemRole.DRUMS_PERCUSSION,
                   StemRole.DRUMS_808]:
            # Drum stems - use rhythm generator
            return orchestrator.generate_beats_only(song_skeleton, num_bars)

        elif role in [StemRole.BASS_SYNTH, StemRole.BASS_ACOUSTIC]:
            # Bass stems
            return orchestrator.generate_selective_patterns(
                song_skeleton, num_bars, ['bass']
            )[0]

        elif role == StemRole.LEAD_MELODY:
            # Lead melody
            return orchestrator.generate_selective_patterns(
                song_skeleton, num_bars, ['melody']
            )[0]

        elif role in [StemRole.HARMONY_PAD, StemRole.HARMONY_PLUCK]:
            # Harmony stems
            return orchestrator.generate_selective_patterns(
                song_skeleton, num_bars, ['harmony']
            )[0]

        else:
            # Default to rhythm pattern for other stems
            return orchestrator.generate_beats_only(song_skeleton, num_bars)

    def _pattern_to_midi(self, pattern: Pattern, config: StemConfig) -> List[mido.Message]:
        """Convert pattern to MIDI messages."""
        messages = []

        for note in pattern.notes:
            # Create note-on message
            note_on = mido.Message(
                'note_on',
                channel=config.midi_channel - 1,  # MIDI channels are 0-based
                note=note.pitch,
                velocity=note.velocity,
                time=0
            )
            print(f"StemManager: Converting note velocity to MIDI - original: {note.velocity}, used: {getattr(note_on, 'velocity', note.velocity)}")
            messages.append(note_on)

            # Create note-off message
            note_off = mido.Message(
                'note_off',
                channel=config.midi_channel - 1,
                note=note.pitch,
                velocity=0,
                time=int(note.duration * 480)  # Assuming PPQ of 480
            )
            messages.append(note_off)

        return messages

    def _apply_routing(self, messages: List[mido.Message], config: StemConfig) -> List[mido.Message]:
        """Apply intelligent routing and mixing to MIDI messages."""
        routed_messages = []

        for msg in messages:
            msg_type = getattr(msg, 'type', None)
            msg_velocity = getattr(msg, 'velocity', 0)
            # All messages are note_on/note_off; scale velocity using copy (immutable)
            if msg_type in ['note_on', 'note_off']:
                scaled_velocity = int(msg_velocity * config.volume)
                # Recreate message since mido.Message is immutable
                new_msg = mido.Message(
                    type=getattr(msg, 'type', None),
                    channel=getattr(msg, 'channel', 0),
                    note=getattr(msg, 'note', 60),
                    velocity=scaled_velocity,
                    time=getattr(msg, 'time', 0)
                )
            else:
                new_msg = msg  # Fallback, though not expected

            routed_messages.append(new_msg)

        return routed_messages

    def _validate_stem(self, stem_data: StemData) -> None:
        """Validate generated stem data."""
        config = stem_data.config

        # Check note velocity range
        if stem_data.midi_messages:
            velocities = []
            for msg in stem_data.midi_messages:
                msg_type = getattr(msg, 'type', None)
                msg_velocity = getattr(msg, 'velocity', 0)
                if msg_type == 'note_on':
                    velocities.append(msg_velocity)
            if velocities:
                min_vel, max_vel = min(velocities), max(velocities)
                if min_vel < config.velocity_range[0] or max_vel > config.velocity_range[1]:
                    stem_data.validation_errors.append(
                        f"Velocity range {min_vel}-{max_vel} outside expected "
                        f"{config.velocity_range[0]}-{config.velocity_range[1]}"
                    )

        # Check for empty stems
        if not stem_data.midi_messages:
            stem_data.validation_errors.append("Stem contains no MIDI messages")

    def export_stems_to_midi(self,
                           output_dir: str,
                           filename_prefix: str = "stem",
                           filename_template: Optional[str] = None,
                           template_settings: Optional[Dict[str, Any]] = None,
                           template_context: Optional[Dict[str, Any]] = None) -> Dict[StemRole, str]:
        """
        Export generated stems to individual MIDI files.
 
        Args:
            output_dir: Directory to save MIDI files
            filename_prefix: Prefix for MIDI filenames
 
        Returns:
            Dictionary mapping stem roles to their file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        exported_files = {}

        for role, stem_data in self.active_stems.items():
            if stem_data.midi_messages:
                # Create MIDI file
                mid = mido.MidiFile()
                track = mido.MidiTrack()

                # Add track name
                track.append(mido.MetaMessage(
                    'track_name',
                    name=stem_data.config.instrument_name
                ))

                # Add messages
                track.extend(stem_data.midi_messages)
                mid.tracks.append(track)

                # Save file
                if filename_template:
                    # Build settings/context for templater
                    settings = template_settings or {
                        "genre": getattr(self.genre_rules, 'genre_name', ''),
                        "mood": self.mood,
                    }
                    ctx = dict(template_context or {})
                    ctx.setdefault("stem", role.value)
                    out_path = templ_format(filename_template, settings, ctx, base_dir=output_dir)
                    filepath = str(out_path)
                else:
                    filename = f"{filename_prefix}_{role.value}.mid"
                    filepath = os.path.join(output_dir, filename)
                mid.save(filepath)
                exported_files[role] = filepath

        return exported_files

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for stem generation."""
        return {
            **self.generation_stats,
            'active_stems': len(self.active_stems),
            'memory_efficiency': self._calculate_memory_efficiency(),
            'parallel_efficiency': self._calculate_parallel_efficiency()
        }

    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency ratio."""
        if not self.active_stems:
            return 1.0

        total_memory = sum(stem.memory_usage for stem in self.active_stems.values())
        return min(1.0, total_memory / (self.memory_limit_mb * 1024 * 1024))

    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel processing efficiency."""
        if not self.enable_parallel or len(self.active_stems) <= 1:
            return 1.0

        # Simple efficiency calculation based on processing times
        processing_times = [stem.processing_time for stem in self.active_stems.values()]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            return avg_time / max_time if max_time > 0 else 1.0

        return 1.0

    def cleanup_resources(self) -> None:
        """Clean up resources and reset manager state."""
        if self._executor:
            self._executor.shutdown(wait=True)

        with self._lock:
            self.active_stems.clear()

        self.generation_stats = {
            'total_processing_time': 0.0,
            'peak_memory_usage': 0,
            'stems_generated': 0,
            'errors_encountered': 0
        }