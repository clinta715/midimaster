"""
Genre-Specific Stem Configuration System

This module provides comprehensive genre-specific instrument role assignments
and configuration templates for the multi-stem MIDI generation system.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json

from generators.stem_manager import StemRole, StemConfig, StemPriority


class GenreType(Enum):
    """Supported music genres with specific stem configurations."""
    ELECTRONIC = "electronic"
    HIP_HOP = "hip_hop"
    TRAP = "trap"
    DNB = "drum_and_bass"
    HOUSE = "house"
    TECHNO = "techno"
    ROCK = "rock"
    POP = "pop"
    JAZZ = "jazz"
    AMBIENT = "ambient"
    EXPERIMENTAL = "experimental"


@dataclass
class GenreStemProfile:
    """Complete stem profile for a specific genre."""
    genre: GenreType
    name: str
    description: str
    typical_stems: List[StemRole]
    required_stems: Set[StemRole]
    recommended_stem_count: int
    bpm_range: tuple[int, int]
    complexity_profile: Dict[str, float]  # mood -> complexity multiplier
    instrument_mappings: Dict[StemRole, Dict[str, Any]] = field(default_factory=dict)


class GenreStemConfigurator:
    """
    Configures stem roles and settings based on musical genre.

    Provides intelligent defaults and genre-appropriate instrument assignments
    for creating authentic multi-stem compositions.
    """

    def __init__(self):
        self._genre_profiles = self._initialize_genre_profiles()
        self._current_profile: Optional[GenreStemProfile] = None

    def _initialize_genre_profiles(self) -> Dict[GenreType, GenreStemProfile]:
        """Initialize all genre profiles with their specific configurations."""
        return {
            GenreType.ELECTRONIC: GenreStemProfile(
                genre=GenreType.ELECTRONIC,
                name="Electronic",
                description="Modern electronic music with synths and beats",
                typical_stems=[
                    StemRole.DRUMS_KICK,
                    StemRole.DRUMS_SNARE,
                    StemRole.DRUMS_HATS,
                    StemRole.DRUMS_PERCUSSION,
                    StemRole.BASS_SYNTH,
                    StemRole.LEAD_MELODY,
                    StemRole.HARMONY_PAD,
                    StemRole.ATMOSPHERE
                ],
                required_stems={
                    StemRole.DRUMS_KICK,
                    StemRole.DRUMS_SNARE,
                    StemRole.BASS_SYNTH
                },
                recommended_stem_count=8,
                bpm_range=(120, 140),
                complexity_profile={
                    'energetic': 1.0,
                    'calm': 0.7,
                    'happy': 0.9,
                    'sad': 0.6
                }
            ),

            GenreType.HIP_HOP: GenreStemProfile(
                genre=GenreType.HIP_HOP,
                name="Hip-Hop",
                description="Hip-hop with boom-bap beats and soul samples",
                typical_stems=[
                    StemRole.DRUMS_KICK,
                    StemRole.DRUMS_SNARE,
                    StemRole.DRUMS_HATS,
                    StemRole.DRUMS_808,
                    StemRole.BASS_SYNTH,
                    StemRole.LEAD_MELODY,
                    StemRole.HARMONY_PLUCK,
                    StemRole.ATMOSPHERE,
                    StemRole.DRUMS_PERCUSSION
                ],
                required_stems={
                    StemRole.DRUMS_KICK,
                    StemRole.DRUMS_SNARE,
                    StemRole.BASS_SYNTH
                },
                recommended_stem_count=9,
                bpm_range=(85, 100),
                complexity_profile={
                    'energetic': 0.8,
                    'calm': 0.9,
                    'happy': 0.7,
                    'sad': 0.8
                }
            ),

            GenreType.DNB: GenreStemProfile(
                genre=GenreType.DNB,
                name="Drum and Bass",
                description="Breakbeat-driven electronic music",
                typical_stems=[
                    StemRole.DRUMS_KICK,
                    StemRole.DRUMS_SNARE,
                    StemRole.DRUMS_PERCUSSION,
                    StemRole.BASS_SYNTH,
                    StemRole.LEAD_MELODY,
                    StemRole.HARMONY_PAD,
                    StemRole.ATMOSPHERE,
                    StemRole.FX_PERCUSSION
                ],
                required_stems={
                    StemRole.DRUMS_KICK,
                    StemRole.DRUMS_SNARE,
                    StemRole.BASS_SYNTH
                },
                recommended_stem_count=8,
                bpm_range=(160, 180),
                complexity_profile={
                    'energetic': 1.2,
                    'calm': 0.8,
                    'happy': 1.0,
                    'sad': 0.7
                }
            ),

            GenreType.ROCK: GenreStemProfile(
                genre=GenreType.ROCK,
                name="Rock",
                description="Classic rock with guitars and live drums",
                typical_stems=[
                    StemRole.DRUMS_KICK,
                    StemRole.DRUMS_SNARE,
                    StemRole.DRUMS_HATS,
                    StemRole.BASS_ACOUSTIC,
                    StemRole.LEAD_MELODY,
                    StemRole.HARMONY_PLUCK,
                    StemRole.ATMOSPHERE
                ],
                required_stems={
                    StemRole.DRUMS_KICK,
                    StemRole.DRUMS_SNARE,
                    StemRole.BASS_ACOUSTIC,
                    StemRole.LEAD_MELODY
                },
                recommended_stem_count=7,
                bpm_range=(100, 160),
                complexity_profile={
                    'energetic': 1.1,
                    'calm': 0.8,
                    'happy': 1.0,
                    'sad': 0.9
                }
            ),

            GenreType.AMBIENT: GenreStemProfile(
                genre=GenreType.AMBIENT,
                name="Ambient",
                description="Atmospheric soundscapes with subtle textures",
                typical_stems=[
                    StemRole.ATMOSPHERE,
                    StemRole.HARMONY_PAD,
                    StemRole.DRUMS_PERCUSSION,
                    StemRole.LEAD_MELODY,
                    StemRole.BASS_SYNTH
                ],
                required_stems={
                    StemRole.ATMOSPHERE,
                    StemRole.HARMONY_PAD
                },
                recommended_stem_count=5,
                bpm_range=(60, 100),
                complexity_profile={
                    'energetic': 0.5,
                    'calm': 1.0,
                    'happy': 0.8,
                    'sad': 0.9
                }
            )
        }

    def get_genre_profile(self, genre: GenreType) -> GenreStemProfile:
        """Get the configuration profile for a specific genre."""
        if genre not in self._genre_profiles:
            # Default to electronic if genre not found
            genre = GenreType.ELECTRONIC
        return self._genre_profiles[genre]

    def configure_stems_for_genre(self,
                                 genre: GenreType,
                                 mood: str = 'energetic',
                                 stem_count: Optional[int] = None) -> Dict[StemRole, StemConfig]:
        """
        Configure stem settings optimized for a specific genre and mood.

        Args:
            genre: Target music genre
            mood: Emotional mood of the composition
            stem_count: Number of stems to configure (None for recommended)

        Returns:
            Dictionary of stem configurations
        """
        profile = self.get_genre_profile(genre)

        # Determine final stem count
        if stem_count is None:
            final_count = profile.recommended_stem_count
        else:
            # Clamp to reasonable range
            final_count = max(3, min(12, stem_count))

        # Get complexity multiplier for mood
        complexity_mult = profile.complexity_profile.get(mood, 1.0)

        # Select stems based on priority and requirements
        selected_stems = self._select_stems_for_genre(profile, final_count)

        # Generate configurations
        stem_configs = {}
        for role in selected_stems:
            config = self._create_genre_specific_config(role, genre, mood, complexity_mult)
            stem_configs[role] = config

        return stem_configs

    def _select_stems_for_genre(self,
                               profile: GenreStemProfile,
                               target_count: int) -> List[StemRole]:
        """Select appropriate stems for the genre within the target count."""
        # Always include required stems
        selected = list(profile.required_stems)

        # Add additional stems from typical list
        remaining_stems = [stem for stem in profile.typical_stems
                          if stem not in profile.required_stems]

        # Sort by priority (required stems first, then by typical order)
        priority_order = {
            StemRole.DRUMS_KICK: 10,
            StemRole.DRUMS_SNARE: 9,
            StemRole.BASS_SYNTH: 8,
            StemRole.BASS_ACOUSTIC: 8,
            StemRole.DRUMS_HATS: 7,
            StemRole.LEAD_MELODY: 6,
            StemRole.HARMONY_PAD: 5,
            StemRole.HARMONY_PLUCK: 5,
            StemRole.ATMOSPHERE: 4,
            StemRole.DRUMS_PERCUSSION: 3,
            StemRole.DRUMS_808: 3,
            StemRole.FX_PERCUSSION: 2
        }

        remaining_stems.sort(key=lambda x: priority_order.get(x, 0), reverse=True)

        # Add stems until we reach target count
        for stem in remaining_stems:
            if len(selected) >= target_count:
                break
            selected.append(stem)

        return selected[:target_count]

    def _create_genre_specific_config(self,
                                    role: StemRole,
                                    genre: GenreType,
                                    mood: str,
                                    complexity_mult: float) -> StemConfig:
        """Create genre-specific stem configuration."""

        # Base configuration varies by stem role and genre
        base_config = self._get_base_config_for_role(role, genre)

        # Adjust for mood and complexity
        adjusted_volume = base_config.volume * complexity_mult
        adjusted_volume = max(0.1, min(1.0, adjusted_volume))  # Clamp to valid range

        return StemConfig(
            role=role,
            priority=base_config.priority,
            midi_channel=base_config.midi_channel,
            instrument_name=base_config.instrument_name,
            volume=adjusted_volume,
            pan=base_config.pan,
            enabled=base_config.enabled,
            genre_specific=True,
            dependencies=base_config.dependencies,
            max_polyphony=base_config.max_polyphony,
            velocity_range=base_config.velocity_range
        )

    def _get_base_config_for_role(self, role: StemRole, genre: GenreType) -> StemConfig:
        """Get base configuration for a stem role in a specific genre."""

        # Default configurations by role (can be overridden by genre specifics)
        role_defaults = {
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
                pan=0.2
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

        # Genre-specific overrides
        genre_overrides = {
            GenreType.ROCK: {
                StemRole.BASS_SYNTH: StemConfig(
                    role=StemRole.BASS_ACOUSTIC,
                    priority=StemPriority.CRITICAL,
                    midi_channel=4,
                    instrument_name="Bass Guitar",
                    volume=0.8,
                    pan=0.0
                ),
                StemRole.LEAD_MELODY: StemConfig(
                    role=StemRole.LEAD_MELODY,
                    priority=StemPriority.PRIMARY,
                    midi_channel=5,
                    instrument_name="Electric Guitar Lead",
                    volume=0.8,
                    pan=0.3
                ),
                StemRole.HARMONY_PLUCK: StemConfig(
                    role=StemRole.HARMONY_PLUCK,
                    priority=StemPriority.SECONDARY,
                    midi_channel=6,
                    instrument_name="Rhythm Guitar",
                    volume=0.7,
                    pan=-0.3
                )
            },

            GenreType.HIP_HOP: {
                StemRole.DRUMS_808: StemConfig(
                    role=StemRole.DRUMS_808,
                    priority=StemPriority.PRIMARY,
                    midi_channel=9,
                    instrument_name="808 Bass",
                    volume=0.8,
                    pan=0.0
                )
            }
        }

        # Apply genre-specific overrides
        if genre in genre_overrides and role in genre_overrides[genre]:
            return genre_overrides[genre][role]

        # Return default configuration
        return role_defaults.get(role, StemConfig(
            role=role,
            priority=StemPriority.SECONDARY,
            midi_channel=10,
            instrument_name=str(role.value).replace('_', ' ').title(),
            volume=0.7,
            pan=0.0
        ))

    def get_available_genres(self) -> List[Dict[str, Any]]:
        """Get list of available genres with their descriptions."""
        return [
            {
                'id': profile.genre.value,
                'name': profile.name,
                'description': profile.description,
                'recommended_stems': profile.recommended_stem_count,
                'bpm_range': profile.bpm_range
            }
            for profile in self._genre_profiles.values()
        ]

    def validate_genre_config(self,
                            genre: GenreType,
                            stem_configs: Dict[StemRole, StemConfig]) -> List[str]:
        """Validate stem configuration for a specific genre."""
        warnings = []
        profile = self.get_genre_profile(genre)

        # Check for required stems
        configured_stems = set(stem_configs.keys())
        missing_required = profile.required_stems - configured_stems
        if missing_required:
            warnings.append(f"Missing required stems: {[s.value for s in missing_required]}")

        # Check stem count
        if len(stem_configs) < 3:
            warnings.append("Very low stem count may result in sparse mix")
        elif len(stem_configs) > 12:
            warnings.append("High stem count may cause performance issues")

        # Check for conflicting MIDI channels
        channels = [config.midi_channel for config in stem_configs.values()]
        if len(channels) != len(set(channels)):
            warnings.append("Duplicate MIDI channels detected")

        return warnings