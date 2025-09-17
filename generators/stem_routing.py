"""
Intelligent Track Routing System for Multi-Stem MIDI Generation

This module provides sophisticated routing capabilities for multi-stem compositions,
including intelligent mixing, effects processing, and signal flow management.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, cast
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from generators.stem_manager import StemRole, StemConfig, StemData


class RoutingType(Enum):
    """Types of routing connections."""
    DIRECT = "direct"         # Direct output to main mix
    SEND = "send"            # Send to effect bus
    GROUP = "group"          # Route to subgroup
    SIDECHAIN = "sidechain"  # Sidechain compression
    PARALLEL = "parallel"    # Parallel processing


class EffectType(Enum):
    """Available effect types for processing."""
    REVERB = "reverb"
    DELAY = "delay"
    CHORUS = "chorus"
    FLANGER = "flanger"
    PHASER = "phaser"
    DISTORTION = "distortion"
    COMPRESSION = "compression"
    EQ = "equalization"
    FILTER = "filter"


class BusType(Enum):
    """Types of audio buses."""
    MAIN = "main"
    DRUMS = "drums"
    BASS = "bass"
    GUITAR = "guitar"
    SYNTH = "synth"
    VOCALS = "vocals"
    PERCUSSION = "percussion"
    ATMOSPHERE = "atmosphere"
    EFFECTS = "effects"


@dataclass
class RoutingRule:
    """A routing rule that defines how a stem connects to buses/effects."""
    stem_role: StemRole
    routing_type: RoutingType
    target_bus: BusType
    level: float = 1.0  # Send level (0.0-1.0)
    pre_fader: bool = False  # Pre-fader send
    conditions: Dict[str, Any] = field(default_factory=dict)  # Conditional routing


@dataclass
class EffectBus:
    """An effects processing bus."""
    bus_type: BusType
    effect_type: EffectType
    parameters: Dict[str, float] = field(default_factory=dict)
    input_stems: Set[StemRole] = field(default_factory=set)
    output_level: float = 1.0
    enabled: bool = True


@dataclass
class MixGroup:
    """A mixing group for related stems."""
    name: str
    member_stems: Set[StemRole]
    master_level: float = 1.0
    master_pan: float = 0.0
    effects: List[EffectBus] = field(default_factory=list)
    subgroup_bus: Optional[BusType] = None


@dataclass
class RoutingConfig:
    """Complete routing configuration for a composition."""
    master_bus: BusType = BusType.MAIN
    routing_rules: List[RoutingRule] = field(default_factory=list)
    effect_buses: List[EffectBus] = field(default_factory=list)
    mix_groups: List[MixGroup] = field(default_factory=list)
    sidechain_pairs: List[Tuple[StemRole, StemRole]] = field(default_factory=list)


class IntelligentRouter:
    """
    Intelligent routing system for multi-stem compositions.

    Features:
    - Automatic routing based on stem types and roles
    - Intelligent mixing groups and subgroups
    - Effects processing buses
    - Sidechain compression setup
    - Dynamic routing adjustments
    """

    def __init__(self):
        self._routing_templates = self._initialize_routing_templates()
        self._current_config: Optional[RoutingConfig] = None

    def _initialize_routing_templates(self) -> Dict[str, RoutingConfig]:
        """Initialize routing templates for different scenarios."""
        return {
            'electronic': self._create_electronic_template(),
            'hip_hop': self._create_hip_hop_template(),
            'rock': self._create_rock_template(),
            'ambient': self._create_ambient_template()
        }

    def _create_electronic_template(self) -> RoutingConfig:
        """Create routing template for electronic music."""
        config = RoutingConfig()

        # Drum group with compression and reverb send
        drum_group = MixGroup(
            name="Drums",
            member_stems={
                StemRole.DRUMS_KICK,
                StemRole.DRUMS_SNARE,
                StemRole.DRUMS_HATS,
                StemRole.DRUMS_PERCUSSION
            },
            master_level=0.9,
            subgroup_bus=BusType.DRUMS
        )

        # Add drum compression
        drum_group.effects.append(EffectBus(
            bus_type=BusType.DRUMS,
            effect_type=EffectType.COMPRESSION,
            parameters={'ratio': 4.0, 'threshold': -18.0, 'attack': 0.1, 'release': 100.0}
        ))

        # Add reverb send for atmosphere
        drum_group.effects.append(EffectBus(
            bus_type=BusType.EFFECTS,
            effect_type=EffectType.REVERB,
            parameters={'decay': 1.5, 'predelay': 0.05, 'wet': 0.3}
        ))

        config.mix_groups.append(drum_group)

        # Routing rules
        config.routing_rules.extend([
            RoutingRule(StemRole.DRUMS_KICK, RoutingType.DIRECT, BusType.MAIN, 0.9),
            RoutingRule(StemRole.DRUMS_SNARE, RoutingType.DIRECT, BusType.MAIN, 0.85),
            RoutingRule(StemRole.DRUMS_HATS, RoutingType.DIRECT, BusType.MAIN, 0.7),
            RoutingRule(StemRole.DRUMS_PERCUSSION, RoutingType.SEND, BusType.EFFECTS, 0.4),
            RoutingRule(StemRole.BASS_SYNTH, RoutingType.DIRECT, BusType.MAIN, 0.8),
            RoutingRule(StemRole.LEAD_MELODY, RoutingType.DIRECT, BusType.MAIN, 0.75),
            RoutingRule(StemRole.HARMONY_PAD, RoutingType.SEND, BusType.EFFECTS, 0.6),
            RoutingRule(StemRole.ATMOSPHERE, RoutingType.DIRECT, BusType.MAIN, 0.4)
        ])

        # Sidechain bass to kick
        config.sidechain_pairs.append((StemRole.BASS_SYNTH, StemRole.DRUMS_KICK))

        return config

    def _create_hip_hop_template(self) -> RoutingConfig:
        """Create routing template for hip-hop music."""
        config = self._create_electronic_template()

        # Add 808 routing
        config.routing_rules.append(
            RoutingRule(StemRole.DRUMS_808, RoutingType.DIRECT, BusType.MAIN, 0.8)
        )

        # Modify sidechain for 808
        config.sidechain_pairs = [(StemRole.DRUMS_808, StemRole.DRUMS_KICK)]

        return config

    def _create_rock_template(self) -> RoutingConfig:
        """Create routing template for rock music."""
        config = RoutingConfig()

        # Guitar group
        guitar_group = MixGroup(
            name="Guitars",
            member_stems={
                StemRole.LEAD_MELODY,
                StemRole.HARMONY_PLUCK
            },
            master_level=0.8,
            subgroup_bus=BusType.GUITAR
        )

        # Add guitar effects
        guitar_group.effects.append(EffectBus(
            bus_type=BusType.GUITAR,
            effect_type=EffectType.DISTORTION,
            parameters={'drive': 0.4, 'tone': 0.7}
        ))

        config.mix_groups.append(guitar_group)

        # Routing rules
        config.routing_rules.extend([
            RoutingRule(StemRole.DRUMS_KICK, RoutingType.DIRECT, BusType.MAIN, 0.9),
            RoutingRule(StemRole.DRUMS_SNARE, RoutingType.DIRECT, BusType.MAIN, 0.85),
            RoutingRule(StemRole.DRUMS_HATS, RoutingType.DIRECT, BusType.MAIN, 0.75),
            RoutingRule(StemRole.BASS_ACOUSTIC, RoutingType.DIRECT, BusType.MAIN, 0.8),
            RoutingRule(StemRole.LEAD_MELODY, RoutingType.DIRECT, BusType.MAIN, 0.8),
            RoutingRule(StemRole.HARMONY_PLUCK, RoutingType.DIRECT, BusType.MAIN, 0.7),
            RoutingRule(StemRole.ATMOSPHERE, RoutingType.SEND, BusType.EFFECTS, 0.5)
        ])

        return config

    def _create_ambient_template(self) -> RoutingConfig:
        """Create routing template for ambient music."""
        config = RoutingConfig()

        # Atmosphere group with heavy effects
        atmosphere_group = MixGroup(
            name="Atmosphere",
            member_stems={
                StemRole.ATMOSPHERE,
                StemRole.HARMONY_PAD
            },
            master_level=0.6,
            subgroup_bus=BusType.ATMOSPHERE
        )

        # Add multiple effects
        atmosphere_group.effects.extend([
            EffectBus(
                bus_type=BusType.ATMOSPHERE,
                effect_type=EffectType.REVERB,
                parameters={'decay': 4.0, 'predelay': 0.2, 'wet': 0.8}
            ),
            EffectBus(
                bus_type=BusType.ATMOSPHERE,
                effect_type=EffectType.DELAY,
                parameters={'time': 0.5, 'feedback': 0.4, 'wet': 0.6}
            )
        ])

        config.mix_groups.append(atmosphere_group)

        # Subtle routing for ambient
        config.routing_rules.extend([
            RoutingRule(StemRole.ATMOSPHERE, RoutingType.DIRECT, BusType.MAIN, 0.4),
            RoutingRule(StemRole.HARMONY_PAD, RoutingType.DIRECT, BusType.MAIN, 0.5),
            RoutingRule(StemRole.DRUMS_PERCUSSION, RoutingType.SEND, BusType.EFFECTS, 0.3),
            RoutingRule(StemRole.LEAD_MELODY, RoutingType.SEND, BusType.EFFECTS, 0.4),
            RoutingRule(StemRole.BASS_SYNTH, RoutingType.DIRECT, BusType.MAIN, 0.6)
        ])

        return config

    def configure_routing(self,
                         genre: str = 'electronic',
                         stem_configs: Optional[Dict[StemRole, StemConfig]] = None,
                         custom_config: Optional[RoutingConfig] = None) -> RoutingConfig:
        """
        Configure routing for a composition.

        Args:
            genre: Music genre for template selection
            stem_configs: Available stem configurations
            custom_config: Custom routing configuration

        Returns:
            Complete routing configuration
        """
        if custom_config:
            self._current_config = custom_config
        else:
            # Use genre template
            template_key = genre.lower().replace(' ', '_')
            if template_key in self._routing_templates:
                self._current_config = self._routing_templates[template_key]
            else:
                # Default to electronic
                self._current_config = self._routing_templates['electronic']

        # Adapt to available stems
        if stem_configs:
            self._adapt_to_available_stems(stem_configs)

        return cast(RoutingConfig, self._current_config)

    def _adapt_to_available_stems(self, stem_configs: Dict[StemRole, StemConfig]) -> None:
        if not self._current_config:
            return
        """Adapt routing configuration to available stems."""
        available_stems = set(stem_configs.keys())

        # Filter routing rules to available stems
        self._current_config.routing_rules = [
            rule for rule in self._current_config.routing_rules
            if rule.stem_role in available_stems
        ]

        # Update mix groups
        for group in self._current_config.mix_groups:
            group.member_stems = group.member_stems.intersection(available_stems)

        # Remove empty groups
        self._current_config.mix_groups = [
            group for group in self._current_config.mix_groups
            if group.member_stems
        ]

        # Update sidechain pairs
        self._current_config.sidechain_pairs = [
            pair for pair in self._current_config.sidechain_pairs
            if pair[0] in available_stems and pair[1] in available_stems
        ]

    def apply_routing(self,
                     stem_data: Dict[StemRole, StemData],
                     master_level: float = 1.0) -> Dict[str, Any]:
        """
        Apply routing configuration to stem data.

        Args:
            stem_data: Generated stem data dictionary
            master_level: Master output level

        Returns:
            Processed mix information
        """
        if not self._current_config:
            raise ValueError("No routing configuration set. Call configure_routing() first.")

        mix_info = {
            'master_level': master_level,
            'bus_levels': {},
            'effect_sends': {},
            'group_levels': {},
            'sidechain_info': [],
            'routing_summary': []
        }

        # Process each stem through routing
        for role, data in stem_data.items():
            routing_info = self._process_stem_routing(role, data, master_level)
            mix_info['routing_summary'].append(routing_info)

        # Process mix groups
        for group in self._current_config.mix_groups:
            group_info = self._process_mix_group(group, stem_data)
            mix_info['group_levels'][group.name] = group_info

        # Set up sidechain compression
        for trigger_role, target_role in self._current_config.sidechain_pairs:
            if trigger_role in stem_data and target_role in stem_data:
                sidechain_info = self._setup_sidechain(trigger_role, target_role)
                mix_info['sidechain_info'].append(sidechain_info)

        return mix_info

    def _process_stem_routing(self,
                            role: StemRole,
                            stem_data: StemData,
                            master_level: float) -> Dict[str, Any]:
        """Process routing for a single stem."""
        routing_info = {
            'stem_role': role.value,
            'direct_level': 0.0,
            'send_levels': {},
            'group_memberships': [],
            'applied_effects': []
        }
        if not self._current_config:
            return routing_info

        # Find routing rules for this stem
        rules = [rule for rule in self._current_config.routing_rules
                if rule.stem_role == role]

        for rule in rules:
            if rule.routing_type == RoutingType.DIRECT:
                routing_info['direct_level'] = rule.level * master_level
            elif rule.routing_type == RoutingType.SEND:
                routing_info['send_levels'][rule.target_bus.value] = rule.level

        # Check mix group membership
        for group in self._current_config.mix_groups:
            if role in group.member_stems:
                routing_info['group_memberships'].append({
                    'group_name': group.name,
                    'group_level': group.master_level,
                    'effects': [effect.effect_type.value for effect in group.effects]
                })

        return routing_info

    def _process_mix_group(self,
                          group: MixGroup,
                          stem_data: Dict[StemRole, StemData]) -> Dict[str, Any]:
        """Process a mix group."""
        group_info = {
            'member_count': len(group.member_stems),
            'active_members': [],
            'master_level': group.master_level,
            'effects_applied': []
        }

        # Check which members are active
        for role in group.member_stems:
            if role in stem_data:
                group_info['active_members'].append(role.value)

        # Apply group effects
        for effect in group.effects:
            if effect.enabled:
                group_info['effects_applied'].append({
                    'effect_type': effect.effect_type.value,
                    'parameters': effect.parameters
                })

        return group_info

    def _setup_sidechain(self,
                        trigger_role: StemRole,
                        target_role: StemRole) -> Dict[str, Any]:
        """Set up sidechain compression between two stems."""
        return {
            'trigger_stem': trigger_role.value,
            'target_stem': target_role.value,
            'compression_settings': {
                'ratio': 8.0,
                'threshold': -20.0,
                'attack': 0.01,
                'release': 0.1,
                'makeup_gain': 3.0
            }
        }

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current routing configuration."""
        if not self._current_config:
            return {}

        stats = {
            'total_routing_rules': len(self._current_config.routing_rules),
            'total_mix_groups': len(self._current_config.mix_groups),
            'total_effect_buses': len(self._current_config.effect_buses),
            'sidechain_pairs': len(self._current_config.sidechain_pairs),
            'bus_usage': {},
            'routing_types': {}
        }

        # Count routing types
        for rule in self._current_config.routing_rules:
            route_type = rule.routing_type.value
            stats['routing_types'][route_type] = stats['routing_types'].get(route_type, 0) + 1

            bus_type = rule.target_bus.value
            stats['bus_usage'][bus_type] = stats['bus_usage'].get(bus_type, 0) + 1

        return stats

    def optimize_routing(self,
                        stem_data: Dict[StemRole, StemData],
                        target_mix_balance: str = 'balanced') -> RoutingConfig:
        """
        Optimize routing configuration based on stem content analysis.

        Args:
            stem_data: Generated stem data for analysis
            target_mix_balance: Target mix balance ('balanced', 'dynamic', 'ambient')

        Returns:
            Optimized routing configuration
        """
        if not self._current_config:
            raise ValueError("No routing configuration to optimize")

        optimized_config = self._current_config

        # Analyze stem characteristics
        stem_analysis = self._analyze_stem_characteristics(stem_data)

        # Apply optimizations based on analysis and target balance
        if target_mix_balance == 'balanced':
            optimized_config = self._optimize_for_balance(stem_analysis)
        elif target_mix_balance == 'dynamic':
            optimized_config = self._optimize_for_dynamics(stem_analysis)
        elif target_mix_balance == 'ambient':
            optimized_config = self._optimize_for_ambience(stem_analysis)

        return optimized_config

    def _analyze_stem_characteristics(self,
                                    stem_data: Dict[StemRole, StemData]) -> Dict[str, Any]:
        """Analyze characteristics of generated stems."""
        analysis = {
            'frequency_content': {},
            'dynamic_range': {},
            'rhythmic_density': {},
            'harmonic_content': {}
        }

        # Basic analysis (simplified for this implementation)
        for role, data in stem_data.items():
            analysis['frequency_content'][role] = self._estimate_frequency_content(role)
            analysis['dynamic_range'][role] = self._estimate_dynamic_range(data)
            analysis['rhythmic_density'][role] = self._estimate_rhythmic_density(role)

        return analysis

    def _estimate_frequency_content(self, role: StemRole) -> str:
        """Estimate frequency content of a stem role."""
        frequency_map = {
            StemRole.DRUMS_KICK: 'low',
            StemRole.DRUMS_SNARE: 'mid_low',
            StemRole.DRUMS_HATS: 'high',
            StemRole.DRUMS_PERCUSSION: 'full',
            StemRole.BASS_SYNTH: 'low',
            StemRole.BASS_ACOUSTIC: 'low_mid',
            StemRole.LEAD_MELODY: 'mid_high',
            StemRole.HARMONY_PAD: 'mid',
            StemRole.HARMONY_PLUCK: 'mid_high',
            StemRole.ATMOSPHERE: 'full'
        }
        return frequency_map.get(role, 'mid')

    def _estimate_dynamic_range(self, stem_data: StemData) -> float:
        """Estimate dynamic range of stem (simplified)."""
        # Simplified estimation - in practice would analyze MIDI velocities
        return 0.7  # Placeholder

    def _estimate_rhythmic_density(self, role: StemRole) -> float:
        """Estimate rhythmic density of stem role."""
        density_map = {
            StemRole.DRUMS_KICK: 0.8,
            StemRole.DRUMS_SNARE: 0.6,
            StemRole.DRUMS_HATS: 0.9,
            StemRole.DRUMS_PERCUSSION: 0.7,
            StemRole.BASS_SYNTH: 0.5,
            StemRole.BASS_ACOUSTIC: 0.4,
            StemRole.LEAD_MELODY: 0.3,
            StemRole.HARMONY_PAD: 0.2,
            StemRole.HARMONY_PLUCK: 0.4,
            StemRole.ATMOSPHERE: 0.1
        }
        return density_map.get(role, 0.5)

        """Optimize routing for balanced mix."""
        # Simplified optimization - adjust levels for better balance
        assert self._current_config is not None
        return cast(RoutingConfig, self._current_config)
    def _optimize_for_balance(self, stem_analysis: Dict[str, Any]) -> RoutingConfig:
        """Optimize routing for balanced mix."""
        # Simplified optimization - adjust levels for better balance
        return cast(RoutingConfig, self._current_config)

    def _optimize_for_dynamics(self, stem_analysis: Dict[str, Any]) -> RoutingConfig:
        """Optimize routing for dynamic mix."""
        return cast(RoutingConfig, self._current_config)
        """Optimize routing for dynamic mix."""
        assert self._current_config is not None
        return cast(RoutingConfig, self._current_config)

        """Optimize routing for ambient mix."""
        assert self._current_config is not None
        return cast(RoutingConfig, self._current_config)
    def _optimize_for_ambience(self, stem_analysis: Dict[str, Any]) -> RoutingConfig:
        """Optimize routing for ambient mix."""
        return cast(RoutingConfig, self._current_config)