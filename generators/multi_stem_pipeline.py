"""
Multi-Stem MIDI Generation Pipeline

This module provides a unified pipeline for generating multi-stem MIDI compositions,
integrating all components: StemManager, genre configuration, intelligent routing,
and performance optimization.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import os
from core.filename_templater import format_filename as templ_format

from structures.song_skeleton import SongSkeleton
from structures.data_structures import Pattern
from generators.stem_manager import StemManager, StemRole, StemData, StemConfig
from generators.stem_genre_config import GenreStemConfigurator, GenreType
from generators.stem_routing import IntelligentRouter, RoutingConfig
from generators.stem_performance import StemPerformanceOptimizer, OptimizationLevel, CacheStrategy
from generators.pattern_orchestrator import PatternOrchestrator


class PipelineStage(Enum):
    """Stages in the multi-stem generation pipeline."""
    INITIALIZATION = "initialization"
    CONFIGURATION = "configuration"
    OPTIMIZATION = "optimization"
    STEM_GENERATION = "stem_generation"
    ROUTING = "routing"
    VALIDATION = "validation"
    OUTPUT = "output"


@dataclass
class PipelineContext:
    """Context object for the multi-stem generation pipeline."""
    song_skeleton: SongSkeleton
    genre: GenreType = GenreType.ELECTRONIC
    mood: str = 'energetic'
    stem_count: Optional[int] = None
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    output_directory: str = "output/stems"
    filename_prefix: str = "stem"
    enable_routing: bool = True
    enable_performance_monitoring: bool = True
    quality_settings: Dict[str, float] = field(default_factory=dict)
    # Optional templating inputs (no CLI changes; thread through when provided)
    filename_template: Optional[str] = None
    base_output_dir: Optional[str] = None
    template_settings: Optional[Dict[str, Any]] = None
    template_context: Optional[Dict[str, Any]] = None
 
    # Pipeline state
    current_stage: PipelineStage = PipelineStage.INITIALIZATION
    start_time: float = field(default_factory=time.time)
    stage_times: Dict[str, float] = field(default_factory=dict)
    generated_stems: Dict[StemRole, StemData] = field(default_factory=dict)
    routing_config: Optional[RoutingConfig] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_results: List[str] = field(default_factory=list)
 
    # Additional configuration and state fields
    stem_configs: Dict[StemRole, StemConfig] = field(default_factory=dict)
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    routing_info: Optional[Dict[str, Any]] = None
    output_files: Dict[StemRole, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result object from the multi-stem generation pipeline."""
    success: bool
    generated_stems: Dict[StemRole, StemData]
    output_files: Dict[StemRole, str]
    performance_metrics: Dict[str, Any]
    validation_results: List[str]
    total_processing_time: float
    stem_count: int
    errors: List[str] = field(default_factory=list)


class MultiStemPipeline:
    """
    Unified pipeline for multi-stem MIDI generation.

    This pipeline orchestrates the entire multi-stem generation process:
    1. Configuration and setup
    2. Performance optimization
    3. Parallel stem generation
    4. Intelligent routing and mixing
    5. Validation and output
    """

    def __init__(self,
                 genre_configurator: Optional[GenreStemConfigurator] = None,
                 router: Optional[IntelligentRouter] = None,
                 performance_optimizer: Optional[StemPerformanceOptimizer] = None):
        """
        Initialize the multi-stem pipeline.

        Args:
            genre_configurator: Optional genre configuration system
            router: Optional intelligent routing system
            performance_optimizer: Optional performance optimization system
        """
        self.genre_configurator = genre_configurator or GenreStemConfigurator()
        self.router = router or IntelligentRouter()
        self.performance_optimizer = performance_optimizer or StemPerformanceOptimizer()

        # Pipeline state
        self._active_context: Optional[PipelineContext] = None
        self._pipeline_history: List[PipelineResult] = []

    def generate_multi_stem_composition(self,
                                      song_skeleton: SongSkeleton,
                                      genre: GenreType = GenreType.ELECTRONIC,
                                      mood: str = 'energetic',
                                      stem_count: Optional[int] = None,
                                      **kwargs) -> PipelineResult:
        """
        Generate a complete multi-stem MIDI composition.

        Args:
            song_skeleton: Song structure definition
            genre: Musical genre for the composition
            mood: Emotional mood of the composition
            stem_count: Number of stems to generate (None for genre default)
            **kwargs: Additional pipeline configuration options

        Returns:
            Complete pipeline result with stems, metrics, and validation
        """
        # Create pipeline context
        context = PipelineContext(
            song_skeleton=song_skeleton,
            genre=genre,
            mood=mood,
            stem_count=stem_count,
            **kwargs
        )

        self._active_context = context

        try:
            # Execute pipeline stages
            self._execute_pipeline(context)

            # Create result object
            result = PipelineResult(
                success=True,
                generated_stems=context.generated_stems,
                output_files=self._get_output_files(context),
                performance_metrics=context.performance_metrics,
                validation_results=context.validation_results,
                total_processing_time=time.time() - context.start_time,
                stem_count=len(context.generated_stems)
            )

            # Store in history
            self._pipeline_history.append(result)

            return result

        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"

            result = PipelineResult(
                success=False,
                generated_stems=context.generated_stems,
                output_files={},
                performance_metrics=context.performance_metrics,
                validation_results=context.validation_results,
                total_processing_time=time.time() - context.start_time,
                stem_count=len(context.generated_stems),
                errors=[error_msg]
            )

            return result

        finally:
            # Cleanup resources
            if context.enable_performance_monitoring:
                self.performance_optimizer.cleanup_resources()

    def _execute_pipeline(self, context: PipelineContext) -> None:
        """Execute the complete pipeline workflow."""
        stages = [
            (PipelineStage.CONFIGURATION, self._configure_pipeline),
            (PipelineStage.OPTIMIZATION, self._optimize_pipeline),
            (PipelineStage.STEM_GENERATION, self._generate_stems),
            (PipelineStage.ROUTING, self._apply_routing),
            (PipelineStage.VALIDATION, self._validate_results),
            (PipelineStage.OUTPUT, self._generate_output)
        ]

        for stage, stage_func in stages:
            stage_start = time.time()

            context.current_stage = stage
            stage_func(context)

            stage_time = time.time() - stage_start
            context.stage_times[stage.value] = stage_time

            print(f"Completed {stage.value} stage in {stage_time:.2f}s")

    def _configure_pipeline(self, context: PipelineContext) -> None:
        """Configure the pipeline based on input parameters."""
        # Get genre-specific stem configurations
        stem_configs = self.genre_configurator.configure_stems_for_genre(
            genre=context.genre,
            mood=context.mood,
            stem_count=context.stem_count
        )

        # Store configurations in context for later use
        context.stem_configs = stem_configs

        # Configure routing
        if context.enable_routing:
            context.routing_config = self.router.configure_routing(
                genre=context.genre.value
            )

    def _optimize_pipeline(self, context: PipelineContext) -> None:
        """Apply performance optimizations."""
        if not context.stem_configs:
            return

        # Generate optimization parameters
        optimization_params = self.performance_optimizer.optimize_stem_generation(
            stem_configs=context.stem_configs,
            generation_context={
                'stem_count': len(context.stem_configs),
                'genre': context.genre.value,
                'mood': context.mood,
                'has_repeated_patterns': self._detect_repeated_patterns(context.song_skeleton),
                'complexity_level': self._estimate_complexity(context),
                'available_cores': os.cpu_count() or 4
            }
        )

        # Store optimization parameters
        context.optimization_params = optimization_params

        # Apply caching strategy
        if optimization_params.get('use_caching', False):
            self.performance_optimizer.cache_strategy = context.cache_strategy

    def _generate_stems(self, context: PipelineContext) -> None:
        """Generate all stems for the composition."""
        if not context.stem_configs:
            return

        # Create StemManager with optimized settings
        stem_manager = StemManager(
            genre_rules=self._create_genre_rules(context.genre),
            mood=context.mood,
            max_stems=len(context.stem_configs),
            enable_parallel=context.optimization_params.get('parallel_processing', True),
            memory_limit_mb=512
        )

        # Generate stems
        context.generated_stems = stem_manager.generate_stems(
            song_skeleton=context.song_skeleton,
            num_bars=self._calculate_num_bars(context.song_skeleton),
            stem_roles=list(context.stem_configs.keys())
        )

        # Record performance metrics
        if context.enable_performance_monitoring:
            context.performance_metrics.update(
                stem_manager.get_performance_stats()
            )

        # Cleanup StemManager resources
        stem_manager.cleanup_resources()

    def _apply_routing(self, context: PipelineContext) -> None:
        """Apply intelligent routing and mixing."""
        if not context.enable_routing or not context.routing_config:
            return

        # Apply routing to generated stems
        routing_result = self.router.apply_routing(
            stem_data=context.generated_stems,
            master_level=0.8
        )

        # Store routing information
        context.routing_info = routing_result

    def _validate_results(self, context: PipelineContext) -> None:
        """Validate the generated stems and composition."""
        validation_results = []

        # Validate genre configuration
        if context.stem_configs:
            genre_warnings = self.genre_configurator.validate_genre_config(
                genre=context.genre,
                stem_configs=context.stem_configs
            )
            validation_results.extend(genre_warnings)

        # Validate stem data integrity
        for role, stem_data in context.generated_stems.items():
            if not stem_data.midi_messages:
                validation_results.append(f"Stem {role.value} has no MIDI messages")
            elif stem_data.validation_errors:
                validation_results.extend(
                    f"Stem {role.value}: {error}"
                    for error in stem_data.validation_errors
                )

        # Check routing consistency
        if context.enable_routing and context.routing_info is not None:
            routing_summary = context.routing_info.get('routing_summary', [])
            if len(routing_summary) != len(context.generated_stems):
                validation_results.append("Routing configuration mismatch with generated stems")

        context.validation_results = validation_results

    def _generate_output(self, context: PipelineContext) -> None:
        """Generate final output files."""
        if not context.generated_stems:
            return

        # Create output directory
        os.makedirs(context.output_directory, exist_ok=True)

        # Export stems to MIDI files
        output_files = {}
        for role, stem_data in context.generated_stems.items():
            if stem_data.midi_messages:
                if context.filename_template:
                    # Build settings/context for templater
                    settings = context.template_settings or {
                        "genre": context.genre.value,
                        "mood": context.mood,
                    }
                    ctx = dict(context.template_context or {})
                    ctx.setdefault("stem", role.value)
                    base_dir = context.base_output_dir or context.output_directory
                    out_path = templ_format(context.filename_template, settings, ctx, base_dir=base_dir)
                    filepath = str(out_path)
                else:
                    filename = f"{context.filename_prefix}_{role.value}.mid"
                    filepath = os.path.join(context.output_directory, filename)
                self._export_stem_to_midi(stem_data, filepath)
                output_files[role] = filepath
 
        context.output_files = output_files

    def _create_genre_rules(self, genre: GenreType) -> Any:
        """Create genre rules object for StemManager."""
        # This would integrate with the existing genre rules system
        # For now, return a mock object
        class MockGenreRules:
            def __init__(self, genre):
                self.genre_name = genre.value

            def get_beat_characteristics(self, subgenre=None):
                return {
                    'bpm_range': (120, 140),
                    'swing_amount': 0.0,
                    'beat_complexity': 0.5
                }

        return MockGenreRules(genre)

    def _calculate_num_bars(self, song_skeleton: SongSkeleton) -> int:
        """Calculate the number of bars to generate."""
        # This would analyze the song skeleton structure
        # For now, return a default
        return 16

    def _detect_repeated_patterns(self, song_skeleton: SongSkeleton) -> bool:
        """Detect if the composition has repeated patterns."""
        # Simplified detection - would analyze song structure
        return False

    def _estimate_complexity(self, context: PipelineContext) -> str:
        """Estimate the complexity level of the composition."""
        stem_count = len(getattr(context, 'stem_configs', {}))
        if stem_count > 10:
            return 'very_high'
        elif stem_count > 7:
            return 'high'
        elif stem_count > 4:
            return 'medium'
        else:
            return 'low'

    def _export_stem_to_midi(self, stem_data: StemData, filepath: str) -> None:
        """Export stem data to MIDI file."""
        import mido

        # Create MIDI file
        mid = mido.MidiFile()
        track = mido.MidiTrack()

        # Add track name
        track.append(mido.MetaMessage(
            'track_name',
            name=stem_data.config.instrument_name
        ))

        # Add MIDI messages
        track.extend(stem_data.midi_messages)
        mid.tracks.append(track)

        # Save file
        mid.save(filepath)

    def _get_output_files(self, context: PipelineContext) -> Dict[StemRole, str]:
        """Get output files from context."""
        return getattr(context, 'output_files', {})

    def get_pipeline_history(self) -> List[PipelineResult]:
        """Get the history of pipeline executions."""
        return self._pipeline_history.copy()

    def get_last_result(self) -> Optional[PipelineResult]:
        """Get the result of the last pipeline execution."""
        return self._pipeline_history[-1] if self._pipeline_history else None

    def export_pipeline_config(self, filepath: str) -> None:
        """Export current pipeline configuration to JSON."""
        if not self._active_context:
            raise ValueError("No active pipeline context")

        config_data = {
            'genre': self._active_context.genre.value,
            'mood': self._active_context.mood,
            'stem_count': len(self._active_context.stem_configs),
            'optimization_level': self._active_context.optimization_level.value,
            'cache_strategy': self._active_context.cache_strategy.value,
            'enable_routing': self._active_context.enable_routing,
            'enable_performance_monitoring': self._active_context.enable_performance_monitoring,
            'available_genres': self.genre_configurator.get_available_genres()
        }

        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)

    def import_pipeline_config(self, filepath: str) -> Dict[str, Any]:
        """Import pipeline configuration from JSON."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def create_quick_preset(self,
                          preset_name: str,
                          genre: GenreType,
                          stem_count: int,
                          optimization_level: OptimizationLevel) -> Dict[str, Any]:
        """
        Create a quick preset for common use cases.

        Args:
            preset_name: Name for the preset
            genre: Musical genre
            stem_count: Number of stems
            optimization_level: Performance optimization level

        Returns:
            Preset configuration dictionary
        """
        preset = {
            'name': preset_name,
            'genre': genre.value,
            'stem_count': stem_count,
            'optimization_level': optimization_level.value,
            'cache_strategy': CacheStrategy.MEMORY.value,
            'enable_routing': True,
            'enable_performance_monitoring': optimization_level != OptimizationLevel.MINIMAL,
            'description': f"{genre.value.title()} preset with {stem_count} stems"
        }

        return preset

    def get_available_presets(self) -> List[Dict[str, Any]]:
        """Get list of available quick presets."""
        presets = [
            self.create_quick_preset("Electronic Basic", GenreType.ELECTRONIC, 6, OptimizationLevel.STANDARD),
            self.create_quick_preset("Electronic Full", GenreType.ELECTRONIC, 10, OptimizationLevel.STANDARD),
            self.create_quick_preset("Hip-Hop Boom", GenreType.HIP_HOP, 9, OptimizationLevel.STANDARD),
            self.create_quick_preset("Rock Band", GenreType.ROCK, 7, OptimizationLevel.STANDARD),
            self.create_quick_preset("Ambient Space", GenreType.AMBIENT, 5, OptimizationLevel.STANDARD),
            self.create_quick_preset("High Performance", GenreType.ELECTRONIC, 8, OptimizationLevel.AGGRESSIVE),
            self.create_quick_preset("Maximum Quality", GenreType.ELECTRONIC, 12, OptimizationLevel.MINIMAL)
        ]

        return presets

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent pipeline executions."""
        if not self._pipeline_history:
            return {'message': 'No pipeline executions recorded'}

        recent_results = self._pipeline_history[-10:]  # Last 10 executions

        summary = {
            'total_executions': len(recent_results),
            'success_rate': sum(1 for r in recent_results if r.success) / len(recent_results),
            'average_processing_time': sum(r.total_processing_time for r in recent_results) / len(recent_results),
            'average_stem_count': sum(r.stem_count for r in recent_results) / len(recent_results),
            'total_stems_generated': sum(r.stem_count for r in recent_results),
            'validation_errors': sum(len(r.validation_results) for r in recent_results),
            'performance_trends': self._analyze_performance_trends(recent_results)
        }

        return summary

    def _analyze_performance_trends(self, results: List[PipelineResult]) -> Dict[str, Any]:
        """Analyze performance trends from recent results."""
        if len(results) < 2:
            return {'message': 'Need at least 2 results for trend analysis'}

        # Calculate trends
        times = [r.total_processing_time for r in results]
        time_trend = 'improving' if times[-1] < times[0] else 'degrading'

        stem_counts = [r.stem_count for r in results]
        complexity_trend = 'increasing' if stem_counts[-1] > stem_counts[0] else 'stable'

        return {
            'processing_time_trend': time_trend,
            'complexity_trend': complexity_trend,
            'time_change_percent': ((times[-1] - times[0]) / times[0]) * 100,
            'complexity_change': stem_counts[-1] - stem_counts[0]
        }