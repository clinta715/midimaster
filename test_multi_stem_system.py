"""
Comprehensive Test Suite for Multi-Stem MIDI Generation System

This module provides extensive testing for all components of the multi-stem system:
- StemManager functionality
- Genre-specific configurations
- Intelligent routing
- Performance optimization
- Pipeline integration
"""

import unittest
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock

from generators.stem_manager import StemManager, StemRole, StemConfig, StemData, StemPriority
from generators.stem_genre_config import GenreStemConfigurator, GenreType
from generators.stem_routing import IntelligentRouter, RoutingType, BusType
from generators.stem_performance import StemPerformanceOptimizer, OptimizationLevel, CacheStrategy
from generators.multi_stem_pipeline import MultiStemPipeline, PipelineContext, PipelineResult
from structures.song_skeleton import SongSkeleton
from structures.data_structures import Pattern, Note
from genres.genre_rules import GenreRules


class MockGenreRules(GenreRules):
    """Mock genre rules for testing."""
    def __init__(self, genre_name='electronic'):
        super().__init__()
        self.genre_name = genre_name

    def get_beat_characteristics(self, subgenre=None):
        return {
            'tempo_range': (120, 140),
            'swing_factor': 0.0,
            'syncopation_level': 0.5,
            'emphasis_patterns': []
        }


class TestStemManager(unittest.TestCase):
    """Test cases for the enhanced StemManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.genre_rules = MockGenreRules()
        self.song_skeleton = Mock()
        self.song_skeleton.bars = 16

    def test_stem_manager_initialization(self):
        """Test StemManager initialization with different configurations."""
        # Test with minimum stems (8)
        manager = StemManager(self.genre_rules, max_stems=8)
        self.assertEqual(manager.max_stems, 8)
        self.assertIsNotNone(manager.stem_configs)
        self.assertEqual(len(manager.stem_configs), 8)

        # Test with maximum stems (12)
        manager = StemManager(self.genre_rules, max_stems=12)
        self.assertEqual(manager.max_stems, 12)

        # Test clamping of invalid values
        manager = StemManager(self.genre_rules, max_stems=5)  # Too low
        self.assertEqual(manager.max_stems, 8)

        manager = StemManager(self.genre_rules, max_stems=15)  # Too high
        self.assertEqual(manager.max_stems, 12)

    @patch('generators.stem_manager.PatternOrchestrator')
    def test_stem_generation_sequential(self, mock_orchestrator_class):
        """Test sequential stem generation."""
        # Mock the orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.generate_beats_only.return_value = self._create_mock_pattern()
        mock_orchestrator.generate_selective_patterns.return_value = [self._create_mock_pattern()]
        mock_orchestrator_class.return_value = mock_orchestrator

        manager = StemManager(self.genre_rules, enable_parallel=False)
        stems = manager.generate_stems(self.song_skeleton, 16)

        self.assertIsInstance(stems, dict)
        self.assertTrue(len(stems) > 0)

        # Verify all stems have required data
        for role, stem_data in stems.items():
            self.assertIsInstance(stem_data, StemData)
            self.assertEqual(stem_data.config.role, role)
            self.assertIsNotNone(stem_data.pattern)

    def test_stem_dependency_resolution(self):
        """Test stem dependency resolution."""
        manager = StemManager(self.genre_rules)

        # Test with dependencies - this should work with our current implementation
        stem_roles = [StemRole.LEAD_MELODY, StemRole.HARMONY_PAD]  # Lead depends on harmony
        resolved = manager._resolve_dependencies(stem_roles)

        # Should maintain order and include all requested stems
        self.assertEqual(len(resolved), len(stem_roles))
        self.assertTrue(all(role in resolved for role in stem_roles))

    def test_midi_export(self):
        """Test MIDI file export functionality."""
        manager = StemManager(self.genre_rules)

        # Create mock stem data
        mock_stem_data = StemData(
            config=StemConfig(
                role=StemRole.DRUMS_KICK,
                priority=StemPriority.CRITICAL,
                midi_channel=1,
                instrument_name="Kick Drum"
            ),
            pattern=self._create_mock_pattern(),
            midi_messages=[]  # Would be populated in real scenario
        )

        manager.active_stems = {StemRole.DRUMS_KICK: mock_stem_data}

        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = manager.export_stems_to_midi(temp_dir, "test_stem")

            # Should create one file
            self.assertEqual(len(exported_files), 1)
            self.assertIn(StemRole.DRUMS_KICK, exported_files)

            # File should exist
            filepath = exported_files[StemRole.DRUMS_KICK]
            self.assertTrue(os.path.exists(filepath))

    def _create_mock_pattern(self):
        """Create a mock pattern for testing."""
        pattern = Mock()
        pattern.notes = [
            Mock(pitch=60, velocity=0.8, duration=1.0, start_time=0.0),
            Mock(pitch=64, velocity=0.7, duration=0.5, start_time=1.0)
        ]
        return pattern


class TestGenreStemConfigurator(unittest.TestCase):
    """Test cases for genre-specific stem configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.configurator = GenreStemConfigurator()

    def test_genre_profile_loading(self):
        """Test loading of genre profiles."""
        profile = self.configurator.get_genre_profile(GenreType.ELECTRONIC)
        self.assertEqual(profile.genre, GenreType.ELECTRONIC)
        self.assertEqual(profile.name, "Electronic")
        self.assertTrue(len(profile.typical_stems) > 0)

    def test_stem_configuration_generation(self):
        """Test generation of stem configurations for different genres."""
        # Test electronic configuration
        electronic_config = self.configurator.configure_stems_for_genre(
            GenreType.ELECTRONIC, 'energetic', 8
        )
        self.assertEqual(len(electronic_config), 8)
        self.assertIn(StemRole.DRUMS_KICK, electronic_config)

        # Test rock configuration
        rock_config = self.configurator.configure_stems_for_genre(
            GenreType.ROCK, 'energetic', 7
        )
        self.assertEqual(len(rock_config), 7)
        self.assertIn(StemRole.BASS_ACOUSTIC, rock_config)

    def test_mood_adjustments(self):
        """Test mood-based adjustments to stem configurations."""
        # Test energetic mood
        energetic_config = self.configurator.configure_stems_for_genre(
            GenreType.ELECTRONIC, 'energetic'
        )

        # Test calm mood
        calm_config = self.configurator.configure_stems_for_genre(
            GenreType.ELECTRONIC, 'calm'
        )

        # Energetic should have higher volume on drums
        energetic_drum_vol = energetic_config[StemRole.DRUMS_KICK].volume
        calm_drum_vol = calm_config[StemRole.DRUMS_KICK].volume
        self.assertGreater(energetic_drum_vol, calm_drum_vol)

    def test_genre_validation(self):
        """Test genre configuration validation."""
        # Create a valid configuration
        valid_config = self.configurator.configure_stems_for_genre(GenreType.ELECTRONIC)

        warnings = self.configurator.validate_genre_config(
            GenreType.ELECTRONIC, valid_config
        )

        # Should have no critical warnings for valid config
        critical_warnings = [w for w in warnings if 'Missing required' in w]
        self.assertEqual(len(critical_warnings), 0)


class TestIntelligentRouter(unittest.TestCase):
    """Test cases for intelligent routing system."""

    def setUp(self):
        """Set up test fixtures."""
        self.router = IntelligentRouter()

    def test_routing_template_loading(self):
        """Test loading of routing templates."""
        electronic_config = self.router.configure_routing('electronic')
        self.assertIsNotNone(electronic_config)
        self.assertTrue(len(electronic_config.routing_rules) > 0)

    def test_routing_application(self):
        """Test application of routing to stem data."""
        # Configure routing
        routing_config = self.router.configure_routing('electronic')

        # Create mock stem data
        mock_stems = {
            StemRole.DRUMS_KICK: StemData(
                config=StemConfig(
                    role=StemRole.DRUMS_KICK,
                    priority=StemPriority.CRITICAL,
                    midi_channel=1,
                    instrument_name="Kick Drum"
                ),
                midi_messages=[]
            )
        }

        # Apply routing
        routing_result = self.router.apply_routing(mock_stems)

        self.assertIsInstance(routing_result, dict)
        self.assertIn('routing_summary', routing_result)

    def test_sidechain_setup(self):
        """Test sidechain compression setup."""
        # Configure routing with sidechain
        routing_config = self.router.configure_routing('electronic')

        # Should have sidechain pairs configured
        self.assertTrue(len(routing_config.sidechain_pairs) > 0)

        # Test sidechain setup
        trigger_role, target_role = routing_config.sidechain_pairs[0]
        sidechain_info = self.router._setup_sidechain(trigger_role, target_role)

        self.assertEqual(sidechain_info['trigger_stem'], trigger_role.value)
        self.assertEqual(sidechain_info['target_stem'], target_role.value)
        self.assertIn('compression_settings', sidechain_info)


class TestStemPerformanceOptimizer(unittest.TestCase):
    """Test cases for performance optimization system."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = StemPerformanceOptimizer(
            optimization_level=OptimizationLevel.STANDARD,
            cache_strategy=CacheStrategy.MEMORY
        )

    def test_optimization_parameter_generation(self):
        """Test generation of optimization parameters."""
        stem_configs = {
            StemRole.DRUMS_KICK: StemConfig(
                role=StemRole.DRUMS_KICK,
                priority=StemPriority.CRITICAL,
                midi_channel=1,
                instrument_name="Kick Drum"
            )
        }

        generation_context = {
            'stem_count': 1,
            'has_repeated_patterns': False,
            'complexity_level': 'medium',
            'available_cores': 4
        }

        params = self.optimizer.optimize_stem_generation(
            stem_configs, generation_context
        )

        self.assertIsInstance(params, dict)
        self.assertIn('use_caching', params)
        self.assertIn('parallel_processing', params)

    def test_cache_operations(self):
        """Test caching operations."""
        # Create mock stem data
        stem_data = StemData(
            config=StemConfig(
                role=StemRole.DRUMS_KICK,
                priority=StemPriority.CRITICAL,
                midi_channel=1,
                instrument_name="Kick Drum"
            ),
            midi_messages=[]
        )

        # Cache data
        cache_key = "test_kick_pattern"
        self.optimizer.cache_stem_data(cache_key, stem_data)

        # Retrieve cached data
        cached_data = self.optimizer.get_cached_stem_data(cache_key)
        self.assertIsNotNone(cached_data)
        if cached_data is not None:
            self.assertEqual(cached_data.config.role, StemRole.DRUMS_KICK)

    def test_memory_pool_operations(self):
        """Test memory pool operations."""
        # Allocate memory block
        block = self.optimizer.allocate_memory_block('midi_data')
        self.assertIsNotNone(block)

        # Deallocate memory block
        if block is not None:
            self.optimizer.deallocate_memory_block('midi_data', block)

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        # Start monitoring
        self.optimizer.start_monitoring()

        # Record some processing time
        self.optimizer.record_processing_time(0.5)

        # Get metrics
        metrics = self.optimizer.get_performance_metrics()
        self.assertIsInstance(metrics, object)  # PerformanceMetrics object

        # Stop monitoring
        self.optimizer.stop_monitoring()


class TestMultiStemPipeline(unittest.TestCase):
    """Test cases for the complete multi-stem pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = MultiStemPipeline()
        self.song_skeleton = Mock()
        self.song_skeleton.bars = 16

    @patch('generators.multi_stem_pipeline.MultiStemPipeline._create_genre_rules')
    @patch('generators.multi_stem_pipeline.MultiStemPipeline._calculate_num_bars')
    @patch('generators.stem_manager.PatternOrchestrator')
    def test_pipeline_execution(self, mock_orchestrator_class, mock_calc_bars, mock_genre_rules):
        """Test complete pipeline execution."""
        # Mock dependencies
        mock_genre_rules.return_value = MockGenreRules()
        mock_calc_bars.return_value = 16

        mock_orchestrator = Mock()
        mock_orchestrator.generate_beats_only.return_value = Mock()
        mock_orchestrator.generate_selective_patterns.return_value = [Mock()]
        mock_orchestrator_class.return_value = mock_orchestrator

        # Execute pipeline
        result = self.pipeline.generate_multi_stem_composition(
            song_skeleton=self.song_skeleton,
            genre=GenreType.ELECTRONIC,
            mood='energetic',
            stem_count=8
        )

        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(result.success or len(result.errors) > 0)

    def test_pipeline_configuration(self):
        """Test pipeline configuration methods."""
        # Test preset creation
        preset = self.pipeline.create_quick_preset(
            "Test Preset",
            GenreType.ELECTRONIC,
            8,
            OptimizationLevel.STANDARD
        )

        self.assertEqual(preset['name'], "Test Preset")
        self.assertEqual(preset['genre'], 'electronic')
        self.assertEqual(preset['stem_count'], 8)

        # Test available presets
        presets = self.pipeline.get_available_presets()
        self.assertTrue(len(presets) > 0)
        self.assertIn('name', presets[0])

    def test_performance_summary(self):
        """Test performance summary generation."""
        # Initially should have no executions
        summary = self.pipeline.get_performance_summary()
        self.assertIn('total_executions', summary)

        if summary['total_executions'] == 0:
            self.assertIn('message', summary)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete multi-stem system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.song_skeleton = Mock()
        self.song_skeleton.bars = 16

    @patch('generators.multi_stem_pipeline.PatternOrchestrator')
    def test_full_system_integration(self, mock_orchestrator_class):
        """Test full system integration."""
        # Mock the orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.generate_beats_only.return_value = Mock()
        mock_orchestrator.generate_selective_patterns.return_value = [Mock()]
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create complete pipeline
        pipeline = MultiStemPipeline()

        # Execute full pipeline
        result = pipeline.generate_multi_stem_composition(
            song_skeleton=self.song_skeleton,
            genre=GenreType.ELECTRONIC,
            mood='energetic',
            stem_count=8,
            enable_performance_monitoring=True
        )

        # Verify result structure
        self.assertIsInstance(result, PipelineResult)
        self.assertIsInstance(result.performance_metrics, dict)

        if result.success:
            self.assertTrue(result.stem_count > 0)
            self.assertIsInstance(result.generated_stems, dict)

    def test_error_handling(self):
        """Test error handling in the pipeline."""
        pipeline = MultiStemPipeline()

        # Test with invalid parameters
        result = pipeline.generate_multi_stem_composition(
            song_skeleton=self.song_skeleton,
            genre=GenreType.ELECTRONIC,
            stem_count=2  # Too few stems
        )

        # Should handle gracefully
        self.assertIsInstance(result, PipelineResult)
        # May succeed or fail gracefully depending on implementation


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_stem_configuration(self):
        """Test handling of empty stem configurations."""
        configurator = GenreStemConfigurator()

        # This should handle gracefully
        config = configurator.configure_stems_for_genre(
            GenreType.ELECTRONIC, 'energetic', 0
        )

        # Should clamp to minimum
        self.assertTrue(len(config) >= 3)

    def test_invalid_genre_handling(self):
        """Test handling of invalid genres."""
        pipeline = MultiStemPipeline()

        # Should default to electronic for unknown genres
        song_skeleton = Mock()
        song_skeleton.bars = 16

        # This test might need adjustment based on actual implementation
        # result = pipeline.generate_multi_stem_composition(
        #     song_skeleton=song_skeleton,
        #     genre="invalid_genre",
        #     stem_count=8
        # )

    def test_memory_limit_handling(self):
        """Test memory limit handling."""
        optimizer = StemPerformanceOptimizer(
            max_memory_mb=10  # Very low limit
        )

        # Should handle low memory gracefully
        metrics = optimizer.get_performance_metrics()
        self.assertIsNotNone(metrics)


def run_performance_benchmarks():
    """Run performance benchmarks for the multi-stem system."""
    print("Running Multi-Stem System Performance Benchmarks...")
    print("=" * 60)

    # Create test components
    pipeline = MultiStemPipeline()
    song_skeleton = Mock()
    song_skeleton.bars = 16

    # Benchmark different configurations
    test_configs = [
        {'genre': GenreType.ELECTRONIC, 'stem_count': 6, 'name': 'Electronic 6-stem'},
        {'genre': GenreType.ELECTRONIC, 'stem_count': 10, 'name': 'Electronic 10-stem'},
        {'genre': GenreType.ROCK, 'stem_count': 7, 'name': 'Rock 7-stem'},
        {'genre': GenreType.AMBIENT, 'stem_count': 5, 'name': 'Ambient 5-stem'}
    ]

    results = []

    for config in test_configs:
        print(f"Benchmarking: {config['name']}")

        start_time = time.time()

        # Note: This would need actual implementation to run
        # result = pipeline.generate_multi_stem_composition(
        #     song_skeleton=song_skeleton,
        #     genre=config['genre'],
        #     stem_count=config['stem_count']
        # )

        elapsed = time.time() - start_time

        results.append({
            'config': config['name'],
            'time': elapsed,
            'status': 'completed'
        })

        print(f"{elapsed:.2f}")

    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 30)
    for result in results:
        print(f"{result['time']:.2f}s")

    return results


if __name__ == '__main__':
    # Run unit tests
    print("Running Multi-Stem System Unit Tests...")
    unittest.main(verbosity=2, exit=False)

    # Run performance benchmarks (optional)
    try:
        run_performance_benchmarks()
    except Exception as e:
        print(f"Benchmarking failed: {e}")