"""
Multi-Stem MIDI Generation System Demo

This script demonstrates the complete multi-stem MIDI generation system with:
- Enhanced StemManager with 8-12 stems support
- Genre-specific instrument role assignments
- Intelligent track routing and mixing
- Performance optimization features
- Complete pipeline integration

Usage examples:
    python examples/multi_stem_demo.py --genre electronic --stems 10 --mood energetic
    python examples/multi_stem_demo.py --genre rock --stems 8 --mood calm
    python examples/multi_stem_demo.py --genre ambient --stems 6 --mood sad
"""

import argparse
import os
import time
import json
from pathlib import Path

# Import multi-stem system components
from generators.multi_stem_pipeline import MultiStemPipeline
from generators.stem_genre_config import GenreType
from generators.stem_performance import OptimizationLevel
from structures.song_skeleton import SongSkeleton


def create_demo_song_skeleton(bars: int = 16, bpm: int = 128) -> SongSkeleton:
    """Create a demo song skeleton for testing."""
    from structures.data_structures import SectionType, Pattern
    skeleton = SongSkeleton(genre="electronic", tempo=bpm, mood="energetic")
    skeleton.title = "Multi-Stem Demo Composition"
    skeleton.bars = bars
    skeleton.tempo = bpm
    skeleton.key = "C"
    skeleton.scale = "major"
    skeleton.sections = [
        (SectionType.INTRO, []),
        (SectionType.PRE_CHORUS, []),
        (SectionType.CHORUS, []),
        (SectionType.OUTRO, [])
    ]
    return skeleton


def demonstrate_basic_usage():
    """Demonstrate basic multi-stem generation."""
    print("🎵 Multi-Stem MIDI Generation Demo")
    print("=" * 50)

    # Create pipeline
    pipeline = MultiStemPipeline()

    # Create song skeleton
    song_skeleton = create_demo_song_skeleton()

    print(f"📋 Song: {song_skeleton.title}")
    print(f"🎼 Structure: {song_skeleton.bars} bars at {song_skeleton.bpm} BPM")
    print()

    # Generate electronic composition
    print("🎹 Generating Electronic Composition...")
    start_time = time.time()

    result = pipeline.generate_multi_stem_composition(
        song_skeleton=song_skeleton,
        genre=GenreType.ELECTRONIC,
        mood='energetic',
        stem_count=10,
        enable_performance_monitoring=True
    )

    generation_time = time.time() - start_time

    # Display results
    if result.success:
        print(f"✅ Generation successful in {generation_time:.2f}s")
        print(f"🎛️  Generated {result.stem_count} stems:")
        for role, stem_data in result.generated_stems.items():
            midi_count = len(stem_data.midi_messages)
            print(f"   • {role.value}: {midi_count} MIDI messages")

        print(f"💾 Output files: {len(result.output_files)} MIDI files created")

        # Show performance metrics
        if result.performance_metrics:
            print("\n📊 Performance Metrics:")
            for key, value in result.performance_metrics.items():
                if isinstance(value, float):
                    print(f"   • {key}: {value:.2f}")
                else:
                    print(f"   • {key}: {value}")

        # Note: Routing information not available in current PipelineResult
        # Would display stem routing levels here if implemented

    else:
        print(f"❌ Generation failed: {', '.join(result.errors)}")

    print()
    return result


def demonstrate_genre_comparison():
    """Demonstrate different genres with the same parameters."""
    print("🎸 Genre Comparison Demo")
    print("=" * 30)

    pipeline = MultiStemPipeline()
    song_skeleton = create_demo_song_skeleton(bars=8)  # Shorter for comparison

    genres_to_test = [
        (GenreType.ELECTRONIC, 'energetic'),
        (GenreType.ROCK, 'energetic'),
        (GenreType.AMBIENT, 'calm'),
        (GenreType.HIP_HOP, 'energetic')
    ]

    results = {}

    for genre, mood in genres_to_test:
        print(f"🎹 Testing {genre.value} genre...")
        start_time = time.time()

        result = pipeline.generate_multi_stem_composition(
            song_skeleton=song_skeleton,
            genre=genre,
            mood=mood,
            stem_count=8,
            output_directory=f"output/demo_{genre.value}"
        )

        elapsed = time.time() - start_time

        if result.success:
            results[genre.value] = {
                'stems': result.stem_count,
                'time': elapsed,
                'files': len(result.output_files)
            }
            print(f"   ✅ Success: {elapsed:.2f}s")
        else:
            print(f"   ❌ Failed: {result.errors[0] if result.errors else 'Unknown error'}")

    print("\n📊 Genre Comparison Results:")
    print("-" * 40)
    for genre, data in results.items():
        print(f"{genre:12} | {data['stems']:2} stems | {data['time']:5.2f}s | {data['files']:2} files")


def demonstrate_performance_optimization():
    """Demonstrate performance optimization features."""
    print("⚡ Performance Optimization Demo")
    print("=" * 35)

    pipeline = MultiStemPipeline()
    song_skeleton = create_demo_song_skeleton(bars=12)

    optimization_levels = [
        OptimizationLevel.MINIMAL,
        OptimizationLevel.STANDARD,
        OptimizationLevel.AGGRESSIVE
    ]

    print("Testing different optimization levels...")
    print("-" * 50)

    for opt_level in optimization_levels:
        print(f"🔧 Testing {opt_level.value} optimization...")

        # Create pipeline with specific optimization
        from generators.stem_performance import StemPerformanceOptimizer, CacheStrategy
        from generators.stem_routing import IntelligentRouter
        from generators.stem_genre_config import GenreStemConfigurator

        optimizer = StemPerformanceOptimizer(
            optimization_level=opt_level,
            cache_strategy=CacheStrategy.MEMORY
        )
        router = IntelligentRouter()
        configurator = GenreStemConfigurator()

        test_pipeline = MultiStemPipeline(configurator, router, optimizer)

        start_time = time.time()
        result = test_pipeline.generate_multi_stem_composition(
            song_skeleton=song_skeleton,
            genre=GenreType.ELECTRONIC,
            mood='energetic',
            stem_count=8,
            enable_performance_monitoring=True
        )
        elapsed = time.time() - start_time

        if result.success:
            print(f"   ✅ Success: {elapsed:.2f}s")
            if result.performance_metrics:
                memory_usage = result.performance_metrics.get('memory_usage_mb', 0)
                stems_per_sec = result.performance_metrics.get('stems_generated_per_second', 0)
                print(f"   • Memory: {memory_usage:.1f} MB")
                print(f"   • Stems/sec: {stems_per_sec:.1f}")
            else:
                print(f"   ❌ Failed: {result.errors[0] if result.errors else 'Unknown error'}")


def demonstrate_presets():
    """Demonstrate quick presets for common use cases."""
    print("🎛️  Quick Presets Demo")
    print("=" * 25)

    pipeline = MultiStemPipeline()
    song_skeleton = create_demo_song_skeleton(bars=8)

    # Show available presets
    presets = pipeline.get_available_presets()
    print(f"📋 Available presets: {len(presets)}")
    for i, preset in enumerate(presets, 1):
        print(f"   {i}. {preset['name']} - {preset['description']}")

    print("\n🎵 Using 'Electronic Full' preset...")
    # Find and use the electronic full preset
    electronic_preset = next(
        (p for p in presets if 'Electronic Full' in p['name']), None
    )

    if electronic_preset:
        # Create pipeline with preset parameters
        result = pipeline.generate_multi_stem_composition(
            song_skeleton=song_skeleton,
            genre=GenreType(electronic_preset['genre']),
            mood='energetic',
            stem_count=electronic_preset['stem_count'],
            output_directory="output/preset_demo"
        )

        if result.success:
            print("✅ Preset generation successful!")
            print(f"   Generated {result.stem_count} stems")
            print(f"   Created {len(result.output_files)} MIDI files")
        else:
            print(f"❌ Preset generation failed: {result.errors[0] if result.errors else 'Unknown error'}")
    else:
        print("❌ Could not find electronic preset")


def demonstrate_custom_configuration():
    """Demonstrate custom configuration options."""
    print("🔧 Custom Configuration Demo")
    print("=" * 32)

    from generators.stem_genre_config import GenreStemConfigurator
    from generators.stem_routing import IntelligentRouter
    from generators.stem_performance import StemPerformanceOptimizer, CacheStrategy

    # Create custom components
    configurator = GenreStemConfigurator()
    router = IntelligentRouter()
    optimizer = StemPerformanceOptimizer(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        cache_strategy=CacheStrategy.MEMORY,
        max_memory_mb=256
    )

    # Create pipeline with custom components
    pipeline = MultiStemPipeline(configurator, router, optimizer)

    song_skeleton = create_demo_song_skeleton(bars=16, bpm=140)

    # Custom generation parameters
    custom_params = {
        'enable_routing': True,
        'enable_performance_monitoring': True,
        'output_directory': 'output/custom_demo',
        'filename_prefix': 'custom_stem',
        'quality_settings': {
            'pattern_complexity': 0.9,
            'velocity_resolution': 0.95,
            'timing_precision': 0.98
        }
    }

    print("🎹 Generating with custom configuration...")
    print(f"   • Genre: Electronic (custom optimized)")
    print(f"   • Stems: 12 (maximum)")
    print(f"   • BPM: {song_skeleton.bpm}")
    print(f"   • Optimization: Aggressive")
    print(f"   • Memory limit: 256MB")

    start_time = time.time()
    result = pipeline.generate_multi_stem_composition(
        song_skeleton=song_skeleton,
        genre=GenreType.ELECTRONIC,
        mood='energetic',
        stem_count=12,  # Maximum stems
        **custom_params
    )
    elapsed = time.time() - start_time

    if result.success:
        print("\n✅ Custom generation successful!")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Generated {result.stem_count} high-quality stems")
        if result.performance_metrics:
            print("\n📊 Performance:")
            memory_peak = result.performance_metrics.get('memory_usage_mb', 0)
            print(f"   • Memory Peak: {memory_peak:.1f} MB")
            cache_hit_rate = result.performance_metrics.get('cache_hit_rate', 0)
            print(f"   • Cache Hit Rate: {cache_hit_rate:.1%}")
        else:
            print(f"❌ Custom generation failed: {result.errors[0] if result.errors else 'Unknown error'}")


def save_demo_results(results, output_dir: str = "output/demo_results"):
    """Save demo results for analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    summary = {
        'demo_type': 'multi_stem_generation',
        'timestamp': time.time(),
        'results': []
    }

    if hasattr(results, 'stem_count'):
        summary['results'].append({
            'type': 'single_generation',
            'stem_count': results.stem_count,
            'success': results.success,
            'processing_time': results.total_processing_time,
            'output_files': len(results.output_files) if results.output_files else 0
        })

    filepath = os.path.join(output_dir, "demo_summary.json")
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Demo results saved to: {filepath}")


def main():
    """Main demo function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Stem MIDI Generation System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/multi_stem_demo.py --genre electronic --stems 10 --mood energetic
  python examples/multi_stem_demo.py --genre rock --stems 8 --mood calm
  python examples/multi_stem_demo.py --all  # Run all demonstrations
        """
    )

    parser.add_argument('--genre', choices=['electronic', 'rock', 'hip_hop', 'ambient'],
                       default='electronic', help='Music genre')
    parser.add_argument('--stems', type=int, choices=range(3, 13), default=10,
                       help='Number of stems (3-12)')
    parser.add_argument('--mood', choices=['energetic', 'calm', 'happy', 'sad'],
                       default='energetic', help='Emotional mood')
    parser.add_argument('--bars', type=int, default=16,
                       help='Number of bars to generate')
    parser.add_argument('--bpm', type=int, default=128,
                       help='Tempo in BPM')
    parser.add_argument('--output', default='output/demo',
                       help='Output directory')
    parser.add_argument('--all', action='store_true',
                       help='Run all demonstrations')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    if args.all:
        # Run all demonstrations
        print("🚀 Running complete multi-stem system demonstration...\n")

        # Basic usage demo
        result1 = demonstrate_basic_usage()

        print("\n" + "="*60 + "\n")

        # Genre comparison
        demonstrate_genre_comparison()

        print("\n" + "="*60 + "\n")

        # Performance optimization
        demonstrate_performance_optimization()

        print("\n" + "="*60 + "\n")

        # Presets demo
        demonstrate_presets()

        print("\n" + "="*60 + "\n")

        # Custom configuration
        demonstrate_custom_configuration()

        # Save results
        save_demo_results(result1, args.output)

        print("\n" + "="*60)
        print("🎉 All demonstrations completed!")
        print("Check the 'output' directory for generated MIDI files.")
        print("="*60)

    else:
        # Single generation demo
        print(f"🎵 Generating {args.genre} composition with {args.stems} stems...")

        # Create pipeline and song skeleton
        pipeline = MultiStemPipeline()
        song_skeleton = create_demo_song_skeleton(args.bars, args.bpm)

        # Generate composition
        start_time = time.time()
        result = pipeline.generate_multi_stem_composition(
            song_skeleton=song_skeleton,
            genre=GenreType(args.genre),
            mood=args.mood,
            stem_count=args.stems,
            output_directory=args.output,
            enable_performance_monitoring=True
        )
        elapsed = time.time() - start_time

        # Display results
        if result.success:
            print("\n✅ Generation successful!")
            print(f"Time: {elapsed:.2f}s")
            print(f"🎛️  Generated {result.stem_count} stems")
            print(f"💾 Created {len(result.output_files)} MIDI files in '{args.output}'")

            # Show stem details
            print("\n📋 Generated stems:")
            for role, filepath in result.output_files.items():
                print(f"   • {role.value}: {os.path.basename(filepath)}")

            # Show performance metrics
            if result.performance_metrics:
                print("\n📊 Performance:")
                for key, value in result.performance_metrics.items():
                    if isinstance(value, float) and key != 'cache_hit_rate':
                        print(f"   • {key}: {value:.2f}")
                    elif key == 'cache_hit_rate':
                        print(f"   • {key}: {value:.1%}")
                    else:
                        print(f"   • {key}: {value}")

        else:
            print(f"❌ Generation failed: {', '.join(result.errors)}")
            return 1

        return 0


if __name__ == '__main__':
    exit(main())