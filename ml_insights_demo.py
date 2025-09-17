#!/usr/bin/env python3
"""
ML Insights Demo Script

Purpose:
- Demonstrate ML-enhanced music analysis capabilities
- Test all components of the ML insights system
- Provide usage examples for developers

Features Demonstrated:
- Feature extraction from MIDI files
- Genre classification with ML models
- Quality scoring and assessment
- Pattern recognition (hooks, archetypes)
- Similarity search capabilities
- Completion suggestions
- Enhanced analysis API integration
"""

import os
import sys
import json
import glob
import datetime
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_insights.feature_extraction import MidiFeatureExtractor
from ml_insights.genre_classifier import GenreClassifier
from ml_insights.predictive_analysis import QualityScorer, CompletionSuggester
from ml_insights.pattern_recognition import HookDetector, ArchetypeAnalyzer, InfluencerAnalyzer
from ml_insights.similarity_search import SimilarityEngine
from ml_insights.enhanced_analysis_api import EnhancedAnalysisPipeline


def find_midi_files(directory: str = "output") -> List[str]:
    """Find MIDI files in the specified directory."""
    if not os.path.exists(directory):
        print(f"Directory {directory} not found. Using current directory.")
        directory = "."

    midi_files = []
    for ext in ['*.mid', '*.midi']:
        midi_files.extend(glob.glob(os.path.join(directory, ext)))

    # Also check reference_midis if available
    ref_dir = "reference_midis"
    if os.path.exists(ref_dir):
        for ext in ['*.mid', '*.midi']:
            midi_files.extend(glob.glob(os.path.join(ref_dir, "**", ext)))

    return midi_files[:5]  # Limit to first 5 files for demo


def demo_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    print("🎵 ML Insights Demo - Feature Extraction")
    print("=" * 50)

    feature_extractor = MidiFeatureExtractor()
    midi_files = find_midi_files()

    if not midi_files:
        print("No MIDI files found for demonstration.")
        return

    print(f"Found {len(midi_files)} MIDI files for analysis")

    for i, midi_file in enumerate(midi_files[:2]):  # Demo with first 2 files
        print(f"\n📁 Analyzing: {os.path.basename(midi_file)}")
        print("-" * 30)

        features = feature_extractor.extract_features(midi_file)

        if 'error' in features:
            print(f"❌ Error: {features['error']}")
            continue

        # Display key features
        print("📊 Basic Info:")
        basic = features.get('basic_info', {})
        print(f"   Format: {basic.get('format', 'N/A')}, Tracks: {basic.get('tracks', 'N/A')}")

        print("\n🎼 Temporal Features:")
        temporal = features.get('temporal', {})
        for key, value in list(temporal.items())[:5]:  # Show first 5
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

        print("\n🎹 Harmonic Features:")
        harmonic = features.get('harmonic', {})
        for key, value in list(harmonic.items())[:5]:  # Show first 5
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

        print("\n🎶 Melodic Features:")
        melodic = features.get('melodic', {})
        for key, value in list(melodic.items())[:5]:  # Show first 5
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")


def demo_genre_classification():
    """Demonstrate ML-based genre classification."""
    print("\n🎵 ML Insights Demo - Genre Classification")
    print("=" * 50)

    genre_classifier = GenreClassifier()
    midi_files = find_midi_files()

    if not midi_files:
        print("No MIDI files found for demonstration.")
        return

    for midi_file in midi_files[:3]:  # Demo with first 3 files
        print(f"\n🎵 Classifying: {os.path.basename(midi_file)}")

        result = genre_classifier.classify_genre(midi_file)

        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            continue

        confidence = result.get('confidence_score', 0.0)
        print(f"🎯 Primary Genre: {result.get('primary_genre', 'Unknown')} (Confidence: {confidence:.2f})")
        print("\n📋 Top Genres:")
        for genre_info in result.get('genres', [])[:3]:
            genre = genre_info.get('genre', 'Unknown')
            conf = genre_info.get('confidence', 0.0)
            print(f"     - {genre}: {conf:.2f}")
        # Show feature analysis
        feature_analysis = result.get('feature_analysis', {})
        if feature_analysis:
            print("🔍 Key Characteristics:")
            for key, value in feature_analysis.items():
                print(f"   {key}: {value}")


def demo_quality_scoring():
    """Demonstrate ML-based quality scoring."""
    print("\n🎵 ML Insights Demo - Quality Scoring")
    print("=" * 50)

    quality_scorer = QualityScorer()
    midi_files = find_midi_files()

    if not midi_files:
        print("No MIDI files found for demonstration.")
        return

    for midi_file in midi_files[:3]:  # Demo with first 3 files
        print(f"\n📊 Scoring: {os.path.basename(midi_file)}")

        result = quality_scorer.score_quality(midi_file)

        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            continue

        print(f"📊 Overall Score: {result.get('overall_score', 0):.2f}")
        print(f"📈 Grade: {result.get('grade', 'N/A')}")
        print("\n💡 Recommendations:")
        recommendations = result.get('recommendations', [])
        for rec in recommendations[:3]:  # Show first 3
            print(f"   • {rec}")


def demo_pattern_recognition():
    """Demonstrate pattern recognition capabilities."""
    print("\n🎵 ML Insights Demo - Pattern Recognition")
    print("=" * 50)

    hook_detector = HookDetector()
    archetype_analyzer = ArchetypeAnalyzer()
    influencer_analyzer = InfluencerAnalyzer()

    midi_files = find_midi_files()

    if not midi_files:
        print("No MIDI files found for demonstration.")
        return

    midi_file = midi_files[0]  # Use first file
    print(f"\n🔍 Analyzing patterns in: {os.path.basename(midi_file)}")

    # Hook detection
    print("\n🎣 Hook Detection:")
    hooks = hook_detector.detect_hooks(midi_file, min_confidence=0.5)
    if hooks and isinstance(hooks, list) and len(hooks) > 0:
        print(f"   Found {len(hooks)} potential hooks")
        for i, hook in enumerate(hooks[:3]):  # Show first 3
            hook_type = hook.get('type', 'unknown')
            confidence = hook.get('confidence', 0)
            print(f"   Hook {i+1}: {hook_type} (Confidence: {confidence:.2f})")
    else:
        print("   No hooks detected with current confidence threshold")

    # Archetype classification
    print("\n🏛️  Archetype Analysis:")
    archetype = archetype_analyzer.classify_archetype(midi_file)
    if 'error' not in archetype:
        print(f"   Primary Archetype: {archetype.get('primary_archetype', 'Unknown')}")
        confidence = archetype.get('confidence', 0.0)
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Description: {archetype.get('description', 'N/A')}")
    else:
        print(f"   Error: {archetype['error']}")

    # Influencer analysis
    print("\n🌟 Influencer Analysis:")
    influencers = influencer_analyzer.analyze_influencers(midi_file)
    if 'error' not in influencers:
        primary_influence = influencers.get('primary_influence', 'Unknown')
        print(f"   Primary Musical Influence: {primary_influence}")

        influence_scores = influencers.get('influence_scores', {})
        print("   Influence Scores:")
        for influence, score in list(influence_scores.items())[:3]:
            print(f"     {influence}: {score:.2f}")
    else:
        print(f"   Error: {influencers['error']}")


def demo_completion_suggestions():
    """Demonstrate completion suggestion capabilities."""
    print("\n🎵 ML Insights Demo - Completion Suggestions")
    print("=" * 50)

    completion_suggester = CompletionSuggester()

    # Demo chord progression completion
    print("\n🎼 Chord Progression Completion:")
    current_progression = ['I', 'IV', 'V']
    print(f"Current progression: {current_progression}")

    suggestions = completion_suggester.suggest_next_chord(current_progression, 'C major', 3)  # type: ignore
    print("Suggested next chords:")
    for i, (chord, confidence) in enumerate(suggestions):
        print(f"   {i+1}. {chord} (Confidence: {confidence:.2f})")
    print("\n🎶 Melody Completion:")
    print("   (Melody completion requires current melody state)")
    print("   Feature available for integration with composition tools")


def demo_enhanced_analysis():
    """Demonstrate the enhanced analysis API."""
    print("\n🎵 ML Insights Demo - Enhanced Analysis API")
    print("=" * 50)

    api = EnhancedAnalysisPipeline()
    midi_files = find_midi_files()

    if not midi_files:
        print("No MIDI files found for demonstration.")
        return

    midi_file = midi_files[0]
    print(f"\n🚀 Running enhanced analysis on: {os.path.basename(midi_file)}")

    result = api.run_enhanced_pipeline('comprehensive', midi_file)

    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return

    # Summary
    print("\n📊 Analysis Summary:")
    assessment = result.get('overall_assessment', {})
    print(f"📊 Overall Score: {assessment.get('overall_score', 0):.2f}")
    print(f"   Grade: {assessment.get('grade', 'N/A')}")

    analyses = result.get('analyses_performed', [])
    print(f"   Analyses Performed: {len(analyses)}")
    for analysis in analyses:
        print(f"     ✓ {analysis.replace('_', ' ').title()}")

    # ML Insights
    ml_insights = result.get('ml_insights', {})
    if ml_insights and 'error' not in ml_insights:
        print("\n🤖 ML Insights:")
        genre_result = ml_insights.get('genre_classification', {})
        if 'primary_genre' in genre_result:
            print(f"   Genre: {genre_result['primary_genre']} (Confidence: {genre_result.get('confidence_score', 0):.2f})")

        quality_result = ml_insights.get('quality_scoring', {})
        if 'overall_score' in quality_result:
            print(f"   Quality Score: {quality_result['overall_score']:.2f}")
    # Recommendations
    recommendations = result.get('recommendations', [])
    if recommendations:
        print("\n💡 Key Recommendations:")
        for i, rec in enumerate(recommendations[:3]):
            print(f"   {i+1}. {rec}")

    # Performance
    perf = result.get('performance_metrics', {})
    if perf:
        print("\n⚡ Performance:")
        print(f"   Processing Time: {perf.get('processing_time', 0):.2f}s")
        print(f"   Analyses Count: {perf.get('analyses_count', 0)}")


def save_demo_results():
    """Save demo results to JSON file."""
    print("\n💾 Saving demo results...")

    demo_results = {
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'system': 'ML Insights Demo',
        'capabilities_demonstrated': [
            'Feature Extraction',
            'Genre Classification',
            'Quality Scoring',
            'Pattern Recognition',
            'Completion Suggestions',
            'Enhanced Analysis API'
        ],
        'ml_models_used': [
            'Genre Classification Model',
            'Quality Scoring Model',
            'Pattern Recognition Models',
            'Similarity Search Engine'
        ],
        'key_features': [
            'Real-time feature extraction from MIDI',
            'ML-powered genre classification',
            'AI-based quality assessment',
            'Hook and motif detection',
            'Arrangement archetype analysis',
            'Completion suggestions for composition',
            'Integrated analysis pipeline'
        ]
    }

    output_file = "test_outputs/ml_insights_demo_results.json"
    os.makedirs("test_outputs", exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(demo_results, f, indent=2)

    print(f"Demo results saved to {output_file}")


def main():
    """Run the complete ML insights demonstration."""
    print("🎵 MIDI Master - ML Insights System Demo")
    print("==========================================")
    print("This demo showcases the machine learning-enhanced")
    print("insights system for music analysis and composition.\n")

    try:
        # Run demonstrations
        demo_feature_extraction()
        demo_genre_classification()
        demo_quality_scoring()
        demo_pattern_recognition()
        demo_completion_suggestions()
        demo_enhanced_analysis()

        # Save results
        save_demo_results()

        print("\n🎉 Demo completed successfully!")
        print("\nThe ML Insights system provides:")
        print("• 🎵 Advanced music feature extraction")
        print("• 🤖 ML-powered genre classification")
        print("• 📊 AI-based quality assessment")
        print("• 🔍 Pattern recognition and analysis")
        print("• 💡 Intelligent composition assistance")
        print("• 🚀 Integrated analysis pipelines")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This might be due to missing dependencies or data files.")
        print("Please ensure all requirements are installed and MIDI files are available.")

    print("\n" + "="*50)
    print("Demo script completed.")


if __name__ == "__main__":
    main()