#!/usr/bin/env python3
"""
Simple ML Insights Demo

A straightforward demonstration of the ML-enhanced music analysis system.
"""

import os
import sys
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run a simple demo of the ML insights system."""
    print("🎵 MIDI Master - ML Insights System Demo")
    print("==========================================")

    try:
        # Import ML components
        from ml_insights.feature_extraction import MidiFeatureExtractor
        from ml_insights.genre_classifier import GenreClassifier
        from ml_insights.predictive_analysis import QualityScorer
        from ml_insights.enhanced_analysis_api import EnhancedAnalysisPipeline

        print("✅ ML components imported successfully")

        # Initialize components
        feature_extractor = MidiFeatureExtractor()
        genre_classifier = GenreClassifier()
        quality_scorer = QualityScorer()
        api = EnhancedAnalysisPipeline()

        print("✅ ML components initialized successfully")

        # Find MIDI files
        import glob
        midi_files = glob.glob("output/*.mid") + glob.glob("reference_midis/**/*.mid")
        midi_files = midi_files[:3]  # Limit for demo

        if not midi_files:
            print("❌ No MIDI files found. Please ensure MIDI files exist in output/ or reference_midis/")
            return

        print(f"📁 Found {len(midi_files)} MIDI files for analysis")

        # Demo basic feature extraction
        print("\n🎵 Feature Extraction Demo:")
        if midi_files:
            midi_file = midi_files[0]
            print(f"   Analyzing: {os.path.basename(midi_file)}")

            features = feature_extractor.extract_features(midi_file)
            if 'error' not in features:
                print("   ✅ Feature extraction successful")
                temporal = features.get('temporal', {})
                print(f"   📊 Tempo: {temporal.get('tempo_bpm', 'N/A')}")
                print(f"   📊 Duration: {temporal.get('total_duration', 0):.1f} seconds")
            else:
                print(f"   ❌ Error: {features['error']}")

        # Demo enhanced analysis
        print("\n🚀 Enhanced Analysis Demo:")
        if midi_files:
            midi_file = midi_files[0]
            print(f"   Running comprehensive analysis on: {os.path.basename(midi_file)}")

            result = api.run_enhanced_pipeline('comprehensive', midi_file)

            if 'error' not in result:
                assessment = result.get('overall_assessment', {})
                print(f"   📈 Quality Score: {assessment.get('quality_score', 0):.2f}")
                print(f"   📈 Grade: {assessment.get('grade', 'N/A')}")

                analyses = result.get('analyses_performed', [])
                print(f"   🔧 Analyses completed: {len(analyses)}")

                # Show ML insights if available
                ml_insights = result.get('ml_insights', {})
                if ml_insights and 'error' not in ml_insights:
                    genre_result = ml_insights.get('genre_classification', {})
                    if 'primary_genre' in genre_result:
                        genre = genre_result['primary_genre']
                        confidence = genre_result.get('confidence_score', 0)
                        print(f"   🎵 Genre: {genre} (confidence: {confidence:.2f})")
                    else:
                        print(f"   ❌ No primary genre detected")
                else:
                    print(f"   ❌ ML Insights Error: {ml_insights.get('error', 'ML insights unavailable')}")
                
            else:
                print("   ❌ Enhanced analysis failed")

        print("\n🎉 Demo completed successfully!")
        print("\nThe ML Insights system includes:")
        print("• Feature extraction from MIDI files")
        print("• ML-powered genre classification")
        print("• AI-based quality assessment")
        print("• Pattern recognition capabilities")
        print("• Integrated analysis pipelines")
        print("• Completion suggestions for composition")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all ML dependencies are installed:")
        print("pip install scikit-learn numpy pandas matplotlib seaborn joblib scipy")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("ML Insights Demo completed.")

if __name__ == "__main__":
    main()