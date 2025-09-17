#!/usr/bin/env python3
"""
Enhanced Analysis API with ML Insights Integration

Purpose:
- Unified interface combining traditional and ML-based analysis
- Advanced music analysis with predictive capabilities
- Real-time insights and recommendations

Features:
- Traditional analysis integration
- ML-powered genre classification
- Predictive quality scoring
- Pattern recognition and similarity search
- Comprehensive reporting and recommendations
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

import mido

# Import existing analyzers
try:
    from analyzers.simple_metrics_demo import SimpleAnalyzer
    from analyzers.reference_library_simple import SimpleReferenceLibrary
    from analyzers.mix_readiness import MixReadinessAnalyzer
    from analyzers.analysis_api import AnalysisOrchestrator
    EXISTING_ANALYZERS_AVAILABLE = True
except ImportError:
    EXISTING_ANALYZERS_AVAILABLE = False
    print("Warning: Some existing analyzers not available")

# Import ML insights
from ml_insights.feature_extraction import MidiFeatureExtractor
from ml_insights.genre_classifier import GenreClassifier
from ml_insights.predictive_analysis import QualityScorer, CompletionSuggester
from ml_insights.pattern_recognition import HookDetector, ArchetypeAnalyzer, InfluencerAnalyzer
from ml_insights.similarity_search import SimilarityEngine


class EnhancedAnalysisOrchestrator:
    """Enhanced orchestrator combining traditional and ML-based analysis."""

    def __init__(self):
        # Initialize existing analyzers
        self.existing_orchestrator = None
        if EXISTING_ANALYZERS_AVAILABLE:
            self.existing_orchestrator = AnalysisOrchestrator()

        # Initialize ML analyzers
        self.feature_extractor = MidiFeatureExtractor()
        self.genre_classifier = GenreClassifier()
        self.quality_scorer = QualityScorer()
        self.completion_suggester = CompletionSuggester()
        self.hook_detector = HookDetector()
        self.archetype_analyzer = ArchetypeAnalyzer()
        self.influencer_analyzer = InfluencerAnalyzer()
        self.similarity_engine = SimilarityEngine()

    def analyze_comprehensive_enhanced(self, file_path: str,
                                      include_ml_insights: bool = True,
                                      include_pattern_analysis: bool = True,
                                      include_similarity: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive enhanced analysis combining traditional and ML methods.

        Args:
            file_path: Path to music file to analyze
            include_ml_insights: Include ML-based insights
            include_pattern_analysis: Include pattern recognition
            include_similarity: Include similarity analysis

        Returns:
            Comprehensive enhanced analysis results
        """
        start_time = time.time()
        results = {
            'file_path': file_path,
            'timestamp': time.time(),
            'analyses_performed': [],
            'traditional_analysis': {},
            'ml_insights': {},
            'pattern_analysis': {},
            'similarity_analysis': {},
            'recommendations': [],
            'overall_assessment': {},
            'performance_metrics': {}
        }

        # Basic file validation
        if not os.path.exists(file_path):
            results['error'] = f"File not found: {file_path}"
            return results

        try:
            # MIDI file validation
            midi_data = mido.MidiFile(file_path)
            results['file_info'] = {
                'format': midi_data.type,
                'tracks': len(midi_data.tracks),
                'ticks_per_beat': midi_data.ticks_per_beat,
                'length_seconds': self._calculate_midi_length(midi_data)
            }
        except Exception as e:
            results['error'] = f"Invalid MIDI file: {e}"
            return results

        # Traditional analysis
        if self.existing_orchestrator:
            try:
                traditional_results = self.existing_orchestrator.analyze_comprehensive(
                    file_path, target_genre="auto"
                )
                results['traditional_analysis'] = traditional_results
                results['analyses_performed'].append('traditional_analysis')
            except Exception as e:
                results['traditional_analysis'] = {'error': str(e)}

        # ML Insights
        if include_ml_insights:
            try:
                ml_results = self._perform_ml_analysis(file_path)
                results['ml_insights'] = ml_results
                results['analyses_performed'].append('ml_insights')
            except Exception as e:
                results['ml_insights'] = {'error': str(e)}

        # Pattern Analysis
        if include_pattern_analysis:
            try:
                pattern_results = self._perform_pattern_analysis(file_path)
                results['pattern_analysis'] = pattern_results
                results['analyses_performed'].append('pattern_analysis')
            except Exception as e:
                results['pattern_analysis'] = {'error': str(e)}

        # Similarity Analysis (if reference tracks available)
        if include_similarity:
            try:
                # For demo, we'll skip this as it requires a database
                results['similarity_analysis'] = {'status': 'database_required'}
            except Exception as e:
                results['similarity_analysis'] = {'error': str(e)}

        # Generate overall assessment and recommendations
        results['overall_assessment'] = self._calculate_enhanced_assessment(results)
        results['recommendations'] = self._generate_enhanced_recommendations(results)

        # Performance metrics
        end_time = time.time()
        results['performance_metrics'] = {
            'total_time_seconds': end_time - start_time,
            'analyses_count': len(results['analyses_performed'])
        }

        return results

    def _perform_ml_analysis(self, file_path: str) -> Dict[str, Any]:
        """Perform ML-based analysis."""
        ml_results = {}

        # Genre classification
        genre_result = self.genre_classifier.classify_genre(file_path)
        ml_results['genre_classification'] = genre_result

        # Quality scoring
        quality_result = self.quality_scorer.score_quality(file_path)
        ml_results['quality_scoring'] = quality_result

        # Completion suggestions (chord and melody)
        # This is more complex and requires current composition state
        ml_results['completion_suggestions'] = {
            'status': 'requires_current_composition_state',
            'available_features': ['chord_progression_completion', 'melody_continuation']
        }

        return ml_results

    def _perform_pattern_analysis(self, file_path: str) -> Dict[str, Any]:
        """Perform pattern recognition analysis."""
        pattern_results = {}

        # Hook detection
        hooks = self.hook_detector.detect_hooks(file_path)
        pattern_results['hooks'] = hooks

        # Archetype classification
        archetype = self.archetype_analyzer.classify_archetype(file_path)
        pattern_results['archetype'] = archetype

        # Influencer analysis
        influencers = self.influencer_analyzer.analyze_influencers(file_path)
        pattern_results['influencers'] = influencers

        return pattern_results

    def _calculate_midi_length(self, midi_data: mido.MidiFile) -> float:
        """Calculate total length of MIDI file in seconds."""
        max_ticks = 0
        tempo = 120  # default BPM

        for track in midi_data.tracks:
            ticks = 0
            for msg in track:
                ticks += msg.time
                if hasattr(msg, 'type') and msg.type == 'set_tempo':
                    tempo = 60000000 / msg.tempo
            max_ticks = max(max_ticks, ticks)

        # Convert ticks to seconds
        return max_ticks / (midi_data.ticks_per_beat * tempo / 60)

    def _calculate_enhanced_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced overall assessment."""
        assessment = {
            'overall_score': 0.0,
            'grade': 'N/A',
            'strengths': [],
            'weaknesses': [],
            'ml_insights_summary': {},
            'traditional_vs_ml': {}
        }

        scores = []

        # Traditional analysis score
        traditional = results.get('traditional_analysis', {})
        if 'overall_assessment' in traditional:
            trad_assessment = traditional['overall_assessment']
            if 'overall_score' in trad_assessment:
                traditional_score = trad_assessment['overall_score']
                scores.append(traditional_score)
                assessment['traditional_vs_ml']['traditional_score'] = traditional_score

        # ML analysis scores
        ml_insights = results.get('ml_insights', {})
        ml_scores = []

        # Genre classification confidence
        genre_result = ml_insights.get('genre_classification', {})
        if 'confidence_score' in genre_result:
            genre_conf = genre_result['confidence_score']
            ml_scores.append(genre_conf)

        # Quality scoring
        quality_result = ml_insights.get('quality_scoring', {})
        if 'overall_score' in quality_result:
            quality_score = quality_result['overall_score']
            scores.append(quality_score)
            ml_scores.append(quality_score)

        if ml_scores:
            assessment['ml_insights_summary']['average_ml_score'] = sum(ml_scores) / len(ml_scores)
            assessment['ml_insights_summary']['ml_analyses_count'] = len(ml_scores)

        # Pattern analysis
        pattern_analysis = results.get('pattern_analysis', {})
        if pattern_analysis:
            pattern_score = self._calculate_pattern_score(pattern_analysis)
            if pattern_score > 0:
                scores.append(pattern_score)

        # Calculate overall score
        if scores:
            assessment['overall_score'] = sum(scores) / len(scores)

            # Enhanced grading with ML insights
            score = assessment['overall_score']
            if score >= 0.9:
                assessment['grade'] = 'A+ (Excellent with ML validation)'
            elif score >= 0.85:
                assessment['grade'] = 'A (Very good with ML insights)'
            elif score >= 0.8:
                assessment['grade'] = 'A-'
            elif score >= 0.75:
                assessment['grade'] = 'B+'
            elif score >= 0.7:
                assessment['grade'] = 'B'
            elif score >= 0.65:
                assessment['grade'] = 'B-'
            elif score >= 0.6:
                assessment['grade'] = 'C+'
            elif score >= 0.55:
                assessment['grade'] = 'C'
            elif score >= 0.5:
                assessment['grade'] = 'C-'
            elif score >= 0.4:
                assessment['grade'] = 'D'
            else:
                assessment['grade'] = 'F'

        # Add ML-specific strengths
        if ml_insights:
            assessment['strengths'].append("ML-powered analysis completed")
            if 'genre_classification' in ml_insights:
                assessment['strengths'].append("Genre classification with ML model")
            if 'quality_scoring' in ml_insights:
                assessment['strengths'].append("AI-powered quality assessment")

        return assessment

    def _calculate_pattern_score(self, pattern_analysis: Dict) -> float:
        """Calculate score based on pattern analysis results."""
        score = 0.5  # Base score

        # Hook analysis
        hooks = pattern_analysis.get('hooks', [])
        if isinstance(hooks, list) and hooks:
            # Good if hooks are detected
            score += 0.2
        elif isinstance(hooks, dict) and 'error' not in hooks:
            score += 0.1

        # Archetype analysis
        archetype = pattern_analysis.get('archetype', {})
        if 'confidence' in archetype:
            confidence = archetype['confidence']
            score += confidence * 0.3

        return min(1.0, score)

    def _generate_enhanced_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on all analyses."""
        recommendations = []

        # ML-specific recommendations
        ml_insights = results.get('ml_insights', {})

        genre_result = ml_insights.get('genre_classification', {})
        if 'primary_genre' in genre_result:
            genre = genre_result['primary_genre']
            recommendations.append(f"Consider refining for {genre} genre characteristics")

        quality_result = ml_insights.get('quality_scoring', {})
        if 'overall_score' in quality_result:
            quality_score = quality_result['overall_score']
            if quality_score < 0.7:
                recommendations.append("ML analysis suggests quality improvements needed")

        # Pattern-based recommendations
        pattern_analysis = results.get('pattern_analysis', {})

        hooks = pattern_analysis.get('hooks', [])
        if isinstance(hooks, list) and not hooks:
            recommendations.append("Consider adding more memorable hooks or motifs")

        archetype = pattern_analysis.get('archetype', {})
        if 'primary_archetype' in archetype:
            arch_type = archetype['primary_archetype']
            recommendations.append(f"Following {arch_type} structure - ensure consistency")

        # Traditional analysis recommendations
        traditional = results.get('traditional_analysis', {})
        if 'overall_assessment' in traditional:
            trad_assessment = traditional['overall_assessment']
            trad_recs = trad_assessment.get('recommendations', [])
            recommendations.extend(trad_recs)

        # Add general ML recommendations
        if not ml_insights:
            recommendations.append("Consider running ML analysis for advanced insights")
        else:
            recommendations.append("ML analysis completed - review AI-generated suggestions")

        return list(set(recommendations))  # Remove duplicates


class EnhancedAnalysisPipeline:
    """Enhanced analysis pipeline with ML capabilities."""

    def __init__(self):
        self.orchestrator = EnhancedAnalysisOrchestrator()
        self.pipelines = self._load_enhanced_pipelines()

    def _load_enhanced_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Load enhanced analysis pipelines with ML capabilities."""
        return {
            'quick_check': {
                'name': 'Quick Quality Check',
                'description': 'Fast analysis for basic quality assessment',
                'steps': ['traditional_analysis'],
                'ml_enabled': False
            },
            'comprehensive': {
                'name': 'Comprehensive Analysis',
                'description': 'Full analysis suite with ML insights',
                'steps': ['traditional_analysis', 'ml_insights', 'pattern_analysis'],
                'ml_enabled': True
            },
            'ml_powered': {
                'name': 'ML-Powered Analysis',
                'description': 'Focus on ML-driven insights and predictions',
                'steps': ['ml_insights', 'pattern_analysis', 'similarity_analysis'],
                'ml_enabled': True
            },
            'creative_assistance': {
                'name': 'Creative Assistance',
                'description': 'AI-powered composition assistance and suggestions',
                'steps': ['ml_insights', 'pattern_analysis', 'completion_suggestions'],
                'ml_enabled': True
            },
            'production_ready': {
                'name': 'Production Ready with AI',
                'description': 'Complete analysis for production with AI insights',
                'steps': ['traditional_analysis', 'ml_insights', 'pattern_analysis', 'quality_assessment'],
                'ml_enabled': True
            }
        }

    def run_enhanced_pipeline(self, pipeline_name: str, file_path: str,
                             custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run an enhanced analysis pipeline.

        Args:
            pipeline_name: Name of the pipeline to run
            file_path: File to analyze
            custom_config: Custom configuration overrides

        Returns:
            Enhanced analysis results
        """
        if pipeline_name not in self.pipelines:
            return {'error': f"Pipeline '{pipeline_name}' not found"}

        pipeline = self.pipelines[pipeline_name].copy()

        # Apply custom configuration
        if custom_config:
            pipeline.update(custom_config)

        # Configure analysis based on pipeline
        ml_enabled = pipeline.get('ml_enabled', True)
        steps = pipeline.get('steps', [])

        include_ml = ml_enabled and 'ml_insights' in steps
        include_patterns = 'pattern_analysis' in steps
        include_similarity = 'similarity_analysis' in steps

        # Run enhanced analysis
        result = self.orchestrator.analyze_comprehensive_enhanced(
            file_path=file_path,
            include_ml_insights=include_ml,
            include_pattern_analysis=include_patterns,
            include_similarity=include_similarity
        )

        # Add pipeline information
        result['pipeline'] = {
            'name': pipeline['name'],
            'description': pipeline['description'],
            'steps_executed': steps,
            'ml_enabled': ml_enabled
        }

        return result


def main():
    """Enhanced Analysis API command-line interface."""
    parser = argparse.ArgumentParser(description="Enhanced Analysis API with ML Insights")
    parser.add_argument("--input", required=True, help="Input MIDI file")
    parser.add_argument("--pipeline", default="comprehensive",
                        choices=["quick_check", "comprehensive", "ml_powered", "creative_assistance", "production_ready"],
                        help="Analysis pipeline to use")
    parser.add_argument("--output", default="test_outputs", help="Output directory")
    parser.add_argument("--format", default="json", choices=["json", "html"],
                        help="Output format")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML analysis")

    args = parser.parse_args()

    # Initialize enhanced API
    api = EnhancedAnalysisPipeline()

    print(f"Running enhanced {args.pipeline} pipeline on {args.input}")

    # Configure pipeline
    custom_config = {}
    if args.no_ml:
        custom_config['ml_enabled'] = False

    result = api.run_enhanced_pipeline(args.pipeline, args.input, custom_config)

    # Print summary
    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    assessment = result.get('overall_assessment', {})
    print(f"\nOverall Score: {assessment.get('overall_score', 'N/A')}")
    print(f"Grade: {assessment.get('grade', 'N/A')}")

    if 'ml_insights' in result and result['ml_insights']:
        ml_insights = result['ml_insights']
        if 'genre_classification' in ml_insights:
            genre = ml_insights['genre_classification'].get('primary_genre', 'Unknown')
            confidence = ml_insights['genre_classification'].get('confidence_score', 0)
            print(".2f")

        if 'quality_scoring' in ml_insights:
            quality = ml_insights['quality_scoring'].get('overall_score', 0)
            print(".2f")

    # Save results
    os.makedirs(args.output, exist_ok=True)

    if args.format == "json":
        output_file = os.path.join(args.output, "enhanced_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
    else:
        # Enhanced HTML output
        output_file = os.path.join(args.output, "enhanced_analysis.html")
        html_content = generate_enhanced_html_report(result)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    print(f"\nResults saved to {output_file}")


def generate_enhanced_html_report(result: Dict[str, Any]) -> str:
    """Generate enhanced HTML report for analysis results."""
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .section {{ background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .score {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 4px; }}
        .recommendation {{ background: #e3f2fd; border-left: 4px solid #2196F3; padding: 10px; margin: 10px 0; }}
        .error {{ color: #f44336; background: #ffebee; padding: 10px; border-radius: 4px; }}
        .ml-insight {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 6px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Enhanced Music Analysis Report</h1>
        <p>File: {os.path.basename(result.get('file_path', 'Unknown'))}</p>
        <p>Analysis Pipeline: {result.get('pipeline', {}).get('name', 'Unknown')}</p>
    </div>

    <div class="section">
        <h2>📊 Overall Assessment</h2>
        <div class="score">Score: {result.get('overall_assessment', {}).get('overall_score', 'N/A')}</div>
        <div class="score">Grade: {result.get('overall_assessment', {}).get('grade', 'N/A')}</div>
    </div>

    {"".join([f'''
    <div class="section ml-insight">
        <h3>🤖 ML Insights</h3>
        {f"<p><strong>Genre:</strong> {result['ml_insights'].get('genre_classification', {}).get('primary_genre', 'Unknown')} (Confidence: {result['ml_insights'].get('genre_classification', {}).get('confidence_score', 0):.2f})</p>" if 'genre_classification' in result.get('ml_insights', {}) else ""}
        {f"<p><strong>Quality Score:</strong> {result['ml_insights'].get('quality_scoring', {}).get('overall_score', 'N/A'):.2f}</p>" if 'quality_scoring' in result.get('ml_insights', {}) else ""}
    </div>
    ''' if 'ml_insights' in result and result['ml_insights'] else ''])}

    {"".join([f'''
    <div class="section">
        <h3>🎼 Pattern Analysis</h3>
        {f"<p><strong>Archetype:</strong> {result['pattern_analysis'].get('archetype', {}).get('primary_archetype', 'Unknown')}</p>" if 'archetype' in result.get('pattern_analysis', {}) else ""}
        {f"<p><strong>Hooks Detected:</strong> {len([h for h in result['pattern_analysis'].get('hooks', []) if isinstance(h, dict) and 'type' in h])}</p>" if 'hooks' in result.get('pattern_analysis', {}) else ""}
    </div>
    ''' if 'pattern_analysis' in result and result['pattern_analysis'] else ''])}

    <div class="section">
        <h3>💡 Recommendations</h3>
        {"".join([f'<div class="recommendation">{rec}</div>' for rec in result.get('recommendations', [])])}
    </div>

    <div class="section">
        <h3>⚡ Performance Metrics</h3>
        <div class="metric">Analysis Time: {result.get('performance_metrics', {}).get('total_time_seconds', 'N/A'):.2f}s</div>
        <div class="metric">Analyses Performed: {len(result.get('analyses_performed', []))}</div>
    </div>
</body>
</html>"""

    return html_template


if __name__ == "__main__":
    main()