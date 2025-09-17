#!/usr/bin/env python3
"""
Comprehensive Analysis Script for MIDI Files

This script runs all available analysis tools on MIDI files, including:
- Traditional analysis (beat audit, metrics dashboard, structure insights, mix readiness)
- ML-enhanced analysis (genre classification, pattern recognition, predictive analysis)
- Enhanced visualizations for sample files
- Comprehensive summary report generation

Usage:
    python comprehensive_analysis_script.py <midi_directory> [options]

Author: AI Assistant
Date: 2024
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import glob

# Import analysis tools with error handling
try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido not available - MIDI functionality will be limited")

try:
    from analyzers.analysis_api import AnalysisOrchestrator
    TRADITIONAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TRADITIONAL_ANALYSIS_AVAILABLE = False
    print("Warning: Traditional analysis tools not available")

try:
    from ml_insights.enhanced_analysis_api import EnhancedAnalysisOrchestrator, EnhancedAnalysisPipeline
    ML_ANALYSIS_AVAILABLE = True
except ImportError:
    ML_ANALYSIS_AVAILABLE = False
    print("Warning: ML analysis tools not available")

try:
    from enhanced_visualizer import EnhancedMidiVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Enhanced visualization tools not available")


@dataclass
class AnalysisConfig:
    """Configuration for the comprehensive analysis."""
    input_directory: str
    output_directory: str = "analysis_output"
    max_workers: int = 4
    sample_visualizations: int = 3
    include_traditional: bool = True
    include_ml: bool = True
    include_visualization: bool = True
    verbose: bool = False
    progress_callback: Optional[Any] = None


@dataclass
class AnalysisResult:
    """Result of comprehensive analysis for a single MIDI file."""
    file_path: str
    file_size: int
    analysis_time: float
    traditional_analysis: Optional[Dict[str, Any]] = None
    ml_analysis: Optional[Dict[str, Any]] = None
    visualization_paths: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ComprehensiveAnalyzer:
    """Main analyzer that orchestrates all analysis tools."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._setup_analyzers()

    def _setup_analyzers(self):
        """Initialize all available analyzers."""
        self.traditional_orchestrator = None
        self.ml_orchestrator = None
        self.ml_pipeline = None
        self.visualizer = None

        if TRADITIONAL_ANALYSIS_AVAILABLE and self.config.include_traditional:
            try:
                self.traditional_orchestrator = AnalysisOrchestrator()
                print("✓ Traditional analysis tools initialized")
            except Exception as e:
                print(f"✗ Failed to initialize traditional analysis: {e}")

        if ML_ANALYSIS_AVAILABLE and self.config.include_ml:
            try:
                self.ml_orchestrator = EnhancedAnalysisOrchestrator()
                self.ml_pipeline = EnhancedAnalysisPipeline()
                print("✓ ML analysis tools initialized")
            except Exception as e:
                print(f"✗ Failed to initialize ML analysis: {e}")

        if VISUALIZATION_AVAILABLE and self.config.include_visualization:
            try:
                from enhanced_visualizer import VisualizationConfig
                vis_config = VisualizationConfig()
                self.visualizer = EnhancedMidiVisualizer(vis_config)
                print("✓ Visualization tools initialized")
            except Exception as e:
                print(f"✗ Failed to initialize visualization: {e}")

    def discover_midi_files(self) -> List[str]:
        """Discover all MIDI files in the input directory."""
        midi_files = []

        if not os.path.exists(self.config.input_directory):
            print(f"Error: Input directory '{self.config.input_directory}' does not exist")
            return midi_files

        # Find all .mid and .midi files recursively
        patterns = ['**/*.mid', '**/*.midi']
        for pattern in patterns:
            for midi_file in Path(self.config.input_directory).glob(pattern):
                midi_files.append(str(midi_file))

        midi_files.sort()
        print(f"Found {len(midi_files)} MIDI files in {self.config.input_directory}")
        return midi_files

    def analyze_single_file(self, midi_file: str, do_visualize: bool = True) -> AnalysisResult:
        """Analyze a single MIDI file with all available tools."""
        start_time = time.time()
        result = AnalysisResult(
            file_path=midi_file,
            file_size=os.path.getsize(midi_file) if os.path.exists(midi_file) else 0,
            analysis_time=0.0
        )

        try:
            # Validate MIDI file
            if not MIDO_AVAILABLE:
                result.errors.append("MIDI parsing not available")
                return result

            midi_data = mido.MidiFile(midi_file)
            if not midi_data.tracks:
                result.errors.append("MIDI file contains no tracks")
                return result

            print(f"Analyzing: {os.path.basename(midi_file)}")

            # Traditional analysis
            if self.traditional_orchestrator and self.config.include_traditional:
                try:
                    trad_result = self.traditional_orchestrator.analyze_comprehensive(
                        midi_file, target_genre="auto"
                    )
                    result.traditional_analysis = trad_result
                    print(f"  ✓ Traditional analysis completed")
                except Exception as e:
                    result.errors.append(f"Traditional analysis failed: {str(e)}")
                    print(f"  ✗ Traditional analysis failed: {e}")

            # ML-enhanced analysis
            if self.ml_orchestrator and self.config.include_ml:
                try:
                    ml_result = self.ml_orchestrator.analyze_comprehensive_enhanced(
                        midi_file,
                        include_ml_insights=True,
                        include_pattern_analysis=True,
                        include_similarity=False  # Skip similarity for performance
                    )
                    result.ml_analysis = ml_result
                    print(f"  ✓ ML analysis completed")
                except Exception as e:
                    result.errors.append(f"ML analysis failed: {str(e)}")
                    print(f"  ✗ ML analysis failed: {e}")

            # Visualization (only for sample files)
            if self.visualizer and self.config.include_visualization and do_visualize:
                try:
                    vis_paths = self._generate_visualization(midi_file)
                    result.visualization_paths = vis_paths
                    if vis_paths:
                        print(f"  ✓ Generated {len(vis_paths)} visualizations")
                except Exception as e:
                    result.errors.append(f"Visualization failed: {str(e)}")
                    print(f"  ✗ Visualization failed: {e}")

        except Exception as e:
            result.errors.append(f"MIDI parsing failed: {str(e)}")
            print(f"  ✗ MIDI parsing failed: {e}")

        result.analysis_time = time.time() - start_time
        return result

    def _generate_visualization(self, midi_file: str) -> List[str]:
        """Generate visualizations for a MIDI file."""
        if not self.visualizer:
            return []

        vis_paths = []

        try:
            # Parse the MIDI file
            notes = self.visualizer.parse_midi_file(midi_file)
            if not notes:
                return vis_paths

            # Generate different visualization types
            from enhanced_visualizer import VisualizationType, VisualizationPreset

            visualizations = [
                (VisualizationType.PIANO_ROLL, VisualizationPreset.ELECTRONIC, "piano_roll"),
                (VisualizationType.SPECTRAL, None, "spectral"),
                (VisualizationType.RHYTHM_MAP, None, "rhythm_map")
            ]

            base_name = os.path.splitext(os.path.basename(midi_file))[0]

            for view_type, preset, suffix in visualizations:
                try:
                    html_content = self.visualizer.generate_visualization(
                        view_type, preset, interactive=False
                    )

                    output_file = os.path.join(
                        self.config.output_directory,
                        "visualizations",
                        f"{base_name}_{suffix}.html"
                    )

                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)

                    vis_paths.append(output_file)

                except Exception as e:
                    print(f"    Warning: Failed to generate {suffix} visualization: {e}")

        except Exception as e:
            print(f"    Warning: Visualization setup failed: {e}")

        return vis_paths

    def analyze_batch(self, midi_files: List[str]) -> Dict[str, Any]:
        """Analyze a batch of MIDI files."""
        results = {}
        total_files = len(midi_files)
        sample_count = min(self.config.sample_visualizations, total_files)

        print(f"Starting batch analysis of {total_files} files with {self.config.max_workers} workers")
        if sample_count > 0:
            print(f"Will generate visualizations for first {sample_count} files")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all analysis tasks
            future_to_file = {
                executor.submit(
                    self.analyze_single_file,
                    midi_file,
                    do_visualize=(i < sample_count)
                ): midi_file
                for i, midi_file in enumerate(midi_files)
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                midi_file = future_to_file[future]
                try:
                    result = future.result()
                    results[midi_file] = result
                    completed += 1

                    if self.config.verbose or completed % 10 == 0:
                        print(f"Completed {completed}/{total_files}: {os.path.basename(midi_file)}")

                    if self.config.progress_callback:
                        self.config.progress_callback(completed, total_files)

                except Exception as e:
                    print(f"Error analyzing {midi_file}: {e}")
                    results[midi_file] = AnalysisResult(
                        file_path=midi_file,
                        file_size=0,
                        analysis_time=0.0,
                        errors=[str(e)]
                    )

        return results

    def generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        summary = {
            'metadata': {
                'analysis_time': time.time(),
                'total_files': len(results),
                'input_directory': self.config.input_directory,
                'output_directory': self.config.output_directory,
                'tools_used': []
            },
            'statistics': {},
            'findings': {},
            'recommendations': [],
            'file_summaries': {}
        }

        # Add tools information
        if self.traditional_orchestrator:
            summary['metadata']['tools_used'].append('traditional_analysis')
        if self.ml_orchestrator:
            summary['metadata']['tools_used'].append('ml_analysis')
        if self.visualizer:
            summary['metadata']['tools_used'].append('visualization')

        # Calculate statistics
        successful_analyses = [r for r in results.values() if not r.errors]
        failed_analyses = [r for r in results.values() if r.errors]

        summary['statistics'] = {
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(failed_analyses),
            'success_rate': len(successful_analyses) / len(results) if results else 0,
            'average_analysis_time': sum(r.analysis_time for r in results.values()) / len(results) if results else 0,
            'total_visualizations': sum(len(r.visualization_paths) for r in results.values())
        }

        # Generate findings
        findings = self._analyze_findings(results)
        summary['findings'] = findings

        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(results)

        # File summaries
        summary['file_summaries'] = {
            os.path.basename(path): {
                'size': result.file_size,
                'analysis_time': result.analysis_time,
                'analyses_completed': self._count_completed_analyses(result),
                'errors': result.errors
            }
            for path, result in results.items()
        }

        return summary

    def _analyze_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze findings across all results."""
        findings = {
            'genre_distribution': {},
            'quality_scores': [],
            'pattern_types': {},
            'technical_issues': []
        }

        for result in results.values():
            if result.errors:
                findings['technical_issues'].extend(result.errors)
                continue

            # ML analysis findings
            if result.ml_analysis:
                ml_data = result.ml_analysis

                # Genre classification
                if 'genre_classification' in ml_data:
                    genre_info = ml_data['genre_classification']
                    primary_genre = genre_info.get('primary_genre', 'unknown')
                    findings['genre_distribution'][primary_genre] = \
                        findings['genre_distribution'].get(primary_genre, 0) + 1

                # Quality scores
                if 'quality_scoring' in ml_data:
                    quality_score = ml_data['quality_scoring'].get('overall_score')
                    if quality_score is not None:
                        findings['quality_scores'].append(quality_score)

                # Pattern analysis
                if 'pattern_analysis' in ml_data:
                    patterns = ml_data['pattern_analysis']
                    if 'hooks' in patterns and patterns['hooks']:
                        findings['pattern_types']['hooks_detected'] = \
                            findings['pattern_types'].get('hooks_detected', 0) + 1

        return findings

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        # Analyze success rates
        success_rate = len([r for r in results.values() if not r.errors]) / len(results) if results else 0

        if success_rate < 0.8:
            recommendations.append("Consider reviewing MIDI file integrity - some files failed analysis")

        # Analyze quality scores
        quality_scores = []
        for result in results.values():
            if result.ml_analysis and 'quality_scoring' in result.ml_analysis:
                score = result.ml_analysis['quality_scoring'].get('overall_score')
                if score is not None:
                    quality_scores.append(score)

        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            if avg_quality < 0.6:
                recommendations.append("Average quality score is low - consider production improvements")

        # Tool availability recommendations
        if not self.traditional_orchestrator:
            recommendations.append("Install traditional analysis tools for comprehensive evaluation")
        if not self.ml_orchestrator:
            recommendations.append("Install ML analysis tools for advanced insights")
        if not self.visualizer:
            recommendations.append("Install visualization tools for enhanced analysis")

        return recommendations

    def _count_completed_analyses(self, result: AnalysisResult) -> int:
        """Count how many analyses were completed for a result."""
        count = 0
        if result.traditional_analysis:
            count += 1
        if result.ml_analysis:
            count += 1
        if result.visualization_paths:
            count += 1
        return count
    def _generate_html_report(self, summary: Dict[str, Any], results: Dict[str, Any]):
        """Generate HTML report for the analysis results."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive MIDI Analysis Report</title>
</head>
<body>
    <h1>Comprehensive MIDI Analysis Report</h1>
    <p>Analysis completed successfully</p>
</body>
</html>"""

        html_file = os.path.join(self.config.output_directory, "analysis_report.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def save_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """Save analysis results to disk."""
        os.makedirs(self.config.output_directory, exist_ok=True)

        # Save detailed results
        results_file = os.path.join(self.config.output_directory, "detailed_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert results to serializable format
            serializable_results = {}
            for path, result in results.items():
                serializable_results[path] = {
                    'file_path': result.file_path,
                    'file_size': result.file_size,
                    'analysis_time': result.analysis_time,
                    'traditional_analysis': result.traditional_analysis,
                    'ml_analysis': result.ml_analysis,
                    'visualization_paths': result.visualization_paths,
                    'errors': result.errors
                }
            json.dump(serializable_results, f, indent=2, default=str)

        # Save summary report
        summary_file = os.path.join(self.config.output_directory, "summary_report.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

        # Generate HTML report
        self._generate_html_report(summary, results)

        print(f"Results saved to: {self.config.output_directory}")
        print(f"  - Detailed results: detailed_results.json")
        print(f"  - Summary report: summary_report.json")
        print(f"  - HTML report: analysis_report.html")


def main():
    """Main entry point for the comprehensive analysis script."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Analysis Script for MIDI Files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comprehensive_analysis_script.py reference_midis/midi4
  python comprehensive_analysis_script.py reference_midis/midi4 --output my_analysis --workers 8
  python comprehensive_analysis_script.py reference_midis/midi4 --no-ml --sample-viz 5 --verbose
        """
    )

    parser.add_argument("input_directory", help="Directory containing MIDI files")
    parser.add_argument("--output", "-o", default="analysis_output",
                       help="Output directory for results (default: analysis_output)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Maximum number of worker threads (default: 4)")
    parser.add_argument("--sample-viz", type=int, default=3,
                       help="Number of sample files for visualization (default: 3)")
    parser.add_argument("--no-traditional", action="store_true",
                       help="Disable traditional analysis")
    parser.add_argument("--no-ml", action="store_true",
                       help="Disable ML analysis")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization generation")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Create configuration
    config = AnalysisConfig(
        input_directory=args.input_directory,
        output_directory=args.output,
        max_workers=args.workers,
        sample_visualizations=args.sample_viz,
        include_traditional=not args.no_traditional,
        include_ml=not args.no_ml,
        include_visualization=not args.no_viz,
        verbose=args.verbose
    )

    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer(config)

    # Discover MIDI files
    midi_files = analyzer.discover_midi_files()
    if not midi_files:
        print("No MIDI files found. Exiting.")
        sys.exit(1)

    print(f"Starting comprehensive analysis of {len(midi_files)} MIDI files...")
    start_time = time.time()

    # Run batch analysis
    results = analyzer.analyze_batch(midi_files)

    # Generate summary report
    summary = analyzer.generate_summary_report(results)

    # Save results
    analyzer.save_results(results, summary)

    # Print summary
    total_time = time.time() - start_time
    print("\n=== ANALYSIS COMPLETE ===\n")
    print(f"Total files processed: {len(results)}")
    print(f"Successful analyses: {summary['statistics']['successful_analyses']}")
    print(f"Failed analyses: {summary['statistics']['failed_analyses']}")
    print(f"Total analysis time: {total_time:.2f} seconds")
    print(f"Results saved to: {config.output_directory}")


if __name__ == "__main__":
    main()