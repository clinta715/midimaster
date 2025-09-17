#!/usr/bin/env python3
"""
Comprehensive MIDI Analyzer - Production Ready

This script provides comprehensive analysis of MIDI files using all available analysis tools:
- Traditional analysis (structure insights, mix readiness, metrics dashboard)
- ML-enhanced analysis (genre classification, pattern recognition, predictive analysis)
- Enhanced visualizations for sample files
- Batch processing with progress reporting
- HTML and JSON report generation
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Import analysis tools with graceful fallbacks
try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

# Traditional analysis tools
try:
    from analyzers.analysis_api import AnalysisOrchestrator
    TRADITIONAL_AVAILABLE = True
except ImportError:
    TRADITIONAL_AVAILABLE = False

# ML analysis tools
try:
    from ml_insights.enhanced_analysis_api import EnhancedAnalysisOrchestrator
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Visualization tools
try:
    from enhanced_visualizer import EnhancedMidiVisualizer, VisualizationType, VisualizationPreset
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


@dataclass
class AnalysisConfig:
    """Configuration for the comprehensive analysis."""
    input_directory: str
    output_directory: str = "comprehensive_analysis_output"
    max_workers: int = 4
    sample_visualizations: int = 3
    include_traditional: bool = True
    include_ml: bool = True
    include_visualization: bool = True
    verbose: bool = False
    generate_html_report: bool = True
    generate_json_report: bool = True


@dataclass
class AnalysisResult:
    """Result of comprehensive analysis for a single MIDI file."""
    file_path: str
    file_name: str
    file_size: int
    analysis_time: float
    traditional_analysis: Optional[Dict[str, Any]] = None
    ml_analysis: Optional[Dict[str, Any]] = None
    visualization_paths: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_size': self.file_size,
            'analysis_time': self.analysis_time,
            'traditional_analysis': self.traditional_analysis,
            'ml_analysis': self.ml_analysis,
            'visualization_paths': self.visualization_paths,
            'errors': self.errors,
            'success': self.success
        }


class ComprehensiveMidiAnalyzer:
    """Main analyzer that orchestrates all analysis tools."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._setup_analyzers()
        self._ensure_output_directory()

    def _setup_analyzers(self):
        """Initialize all available analyzers."""
        print("🔧 Setting up analysis tools...")

        self.traditional_orchestrator = None
        if TRADITIONAL_AVAILABLE and self.config.include_traditional:
            try:
                self.traditional_orchestrator = AnalysisOrchestrator()
                print("✅ Traditional analysis tools ready")
            except Exception as e:
                print(f"⚠️  Traditional analysis setup failed: {e}")

        self.ml_orchestrator = None
        if ML_AVAILABLE and self.config.include_ml:
            try:
                self.ml_orchestrator = EnhancedAnalysisOrchestrator()
                print("✅ ML analysis tools ready")
            except Exception as e:
                print(f"⚠️  ML analysis setup failed: {e}")

        self.visualizer = None
        if VISUALIZATION_AVAILABLE and self.config.include_visualization:
            try:
                from enhanced_visualizer import VisualizationConfig as VisConfig
                vis_config = VisConfig()
                self.visualizer = EnhancedMidiVisualizer(vis_config)
                print("✅ Visualization tools ready")
            except Exception as e:
                print(f"⚠️  Visualization setup failed: {e}")

        print()

    def _ensure_output_directory(self):
        """Create output directory structure."""
        os.makedirs(self.config.output_directory, exist_ok=True)
        if self.config.include_visualization:
            os.makedirs(os.path.join(self.config.output_directory, "visualizations"), exist_ok=True)

    def discover_midi_files(self) -> List[str]:
        """Discover all MIDI files in the input directory."""
        if not os.path.exists(self.config.input_directory):
            raise ValueError(f"Input directory does not exist: {self.config.input_directory}")

        midi_files = []
        patterns = ['**/*.mid', '**/*.midi', '**/*.MID', '**/*.MIDI']

        for pattern in patterns:
            for midi_file in Path(self.config.input_directory).glob(pattern):
                midi_files.append(str(midi_file))

        midi_files = sorted(list(set(midi_files)))  # Remove duplicates
        return midi_files

    def analyze_single_file(self, midi_file: str, file_index: int, total_files: int) -> AnalysisResult:
        """Analyze a single MIDI file with all available tools."""
        start_time = time.time()
        file_name = os.path.basename(midi_file)

        if self.config.verbose:
            print(f"[{file_index}/{total_files}] Analyzing: {file_name}")

        result = AnalysisResult(
            file_path=midi_file,
            file_name=file_name,
            file_size=os.path.getsize(midi_file),
            analysis_time=0.0
        )

        try:
            # Validate MIDI file
            if not MIDO_AVAILABLE:
                result.errors.append("MIDI parsing not available")
                result.success = False
                return result

            midi_data = mido.MidiFile(midi_file)
            if len(midi_data.tracks) == 0:
                result.errors.append("MIDI file contains no tracks")
                result.success = False
                return result

            # Traditional analysis
            if self.traditional_orchestrator and self.config.include_traditional:
                try:
                    trad_result = self.traditional_orchestrator.analyze_comprehensive(
                        midi_file, target_genre="auto"
                    )
                    result.traditional_analysis = trad_result
                    if self.config.verbose:
                        print(f"  ✅ Traditional analysis completed")
                except Exception as e:
                    result.errors.append(f"Traditional analysis failed: {str(e)}")
                    if self.config.verbose:
                        print(f"  ❌ Traditional analysis failed: {e}")

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
                    if self.config.verbose:
                        print(f"  ✅ ML analysis completed")
                except Exception as e:
                    result.errors.append(f"ML analysis failed: {str(e)}")
                    if self.config.verbose:
                        print(f"  ❌ ML analysis failed: {e}")

            # Visualization (only for first few files to avoid excessive processing)
            if (self.visualizer and self.config.include_visualization and
                file_index <= self.config.sample_visualizations):
                try:
                    vis_paths = self._generate_visualizations(midi_file)
                    result.visualization_paths = vis_paths
                    if vis_paths and self.config.verbose:
                        print(f"  ✅ Generated {len(vis_paths)} visualizations")
                except Exception as e:
                    result.errors.append(f"Visualization failed: {str(e)}")
                    if self.config.verbose:
                        print(f"  ❌ Visualization failed: {e}")

        except Exception as e:
            result.errors.append(f"MIDI parsing failed: {str(e)}")
            result.success = False
            if self.config.verbose:
                print(f"  ❌ MIDI parsing failed: {e}")

        result.analysis_time = time.time() - start_time
        if result.errors:
            result.success = False

        return result

    def _generate_visualizations(self, midi_file: str) -> List[str]:
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

                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)

                    vis_paths.append(output_file)

                except Exception as e:
                    print(f"    Warning: Failed to generate {suffix} visualization: {e}")

        except Exception as e:
            print(f"    Warning: Visualization setup failed: {e}")

        return vis_paths

    def analyze_batch(self, midi_files: List[str]) -> Dict[str, AnalysisResult]:
        """Analyze a batch of MIDI files."""
        results = {}
        total_files = len(midi_files)

        print(f"🚀 Starting batch analysis of {total_files} files with {self.config.max_workers} workers")
        print(f"   Traditional analysis: {'✅' if self.traditional_orchestrator else '❌'}")
        print(f"   ML analysis: {'✅' if self.ml_orchestrator else '❌'}")
        print(f"   Visualization: {'✅' if self.visualizer else '❌'}")
        print()

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all analysis tasks
            future_to_file = {
                executor.submit(self.analyze_single_file, midi_file, i+1, total_files): (midi_file, i+1)
                for i, midi_file in enumerate(midi_files)
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                midi_file, file_index = future_to_file[future]
                try:
                    result = future.result()
                    results[midi_file] = result
                    completed += 1

                    if not self.config.verbose and completed % 10 == 0:
                        print(f"📊 Completed {completed}/{total_files} files")

                except Exception as e:
                    print(f"❌ Error analyzing {midi_file}: {e}")
                    results[midi_file] = AnalysisResult(
                        file_path=midi_file,
                        file_name=os.path.basename(midi_file),
                        file_size=0,
                        analysis_time=0.0,
                        errors=[str(e)],
                        success=False
                    )

        total_time = time.time() - start_time
        print(".1f")
        return results

    def generate_reports(self, results: Dict[str, AnalysisResult]):
        """Generate comprehensive reports."""
        print("\n📝 Generating reports...")

        # Generate summary statistics
        summary = self._generate_summary(results)

        # Save JSON report
        if self.config.generate_json_report:
            self._save_json_report(results, summary)

        # Generate HTML report
        if self.config.generate_html_report:
            self._generate_html_report(results, summary)

        print("✅ Reports generated successfully")

    def _generate_summary(self, results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics."""
        successful_analyses = [r for r in results.values() if r.success]
        failed_analyses = [r for r in results.values() if not r.success]

        summary = {
            'metadata': {
                'analysis_timestamp': time.time(),
                'total_files': len(results),
                'input_directory': self.config.input_directory,
                'output_directory': self.config.output_directory,
                'tools_available': {
                    'traditional_analysis': self.traditional_orchestrator is not None,
                    'ml_analysis': self.ml_orchestrator is not None,
                    'visualization': self.visualizer is not None
                }
            },
            'statistics': {
                'successful_analyses': len(successful_analyses),
                'failed_analyses': len(failed_analyses),
                'success_rate': len(successful_analyses) / len(results) if results else 0,
                'average_analysis_time': sum(r.analysis_time for r in results.values()) / len(results) if results else 0,
                'total_visualizations': sum(len(r.visualization_paths) for r in results.values())
            }
        }

        return summary

    def _save_json_report(self, results: Dict[str, AnalysisResult], summary: Dict[str, Any]):
        """Save detailed JSON report."""
        report_data = {
            'summary': summary,
            'results': {path: result.to_dict() for path, result in results.items()}
        }

        json_file = os.path.join(self.config.output_directory, "comprehensive_analysis_report.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)

    def _generate_html_report(self, results: Dict[str, AnalysisResult], summary: Dict[str, Any]):
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive MIDI Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .file-list {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .file-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px;
            border-bottom: 1px solid #eee;
            align-items: center;
        }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .tools-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .tool-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .tool-available {{ border-left: 4px solid #28a745; }}
        .tool-unavailable {{ border-left: 4px solid #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Comprehensive MIDI Analysis Report</h1>
            <p>Analysis of {summary['metadata']['total_files']} MIDI files from {os.path.basename(summary['metadata']['input_directory'])}</p>
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>🔧 Analysis Tools Used</h2>
            <div class="tools-grid">
                <div class="tool-card {'tool-available' if summary['metadata']['tools_available']['traditional_analysis'] else 'tool-unavailable'}">
                    <div>Traditional Analysis</div>
                    <div style="font-size: 1.5em;">{'✅' if summary['metadata']['tools_available']['traditional_analysis'] else '❌'}</div>
                </div>
                <div class="tool-card {'tool-available' if summary['metadata']['tools_available']['ml_analysis'] else 'tool-unavailable'}">
                    <div>ML Analysis</div>
                    <div style="font-size: 1.5em;">{'✅' if summary['metadata']['tools_available']['ml_analysis'] else '❌'}</div>
                </div>
                <div class="tool-card {'tool-available' if summary['metadata']['tools_available']['visualization'] else 'tool-unavailable'}">
                    <div>Visualization</div>
                    <div style="font-size: 1.5em;">{'✅' if summary['metadata']['tools_available']['visualization'] else '❌'}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Analysis Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="metric">{summary['statistics']['successful_analyses']}</div>
                    <div>Successful Analyses</div>
                </div>
                <div class="stat-card">
                    <div class="metric">{summary['statistics']['failed_analyses']}</div>
                    <div>Failed Analyses</div>
                </div>
                <div class="stat-card">
                    <div class="metric">{summary['statistics']['success_rate']:.1%}</div>
                    <div>Success Rate</div>
                </div>
                <div class="stat-card">
                    <div class="metric">{summary['statistics']['average_analysis_time']:.1f}s</div>
                    <div>Avg. Analysis Time</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📁 Analyzed Files ({len(results)})</h2>
            <div class="file-list">
                {"".join([f'''
                <div class="file-item">
                    <div>
                        <strong>{result.file_name}</strong>
                        <br>
                        <small>{result.file_size} bytes | {result.analysis_time:.2f}s</small>
                    </div>
                    <div class="{"success" if result.success else "error"}">
                        {"✅ " + str(len(result.errors)) + " errors" if result.errors else "✅ Success"}
                    </div>
                </div>''' for result in results.values()])}
            </div>
        </div>
    </div>
</body>
</html>"""

        html_file = os.path.join(self.config.output_directory, "comprehensive_analysis_report.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive MIDI Analyzer - Production Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comprehensive_midi_analyzer.py reference_midis/midi4
  python comprehensive_midi_analyzer.py reference_midis/midi4 --output my_analysis --workers 8
  python comprehensive_midi_analyzer.py reference_midis/midi4 --no-ml --sample-viz 5 --verbose
        """
    )

    parser.add_argument("input_directory", help="Directory containing MIDI files")
    parser.add_argument("--output", "-o", default="comprehensive_analysis_output",
                       help="Output directory for results")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Maximum number of worker threads")
    parser.add_argument("--sample-viz", type=int, default=3,
                       help="Number of sample files for visualization")
    parser.add_argument("--no-traditional", action="store_true",
                       help="Disable traditional analysis")
    parser.add_argument("--no-ml", action="store_true",
                       help="Disable ML analysis")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization generation")
    parser.add_argument("--no-html", action="store_true",
                       help="Disable HTML report generation")
    parser.add_argument("--no-json", action="store_true",
                       help="Disable JSON report generation")
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
        verbose=args.verbose,
        generate_html_report=not args.no_html,
        generate_json_report=not args.no_json
    )

    try:
        # Initialize analyzer
        analyzer = ComprehensiveMidiAnalyzer(config)

        # Discover MIDI files
        midi_files = analyzer.discover_midi_files()
        if not midi_files:
            print("❌ No MIDI files found. Exiting.")
            sys.exit(1)

        print(f"📂 Found {len(midi_files)} MIDI files in {args.input_directory}")

        # Run comprehensive analysis
        results = analyzer.analyze_batch(midi_files)

        # Generate reports
        analyzer.generate_reports(results)

        # Print final summary
        successful = sum(1 for r in results.values() if r.success)
        total_time = sum(r.analysis_time for r in results.values())

        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Total files processed: {len(results)}")
        print(f"✅ Successful analyses: {successful}")
        print(f"❌ Failed analyses: {len(results) - successful}")
        print(".1f")
        print(".1f")
        print(f"📁 Results saved to: {config.output_directory}")

        if config.include_visualization and any(len(r.visualization_paths) > 0 for r in results.values()):
            print(f"🎨 Visualizations generated: {sum(len(r.visualization_paths) for r in results.values())}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()