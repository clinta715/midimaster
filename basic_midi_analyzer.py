#!/usr/bin/env python3
"""
Basic MIDI Analyzer - Working Version

This script provides basic analysis of MIDI files with file discovery,
batch processing, and HTML report generation.
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

# Import basic MIDI parsing
try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


@dataclass
class AnalysisConfig:
    """Configuration for the analysis."""
    input_directory: str
    output_directory: str = "basic_midi_analysis_output"
    max_workers: int = 4
    verbose: bool = False


@dataclass
class MidiAnalysis:
    """Basic analysis of a MIDI file."""
    file_path: str
    file_name: str
    file_size: int
    analysis_time: float
    midi_info: Optional[Dict[str, Any]] = None
    note_info: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_size': self.file_size,
            'analysis_time': self.analysis_time,
            'midi_info': self.midi_info,
            'note_info': self.note_info,
            'errors': self.errors,
            'success': self.success
        }


class BasicMidiAnalyzer:
    """Basic MIDI file analyzer."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._ensure_output_directory()

        if not MIDO_AVAILABLE:
            print("⚠️  MIDI parsing not available - install mido for MIDI analysis")
        else:
            print("✅ MIDI parsing available")

    def _ensure_output_directory(self):
        """Create output directory."""
        os.makedirs(self.config.output_directory, exist_ok=True)

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

    def analyze_single_file(self, midi_file: str, file_index: int, total_files: int) -> MidiAnalysis:
        """Analyze a single MIDI file."""
        start_time = time.time()
        file_name = os.path.basename(midi_file)

        if self.config.verbose:
            print(f"[{file_index}/{total_files}] Analyzing: {file_name}")

        analysis = MidiAnalysis(
            file_path=midi_file,
            file_name=file_name,
            file_size=os.path.getsize(midi_file),
            analysis_time=0.0
        )

        try:
            if not MIDO_AVAILABLE:
                analysis.errors.append("MIDI parsing not available")
                analysis.success = False
                return analysis

            # Parse MIDI file
            midi_data = mido.MidiFile(midi_file)

            # Basic MIDI info
            analysis.midi_info = {
                'format': midi_data.type,
                'tracks': len(midi_data.tracks),
                'ticks_per_beat': midi_data.ticks_per_beat,
                'length_seconds': self._calculate_midi_length(midi_data)
            }

            # Analyze notes
            analysis.note_info = self._analyze_notes(midi_data)

        except Exception as e:
            analysis.errors.append(f"MIDI parsing failed: {str(e)}")
            analysis.success = False
            if self.config.verbose:
                print(f"  ❌ Analysis failed: {e}")

        analysis.analysis_time = time.time() - start_time
        return analysis

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

    def _analyze_notes(self, midi_data: mido.MidiFile) -> Dict[str, Any]:
        """Analyze notes in the MIDI file."""
        notes = []
        total_notes = 0
        pitch_range = []
        velocity_sum = 0
        channels_used = set()

        for track in midi_data.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    total_notes += 1
                    pitch_range.append(msg.note)
                    velocity_sum += msg.velocity
                    channels_used.add(msg.channel)

        note_info = {
            'total_notes': total_notes,
            'unique_pitches': len(set(pitch_range)) if pitch_range else 0,
            'pitch_range': {
                'min': min(pitch_range) if pitch_range else 0,
                'max': max(pitch_range) if pitch_range else 127,
                'span': max(pitch_range) - min(pitch_range) if pitch_range else 0
            },
            'average_velocity': velocity_sum / total_notes if total_notes > 0 else 0,
            'channels_used': sorted(list(channels_used))
        }

        return note_info

    def analyze_batch(self, midi_files: List[str]) -> Dict[str, MidiAnalysis]:
        """Analyze a batch of MIDI files."""
        results = {}
        total_files = len(midi_files)

        print(f"🚀 Starting batch analysis of {total_files} files with {self.config.max_workers} workers")
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
                    results[midi_file] = MidiAnalysis(
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

    def generate_reports(self, results: Dict[str, MidiAnalysis]):
        """Generate comprehensive reports."""
        print("\n📝 Generating reports...")

        # Generate summary statistics
        summary = self._generate_summary(results)

        # Save JSON report
        self._save_json_report(results, summary)

        # Generate HTML report
        self._generate_html_report(results, summary)

        print("✅ Reports generated successfully")

    def _generate_summary(self, results: Dict[str, MidiAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics."""
        successful_analyses = [r for r in results.values() if r.success]
        failed_analyses = [r for r in results.values() if not r.success]

        # Calculate aggregate statistics
        total_notes = sum(r.note_info['total_notes'] for r in successful_analyses if r.note_info)
        avg_velocity = sum(r.note_info['average_velocity'] for r in successful_analyses if r.note_info) / len(successful_analyses) if successful_analyses else 0

        summary = {
            'metadata': {
                'analysis_timestamp': time.time(),
                'total_files': len(results),
                'input_directory': self.config.input_directory,
                'output_directory': self.config.output_directory,
                'midi_parsing_available': MIDO_AVAILABLE
            },
            'statistics': {
                'successful_analyses': len(successful_analyses),
                'failed_analyses': len(failed_analyses),
                'success_rate': len(successful_analyses) / len(results) if results else 0,
                'average_analysis_time': sum(r.analysis_time for r in results.values()) / len(results) if results else 0,
                'total_notes_analyzed': total_notes,
                'average_velocity': avg_velocity
            },
            'file_summaries': {}
        }

        # File summaries
        summary['file_summaries'] = {
            os.path.basename(path): {
                'size': result.file_size,
                'analysis_time': result.analysis_time,
                'notes': result.note_info['total_notes'] if result.note_info else 0,
                'errors': result.errors
            }
            for path, result in results.items()
        }

        return summary

    def _save_json_report(self, results: Dict[str, MidiAnalysis], summary: Dict[str, Any]):
        """Save detailed JSON report."""
        report_data = {
            'summary': summary,
            'results': {path: result.to_dict() for path, result in results.items()}
        }

        json_file = os.path.join(self.config.output_directory, "midi_analysis_report.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)

    def _generate_html_report(self, results: Dict[str, MidiAnalysis], summary: Dict[str, Any]):
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Basic MIDI Analysis Report</title>
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎵 Basic MIDI Analysis Report</h1>
            <p>Analysis of {summary['metadata']['total_files']} MIDI files from {os.path.basename(summary['metadata']['input_directory'])}</p>
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
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
                    <div class="metric">{summary['statistics']['total_notes_analyzed']:,}</div>
                    <div>Total Notes</div>
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
                        <small>{result.file_size} bytes | {result.analysis_time:.2f}s | {result.note_info["total_notes"] if result.note_info else 0} notes</small>
                    </div>
                    <div class="{"success" if result.success else "error"}">
                        {"✅ Success" if result.success else "❌ " + str(len(result.errors)) + " errors"}
                    </div>
                </div>''' for result in results.values()])}
            </div>
        </div>
    </div>
</body>
</html>"""

        html_file = os.path.join(self.config.output_directory, "midi_analysis_report.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Basic MIDI Analyzer - Working Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python basic_midi_analyzer.py reference_midis/midi4
  python basic_midi_analyzer.py reference_midis/midi4 --output my_analysis --workers 8
  python basic_midi_analyzer.py reference_midis/midi4 --verbose
        """
    )

    parser.add_argument("input_directory", help="Directory containing MIDI files")
    parser.add_argument("--output", "-o", default="basic_midi_analysis_output",
                       help="Output directory for results")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Maximum number of worker threads")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Create configuration
    config = AnalysisConfig(
        input_directory=args.input_directory,
        output_directory=args.output,
        max_workers=args.workers,
        verbose=args.verbose
    )

    try:
        # Initialize analyzer
        analyzer = BasicMidiAnalyzer(config)

        # Discover MIDI files
        midi_files = analyzer.discover_midi_files()
        if not midi_files:
            print("❌ No MIDI files found. Exiting.")
            sys.exit(1)

        print(f"📂 Found {len(midi_files)} MIDI files in {args.input_directory}")

        # Run analysis
        results = analyzer.analyze_batch(midi_files)

        # Generate reports
        analyzer.generate_reports(results)

        # Print final summary
        successful = sum(1 for r in results.values() if r.success)
        total_time = sum(r.analysis_time for r in results.values())
        total_notes = sum(r.note_info['total_notes'] for r in results.values() if r.note_info)

        print("\n" + "="*60)
        print("🎉 BASIC MIDI ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Total files processed: {len(results)}")
        print(f"✅ Successful analyses: {successful}")
        print(f"❌ Failed analyses: {len(results) - successful}")
        print(f"🎵 Total notes analyzed: {total_notes:,}")
        print(".1f")
        print(".1f")
        print(f"📁 Results saved to: {config.output_directory}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()