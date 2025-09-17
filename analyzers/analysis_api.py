#!/usr/bin/env python3
"""
Comprehensive Analysis API

Purpose:
- Unified interface for all analysis features
- Orchestrate complex analysis workflows
- Provide structured results and recommendations
- Support real-time analysis capabilities

Features:
- Integrated analysis pipeline
- Batch processing capabilities
- Real-time analysis support
- Performance monitoring and optimization
- Comprehensive reporting
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import mido

# New: lightweight ingestion helper imports
from data_store.pattern_repository import PatternRepository
try:
    from .store_writer import (
        get_repository as _get_repo,
        upsert_source_for_midi as _upsert_source_for_midi,
        upsert_patterns as _upsert_patterns,
    )
    from .midi_pattern_extractor import extract_rhythm_patterns_from_midi
except Exception:
    # Fallback when running as a script outside package context
    from analyzers.store_writer import (  # type: ignore
        get_repository as _get_repo,
        upsert_source_for_midi as _upsert_source_for_midi,
        upsert_patterns as _upsert_patterns,
    )
    from analyzers.midi_pattern_extractor import extract_rhythm_patterns_from_midi  # type: ignore


def analyze_midi_and_store(
    midi_path: str,
    instrument_hint: str = "drums",
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    db_path: Optional[str] = None,
    repository: Optional[PatternRepository] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    High-level helper: load MIDI → extract rhythm patterns → upsert into repository.

    Args:
        midi_path: Path to the MIDI file to analyze and ingest.
        instrument_hint: Instrument category; defaults to 'drums'.
        genre: Optional genre to attach to stored patterns (default 'unknown' if None).
        mood: Optional mood label.
        db_path: Optional SQLite DB path; ignored if 'repository' is provided.
        repository: Optional pre-opened PatternRepository instance to use.
        tags: Optional list of tags to attach to persisted patterns.

    Returns:
        {
          "source_id": int,
          "pattern_ids": List[int],
          "patterns_extracted": int
        }
    """
    # 1) open repository (argument repository overrides db_path; else default)
    repo_created = False
    repo = repository
    if repo is None:
        repo = _get_repo(db_path=db_path)
        repo_created = True

    try:
        # 2) call extractor to get pattern dicts (read-only; do not persist here)
        extracted = extract_rhythm_patterns_from_midi(
            midi_path,
            instrument_hint=instrument_hint,
            genre=genre,
            mood=mood,
            tags=tags,
        )

        # If no patterns, still upsert a source row for idempotency/traceability
        # 3) upsert source row
        source_id = _upsert_source_for_midi(repo, midi_path, track_name=None)

        # 4) iterate extracted patterns and upsert each into rhythm_patterns
        pattern_ids: List[int] = []
        if extracted:
            pattern_ids = _upsert_patterns(repo, source_id, extracted)

        return {
            "source_id": source_id,
            "pattern_ids": pattern_ids,
            "patterns_extracted": len(extracted),
        }
    finally:
        # Close only if we opened it here
        if repo_created:
            try:
                repo.close()
            except Exception:
                pass


class AnalysisOrchestrator:
    """Main orchestrator for comprehensive music analysis."""

    def __init__(self):
        # Initialize all analyzers
        self.metrics_analyzer = None
        self.reference_library = None
        self.mix_readiness_analyzer = None
        self.workflow_integrator = None

        # Load analyzers dynamically to avoid import issues
        self._load_analyzers()

    def _load_analyzers(self):
        """Load available analyzers dynamically."""
        try:
            from simple_metrics_demo import SimpleAnalyzer
            self.metrics_analyzer = SimpleAnalyzer()
        except ImportError:
            print("Metrics analyzer not available")

        try:
            from reference_library_simple import SimpleReferenceLibrary
            self.reference_library = SimpleReferenceLibrary()
        except ImportError:
            print("Reference library not available")

        try:
            from mix_readiness import MixReadinessAnalyzer
            self.mix_readiness_analyzer = MixReadinessAnalyzer()
        except ImportError:
            print("Mix readiness analyzer not available")

    def analyze_comprehensive(self, file_path: str, target_genre: str = "pop",
                            include_reference_matching: bool = True,
                            include_mix_readiness: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a music file.

        Args:
            file_path: Path to music file to analyze
            target_genre: Target genre for genre-specific analysis
            include_reference_matching: Whether to include reference library matching
            include_mix_readiness: Whether to include mix readiness analysis

        Returns:
            Comprehensive analysis results
        """

        start_time = time.time()
        results = {
            'file_path': file_path,
            'target_genre': target_genre,
            'timestamp': time.time(),
            'analyses_performed': [],
            'results': {},
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

        # Perform genre consistency analysis
        if self.metrics_analyzer:
            try:
                genre_result = self.metrics_analyzer.analyze_file(file_path, target_genre)
                results['results']['genre_consistency'] = genre_result
                results['analyses_performed'].append('genre_consistency')
            except Exception as e:
                results['results']['genre_consistency_error'] = str(e)

        # Perform reference library matching
        if include_reference_matching and self.reference_library:
            try:
                matches = self.reference_library.find_similar_tracks(file_path, target_genre)
                results['results']['reference_matches'] = matches
                results['analyses_performed'].append('reference_matching')
            except Exception as e:
                results['results']['reference_matching_error'] = str(e)

        # Perform trend analysis
        if self.reference_library:
            try:
                trends = self.reference_library.analyze_trends()
                results['results']['genre_trends'] = trends
                results['analyses_performed'].append('trend_analysis')
            except Exception as e:
                results['results']['trend_analysis_error'] = str(e)

        # Perform mix readiness analysis
        if include_mix_readiness and self.mix_readiness_analyzer:
            try:
                mix_analysis = self.mix_readiness_analyzer.analyze_mix_readiness(file_path)
                results['results']['mix_readiness'] = mix_analysis
                results['analyses_performed'].append('mix_readiness')
            except Exception as e:
                results['results']['mix_readiness_error'] = str(e)

        # Calculate overall assessment
        results['overall_assessment'] = self._calculate_overall_assessment(results)

        # Performance metrics
        end_time = time.time()
        results['performance_metrics'] = {
            'total_time_seconds': end_time - start_time,
            'analyses_count': len(results['analyses_performed'])
        }

        return results

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

    def _calculate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall assessment from all analyses."""

        assessment = {
            'overall_score': 0.0,
            'grade': 'N/A',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }

        scores = []
        available_analyses = results.get('analyses_performed', [])

        # Genre consistency score
        if 'genre_consistency' in results['results']:
            genre_result = results['results']['genre_consistency']
            if hasattr(genre_result, 'overall_score'):
                scores.append(genre_result.overall_score)
                assessment['strengths'].append("Genre consistency analysis completed")

        # Mix readiness score
        if 'mix_readiness' in results['results']:
            mix_result = results['results']['mix_readiness']
            if 'overall_readiness_score' in mix_result:
                scores.append(mix_result['overall_readiness_score'])
                assessment['strengths'].append("Mix readiness analysis completed")

        # Reference matching
        if 'reference_matches' in results['results']:
            matches = results['results']['reference_matches']
            if matches:
                best_match = max(matches, key=lambda x: x['similarity_score'])
                scores.append(best_match['similarity_score'] * 0.5)  # Weight reference matching lower
                assessment['strengths'].append("Reference library matching completed")

        # Calculate overall score
        if scores:
            assessment['overall_score'] = sum(scores) / len(scores)

            # Assign grade
            if assessment['overall_score'] >= 0.9:
                assessment['grade'] = 'A+'
                assessment['strengths'].append("Excellent overall quality")
            elif assessment['overall_score'] >= 0.8:
                assessment['grade'] = 'A'
                assessment['strengths'].append("Very good overall quality")
            elif assessment['overall_score'] >= 0.7:
                assessment['grade'] = 'B'
                assessment['strengths'].append("Good overall quality")
            elif assessment['overall_score'] >= 0.6:
                assessment['grade'] = 'C'
                assessment['weaknesses'].append("Moderate quality issues")
            else:
                assessment['grade'] = 'D/F'
                assessment['weaknesses'].append("Significant quality issues")

        # Generate recommendations based on missing analyses
        expected_analyses = ['genre_consistency', 'reference_matching', 'mix_readiness']
        missing_analyses = [a for a in expected_analyses if a not in available_analyses]

        if missing_analyses:
            assessment['recommendations'].append(
                f"Consider running additional analyses: {', '.join(missing_analyses)}"
            )

        return assessment

    def batch_analyze(self, file_paths: List[str], target_genre: str = "pop",
                     max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Perform batch analysis of multiple files.

        Args:
            file_paths: List of file paths to analyze
            target_genre: Target genre for analysis
            max_workers: Maximum number of concurrent workers

        Returns:
            List of analysis results
        """

        def analyze_single_file(file_path: str) -> Dict[str, Any]:
            return self.analyze_comprehensive(file_path, target_genre)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(analyze_single_file, file_paths))

        return results

    def get_analysis_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of batch analysis results."""

        if not results:
            return {'error': 'No results to summarize'}

        summary = {
            'total_files': len(results),
            'successful_analyses': len([r for r in results if 'error' not in r]),
            'failed_analyses': len([r for r in results if 'error' in r]),
            'average_scores': {},
            'grade_distribution': {},
            'performance_stats': {}
        }

        # Calculate averages
        overall_scores = []
        analysis_times = []

        for result in results:
            if 'overall_assessment' in result:
                assessment = result['overall_assessment']
                if 'overall_score' in assessment:
                    overall_scores.append(assessment['overall_score'])

                if 'grade' in assessment:
                    grade = assessment['grade']
                    summary['grade_distribution'][grade] = summary['grade_distribution'].get(grade, 0) + 1

            if 'performance_metrics' in result:
                perf = result['performance_metrics']
                if 'total_time_seconds' in perf:
                    analysis_times.append(perf['total_time_seconds'])

        if overall_scores:
            summary['average_scores']['overall'] = sum(overall_scores) / len(overall_scores)

        if analysis_times:
            summary['performance_stats'] = {
                'average_time_per_file': sum(analysis_times) / len(analysis_times),
                'total_time': sum(analysis_times),
                'min_time': min(analysis_times),
                'max_time': max(analysis_times)
            }

        return summary


class AnalysisPipeline:
    """Advanced analysis pipeline with custom workflows."""

    def __init__(self):
        self.orchestrator = AnalysisOrchestrator()
        self.pipelines = self._load_default_pipelines()

    def _load_default_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Load default analysis pipelines."""

        return {
            'quick_check': {
                'name': 'Quick Quality Check',
                'description': 'Fast analysis for basic quality assessment',
                'steps': ['genre_consistency'],
                'target_genre': 'pop'
            },
            'comprehensive': {
                'name': 'Comprehensive Analysis',
                'description': 'Full analysis suite for detailed assessment',
                'steps': ['genre_consistency', 'reference_matching', 'mix_readiness'],
                'target_genre': 'auto'
            },
            'production_ready': {
                'name': 'Production Readiness',
                'description': 'Focus on mix and production quality',
                'steps': ['mix_readiness', 'reference_matching'],
                'target_genre': 'auto'
            },
            'genre_matching': {
                'name': 'Genre Matching',
                'description': 'Focus on genre consistency and reference matching',
                'steps': ['genre_consistency', 'reference_matching'],
                'target_genre': 'auto'
            }
        }

    def run_pipeline(self, pipeline_name: str, file_path: str,
                    custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a specific analysis pipeline.

        Args:
            pipeline_name: Name of the pipeline to run
            file_path: File to analyze
            custom_config: Custom configuration overrides

        Returns:
            Analysis results
        """

        if pipeline_name not in self.pipelines:
            return {'error': f"Pipeline '{pipeline_name}' not found"}

        pipeline = self.pipelines[pipeline_name].copy()

        # Apply custom configuration
        if custom_config:
            pipeline.update(custom_config)

        # Determine target genre
        target_genre = pipeline.get('target_genre', 'pop')
        if target_genre == 'auto':
            # Auto-detect genre from file path or analysis
            target_genre = self._auto_detect_genre(file_path)

        # Configure analysis based on pipeline steps
        steps = pipeline.get('steps', [])
        include_reference = 'reference_matching' in steps
        include_mix = 'mix_readiness' in steps

        # Run analysis
        result = self.orchestrator.analyze_comprehensive(
            file_path=file_path,
            target_genre=target_genre,
            include_reference_matching=include_reference,
            include_mix_readiness=include_mix
        )

        # Add pipeline information
        result['pipeline'] = {
            'name': pipeline['name'],
            'description': pipeline['description'],
            'steps_executed': steps,
            'target_genre': target_genre
        }

        return result

    def _auto_detect_genre(self, file_path: str) -> str:
        """Auto-detect genre from file path and content."""
        filename = os.path.basename(file_path).lower()

        # Simple genre detection from filename
        if 'jazz' in filename:
            return 'jazz'
        elif 'rock' in filename:
            return 'rock'
        elif 'pop' in filename:
            return 'pop'
        elif 'electronic' in filename or 'dnb' in filename:
            return 'electronic'
        elif 'hip' in filename:
            return 'hip-hop'
        elif 'classical' in filename:
            return 'classical'
        else:
            return 'pop'  # Default


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Analysis API")
    parser.add_argument("--input", help="Input MIDI file")
    parser.add_argument("--batch", nargs="+", help="Batch input files")
    parser.add_argument("--pipeline", default="comprehensive",
                       choices=["quick_check", "comprehensive", "production_ready", "genre_matching"],
                       help="Analysis pipeline to use")
    parser.add_argument("--genre", default="pop", help="Target genre")
    parser.add_argument("--output", default="test_outputs", help="Output directory")
    parser.add_argument("--format", default="json", choices=["json", "html"],
                       help="Output format")

    args = parser.parse_args()

    # Initialize API
    api = AnalysisPipeline()

    results = []

    if args.input:
        # Single file analysis
        print(f"Running {args.pipeline} pipeline on {args.input}")
        result = api.run_pipeline(args.pipeline, args.input)
        results.append(result)

    elif args.batch:
        # Batch analysis
        print(f"Running {args.pipeline} pipeline on {len(args.batch)} files")
        orchestrator = AnalysisOrchestrator()

        for file_path in args.batch:
            result = api.run_pipeline(args.pipeline, file_path)
            results.append(result)

    else:
        print("No input files specified. Use --input or --batch")
        return

    # Generate summary
    if len(results) > 1:
        summary = orchestrator.get_analysis_summary(results)
        print("Batch Summary:")
        print(f"  Files analyzed: {summary['total_files']}")
        print(f"  Successful: {summary['successful_analyses']}")
        print(f"  Failed: {summary['failed_analyses']}")

        if 'average_scores' in summary and 'overall' in summary['average_scores']:
            print(f"  Average Score: {summary['average_scores']['overall']:.2f}")

        if 'performance_stats' in summary:
            perf = summary['performance_stats']
            print(f"  Avg Analysis Time: {perf['average_time_per_file']:.2f}s")

    # Save results
    os.makedirs(args.output, exist_ok=True)

    if args.format == "json":
        output_file = os.path.join(args.output, "comprehensive_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
    else:
        # Simple HTML output
        output_file = os.path.join(args.output, "comprehensive_analysis.html")
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Analysis Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .result {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
        .score {{ font-weight: bold; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>Comprehensive Analysis Results</h1>
    <p>Analysis Pipeline: {args.pipeline}</p>
    <p>Files Analyzed: {len(results)}</p>

    {"".join([f'''
    <div class="result">
        <h3>{os.path.basename(r['file_path'])}</h3>
        {"<p class='error'>Error: " + r.get('error', 'Unknown error') + "</p>" if 'error' in r else ""}
        {"<p>Overall Score: <span class='score'>" + str(r.get('overall_assessment', {}).get('overall_score', 'N/A')) + "</span></p>" if 'overall_assessment' in r else ""}
        {"<p>Grade: <span class='score'>" + str(r.get('overall_assessment', {}).get('grade', 'N/A')) + "</span></p>" if 'overall_assessment' in r else ""}
    </div>
    ''' for r in results])}
</body>
</html>"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()