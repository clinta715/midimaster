import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from core.rhythms_db_resolver import RhythmsDbResolver
import pickle
import numpy as np
from collections import defaultdict, OrderedDict
from functools import lru_cache
from analyzers.midi_pattern_extractor import MidiPatternData, NoteEvent, TempoEvent, TimeSignatureEvent, extract_from_directory, extract_from_file
import mido
import pandas as pd
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import psutil
from functools import wraps


@dataclass
class PatternMetadata:
    """Additional metadata for patterns."""
    bpm: float
    time_signature: Tuple[int, int]
    complexity: float  # Notes per beat as proxy
    genre: str
    instrument_type: str
    key: Optional[str] = None  # Simple key, e.g., 'C major'
    pattern_category: str = 'rhythmic'
    chord_types: Optional[List[str]] = None
    harmonic_progression: Optional[str] = None
    melodic_intervals: Optional[List[int]] = None
    scale_degrees: Optional[List[int]] = None
    contour: Optional[str] = None

    def validate(self) -> Dict[str, List[str]]:
        """Validate metadata fields."""
        errors = []
        warnings = []

        # BPM validation
        if not isinstance(self.bpm, (int, float)) or not 40 <= self.bpm <= 250:
            errors.append(f"Invalid BPM: {self.bpm} (must be 40-250)")
        elif not 60 <= self.bpm <= 200:
            warnings.append(f"BPM {self.bpm} outside common range (60-200)")

        # Time signature
        if not isinstance(self.time_signature, tuple) or len(self.time_signature) != 2:
            errors.append("Time signature must be a tuple of two integers")
        else:
            num, den = self.time_signature
            if not (isinstance(num, int) and isinstance(den, int)):
                errors.append("Time signature components must be integers")
            elif num < 1 or num > 12 or den not in [1, 2, 4, 8, 16, 32]:
                errors.append(f"Invalid time signature {self.time_signature}")

        # Complexity
        if not isinstance(self.complexity, (int, float)) or self.complexity < 0:
            errors.append(f"Invalid complexity: {self.complexity} (must be >= 0)")

        # Genre
        if not isinstance(self.genre, str) or not self.genre.strip():
            errors.append("Genre must be a non-empty string")

        # Instrument type
        if not isinstance(self.instrument_type, str) or not self.instrument_type.strip():
            errors.append("Instrument type must be a non-empty string")

        # Optional fields
        if self.key is not None and not isinstance(self.key, str):
            warnings.append("Key should be a string or None")

        if self.chord_types is not None and not isinstance(self.chord_types, list):
            errors.append("Chord types must be a list or None")

        if self.melodic_intervals is not None and not isinstance(self.melodic_intervals, list):
            errors.append("Melodic intervals must be a list or None")

        return {"errors": errors, "warnings": warnings}
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for DF storage."""
        d = asdict(self)
        d['time_signature'] = str(d['time_signature'])
        if d['chord_types']:
            d['chord_types'] = json.dumps(d['chord_types'])
        if d['melodic_intervals']:
            d['melodic_intervals'] = json.dumps(d['melodic_intervals'])
        if d['scale_degrees']:
            d['scale_degrees'] = json.dumps(d['scale_degrees'])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PatternMetadata':
        """Create from dict (for DF row)."""
        for key in ['chord_types', 'melodic_intervals', 'scale_degrees']:
            if key in d and isinstance(d[key], str):
                d[key] = json.loads(d[key]) if d[key] else None
        if isinstance(d['time_signature'], str):
            d['time_signature'] = tuple(map(int, d['time_signature'].strip('()').split(',')))
        return cls(**d)


def timing_decorator(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


class ReferencePatternLibrary:
    """
    Management system for MIDI patterns.
    Organizes by (genre, instrument_type) -> list of patterns or file paths (lazy mode).
    Supports optimization levels: 'eager' (default, loads all into memory), 'lazy' (loads on demand).
    Uses pandas DataFrame for efficient metadata storage and querying.
    Includes indexing for fast retrieval by bpm, complexity bins, genre, instrument.
    Implements LRU caching for pattern loading to manage memory.
    Enhanced with ANN for similarity, PCA for dimensionality reduction, parallel computation, incremental extraction.
    """
    def __init__(self, optimization_level: str = 'eager', cache_size: int = 100, use_pca: bool = True, n_components: int = 5, use_ann: bool = True, n_neighbors: int = 10, parallel: bool = True, max_workers: Optional[int] = None, max_memory_mb: Optional[float] = None, eviction_policy: str = 'lru', enable_compression: bool = False, serialization_format: str = 'pickle'):
        self.optimization_level = optimization_level
        self.cache_size = cache_size
        self.use_pca = use_pca
        self.n_components = n_components
        self.use_ann = use_ann
        self.n_neighbors = n_neighbors
        self.parallel = parallel
        self.max_workers = max_workers or min(32, os.cpu_count() or 1)
        self.metadata: Dict[str, PatternMetadata] = {}
        self.metadata_df: Optional[pd.DataFrame] = None
        self.indices: Dict[str, Dict[Any, List[str]]] = {}
        self.pattern_cache: OrderedDict = OrderedDict()
        self.feature_vectors: Dict[str, np.ndarray] = {}
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.ann_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine') if use_ann else None
        self._fit_ann = False
        self.pca_fitted = False
        self.max_memory_mb = max_memory_mb
        self.eviction_policy = eviction_policy
        self.access_counts = defaultdict(int) if eviction_policy == 'lfu' else None
        if optimization_level == 'eager':
            self.patterns: Dict[Tuple[str, str], List[MidiPatternData]] = {}
            self.file_paths: Dict[Tuple[str, str], List[str]] = {}
        else:
            self.file_paths: Dict[Tuple[str, str], List[str]] = {}

    def _enforce_memory_limit(self):
        """
        Enforce memory limits by evicting patterns based on policy.
        """
        # First, enforce cache size limit
        while len(self.pattern_cache) > self.cache_size and self.pattern_cache:
            if self.eviction_policy == 'lru':
                old_key = self.pattern_cache.popitem(last=False)[0]
            elif self.eviction_policy == 'lfu':
                if self.access_counts:
                    min_count = min(self.access_counts.values())
                    candidates = [k for k, v in self.access_counts.items() if v == min_count]
                    if candidates:
                        def get_size(kk):
                            if kk in self.pattern_cache:
                                pat = self.pattern_cache[kk]
                                return pat.length_ticks if hasattr(pat, 'length_ticks') else 0
                            return 0
                        old_key = min(candidates, key=get_size)
                        if self.access_counts is not None and old_key in self.access_counts:
                            del self.access_counts[old_key]
                    else:
                        old_key = self.pattern_cache.popitem(last=False)[0]
                        if self.access_counts is not None and old_key in self.access_counts:
                            del self.access_counts[old_key]
                else:
                    old_key = self.pattern_cache.popitem(last=False)[0]
                    if self.access_counts is not None and old_key in self.access_counts:
                        del self.access_counts[old_key]
            else:
                old_key = self.pattern_cache.popitem(last=False)[0]
                if self.access_counts is not None and old_key in self.access_counts:
                    del self.access_counts[old_key]
            
            if old_key in self.feature_vectors:
                self.feature_vectors.pop(old_key, None)

        if self.max_memory_mb is None:
            return
        current_mem = memory_usage()
        while current_mem > self.max_memory_mb and self.pattern_cache:
            if self.eviction_policy == 'lru':
                old_key = self.pattern_cache.popitem(last=False)[0]
            elif self.eviction_policy == 'lfu':
                if self.access_counts:
                    min_count = min(self.access_counts.values())
                    candidates = [k for k, v in self.access_counts.items() if v == min_count]
                    if candidates:
                        def get_size_mem(kk):
                            if kk in self.pattern_cache:
                                pat = self.pattern_cache[kk]
                                return pat.length_ticks if hasattr(pat, 'length_ticks') else 0
                            return 0
                        old_key = min(candidates, key=get_size_mem)
                        if self.access_counts is not None and old_key in self.access_counts:
                            del self.access_counts[old_key]
                    else:
                        old_key = self.pattern_cache.popitem(last=False)[0]
                        if self.access_counts is not None and old_key in self.access_counts:
                            del self.access_counts[old_key]
                else:
                    old_key = self.pattern_cache.popitem(last=False)[0]
                    if self.access_counts is not None and old_key in self.access_counts:
                        del self.access_counts[old_key]
            else:
                old_key = self.pattern_cache.popitem(last=False)[0]
                if self.access_counts is not None and old_key in self.access_counts:
                    del self.access_counts[old_key]
            
            if old_key in self.feature_vectors:
                self.feature_vectors.pop(old_key, None)
            # Keep metadata as it's lightweight
            current_mem = memory_usage()
            if current_mem <= self.max_memory_mb:
                break
        if current_mem > self.max_memory_mb * 0.95:
            print(f"Warning: Memory usage {current_mem:.1f}MB approaching limit {self.max_memory_mb}MB")

    def get_metadata(self, pattern: Union[MidiPatternData, str]) -> PatternMetadata:
        """Get metadata for a pattern or file path (backward compatible)."""
        key = pattern.file_path if isinstance(pattern, MidiPatternData) else pattern
        return self.metadata.get(key, PatternMetadata(120.0, (4, 4), 0.0, 'unknown', 'unknown'))

    def _update_df(self):
        """Update DataFrame from metadata dict."""
        if not self.metadata:
            self.metadata_df = pd.DataFrame()
            return
        df_data = [m.to_dict() for m in self.metadata.values()]
        self.metadata_df = pd.DataFrame(df_data, index=list(self.metadata.keys()))
        self.metadata_df.index.name = 'file_path'

    def _build_indices(self):
        """Build indices for fast retrieval."""
        self.indices = {}
        if self.metadata_df is None or self.metadata_df.empty:
            return
        
        self.indices['genre'] = self.metadata_df.groupby('genre').apply(lambda g: g.index.tolist()).to_dict()
        
        self.indices['instrument'] = self.metadata_df.groupby('instrument_type').apply(lambda g: g.index.tolist()).to_dict()
        
        self.metadata_df['bpm_bin'] = pd.cut(self.metadata_df['bpm'], bins=[0, 80, 100, 120, 140, 160, 200], labels=['<80', '80-100', '100-120', '120-140', '140-160', '160+'])
        self.indices['bpm_bin'] = self.metadata_df.groupby('bpm_bin').apply(lambda g: g.index.tolist()).to_dict()
        
        self.metadata_df['complexity_bin'] = pd.cut(self.metadata_df['complexity'], bins=[0, 0.5, 2.0, np.inf], labels=['low', 'med', 'high'])
        self.indices['complexity_bin'] = self.metadata_df.groupby('complexity_bin').apply(lambda g: g.index.tolist()).to_dict()

    def classify_genre(self, file_path: str) -> str:
        """Categorize genre based on directory naming."""
        path = Path(file_path)
        genre_dirs = {
            'aba_pack': 'rock',
            'midi4': 'panoramic',
            'midi5': 'deep_water',
            'midi6': 'trap_808',
            'midi8': 'trap_808_v2',
        }
        parent = path.parent.name
        return genre_dirs.get(parent, parent)

    def classify_instrument_from_filename(self, file_path: str) -> str:
        """Classify instrument type from filename for lazy loading."""
        filename = Path(file_path).stem.lower()
        if any(word in filename for word in ['kick', 'snare', 'hat', 'clap', 'perc', 'drum']):
            return 'drums'
        if any(word in filename for word in ['bass', '808']):
            return 'bass'
        if any(word in filename for word in ['piano chords', 'strings', 'brass', 'pad', 'chord']):
            return 'harmony'
        if any(word in filename for word in ['synth', 'piano', 'violin', 'lead', 'pluck', 'bells', 'melody']):
            return 'melody'
        return 'melody'

    def classify_instrument_type(self, pattern: MidiPatternData) -> str:
        """Classify instrument type: drums, bass, harmony, melody."""
        if not pattern.notes:
            return 'unknown'
        
        instruments = set(n.instrument for n in pattern.notes if n.instrument)
        channels = set(n.channel for n in pattern.notes)
        
        if 9 in channels or any('drum' in i.lower() or 'perc' in i.lower() for i in instruments):
            return 'drums'
        
        avg_pitch = np.mean([n.note for n in pattern.notes])
        if avg_pitch < 60 or any('bass' in i.lower() for i in instruments):
            return 'bass'
        
        harmony_instruments = ['piano chords', 'strings', 'brass', 'pad']
        if any(any(instr in i.lower() for instr in harmony_instruments) for i in instruments):
            return 'harmony'
        
        time_to_notes = defaultdict(list)
        for n in pattern.notes:
            time_to_notes[n.start_time].append(n.note)
        max_simultaneous = max(len(notes) for notes in time_to_notes.values())
        if max_simultaneous > 1:
            return 'harmony'
        
        melody_instruments = ['synth', 'piano', 'violin', 'lead', 'pluck', 'bells']
        if any(any(instr in i.lower() for instr in melody_instruments) for i in instruments):
            return 'melody'
        
        return 'melody'

    def extract_basic_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata (bpm, time_signature) without full pattern load."""
        midi = mido.MidiFile(file_path)
        tempos = []
        try:
            time_sigs = []
            for track in midi.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempos.append(60000000 / msg.tempo)
                    elif msg.type == 'time_signature':
                        time_sigs.append((msg.numerator, msg.denominator))
            bpm = np.mean(tempos) if tempos else 120.0
            ts = time_sigs[0] if time_sigs else (4, 4)
            return {'bpm': bpm, 'time_signature': ts}
        except Exception as e:
            print(f"Error extracting basic metadata from {file_path}: {e}")
            return {'bpm': 120.0, 'time_signature': (4, 4)}

    def detect_chords(self, pattern: MidiPatternData) -> List[str]:
        """Detect chord types from polyphonic MIDI data."""
        time_to_notes = defaultdict(list)
        for n in pattern.notes:
            time_to_notes[n.start_time].append(n.note)
        
        chords = []
        for time, notes in time_to_notes.items():
            if len(notes) > 1:
                sorted_notes = sorted(set(notes))
                if len(sorted_notes) >= 3:
                    root = min(sorted_notes)
                    intervals = sorted([n - root for n in sorted_notes if n != root])
                    if intervals[:2] == [4, 7]:
                        chord_type = 'major'
                    elif intervals[:2] == [3, 7]:
                        chord_type = 'minor'
                    else:
                        chord_type = 'other'
                    chords.append(chord_type)
                else:
                    chords.append('dyad')
        return chords

    def extract_melodic_contour(self, pattern: MidiPatternData) -> List[int]:
        """Extract pitch intervals for melodic patterns."""
        notes_sorted = sorted([n for n in pattern.notes if n.note], key=lambda n: n.start_time)
        if len(notes_sorted) < 2:
            return []
        intervals = [notes_sorted[i+1].note - notes_sorted[i].note for i in range(len(notes_sorted)-1)]
        return intervals

    def extract_metadata(self, pattern: MidiPatternData) -> PatternMetadata:
        """Extract metadata from pattern (basic fields)."""
        if pattern.tempos:
            bpm = float(np.mean([t.bpm for t in pattern.tempos]))
        else:
            bpm = 120.0
        
        if pattern.time_signatures:
            ts = pattern.time_signatures[0]
            time_sig = (ts.numerator, ts.denominator)
        else:
            time_sig = (4, 4)
        
        length_beats = pattern.length_ticks / pattern.ticks_per_beat
        complexity = len(pattern.notes) / length_beats if length_beats > 0 else 0
        
        key = None
        
        return PatternMetadata(
            bpm=bpm,
            time_signature=time_sig,
            complexity=complexity,
            genre='',  # Set later
            instrument_type=''  # Set later
        )

    def _set_advanced_metadata(self, metadata: PatternMetadata, pattern: MidiPatternData, instrument_type: str):
        """Set advanced metadata based on type."""
        if instrument_type in ['harmony', 'drums', 'bass']:
            metadata.pattern_category = 'harmonic' if instrument_type == 'harmony' else 'rhythmic'
            if instrument_type == 'harmony':
                metadata.chord_types = self.detect_chords(pattern)
                metadata.harmonic_progression = '-'.join(metadata.chord_types) if metadata.chord_types else None
        else:
            metadata.pattern_category = 'melodic'
            metadata.melodic_intervals = self.extract_melodic_contour(pattern)
            if metadata.melodic_intervals:
                if all(i > 0 for i in metadata.melodic_intervals):
                    metadata.contour = 'ascending'
                elif all(i < 0 for i in metadata.melodic_intervals):
                    metadata.contour = 'descending'
                else:
                    metadata.contour = 'varied'
            metadata.scale_degrees = None

    def add_pattern(self, pattern: MidiPatternData, genre: str, instrument_type: str):
        """Add a pattern with classification."""
        file_path = pattern.file_path
        key = (genre, instrument_type)
        if self.optimization_level == 'eager':
            if key not in self.patterns:
                self.patterns[key] = []
            self.patterns[key].append(pattern)
        if key not in self.file_paths:
            self.file_paths[key] = []
        self.file_paths[key].append(file_path)
        
        metadata = self.extract_metadata(pattern)
        metadata.genre = genre
        metadata.instrument_type = instrument_type
        self._set_advanced_metadata(metadata, pattern, instrument_type)
        self.metadata[file_path] = metadata
        if self.optimization_level == 'eager':
            self.pattern_cache[file_path] = pattern
        self._update_df()
        self._build_indices()
        self._compute_features(file_path)

    def load_from_directory(self, dir_path: str, incremental: bool = False, use_mmap: bool = False, progress_callback: Optional[Callable[[float], None]] = None, rhythms_db_path: Optional[Union[str, os.PathLike]] = None):
        """Load and store patterns from directory, with options for incremental and mmap."""
        # Optional, backward-compatible integration: allow callers to provide a rhythms_db_path
        # which will be resolved via RhythmsDbResolver. If not provided, legacy behavior is unchanged.
        if rhythms_db_path is not None:
            try:
                resolved_dir = RhythmsDbResolver().get_rhythms_db_path(rhythms_db_path)
                dir_path = str(resolved_dir)
            except Exception:
                # Preserve backward compatibility by falling back silently
                pass
        dir_path_obj = Path(dir_path)
        if not dir_path_obj.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if self.optimization_level == 'eager':
            patterns = list(extract_from_directory(dir_path))
            for pattern in patterns:
                file_path = pattern.file_path
                genre = self.classify_genre(file_path)
                instr_type = self.classify_instrument_type(pattern)
                self.add_pattern(pattern, genre, instr_type)
        else:  # lazy
            if incremental:
                # Fall back to non-incremental for compatibility
                patterns = list(extract_from_directory(dir_path))
                for pattern in patterns:
                    file_path = pattern.file_path
                    genre = self.classify_genre(file_path)
                    instr_type = self.classify_instrument_from_filename(file_path)
                    key = (genre, instr_type)
                    if key not in self.file_paths:
                        self.file_paths[key] = []
                    self.file_paths[key].append(file_path)
                    
                    basic_info = self.extract_basic_metadata(file_path)
                    basic_meta = PatternMetadata(
                        bpm=basic_info['bpm'],
                        time_signature=basic_info['time_signature'],
                        complexity=0.0,
                        genre=genre,
                        instrument_type=instr_type
                    )
                    if instr_type in ['harmony', 'drums', 'bass']:
                        basic_meta.pattern_category = 'harmonic' if instr_type == 'harmony' else 'rhythmic'
                    else:
                        basic_meta.pattern_category = 'melodic'
                    self.metadata[file_path] = basic_meta
            else:
                for file_path in dir_path_obj.rglob("*.mid"):
                    str_path = str(file_path)
                    genre = self.classify_genre(str_path)
                    instr_type = self.classify_instrument_from_filename(str_path)
                    key = (genre, instr_type)
                    if key not in self.file_paths:
                        self.file_paths[key] = []
                    self.file_paths[key].append(str_path)
                    
                    basic_info = self.extract_basic_metadata(str_path)
                    basic_meta = PatternMetadata(
                        bpm=basic_info['bpm'],
                        time_signature=basic_info['time_signature'],
                        complexity=0.0,
                        genre=genre,
                        instrument_type=instr_type
                    )
                    if instr_type in ['harmony', 'drums', 'bass']:
                        basic_meta.pattern_category = 'harmonic' if instr_type == 'harmony' else 'rhythmic'
                    else:
                        basic_meta.pattern_category = 'melodic'
                    self.metadata[str_path] = basic_meta
            self._update_df()
        self._build_indices()

    def _compute_features(self, file_path: str):
        try:
            """Compute feature vector for similarity, apply PCA if enabled."""
            if file_path not in self.pattern_cache and self.optimization_level == 'eager':
                return
            pattern = self.pattern_cache[file_path] if self.optimization_level == 'eager' else None
            if pattern is None or not pattern.notes:
                self.feature_vectors[file_path] = np.zeros(5)
                return

            pitches = np.array([n.note for n in pattern.notes])
            durations = np.array([n.duration for n in pattern.notes])
            velocities = np.array([n.velocity for n in pattern.notes])

            avg_pitch = np.mean(pitches)
            pitch_range = np.max(pitches) - np.min(pitches)
            avg_dur = np.mean(durations)
            avg_vel = np.mean(velocities)
            density = len(pattern.notes) / (pattern.length_ticks / pattern.ticks_per_beat)

            features = np.array([
                avg_pitch / 128,
                pitch_range / 128,
                avg_dur / pattern.ticks_per_beat,
                avg_vel / 127,
                density / 4
            ])
            if self.use_pca and self.pca is not None and self.pca_fitted:
                features = self.pca.transform(features.reshape(1, -1)).flatten()
            self.feature_vectors[file_path] = features

        except Exception:
            self.feature_vectors[file_path] = np.zeros(5)

    def _fit_pca_ann(self):
        """Fit PCA and ANN on available features."""
        if not self.feature_vectors:
            return
        features_list = list(self.feature_vectors.values())
        if len(features_list) < 2:
            return
        features = np.array(features_list)
        if self.use_pca and self.pca is not None:
            self.pca.fit(features)
            features = self.pca.transform(features)
            self.pca_fitted = True
        if self.use_ann and self.ann_model is not None:
            self.ann_model.fit(features)
            self._fit_ann = True

    def _load_pattern_if_needed(self, file_path: str, genre: str, instr_type: str) -> Tuple[MidiPatternData, PatternMetadata]:
        """Load pattern and full metadata if not already loaded, with LRU caching."""
        if file_path not in self.pattern_cache:
            pattern = extract_from_file(file_path)
            metadata = self.extract_metadata(pattern)
            metadata.genre = genre
            metadata.instrument_type = instr_type
            self._set_advanced_metadata(metadata, pattern, instr_type)
            self.metadata[file_path] = metadata
            self.pattern_cache[file_path] = pattern
            self.pattern_cache.move_to_end(file_path)
            self._compute_features(file_path)
            self._update_df()
            self._build_indices()
            if len(self.feature_vectors) >= self.n_neighbors + 1:
                self._fit_pca_ann()
            if self.eviction_policy == 'lfu' and self.access_counts is not None:
                self.access_counts[file_path] += 1
            self._enforce_memory_limit()
            return pattern, metadata
        else:
            pattern = self.pattern_cache[file_path]
            metadata = self.metadata[file_path]
            self.pattern_cache.move_to_end(file_path)
            if self.eviction_policy == 'lfu' and self.access_counts is not None:
                self.access_counts[file_path] += 1
            self._enforce_memory_limit()
        return pattern, metadata

    def get_patterns(self, 
                     instrument: Optional[str] = None,
                     genre: Optional[str] = None,
                     min_complexity: Optional[float] = None,
                     max_complexity: Optional[float] = None,
                     filter_func: Optional[Callable[[MidiPatternData, PatternMetadata], bool]] = None) -> List[MidiPatternData]:
        """Retrieve patterns based on criteria. Uses DF and indices for efficient filtering."""
        if self.metadata_df is None:
            self._update_df()
        if not self.indices:
            self._build_indices()
        
        if self.metadata_df is None or self.metadata_df.empty:
            return []
        
        candidate_paths = set(self.metadata_df.index.tolist())
        if genre is not None and genre in self.indices.get('genre', {}):
            candidate_paths &= set(self.indices['genre'][genre])
        if instrument is not None and instrument in self.indices.get('instrument', {}):
            candidate_paths &= set(self.indices['instrument'][instrument])
        
        filtered_df = self.metadata_df.loc[list(candidate_paths)]
        query_str = 'True'
        params = {}
        if min_complexity is not None:
            query_str += ' & (complexity >= @min_complexity)'
            params['min_complexity'] = min_complexity
        if max_complexity is not None:
            query_str += ' & (complexity <= @max_complexity)'
            params['max_complexity'] = max_complexity
        
        if query_str != 'True':
            try:
                filtered_df = filtered_df.query(query_str, **params)
            except:
                pass
        
        results = []
        for path in filtered_df.index:
            meta_dict = filtered_df.loc[path].to_dict()
            meta = PatternMetadata.from_dict(meta_dict)
            load_needed = filter_func or meta.complexity == 0.0
            if load_needed:
                g = meta.genre
                i = meta.instrument_type
                pat, full_meta = self._load_pattern_if_needed(path, g, i)
                meta = full_meta
            else:
                pat = self.pattern_cache.get(path)
            
            if pat is None:
                continue
            if filter_func and not filter_func(pat, meta):
                continue
            results.append(pat)
        return results

    def get_harmonic_patterns(self, 
                              genre: Optional[str] = None,
                              min_complexity: Optional[float] = None,
                              max_complexity: Optional[float] = None,
                              filter_func: Optional[Callable[[MidiPatternData, PatternMetadata], bool]] = None) -> List[MidiPatternData]:
        """Retrieve harmonic patterns."""
        def harmonic_filter(pat: MidiPatternData, meta: PatternMetadata) -> bool:
            return meta.pattern_category == 'harmonic'
        return self.get_patterns(instrument='harmony', genre=genre, min_complexity=min_complexity, max_complexity=max_complexity, filter_func=filter_func or harmonic_filter)

    def get_melodic_patterns(self, 
                             genre: Optional[str] = None,
                             min_complexity: Optional[float] = None,
                             max_complexity: Optional[float] = None,
                             filter_func: Optional[Callable[[MidiPatternData, PatternMetadata], bool]] = None) -> List[MidiPatternData]:
        """Retrieve melodic patterns."""
        def melodic_filter(pat: MidiPatternData, meta: PatternMetadata) -> bool:
            return meta.pattern_category == 'melodic'
        return self.get_patterns(instrument='melody', genre=genre, min_complexity=min_complexity, max_complexity=max_complexity, filter_func=filter_func or melodic_filter)

    def load_harmonic_patterns(self, dir_path: str, instrument_list: Optional[List[str]] = None):
        """Specialized extraction for harmonic patterns."""
        if self.optimization_level != 'eager':
            raise NotImplementedError("Specialized loading for lazy mode not implemented yet.")
        if instrument_list is None:
            instrument_list = ['piano chords', 'strings', 'brass', 'pad']
        patterns = list(extract_from_directory(dir_path))
        for pattern in patterns:
            file_lower = pattern.file_path.lower()
            instr_match = any(instr.lower() in file_lower for instr in instrument_list)
            if instr_match or any(any(instr.lower() in (n.instrument or '').lower() for instr in instrument_list) for n in pattern.notes if n.instrument is not None):
                genre = self.classify_genre(pattern.file_path)
                self.add_pattern(pattern, genre, 'harmony')

    def load_melodic_patterns(self, dir_path: str, instrument_list: Optional[List[str]] = None):
        """Specialized extraction for melodic patterns."""
        if self.optimization_level != 'eager':
            raise NotImplementedError("Specialized loading for lazy mode not implemented yet.")
        if instrument_list is None:
            instrument_list = ['synth', 'piano', 'violin', 'lead', 'pluck', 'bells']
        patterns = list(extract_from_directory(dir_path))
        for pattern in patterns:
            file_lower = pattern.file_path.lower()
            instr_match = any(instr.lower() in file_lower for instr in instrument_list)
            if instr_match or any(any(instr.lower() in (n.instrument or '').lower() for instr in instrument_list) for n in pattern.notes if n.instrument is not None):
                genre = self.classify_genre(pattern.file_path)
                self.add_pattern(pattern, genre, 'melody')

    def compute_similarity(self, pattern1: MidiPatternData, pattern2: MidiPatternData, use_ann: bool = True) -> float:
        """Compute similarity score using features or ANN if enabled."""
        f1 = self._get_feature_vector(pattern1)
        f2 = self._get_feature_vector(pattern2)
        if use_ann and self._fit_ann and self.ann_model is not None:
            # Use precomputed ANN for batch, but for single, compute cosine
            sim = cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))[0, 0]
        else:
            dot = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            sim = dot / (norm1 * norm2)
        return max(0, min(1, sim))

    def _get_feature_vector(self, pattern: MidiPatternData) -> np.ndarray:
        """Get or compute feature vector for pattern."""
        file_path = pattern.file_path
        if file_path not in self.feature_vectors:
            self._compute_features(file_path)
        vec = self.feature_vectors.get(file_path, np.zeros(self.n_components if self.pca else 5))
        # Ensure consistency if PCA fitted
        if self.pca is not None and self.pca_fitted and self.use_pca and len(vec) == 5:
            vec = self.pca.transform(vec.reshape(1, -1)).flatten()
            self.feature_vectors[file_path] = vec
        return vec

    def find_similar(self, pattern: MidiPatternData, threshold: float = 0.7, max_results: int = 10, parallel: bool = True) -> List[Tuple[MidiPatternData, float]]:
        """Find similar patterns using ANN or parallel cosine similarity."""
        all_patterns = []
        if self.optimization_level == 'eager':
            all_patterns = [p for pats in self.patterns.values() for p in pats]
        else:
            for paths in self.file_paths.values():
                for path in paths:
                    g, i = next((k for k, v in self.file_paths.items() if path in v), ('unknown', 'unknown'))
                    self._load_pattern_if_needed(path, g, i)
                    all_patterns.append(self.pattern_cache[path])
        
        if self.use_ann and self._fit_ann and self.ann_model is not None:
            f_query = self._get_feature_vector(pattern).reshape(1, -1)
            distances, indices = self.ann_model.kneighbors(f_query, n_neighbors=max_results + 1)
            similar = []
            for dist, idx in zip(distances[0], indices[0]):
                if dist > threshold:
                    continue
                other = all_patterns[idx]
                if other.file_path == pattern.file_path:
                    continue
                sim = 1 - dist
                similar.append((other, sim))
            similar.sort(key=lambda x: x[1], reverse=True)
            return similar[:max_results]
        else:
            # Parallel computation
            def compute_sim(other):
                if other.file_path == pattern.file_path:
                    return None
                sim = self.compute_similarity(pattern, other, use_ann=False)
                if sim >= threshold:
                    return (other, sim)
                return None
            
            similar = []
            if parallel:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(compute_sim, other) for other in all_patterns]
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            similar.append(result)
            else:
                for other in all_patterns:
                    result = compute_sim(other)
                    if result:
                        similar.append(result)
            similar.sort(key=lambda x: x[1], reverse=True)
            return similar[:max_results]

    def save_to_disk(self, file_path: str):
        """Save library to pickle file. Cache not saved."""
        data = {
            'optimization_level': self.optimization_level,
            'cache_size': self.cache_size,
            'use_pca': self.use_pca,
            'n_components': self.n_components,
            'use_ann': self.use_ann,
            'n_neighbors': self.n_neighbors,
            'parallel': self.parallel,
            'max_workers': self.max_workers,
            'eviction_policy': self.eviction_policy,
            'max_memory_mb': self.max_memory_mb,
            'file_paths': self.file_paths,
            'metadata': {p: m.to_dict() for p, m in self.metadata.items()},
        }
        if self.optimization_level == 'eager':
            data['patterns'] = {k: [p.file_path for p in v] for k, v in self.patterns.items()}
        if self.pca is not None and self.pca_fitted:
            data['pca_components'] = self.pca.components_.tolist()
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_from_disk(self, file_path: str):
        """Load library from pickle file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.optimization_level = data.get('optimization_level', 'eager')
        self.cache_size = data.get('cache_size', 100)
        self.use_pca = data.get('use_pca', True)
        self.n_components = data.get('n_components', 5)
        self.use_ann = data.get('use_ann', True)
        self.n_neighbors = data.get('n_neighbors', 10)
        self.parallel = data.get('parallel', True)
        self.max_workers = data.get('max_workers', None)
        self.eviction_policy = data.get('eviction_policy', 'lru')
        self.max_memory_mb = data.get('max_memory_mb', None)
        self.access_counts = defaultdict(int) if self.eviction_policy == 'lfu' else None
        self.pca = PCA(n_components=self.n_components) if self.use_pca else None
        self.ann_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine') if self.use_ann else None
        self._fit_ann = False
        self.pca_fitted = False
        self.file_paths = data['file_paths']
        self.metadata = {p: PatternMetadata.from_dict(m) for p, m in data['metadata'].items()}
        self.pattern_cache = OrderedDict()
        self.feature_vectors = {}
        self.metadata_df = None
        self.indices = {}
        if self.optimization_level == 'eager':
            self.patterns = {}
            print("Warning: For eager mode, call load_from_directory after load_from_disk to populate patterns.")
        print("Library metadata and paths loaded.")

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics, including memory usage."""
        total_patterns = sum(len(paths) for paths in self.file_paths.values())
        genres = set(g for g, _ in self.file_paths)
        instruments = set(i for _, i in self.file_paths)
        mem_usage = memory_usage()
        return {
            'optimization_level': self.optimization_level,
            'cache_size': self.cache_size,
            'total_patterns': total_patterns,
            'genres': list(genres),
            'instruments': list(instruments),
            'loaded_count': len(self.pattern_cache),
            'df_shape': self.metadata_df.shape if self.metadata_df is not None else (0, 0),
            'indices_keys': list(self.indices.keys()) if self.indices else [],
            'memory_mb': mem_usage,
            'pca_fitted': self.pca_fitted,
            'ann_fitted': self._fit_ann
        }


if __name__ == "__main__":
    lib = ReferencePatternLibrary(optimization_level='eager', cache_size=50, use_pca=True, use_ann=True, parallel=True)
    lib.load_from_directory("reference_midis", incremental=False, use_mmap=False)
    print(lib.get_stats())
    drum_patterns = lib.get_patterns(instrument="drums", genre="trap_808")
    print(f"Found {len(drum_patterns)} drum patterns in trap_808")
    harmonic_patterns = lib.get_harmonic_patterns(genre="panoramic")
    print(f"Found {len(harmonic_patterns)} harmonic patterns in panoramic")
    melodic_patterns = lib.get_melodic_patterns(genre="trap_808")
    print(f"Found {len(melodic_patterns)} melodic patterns in trap_808")
    if melodic_patterns:
        similar = lib.find_similar(melodic_patterns[0], threshold=0.7, max_results=3, parallel=True)
        print(f"Found {len(similar)} similar patterns")