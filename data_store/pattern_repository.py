import os
import sqlite3
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from datetime import datetime


JsonLike = Union[str, dict, list, tuple]


class PatternRepository:
    """
    SQLite-backed repository for rhythm pattern storage and retrieval.

    - Resolves DB path from MIDIMASTER_DB_PATH or defaults to data/pattern_store.sqlite
    - Applies WAL mode and pragmas: foreign_keys=ON, journal_mode=WAL, synchronous=NORMAL
    - Inline migrations with schema_versions (v1)
    - Provides CRUD-like upserts, staged fallback queries, import/export, and stats
    """

    CURRENT_SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize the repository.

        - db_path: Optional path override. If None, checks env MIDIMASTER_DB_PATH,
          else defaults to data/pattern_store.sqlite. Ensures parent directory exists.
        - Opens SQLite connection with required pragmas and applies migrations.
        """
        resolved = db_path or os.environ.get("MIDIMASTER_DB_PATH")
        if not resolved:
            resolved = os.path.join("data", "pattern_store.sqlite")

        parent_dir = os.path.dirname(os.path.abspath(resolved)) or "."
        os.makedirs(parent_dir, exist_ok=True)

        # Connect and configure connection/session pragmas
        # Use row_factory for dict-like row access; keep default isolation level and commit explicitly
        self._conn = sqlite3.connect(resolved)
        self._conn.row_factory = sqlite3.Row

        # Apply pragmas for reliability and Windows safety
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")

        # Run migrations if needed
        self._apply_migrations()

    # ---------------------------
    # Public API
    # ---------------------------

    def upsert_source(
        self,
        source_type: str,
        source_path: str,
        source_track: Optional[str] = None,
        extracted_by_version: Optional[str] = None,
    ) -> int:
        """
        Insert or return existing source id matching on (source_type, source_path, coalesced source_track).

        - source_track is treated as '' (empty string) if None to provide stable uniqueness semantics.
        - Updates last_seen_at on existing record.
        - Returns source_id.
        """
        stype = self._require_non_empty_str("source_type", source_type)
        spath = self._require_non_empty_str("source_path", source_path)
        strack = (source_track or "").strip()
        now = self._now()

        # Try to fetch existing
        row = self._conn.execute(
            """
            SELECT id FROM pattern_sources
            WHERE source_type = ? AND source_path = ? AND source_track = ?
            """,
            (stype, spath, strack),
        ).fetchone()

        if row:
            src_id = int(row["id"])
            self._conn.execute(
                "UPDATE pattern_sources SET last_seen_at = ? WHERE id = ?",
                (now, src_id),
            )
            self._conn.commit()
            return src_id

        # Insert new
        cur = self._conn.execute(
            """
            INSERT INTO pattern_sources
                (source_type, source_path, source_track, extracted_by_version, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (stype, spath, strack, extracted_by_version, now, now),
        )
        self._conn.commit()
        return self._lastrowid(cur)

    def upsert_rhythm_pattern(
        self,
        *,
        source_id: int,
        instrument: str,
        genre: str,
        mood: Optional[str],
        time_signature: str,
        subdivision: int,
        length_beats: float,
        bpm_min: Optional[float],
        bpm_max: Optional[float],
        syncopation: float,
        density: float,
        swing: Optional[float],
        humanization: Optional[float],
        quality_score: Optional[float],
        pattern_json: JsonLike,
        accent_profile_json: Optional[JsonLike],
        tags_json: Optional[JsonLike],
    ) -> int:
        """
        Insert a rhythm pattern if unique; otherwise returns existing pattern id.

        Validations:
        - Required fields must be non-empty/positive where applicable.
        - Normalized fields (syncopation, density, swing, humanization, quality_score) must be in [0,1]; raises ValueError if not, then coerces to [0,1] to store.
        - pattern_json must be a valid JSON array (list). Accepts:
           * a JSON string encoding a list,
           * a Python list/tuple,
           * a dict ONLY if it contains a top-level 'pattern' key with list value.
             Any other dict shape raises ValueError.
        - accent_profile_json and tags_json accept str/dict/list; stored as JSON text with stable serialization.

        Uniqueness constraint defined as:
          UNIQUE(source_id, instrument, genre, time_signature, length_beats, subdivision, pattern_json)

        Returns the inserted or existing pattern id.
        """
        # Basic validations
        if not isinstance(source_id, int) or source_id <= 0:
            raise ValueError("source_id must be a positive integer")

        instr = self._require_non_empty_str("instrument", instrument)
        g = self._require_non_empty_str("genre", genre)
        ts = self._require_non_empty_str("time_signature", time_signature)

        if not isinstance(subdivision, int) or subdivision <= 0:
            raise ValueError("subdivision must be a positive integer")

        if not isinstance(length_beats, (int, float)) or length_beats <= 0:
            raise ValueError("length_beats must be a positive number")

        # Normalized fields: validate in [0,1] (inclusive). If slightly out-of-range, still raise per spec.
        syncopation = self._validated_unit_interval("syncopation", syncopation)
        density = self._validated_unit_interval("density", density)
        swing_val: Optional[float] = None if swing is None else self._validated_unit_interval("swing", swing)
        human_val: Optional[float] = None if humanization is None else self._validated_unit_interval("humanization", humanization)
        quality_val: Optional[float] = None if quality_score is None else self._validated_unit_interval("quality_score", quality_score)

        # JSON handling
        pattern_text, _ = self._normalize_pattern_json(pattern_json)
        accent_text = self._normalize_arbitrary_json(accent_profile_json) if accent_profile_json is not None else None
        tags_text = self._normalize_arbitrary_json(tags_json) if tags_json is not None else None

        # Attempt insert; on UNIQUE conflict, fetch existing id
        now = self._now()
        try:
            cur = self._conn.execute(
                """
                INSERT INTO rhythm_patterns (
                  source_id, instrument, genre, mood, time_signature, subdivision, length_beats,
                  bpm_min, bpm_max, syncopation, density, swing, humanization, quality_score,
                  pattern_json, accent_profile_json, tags_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id, instr, g, mood, ts, subdivision, float(length_beats),
                    bpm_min, bpm_max, syncopation, density, swing_val, human_val, quality_val,
                    pattern_text, accent_text, tags_text, now,
                ),
            )
            self._conn.commit()
            return self._lastrowid(cur)
        except sqlite3.IntegrityError as e:
            # Either FK failure or UNIQUE conflict
            # On FK failure, surface error
            msg = str(e).lower()
            if "foreign key" in msg:
                raise

        # Fetch existing matching row id for the unique tuple
        row = self._conn.execute(
            """
            SELECT id FROM rhythm_patterns
            WHERE source_id = ?
              AND instrument = ?
              AND genre = ?
              AND time_signature = ?
              AND length_beats = ?
              AND subdivision = ?
              AND pattern_json = ?
            """,
            (source_id, instr, g, ts, float(length_beats), subdivision, pattern_text),
        ).fetchone()

        if row:
            return int(row["id"])

        # If we reach here, the error was not a uniqueness conflict we can reconcile
        raise RuntimeError("Failed to upsert rhythm pattern due to unexpected constraint issue.")

    def find_rhythm_patterns(
        self,
        *,
        instrument: str,
        genre: str,
        time_signature: str,
        mood: Optional[str] = None,
        subdivision: Optional[int] = None,
        target_length_beats: Optional[float] = None,
        target_bpm: Optional[float] = None,
        syncopation_range: Optional[Union[Tuple[float, float], Tuple[float]]] = None,
        density_range: Optional[Union[Tuple[float, float], Tuple[float]]] = None,
        swing_tolerance: Optional[float] = None,
        min_quality_score: float = 0.0,
        tag_filters: Optional[Sequence[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Query candidates matching required filters with staged fallback if empty:

        Stages (stop at first non-empty result):
          0) Strict filters (as provided)
          1) Widen syncopation by ±0.1 around given range (or ±0.3 if a single-center value was given)
          2) Drop bpm bounds (ignore target_bpm)
          3) Drop mood requirement
          4) Relax subdivision to compatible multiples/factors (e.g., allow 3 when requesting 4, and vice versa)
          5) Relax density range to [0, 1]

        Returns list of dicts of all columns plus key 'pattern' containing parsed pattern_json (Python object).
        JSON string columns remain intact in the dict as stored.
        """
        instr = self._require_non_empty_str("instrument", instrument)
        g = self._require_non_empty_str("genre", genre)
        ts = self._require_non_empty_str("time_signature", time_signature)
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        # Normalize optional numeric ranges and tolerances
        sync_range = self._normalize_range_tuple(syncopation_range, "syncopation_range")
        dens_range = self._normalize_range_tuple(density_range, "density_range")
        swing_tol = None if swing_tolerance is None else self._validated_unit_interval("swing_tolerance", swing_tolerance)
        min_q = self._validated_unit_interval("min_quality_score", min_quality_score)

        # Stage 0: strict
        candidates = self._query_candidates(
            instrument=instr, genre=g, time_signature=ts, mood=mood,
            subdivision=subdivision, target_length_beats=target_length_beats,
            target_bpm=target_bpm, sync_range=sync_range, dens_range=dens_range,
            swing_tolerance=swing_tol, min_quality=min_q, tag_filters=tag_filters,
            limit=limit, relax_subdivisions=None,
        )
        if candidates:
            return candidates

        # Stage 1: widen syncopation range
        widened_sync = None
        if sync_range is not None:
            widened_sync = (max(0.0, sync_range[0] - 0.1), min(1.0, sync_range[1] + 0.1))
        elif syncopation_range is not None:
            # A single center value provided e.g. (0.5,) - widen to ±0.3
            center = float(syncopation_range[0])  # type: ignore[index]
            self._validated_unit_interval("syncopation_center", center)
            widened_sync = (max(0.0, center - 0.3), min(1.0, center + 0.3))
        candidates = self._query_candidates(
            instrument=instr, genre=g, time_signature=ts, mood=mood,
            subdivision=subdivision, target_length_beats=target_length_beats,
            target_bpm=target_bpm, sync_range=widened_sync, dens_range=dens_range,
            swing_tolerance=swing_tol, min_quality=min_q, tag_filters=tag_filters,
            limit=limit, relax_subdivisions=None,
        )
        if candidates:
            return candidates

        # Stage 2: drop bpm bounds
        candidates = self._query_candidates(
            instrument=instr, genre=g, time_signature=ts, mood=mood,
            subdivision=subdivision, target_length_beats=target_length_beats,
            target_bpm=None, sync_range=widened_sync, dens_range=dens_range,
            swing_tolerance=swing_tol, min_quality=min_q, tag_filters=tag_filters,
            limit=limit, relax_subdivisions=None,
        )
        if candidates:
            return candidates

        # Stage 3: drop mood
        candidates = self._query_candidates(
            instrument=instr, genre=g, time_signature=ts, mood=None,
            subdivision=subdivision, target_length_beats=target_length_beats,
            target_bpm=None, sync_range=widened_sync, dens_range=dens_range,
            swing_tolerance=swing_tol, min_quality=min_q, tag_filters=tag_filters,
            limit=limit, relax_subdivisions=None,
        )
        if candidates:
            return candidates

        # Stage 4: relax subdivision to compatible set
        relax_subdivisions = None
        if subdivision and subdivision > 0:
            relax_subdivisions = self._compatible_subdivisions(subdivision)
        candidates = self._query_candidates(
            instrument=instr, genre=g, time_signature=ts, mood=None,
            subdivision=None if relax_subdivisions else subdivision,
            target_length_beats=target_length_beats, target_bpm=None,
            sync_range=widened_sync, dens_range=dens_range,
            swing_tolerance=swing_tol, min_quality=min_q, tag_filters=tag_filters,
            limit=limit, relax_subdivisions=relax_subdivisions,
        )
        if candidates:
            return candidates

        # Stage 5: relax density to [0,1]
        candidates = self._query_candidates(
            instrument=instr, genre=g, time_signature=ts, mood=None,
            subdivision=None if relax_subdivisions else subdivision,
            target_length_beats=target_length_beats, target_bpm=None,
            sync_range=widened_sync, dens_range=(0.0, 1.0),
            swing_tolerance=swing_tol, min_quality=min_q, tag_filters=tag_filters,
            limit=limit, relax_subdivisions=relax_subdivisions,
        )
        return candidates or []

    def export_patterns(self, file_path: str) -> None:
        """
        Export pattern_sources and rhythm_patterns to a JSON file for backup.
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")

        parent_dir = os.path.dirname(os.path.abspath(file_path)) or "."
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        sources = [dict(r) for r in self._conn.execute("SELECT * FROM pattern_sources")]
        patterns = [dict(r) for r in self._conn.execute("SELECT * FROM rhythm_patterns")]

        payload = {
            "schema_version": self.get_schema_version(),
            "pattern_sources": sources,
            "rhythm_patterns": patterns,
            "exported_at": self._now(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def import_patterns(self, file_path: str) -> None:
        """
        Bulk import JSON export while respecting upsert semantics.
        - Sources are upserted; a mapping from old_id to new_id is constructed.
        - Patterns are upserted using mapped source_id.
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sources = data.get("pattern_sources", [])
        patterns = data.get("rhythm_patterns", [])

        # Map old source ids to new upserted ids
        id_map: Dict[int, int] = {}
        for s in sources:
            new_id = self.upsert_source(
                source_type=s.get("source_type", ""),
                source_path=s.get("source_path", ""),
                source_track=s.get("source_track", "") or None,
                extracted_by_version=s.get("extracted_by_version") or None,
            )
            old_id = int(s.get("id"))
            id_map[old_id] = new_id

        for p in patterns:
            sid_old = int(p.get("source_id"))
            sid_new = id_map.get(sid_old)
            if not sid_new:
                # Skip patterns with missing source mapping (should not happen)
                continue

            # Reuse provided JSON strings if present; otherwise rely on normalized serialization
            pattern_json = p.get("pattern_json")
            accent_json = p.get("accent_profile_json")
            tags_json = p.get("tags_json")

            self.upsert_rhythm_pattern(
                source_id=sid_new,
                instrument=p.get("instrument", ""),
                genre=p.get("genre", ""),
                mood=p.get("mood"),
                time_signature=p.get("time_signature", ""),
                subdivision=int(p.get("subdivision")),
                length_beats=float(p.get("length_beats")),
                bpm_min=self._to_optional_float(p.get("bpm_min")),
                bpm_max=self._to_optional_float(p.get("bpm_max")),
                syncopation=float(p.get("syncopation")),
                density=float(p.get("density")),
                swing=self._to_optional_float(p.get("swing")),
                humanization=self._to_optional_float(p.get("humanization")),
                quality_score=self._to_optional_float(p.get("quality_score")),
                pattern_json=pattern_json,
                accent_profile_json=accent_json,
                tags_json=tags_json,
            )

    def get_pattern_stats(self) -> Dict[str, Any]:
        """
        Returns:
          {
            "total_patterns": int,
            "total_sources": int,
            "schema_version": int,
            "counts_by_genre_instrument": [
                {"genre": "...", "instrument": "...", "count": N}, ...
            ]
          }
        """
        total_patterns = self._conn.execute("SELECT COUNT(*) AS c FROM rhythm_patterns").fetchone()["c"]
        total_sources = self._conn.execute("SELECT COUNT(*) AS c FROM pattern_sources").fetchone()["c"]
        rows = self._conn.execute(
            """
            SELECT genre, instrument, COUNT(*) AS c
            FROM rhythm_patterns
            GROUP BY genre, instrument
            ORDER BY genre, instrument
            """
        ).fetchall()
        stats = [
            {"genre": r["genre"], "instrument": r["instrument"], "count": r["c"]}
            for r in rows
        ]
        return {
            "total_patterns": int(total_patterns),
            "total_sources": int(total_sources),
            "schema_version": self.get_schema_version(),
            "counts_by_genre_instrument": stats,
        }

    def get_schema_version(self) -> int:
        row = self._conn.execute("SELECT MAX(version) AS v FROM schema_versions").fetchone()
        v = row["v"]
        return int(v) if v is not None else 0

    def close(self) -> None:
        """
        Close the SQLite connection.
        """
        try:
            self._conn.close()
        except Exception:
            pass

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _apply_migrations(self) -> None:
        # Ensure schema_versions exists; if not, create and set up v1 schema
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_versions (
              version INTEGER PRIMARY KEY,
              applied_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

        current = self.get_schema_version()
        if current >= self.CURRENT_SCHEMA_VERSION:
            return

        # Apply v1 (initial) schema
        if current < 1:
            self._create_v1_schema()

        # Record current version
        self._conn.execute(
            "INSERT INTO schema_versions (version, applied_at) VALUES (?, ?)",
            (self.CURRENT_SCHEMA_VERSION, self._now()),
        )
        self._conn.commit()

    def _create_v1_schema(self) -> None:
        # Sources table: enforce uniqueness using coalesced source_track -> stored as NOT NULL with default ''
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pattern_sources (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              source_type TEXT NOT NULL,
              source_path TEXT NOT NULL,
              source_track TEXT NOT NULL DEFAULT '',
              extracted_by_version TEXT,
              created_at TEXT NOT NULL,
              last_seen_at TEXT NOT NULL,
              UNIQUE (source_type, source_path, source_track)
            )
            """
        )

        # Patterns table with constraints and uniqueness guard
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rhythm_patterns (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              source_id INTEGER NOT NULL,
              instrument TEXT NOT NULL,
              genre TEXT NOT NULL,
              mood TEXT,
              time_signature TEXT NOT NULL,
              subdivision INTEGER NOT NULL,
              length_beats REAL NOT NULL,
              bpm_min REAL,
              bpm_max REAL,
              syncopation REAL NOT NULL CHECK (syncopation >= 0.0 AND syncopation <= 1.0),
              density REAL NOT NULL CHECK (density >= 0.0 AND density <= 1.0),
              swing REAL CHECK (swing >= 0.0 AND swing <= 1.0),
              humanization REAL CHECK (humanization >= 0.0 AND humanization <= 1.0),
              quality_score REAL CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
              pattern_json TEXT NOT NULL,
              accent_profile_json TEXT,
              tags_json TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY (source_id) REFERENCES pattern_sources(id) ON DELETE CASCADE,
              UNIQUE (source_id, instrument, genre, time_signature, length_beats, subdivision, pattern_json)
            )
            """
        )

        # Indexes
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_genre_instr_ts_subdiv ON rhythm_patterns (genre, instrument, time_signature, subdivision)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_syncopation ON rhythm_patterns (syncopation)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_density ON rhythm_patterns (density)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_length ON rhythm_patterns (length_beats)"
        )
        self._conn.commit()

    def _lastrowid(self, cur: sqlite3.Cursor) -> int:
        """
        Safely return the last inserted row id as int.
        Falls back to last_insert_rowid() if cursor.lastrowid is None.
        """
        val = cur.lastrowid
        if val is None:
            row = self._conn.execute("SELECT last_insert_rowid() AS rid").fetchone()
            if isinstance(row, sqlite3.Row):
                return int(row["rid"])
            return int(row[0])
        return int(val)

    def _query_candidates(
        self,
        *,
        instrument: str,
        genre: str,
        time_signature: str,
        mood: Optional[str],
        subdivision: Optional[int],
        target_length_beats: Optional[float],
        target_bpm: Optional[float],
        sync_range: Optional[Tuple[float, float]],
        dens_range: Optional[Tuple[float, float]],
        swing_tolerance: Optional[float],
        min_quality: float,
        tag_filters: Optional[Sequence[str]],
        limit: int,
        relax_subdivisions: Optional[Iterable[int]],
    ) -> List[Dict[str, Any]]:
        """
        Internal query builder executing a parameterized SELECT with optional filters.
        """
        clauses: List[str] = [
            "genre = ?",
            "instrument = ?",
            "time_signature = ?",
        ]
        params: List[Any] = [genre, instrument, time_signature]

        if mood:
            clauses.append("mood = ?")
            params.append(mood)

        # Subdivision
        if relax_subdivisions:
            sub_list = sorted({int(s) for s in relax_subdivisions if isinstance(s, int) and s > 0})
            if sub_list:
                placeholders = ",".join("?" for _ in sub_list)
                clauses.append(f"subdivision IN ({placeholders})")
                params.extend(sub_list)
        elif subdivision:
            clauses.append("subdivision = ?")
            params.append(int(subdivision))

        # Length equality with small tolerance
        if target_length_beats is not None:
            clauses.append("ABS(length_beats - ?) <= 0.001")
            params.append(float(target_length_beats))

        # BPM bounds if provided
        if target_bpm is not None:
            # Accept when bpm_min/max are NULL (open bound) or inclusive of target
            clauses.append("( (bpm_min IS NULL OR bpm_min <= ?) AND (bpm_max IS NULL OR bpm_max >= ?) )")
            params.extend([float(target_bpm), float(target_bpm)])

        # Syncopation range
        if sync_range is not None:
            clauses.append("syncopation >= ? AND syncopation <= ?")
            params.extend([sync_range[0], sync_range[1]])

        # Density range
        if dens_range is not None:
            clauses.append("density >= ? AND density <= ?")
            params.extend([dens_range[0], dens_range[1]])

        # Swing tolerance: include rows with swing NULL or swing <= tolerance
        if swing_tolerance is not None:
            clauses.append("(swing IS NULL OR swing <= ?)")
            params.append(swing_tolerance)

        # Min quality: treat NULL as 0 for comparison
        if min_quality is not None and min_quality > 0.0:
            clauses.append("COALESCE(quality_score, 0.0) >= ?")
            params.append(min_quality)

        # Tag filters: naive LIKE search for '"tag"' inside tags_json
        if tag_filters:
            for tag in tag_filters:
                t = str(tag).strip()
                if not t:
                    continue
                clauses.append("tags_json LIKE ?")
                params.append(f'%"{t}"%')

        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT
              id, source_id, instrument, genre, mood, time_signature, subdivision, length_beats,
              bpm_min, bpm_max, syncopation, density, swing, humanization, quality_score,
              pattern_json, accent_profile_json, tags_json, created_at
            FROM rhythm_patterns
            WHERE {where_sql}
            ORDER BY (quality_score IS NULL) ASC, quality_score DESC, created_at DESC
            LIMIT ?
        """
        params.append(int(limit))

        rows = self._conn.execute(sql, params).fetchall()
        results: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["pattern"] = json.loads(d["pattern_json"])
            except Exception:
                d["pattern"] = None
            results.append(d)
        return results

    @staticmethod
    def _compatible_subdivisions(requested: int) -> List[int]:
        """
        Generate a set of 'compatible' subdivisions. This is a pragmatic heuristic:
          - Include requested
          - Include factors and multiples within a typical grid set up to 24
          - Explicitly include 3 and 4 for common triplet/straight swaps
        """
        if requested <= 0:
            return []
        base_set = {requested, 3, 4}
        typical = {2, 3, 4, 6, 8, 12, 16, 24}
        for s in typical:
            if s == requested:
                continue
            if s % requested == 0 or requested % s == 0:
                base_set.add(s)
        # Also include +/- neighbors within reason
        for n in (requested // 2, requested * 2, requested * 3 // 2):
            if isinstance(n, int) and n > 0:
                base_set.add(n)
        return sorted({int(x) for x in base_set if isinstance(x, int) and x > 0})

    @staticmethod
    def _now() -> str:
        # UTC ISO8601 (seconds precision)
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    @staticmethod
    def _require_non_empty_str(name: str, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _validated_unit_interval(name: str, value: Any) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number in [0,1]")
        v = float(value)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{name} must be within [0,1], got {v}")
        # Coerce to [0,1] exactly to avoid FP drift
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v

    @staticmethod
    def _normalize_range_tuple(
        rng: Optional[Union[Tuple[float, float], Tuple[float]]],
        name: str,
    ) -> Optional[Tuple[float, float]]:
        if rng is None:
            return None
        if not isinstance(rng, tuple) and not isinstance(rng, list):
            raise ValueError(f"{name} must be a tuple like (min, max) or (center,)")
        values = list(rng)
        if len(values) == 0:
            return None
        if len(values) == 1:
            # Single center value -> initial narrow window ±0.05
            center = float(values[0])
            if center < 0.0 or center > 1.0:
                raise ValueError(f"{name} center must be in [0,1]")
            return (max(0.0, center - 0.05), min(1.0, center + 0.05))
        if len(values) >= 2:
            a = float(values[0])
            b = float(values[1])
            lo = min(a, b)
            hi = max(a, b)
            if lo < 0.0 or hi > 1.0:
                raise ValueError(f"{name} bounds must be within [0,1]")
            return (lo, hi)
        return None

    @staticmethod
    def _stable_json_dumps(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    def _normalize_pattern_json(self, value: JsonLike) -> Tuple[str, List[Any]]:
        """
        Ensure top-level JSON is an array (list). Accept:
         - str containing JSON array,
         - list/tuple,
         - dict containing 'pattern' key with list value.
        Returns (json_text, parsed_list)
        """
        parsed: Any
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"pattern_json is not valid JSON: {e}") from e
        elif isinstance(value, (list, tuple)):
            parsed = list(value)
        elif isinstance(value, dict):
            # Accept dict only if it contains a top-level 'pattern' list
            if "pattern" in value and isinstance(value["pattern"], list):
                parsed = value["pattern"]
            else:
                raise ValueError("pattern_json dict must contain a top-level 'pattern' key with a list value")
        else:
            raise ValueError("pattern_json must be a JSON string, list/tuple, or dict with 'pattern' list")

        if not isinstance(parsed, list):
            raise ValueError("pattern_json must encode a JSON array (list)")

        text = self._stable_json_dumps(parsed)
        return text, parsed

    def _normalize_arbitrary_json(self, value: JsonLike) -> str:
        """
        Normalize any value (str/dict/list) to stable JSON text.
        - If str, ensure it is valid JSON, then re-dump stably.
        - If dict/list/tuple, dump stably.
        """
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}") from e
            return self._stable_json_dumps(parsed)
        elif isinstance(value, (list, tuple, dict)):
            return self._stable_json_dumps(value)
        else:
            raise ValueError("Expected JSON-like value (str/dict/list/tuple)")

    @staticmethod
    def _to_optional_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None