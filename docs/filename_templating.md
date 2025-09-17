# Filename Templating

This document describes the filename templating system for MIDI output paths. It covers placeholders, directory structures, sanitization, collision handling, and GUI preview behavior.

Key implementation entry points:
- [core/filename_templater.py](core/filename_templater.py)
  - [python.format_filename()](core/filename_templater.py:170)
  - [python.resolve_placeholders()](core/filename_templater.py:78)
  - [python.validate_template()](core/filename_templater.py:118)
  - [python.sanitize_component()](core/filename_templater.py:24)
- [output/midi_output.py](output/midi_output.py)
  - Uses the templater when a template string is provided
  - [python.MidiOutput.save_to_midi()](output/midi_output.py:312)
  - [python.MidiOutput.save_to_separate_midi_files()](output/midi_output.py:479)
- [generators/stem_manager.py](generators/stem_manager.py)
  - Optional templating for exported stems
  - [python.StemManager.export_stems_to_midi()](generators/stem_manager.py:581)

1) Default template
The recommended default template for descriptive, per-output naming is:
"{genre}_{mood}_{tempo}_{bars}_{timestamp}_{stem}.mid"
- timestamp format: YYYYMMDD_HHMMSS
- stem: "combined" for single-file output, otherwise the stem name (melody, harmony, bass, rhythm, etc.)

Note: If no template is provided, legacy auto-naming is used by [python.MidiOutput.save_to_midi()](output/midi_output.py:312).

2) Allowed placeholders and meanings
Placeholders are written as {name}. The following keys are supported:
- {genre}: Generation genre (e.g., pop, rock, jazz)
- {mood}: Mood string (e.g., happy, calm, energetic)
- {tempo}: Integer BPM
- {bars}: Integer bars
- {timestamp}: Generation timestamp (YYYYMMDD_HHMMSS by default)
- {stem}: Stem name for per-stem output (combined, melody, harmony, bass, rhythm)
- {run_index}: Optional per-run index provided by callers
- {unique_id}: 8-character unique identifier to disambiguate files

Mapping resolution is implemented by [python.resolve_placeholders()](core/filename_templater.py:78).
Validation of placeholders is implemented by [python.validate_template()](core/filename_templater.py:118).

3) Subdirectories in template
Templates may include subdirectories using / or \ as separators. For example:
"runs/{genre}/{mood}_{tempo}_{bars}_{stem}"
This yields a relative path under the base output directory (default "output").

- Components are sanitized per path segment.
- The final file is saved under base_dir / rendered_subdirs with a .mid extension appended if missing.
- Implemented by [python.format_filename()](core/filename_templater.py:170).

4) Sanitization rules and reserved characters
Each path segment is sanitized by [python.sanitize_component()](core/filename_templater.py:24):
- Replace path separators and invalid filesystem chars: <>:"/\|?* become underscores
- Collapse whitespace to underscore; reduce repeated underscores/dashes
- Strip leading/trailing . _ -
- Avoid Windows reserved device names (con, prn, aux, nul, com1..9, lpt1..9)
- Remove control characters; enforce a safe character set [A-Za-z0-9._-]
- Truncate segment length to 100 characters

This ensures cross-platform compatible filenames and directory names.

5) Collision handling
Uniqueness is enforced by [python.ensure_unique()](core/filename_templater.py:135):
- If the exact path exists, append suffixes _2, _3, ... up to _99
- If still colliding, append an 8-character unique id suffix
- Parent directories are created on demand

6) Examples

- Per-stem output
Template: "runs/{genre}_{mood}_{tempo}_{bars}_{stem}"
- Combined output file (no separate files): stem = "combined"
- Separate stems: stem = melody, harmony, bass, rhythm
Used in:
- [python.MidiOutput.save_to_midi()](output/midi_output.py:312) for combined outputs
- [python.MidiOutput.save_to_separate_midi_files()](output/midi_output.py:479) for per-stem files

- Per-run folder organization
Template: "runs/{genre}/{mood}_{tempo}_{bars}/{timestamp}_{stem}"
- Groups outputs by genre and a per-run subfolder keyed by mood/tempo/bars
- Produces clean runs with time-based filenames per stem

- Template preview behavior (GUI vs. on-disk saving)
GUI preview:
- [python.build_preview_filename()](gui/settings_helpers.py:39) is a pure string renderer (no I/O)
- Sanitizes placeholders and appends .mid if missing
- Does NOT ensure uniqueness or create directories

On-disk saving:
- [python.format_filename()](core/filename_templater.py:170) performs full sanitization
- Ensures .mid extension, creates parent directories, and resolves collisions
- Called by [python.MidiOutput.save_to_midi()](output/midi_output.py:312)

7) Backward compatibility
- Empty or omitted template preserves legacy naming:
  - Combined outputs: legacy name generation in [python.MidiOutput.save_to_midi()](output/midi_output.py:312)
  - Separate stems: legacy base naming is preserved when no template is provided
- This behavior keeps existing scripts and workflows working without changes

8) References (source)
- [core/filename_templater.py](core/filename_templater.py)
  - [python.format_filename()](core/filename_templater.py:170)
  - [python.resolve_placeholders()](core/filename_templater.py:78)
  - [python.validate_template()](core/filename_templater.py:118)
- [output/midi_output.py](output/midi_output.py)
  - [python.MidiOutput.save_to_midi()](output/midi_output.py:312)
  - [python.MidiOutput.save_to_separate_midi_files()](output/midi_output.py:479)
- [generators/stem_manager.py](generators/stem_manager.py)
  - [python.StemManager.export_stems_to_midi()](generators/stem_manager.py:581)