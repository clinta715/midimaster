# Beat Audit Report

## Inventory Summary

- pop: 3 styles (straight_eight, swing_eight, syncopated); selection=random choice of listed patterns per generation; swing=0.55; syncopation=0.3; emphasis=[1, 3]; tempo_range=[90, 140]
- rock: 3 styles (power_chord, eight_bar, straight_eight); selection=random choice of listed patterns per generation; swing=0.5; syncopation=0.2; emphasis=[1, 3]; tempo_range=[100, 160]
- jazz: 3 styles (swing, bebop, latin); selection=random choice of listed patterns per generation; swing=0.66; syncopation=0.6; emphasis=[1, 3]; tempo_range=[120, 200]
- electronic: 3 styles (four_on_floor, breakbeat, syncopated); selection=random choice of listed patterns per generation; swing=0.5; syncopation=0.4; emphasis=[1, 2, 3, 4]; tempo_range=[120, 140]
- hip-hop: 3 styles (boom_bap, trap, syncopated); selection=random choice of listed patterns per generation; swing=0.6; syncopation=0.7; emphasis=[2, 4]; tempo_range=[80, 110]
- classical: 3 styles (waltz, common_time, cut_time); selection=random choice of listed patterns per generation; swing=0.5; syncopation=0.1; emphasis=[1]; tempo_range=[60, 160]
- dnb: 4 styles (amen_break, double_kick, syncopated_snare, jungle_pattern); selection=random choice of listed patterns per generation; swing=0.5; syncopation=0.8; emphasis=[1, 2, 3, 4]; tempo_range=[160, 180]

## POP
- Files analyzed: 1
- Avg tempo: 120.0 BPM; Avg beat length: 0.500s
- Avg grid alignment: 28.1%
- Avg cross-part alignment: 0.0%
- Unique pattern signatures: 1
- Variation index (IOI stddev): 0.750s
- Pass/Fail: {'beat_length_fail': 1, 'grid_alignment_fail': 1, 'cross_alignment_fail': 1, 'diversity_fail': 1}

Representative files:
  - pop_energetic_balanced_tempo120_bars16_run1.mid: tempo=120.0 BPM, grid=28.1%, cross=0.0%, dur_avg=0.947s, var=0.508, sig=q16:8,20|acc:00

### Recommendations
- Improve quantization or reduce humanization jitter; ensure beat grid adherence.
- Add more rhythm patterns or vary selection weights for more diversity.

## ROCK
- Files analyzed: 1
- Avg tempo: 120.0 BPM; Avg beat length: 0.500s
- Avg grid alignment: 36.2%
- Avg cross-part alignment: 0.0%
- Unique pattern signatures: 1
- Variation index (IOI stddev): 0.540s
- Pass/Fail: {'beat_length_fail': 1, 'grid_alignment_fail': 1, 'cross_alignment_fail': 1, 'diversity_fail': 1}

Representative files:
  - rock_energetic_balanced_tempo120_bars16_run1.mid: tempo=120.0 BPM, grid=36.2%, cross=0.0%, dur_avg=0.526s, var=0.110, sig=q16:4,14,12|acc:000

### Recommendations
- Improve quantization or reduce humanization jitter; ensure beat grid adherence.
- Add more rhythm patterns or vary selection weights for more diversity.
