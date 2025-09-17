"""
GUI Rhythm Variation Controls for MIDI Master

Add these controls to the Advanced tab in main_window.py to expose rhythm variation settings.

Integration instructions:
1. Add the rhythm controls to the Advanced tab layout
2. Connect the signals to update current_params
3. Pass the rhythm parameters to the pattern generator

Example integration in _build_advanced_tab():

# Add rhythm controls before performance controls
layout.addWidget(rhythm_group)

# Then add performance controls
layout.addWidget(perf_group)

# In connect_signals(), add rhythm control connections:
self.pattern_strength_slider.valueChanged.connect(self._on_advanced_changed)
self.swing_percent_slider.valueChanged.connect(self._on_advanced_changed)
self.fill_freq_slider.valueChanged.connect(self._on_advanced_changed)
self.ghost_level_slider.valueChanged.connect(self._on_advanced_changed)

# Update _collect_performance_params() to include rhythm settings
# Update on_generate_clicked() to pass rhythm params to PatternGenerator
"""

from PyQt6.QtWidgets import QGroupBox, QGridLayout, QLabel, QSlider
from PyQt6.QtCore import Qt

def create_rhythm_controls():
    """Create rhythm variation controls for the GUI."""

    # Rhythm Variation Controls
    rhythm_group = QGroupBox("Rhythm Variation")
    rhythm_layout = QGridLayout(rhythm_group)

    # Pattern Strength
    pattern_strength_label = QLabel("Pattern Strength")
    pattern_strength_slider = QSlider(Qt.Orientation.Horizontal)
    pattern_strength_slider.setRange(0, 100)
    pattern_strength_slider.setValue(100)
    pattern_strength_slider.setToolTip("How strictly to follow original rhythm patterns (0=loose variations, 100=strict)")

    # Swing Percent
    swing_percent_label = QLabel("Swing Amount")
    swing_percent_slider = QSlider(Qt.Orientation.Horizontal)
    swing_percent_slider.setRange(0, 100)
    swing_percent_slider.setValue(50)
    swing_percent_slider.setToolTip("Amount of swing feel (0=straight, 100=maximum swing)")

    # Fill Frequency
    fill_freq_label = QLabel("Fill Frequency")
    fill_freq_slider = QSlider(Qt.Orientation.Horizontal)
    fill_freq_slider.setRange(0, 50)
    fill_freq_slider.setValue(25)
    fill_freq_slider.setToolTip("How often to add rhythmic fills (0=never, 50=every 2 bars)")

    # Ghost Note Level
    ghost_level_label = QLabel("Ghost Note Level")
    ghost_level_slider = QSlider(Qt.Orientation.Horizontal)
    ghost_level_slider.setRange(0, 200)
    ghost_level_slider.setValue(100)
    ghost_level_slider.setToolTip("Intensity of ghost notes (0=none, 200=very prominent)")

    # Add controls to grid layout
    rhythm_layout.addWidget(pattern_strength_label, 0, 0)
    rhythm_layout.addWidget(pattern_strength_slider, 0, 1)
    rhythm_layout.addWidget(swing_percent_label, 1, 0)
    rhythm_layout.addWidget(swing_percent_slider, 1, 1)
    rhythm_layout.addWidget(fill_freq_label, 2, 0)
    rhythm_layout.addWidget(fill_freq_slider, 2, 1)
    rhythm_layout.addWidget(ghost_level_label, 3, 0)
    rhythm_layout.addWidget(ghost_level_slider, 3, 1)

    return rhythm_group, {
        'pattern_strength_slider': pattern_strength_slider,
        'swing_percent_slider': swing_percent_slider,
        'fill_freq_slider': fill_freq_slider,
        'ghost_level_slider': ghost_level_slider
    }


def collect_rhythm_params(sliders):
    """Collect rhythm parameters from GUI sliders."""
    return {
        'pattern_strength': sliders['pattern_strength_slider'].value() / 100.0,  # Convert to 0.0-1.0
        'swing_percent': sliders['swing_percent_slider'].value() / 100.0,       # Convert to 0.0-1.0
        'fill_frequency': sliders['fill_freq_slider'].value() / 100.0,          # Convert to 0.0-0.5
        'ghost_note_level': sliders['ghost_level_slider'].value() / 100.0       # Convert to 0.0-2.0
    }


# Example usage in main_window.py:
"""
# In __init__():
self.rhythm_group, self.rhythm_sliders = create_rhythm_controls()

# In _build_advanced_tab():
layout.addWidget(self.rhythm_group)

# In connect_signals():
for slider in self.rhythm_sliders.values():
    slider.valueChanged.connect(self._on_advanced_changed)

# In _on_advanced_changed():
self.current_params['rhythm_variation'] = collect_rhythm_params(self.rhythm_sliders)

# In on_generate_clicked():
rhythm_params = self.current_params.get('rhythm_variation', {})
pattern_generator = PatternGenerator(
    genre_rules,
    params['mood'],
    note_density=density_manager.note_density,
    rhythm_density=density_manager.rhythm_density,
    chord_density=density_manager.chord_density,
    bass_density=density_manager.bass_density,
    **rhythm_params  # Unpack rhythm variation parameters
)
"""

# Preset rhythm settings for different styles:
RHYTHM_PRESETS = {
    'default': {'pattern_strength': 1.0, 'swing_percent': 0.5, 'fill_frequency': 0.25, 'ghost_note_level': 1.0},
    'loose_groove': {'pattern_strength': 0.7, 'swing_percent': 0.6, 'fill_frequency': 0.4, 'ghost_note_level': 1.3},
    'tight_technical': {'pattern_strength': 0.95, 'swing_percent': 0.48, 'fill_frequency': 0.1, 'ghost_note_level': 0.8},
    'wild_experimental': {'pattern_strength': 0.3, 'swing_percent': 0.8, 'fill_frequency': 0.5, 'ghost_note_level': 1.8},
    'minimal_ambient': {'pattern_strength': 1.0, 'swing_percent': 0.5, 'fill_frequency': 0.0, 'ghost_note_level': 0.2}
}