from output.midi_output import MidiOutput

midi_out = MidiOutput()

# Test filename generation
filename = midi_out.generate_output_filename("pop", "happy", 120, "4/4")
print(f"Generated filename: {filename}")

# Test unique
unique = midi_out.get_unique_filename(filename)
print(f"Unique filename: {unique}")

# Test with existing file
import os
if os.path.exists(unique):
    print("File exists")
else:
    print("File does not exist, would create")

# Test separate stems naming
base = "pop_happy_120_4-4_20250915_210000"
stems = ['bass', 'harmony', 'melody', 'rhythm']
for stem in stems:
    stem_file = f"{base}_{stem}.mid"
    unique_stem = midi_out.get_unique_filename(stem_file)
    print(f"Stem {stem}: {unique_stem}")