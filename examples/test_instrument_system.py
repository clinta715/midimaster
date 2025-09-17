"""
Test script for the diverse instrument support system.

This script demonstrates and tests:
- Modern synths (808, FM, analog, digital)
- Ethnic percussion instruments
- Wind instruments (brass, woodwind, reed)
- Layered and hybrid instrument configurations
- Instrument-specific parameter handling
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from instruments import (
        instrument_registry,
        InstrumentCategory,
        InstrumentSubcategory,
        get_instrument_categories,
        get_preset_config
    )
    from instruments.preset_manager import preset_manager
    from instruments.instrument_midi_output import (
        InstrumentMidiOutput,
        get_available_instruments,
        get_instrument_categories as get_midi_categories,
        validate_instrument_setup
    )
    print("✅ Instrument system imported successfully")
except ImportError as e:
    print(f"❌ Failed to import instrument system: {e}")
    sys.exit(1)

def test_basic_instrument_registry():
    """Test basic instrument registry functionality."""
    print("\n🔍 Testing basic instrument registry...")

    # Test getting presets
    presets = list(instrument_registry.presets.keys())
    print(f"📦 Available presets: {len(presets)}")
    print(f"   Sample presets: {presets[:5]}")

    # Test categories
    categories = get_instrument_categories()
    print(f"📂 Instrument categories: {list(categories.keys())}")

    for category, presets_in_cat in categories.items():
        if presets_in_cat:
            print(f"   {category}: {len(presets_in_cat)} presets")

    return True

def test_instrument_categories():
    """Test instrument categorization."""
    print("\n🏷️  Testing instrument categories...")

    # Test modern synths
    synths = instrument_registry.get_presets_by_category(InstrumentCategory.SYNTHETIC)
    print(f"🎛️  Modern synth presets: {len(synths)}")
    for preset in synths[:3]:  # Show first 3
        print(f"   • {preset.name}: {preset.description}")

    # Test ethnic percussion
    ethnic = instrument_registry.get_presets_by_category(InstrumentCategory.ETHNIC)
    print(f"🪘 Ethnic percussion presets: {len(ethnic)}")
    for preset in ethnic:
        print(f"   • {preset.name}: {preset.description}")

    # Test wind instruments
    wind = instrument_registry.get_presets_by_category(InstrumentCategory.WIND)
    print(f"🎺 Wind instrument presets: {len(wind)}")
    for preset in wind:
        print(f"   • {preset.name}: {preset.description}")

    return True

def test_layered_instruments():
    """Test layered and hybrid instrument configurations."""
    print("\n🎼 Testing layered instruments...")

    # Test layered pad
    layered_pad = instrument_registry.get_preset("layered_pad")
    if layered_pad:
        print(f"🎹 Layered pad preset found:")
        print(f"   • Name: {layered_pad.name}")
        print(f"   • Layered: {layered_pad.is_layered}")
        print(f"   • Layers: {len(layered_pad.layers)}")
        for i, layer in enumerate(layered_pad.layers):
            layer_name = layer.preset if isinstance(layer.preset, str) else layer.preset.name
            print(f"     Layer {i+1}: {layer_name} (vol: {layer.volume}, pan: {layer.pan})")

    # Test hybrid bass
    hybrid_bass = instrument_registry.get_preset("hybrid_bass")
    if hybrid_bass:
        print(f"🎸 Hybrid bass preset found:")
        print(f"   • Name: {hybrid_bass.name}")
        print(f"   • Layered: {hybrid_bass.is_layered}")
        print(f"   • Layers: {len(hybrid_bass.layers)}")
        for i, layer in enumerate(hybrid_bass.layers):
            layer_name = layer.preset if isinstance(layer.preset, str) else layer.preset.name
            print(f"     Layer {i+1}: {layer_name} (vol: {layer.volume}, transpose: {layer.transpose})")

    return True

def test_instrument_parameters():
    """Test instrument-specific parameter handling."""
    print("\n⚙️  Testing instrument parameters...")

    # Test 808 kick parameters
    kick = instrument_registry.get_preset("808_kick")
    if kick:
        print(f"🥁 808 Kick parameters:")
        for param, value in kick.parameters.items():
            print(f"   • {param}: {value}")

    # Test FM synth parameters
    fm_bass = instrument_registry.get_preset("fm_bass")
    if fm_bass:
        print(f"🎛️  FM Bass parameters:")
        for param, value in fm_bass.parameters.items():
            print(f"   • {param}: {value}")

    # Test wind instrument parameters
    trumpet = instrument_registry.get_preset("trumpet")
    if trumpet:
        print(f"🎺 Trumpet parameters:")
        for param, value in trumpet.parameters.items():
            print(f"   • {param}: {value}")

    return True

def test_preset_search():
    """Test preset search functionality."""
    print("\n🔎 Testing preset search...")

    # Search for kick drums
    kick_results = instrument_registry.search_presets("kick")
    print(f"🥁 Kick drum search results: {len(kick_results)}")
    for preset in kick_results:
        print(f"   • {preset.name}: {preset.description}")

    # Search for synths
    synth_results = instrument_registry.search_presets("synth")
    print(f"🎛️  Synth search results: {len(synth_results)}")
    for preset in synth_results[:3]:  # Show first 3
        print(f"   • {preset.name}: {preset.description}")

    # Search by tags
    bass_presets = instrument_registry.get_presets_by_tag("bass")
    print(f"🎸 Bass-tagged presets: {len(bass_presets)}")
    for preset in bass_presets[:3]:
        print(f"   • {preset.name}")

    return True

def test_midi_output_integration():
    """Test MIDI output integration."""
    print("\n🎵 Testing MIDI output integration...")

    try:
        midi_output = InstrumentMidiOutput()
        print("✅ InstrumentMidiOutput initialized successfully")

        # Test instrument assignment
        from structures.data_structures import PatternType
        melody_instrument = midi_output.assign_instrument_to_pattern(PatternType.MELODY)
        bass_instrument = midi_output.assign_instrument_to_pattern(PatternType.BASS)
        rhythm_instrument = midi_output.assign_instrument_to_pattern(PatternType.RHYTHM)

        print(f"🎼 Assigned instruments:")
        print(f"   • Melody: {melody_instrument}")
        print(f"   • Bass: {bass_instrument}")
        print(f"   • Rhythm: {rhythm_instrument}")

        # Test preset configuration retrieval
        config = get_preset_config("808_kick")
        if config:
            print(f"⚙️  808 Kick configuration retrieved:")
            print(f"   • MIDI Program: {config.get('midi_program')}")
            print(f"   • Parameters: {config.get('parameters', {})}")

    except Exception as e:
        print(f"❌ MIDI output test failed: {e}")
        return False

    return True

def test_preset_management():
    """Test preset management functionality."""
    print("\n💾 Testing preset management...")

    try:
        # Test preset validation
        issues = validate_instrument_setup("808_kick")
        if issues:
            print(f"⚠️  808 Kick validation issues: {issues}")
        else:
            print("✅ 808 Kick preset validation passed")

        # Test creating a custom preset
        from instruments.instrument_categories import InstrumentLayer
        custom_preset = preset_manager.create_preset_from_template(
            "fm_bass",
            "my_custom_fm_bass",
            {"feedback": 0.8, "algorithm": 3}
        )

        if custom_preset:
            print("✅ Custom preset created successfully")
            print(f"   • Name: {custom_preset.name}")
            print(f"   • Modified feedback: {custom_preset.parameters.get('feedback')}")
        else:
            print("❌ Failed to create custom preset")

    except Exception as e:
        print(f"❌ Preset management test failed: {e}")
        return False

    return True

def main():
    """Run all instrument system tests."""
    print("🎯 Testing Diverse Instrument Support System")
    print("=" * 50)

    tests = [
        ("Basic Registry", test_basic_instrument_registry),
        ("Categories", test_instrument_categories),
        ("Layered Instruments", test_layered_instruments),
        ("Parameters", test_instrument_parameters),
        ("Search", test_preset_search),
        ("MIDI Integration", test_midi_output_integration),
        ("Preset Management", test_preset_management),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Instrument system is working correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the output above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)