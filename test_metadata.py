#!/usr/bin/env python3
"""
Test script for the professional MIDI metadata preservation system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from output.metadata_manager import MetadataManager, ProjectMetadata, DAWType, TrackMetadata, TrackType, CopyrightInfo, SMPTEConfig

def test_metadata_creation():
    """Test basic metadata creation and validation."""
    print("Testing metadata creation...")

    # Create a metadata manager
    manager = MetadataManager()

    # Create project metadata
    metadata = manager.create_project_metadata_from_generation(
        genre="electronic",
        mood="energetic",
        tempo=128,
        time_signature=(4, 4),
        key="C",
        scale="minor",
        daw_target=DAWType.ABLETON_LIVE
    )

    print(f"Created project: {metadata.project_name}")
    print(f"Genre: {metadata.genre}, Mood: {metadata.mood}")
    print(f"Tempo: {metadata.tempo}, Time Signature: {metadata.time_signature}")
    print(f"Key: {metadata.key} {metadata.scale}")
    print(f"DAW Target: {metadata.daw_target.value}")

    # Check track names
    print("\nTracks:")
    for name, track in metadata.tracks.items():
        print(f"  {name}: {track.name} (Channel {track.channel}, Program {track.program})")

    # Validate metadata
    is_valid, errors = manager.validate_metadata(metadata)
    if is_valid:
        print("\n✓ Metadata validation passed")
    else:
        print("\n✗ Metadata validation failed:")
        for error in errors:
            print(f"  - {error}")

    return metadata

def test_smpte_config():
    """Test SMPTE configuration."""
    print("\nTesting SMPTE configuration...")

    smpte = SMPTEConfig(
        frame_rate=30,
        start_hour=1,
        start_minute=30,
        start_second=45,
        start_frame=15,
        drop_frame=False
    )

    print(f"SMPTE: {smpte.start_hour:02d}:{smpte.start_minute:02d}:{smpte.start_second:02d}:{smpte.start_frame:02d}@{smpte.frame_rate}fps")
    print(f"MIDI SMPTE data: {smpte.to_midi_smpte_offset().hex()}")

    return smpte

def test_copyright_info():
    """Test copyright information."""
    print("\nTesting copyright information...")

    copyright = CopyrightInfo(
        copyright_text="© 2024 MIDI Master Project",
        author="Test Composer",
        composer="Test Composer",
        license_type="Creative Commons",
        isrc="US-ABC-24-12345",
        upc="1234567890123"
    )

    print(f"Copyright: {copyright.copyright_text}")
    print(f"Author: {copyright.author}")
    print(f"License: {copyright.license_type}")
    print(f"ISRC: {copyright.isrc}")

    return copyright

def test_config_persistence():
    """Test metadata configuration persistence."""
    print("\nTesting configuration persistence...")

    manager = MetadataManager()

    # Get current metadata
    original = manager.current_metadata

    # Modify some settings
    original.copyright.copyright_text = "© 2024 Test Project"
    original.custom_metadata["test_setting"] = "test_value"

    # Save configuration
    if manager.save_config():
        print("✓ Configuration saved successfully")

        # Create new manager and load config
        new_manager = MetadataManager()
        loaded = new_manager.current_metadata

        if loaded.copyright.copyright_text == "© 2024 Test Project":
            print("✓ Configuration loaded successfully")
        else:
            print("✗ Configuration loading failed")
    else:
        print("✗ Configuration saving failed")

def main():
    """Run all tests."""
    print("=== MIDI Metadata Preservation System Test ===\n")

    try:
        # Test basic functionality
        metadata = test_metadata_creation()
        smpte = test_smpte_config()
        copyright = test_copyright_info()

        # Update metadata with test data
        metadata.smpte = smpte
        metadata.copyright = copyright

        # Test persistence
        test_config_persistence()

        print("\n=== All tests completed successfully! ===")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())