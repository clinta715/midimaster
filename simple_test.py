#!/usr/bin/env python3
"""
Simple test script for MIDI Master selective generation enhancements.
"""

from genres.genre_factory import GenreFactory
from generators.pattern_generator import PatternGenerator
from structures.song_skeleton import SongSkeleton


def test_basic_selective_generation():
    """Test basic selective generation functionality."""
    print("🧪 Testing basic selective generation...")

    try:
        # Create genre rules
        genre_rules = GenreFactory.create_genre_rules('pop')
        generator = PatternGenerator(genre_rules, 'happy')
        skeleton = SongSkeleton('pop', 120, 'happy')

        # Test generate_beats_only
        print("Testing generate_beats_only...")
        beat_pattern = generator.generate_beats_only(skeleton, 4, beat_complexity=0.5)
        print(f"✅ Generated beat pattern with {len(beat_pattern.notes)} notes")

        # Test generate_chords_only
        print("Testing generate_chords_only...")
        chord_pattern = generator.generate_chords_only(skeleton, 4, chord_complexity='medium')
        print(f"✅ Generated chord pattern with {len(chord_pattern.chords)} chords")

        # Test generate_selective_patterns
        print("Testing generate_selective_patterns...")
        patterns = generator.generate_selective_patterns(
            skeleton, 4,
            ['melody', 'harmony'],
            beat_complexity=0.5,
            chord_complexity='medium'
        )
        print(f"✅ Generated {len(patterns)} selective patterns")

        # Test backward compatibility
        print("Testing backward compatibility...")
        all_patterns = generator.generate_patterns(skeleton, 4)
        print(f"✅ Generated {len(all_patterns)} patterns with backward compatibility")

        print("✅ All basic tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_subgenre_support():
    """Test subgenre support."""
    print("\n🎭 Testing subgenre support...")

    try:
        from genres.genre_rules import PopRules, RockRules

        # Test PopRules subgenres
        pop_rules = PopRules()
        subgenres = pop_rules.get_subgenres()
        print(f"✅ Pop subgenres: {subgenres}")

        # Test beat characteristics
        base_chars = pop_rules.get_beat_characteristics()
        print(f"✅ Base pop characteristics: swing={base_chars['swing_factor']}, sync={base_chars['syncopation_level']}")

        if subgenres:
            sub_chars = pop_rules.get_beat_characteristics(subgenres[0])
            print(f"✅ Subgenre characteristics for {subgenres[0]}: swing={sub_chars['swing_factor']}")

        print("✅ Subgenre tests passed!")
        return True

    except Exception as e:
        print(f"❌ Subgenre test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🎵 MIDI Master Selective Generation Test Suite")
    print("=" * 50)

    results = []

    # Run tests
    results.append(test_basic_selective_generation())
    results.append(test_subgenre_support())

    # Summary
    passed = sum(results)
    total = len(results)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)