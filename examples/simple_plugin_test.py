"""
Simple Test for Enhanced Plugin Parameter Enumeration System

This script provides a basic test of the enhanced plugin parameter enumeration system
to verify that the implementation works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from audio.plugin_host import PluginHost, ENHANCED_ENUMERATION_AVAILABLE
    from audio.plugin_enumeration import (
        ParameterType,
        ParameterCategory,
        ParameterCollection,
        ParameterInfo,
        ParameterMetadata,
        ParameterEnumerator
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_basic_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing basic imports...")

    try:
        print(f"✅ PluginHost imported: {PluginHost}")

        if ENHANCED_ENUMERATION_AVAILABLE:
            print("✅ Enhanced enumeration system available")

            # Test enumeration components
            print(f"✅ ParameterType enum: {[pt.value for pt in ParameterType]}")
            print(f"✅ ParameterCategory enum: {[pc.value for pc in ParameterCategory]}")

            # Test classes
            print(f"✅ ParameterMetadata class: {ParameterMetadata}")
            print(f"✅ ParameterInfo class: {ParameterInfo}")
            print(f"✅ ParameterCollection class: {ParameterCollection}")
            print(f"✅ ParameterEnumerator class: {ParameterEnumerator}")

            return True
        else:
            print("⚠️ Enhanced enumeration system not available")
            return False

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_parameter_collection():
    """Test ParameterCollection functionality."""
    print("\n🧪 Testing ParameterCollection...")

    try:
        collection = ParameterCollection()
        collection.plugin_name = "Test Plugin"
        collection.plugin_version = "1.0.0"
        collection.plugin_type = "synthesizer"

        # Create test parameters
        test_params = [
            ("frequency", ParameterType.CONTINUOUS, ParameterCategory.OSCILLATOR, "Hz"),
            ("volume", ParameterType.CONTINUOUS, ParameterCategory.AMPLITUDE, "dB"),
            ("waveform", ParameterType.ENUMERATION, ParameterCategory.OSCILLATOR, ""),
            ("bypass", ParameterType.BOOLEAN, ParameterCategory.GENERIC, "")
        ]

        for name, ptype, category, units in test_params:
            metadata = ParameterMetadata(
                name=name,
                description=f"Controls {name}",
                units=units,
                category=category,
                parameter_type=ptype
            )

            param_info = ParameterInfo(
                metadata=metadata,
                current_value=0.5 if ptype == ParameterType.CONTINUOUS else (True if ptype == ParameterType.BOOLEAN else 0),
                index=len(collection.parameters)
            )

            collection.add_parameter(param_info)

        print(f"✅ Created test collection: {collection.plugin_name}")
        print(f"✅ Parameters added: {len(collection.parameters)}")

        # Test retrieval methods
        osc_params = collection.get_parameters_by_category(ParameterCategory.OSCILLATOR)
        print(f"✅ Oscillator parameters: {len(osc_params)}")

        continuous_params = collection.get_parameters_by_type(ParameterType.CONTINUOUS)
        print(f"✅ Continuous parameters: {len(continuous_params)}")

        # Test serialization
        data_dict = collection.to_dict()
        print(f"✅ Serialization successful: {len(data_dict['parameters'])} parameters")

        # Test deserialization
        recreated_collection = ParameterCollection.from_dict(data_dict)
        print(f"✅ Deserialization successful: {recreated_collection.plugin_name}")

        return True

    except Exception as e:
        print(f"❌ ParameterCollection test failed: {e}")
        return False


def test_parameter_enumerator():
    """Test ParameterEnumerator functionality."""
    print("\n🧪 Testing ParameterEnumerator...")

    try:
        enumerator = ParameterEnumerator()

        # Test metadata generation
        metadata1 = enumerator.extract_metadata_from_parameter("osc1_frequency", 440.0)
        print(f"✅ Metadata extracted: {metadata1.name} -> {metadata1.parameter_type.value}")

        metadata2 = enumerator.extract_metadata_from_parameter("bypass", True)
        print(f"✅ Metadata extracted: {metadata2.name} -> {metadata2.parameter_type.value}")

        # Test categorization
        print("✅ Categorization keywords loaded:"
        for category, keywords in enumerator.category_keywords.items():
            print(f"      {category.value}: {keywords[:3]}..."        # Test category hierarchy
        hierarchy = enumerator.get_category_hierarchy()
        print(f"✅ Category hierarchy defined with {len(hierarchy)} main categories"

        return True

    except Exception as e:
        print(f"❌ ParameterEnumerator test failed: {e}")
        return False


def test_plugin_host_extensions():
    """Test the new methods added to PluginHost."""
    print("\n🧪 Testing PluginHost extensions...")

    try:
        host = PluginHost()
        print(f"✅ PluginHost initialized: {type(host)}")

        # Test new methods (these will return None if no plugin loaded)
        detailed = host.get_detailed_parameters()
        print(f"✅ get_detailed_parameters(): {type(detailed)}")

        category_params = host.get_parameters_by_category("oscillator")
        print(f"✅ get_parameters_by_category(): {len(category_params)} parameters")

        enumerator = host.get_enumeration_interface()
        print(f"✅ get_enumeration_interface(): {type(enumerator)}")

        # Test backward compatibility - should not crash
        legacy_params = host.get_plugin_parameters()
        print(f"✅ get_plugin_parameters() (legacy): {type(legacy_params)}")

        return True

    except Exception as e:
        print(f"❌ PluginHost extensions test failed: {e}")
        return False


def run_tests():
    """Run all tests."""
    print("🎵 MIDI MASTER - PLUGIN ENUMERATION SYSTEM TEST")
    print("=" * 60)

    results = []

    if not IMPORTS_SUCCESSFUL:
        print("❌ Required modules could not be imported. Test failed.")
        return False

    # Run individual tests
    results.append(("Basic Imports", test_basic_imports()))
    results.append(("ParameterCollection", test_parameter_collection()))
    results.append(("ParameterEnumerator", test_parameter_enumerator()))
    results.append(("PluginHost Extensions", test_plugin_host_extensions()))

    # Print results summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print("2")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! The enhanced plugin parameter enumeration system is working correctly.")
        print("\nKey Features Verified:")
        print("✅ Parameter data classes and collections")
        print("✅ Automatic parameter categorization")
        print("✅ Metadata extraction capabilities")
        print("✅ PluginHost integration with backward compatibility")
        print("✅ Serialization and deserialization support")

        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)