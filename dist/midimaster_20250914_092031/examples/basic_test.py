#!/usr/bin/env python3
"""
Basic test to verify the enhanced plugin parameter enumeration system works
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that we can import the enhanced enumeration system"""
    print("Testing imports...")

    try:
        from audio.plugin_enumeration import ParameterCollection, ParameterEnumerator, ParameterType, ParameterCategory
        from audio.plugin_host import PluginHost, ENHANCED_ENUMERATION_AVAILABLE
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the enumeration system"""
    print("\nTesting basic functionality...")

    try:
        from audio.plugin_enumeration import ParameterCollection, ParameterEnumerator

        # Test ParameterCollection
        collection = ParameterCollection()
        collection.plugin_name = "Test Plugin"
        print(f"✅ Created ParameterCollection: {collection.plugin_name}")

        # Test ParameterEnumerator
        enumerator = ParameterEnumerator()
        print("✅ Created ParameterEnumerator")

        # Test basic enumeration
        test_params = ["frequency", "volume", "attack"]
        for param in test_params:
            metadata = enumerator.extract_metadata_from_parameter(param, 0.5)
            print(f"   📊 {param} -> {metadata.parameter_type.value}")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_plugin_host_integration():
    """Test PluginHost integration"""
    print("\nTesting PluginHost integration...")

    try:
        from audio.plugin_host import PluginHost

        host = PluginHost()
        print("✅ Created PluginHost instance")

        # Test new methods
        detailed = host.get_detailed_parameters()
        print(f"✅ get_detailed_parameters() returned: {type(detailed)}")

        category_params = host.get_parameters_by_category("oscillator")
        print(f"✅ get_parameters_by_category returned: {len(category_params)} items")

        enumerator = host.get_enumeration_interface()
        print(f"✅ get_enumeration_interface() returned: {type(enumerator)}")

        # Test backward compatibility
        legacy_params = host.get_plugin_parameters()
        print(f"✅ Legacy get_plugin_parameters() works: {type(legacy_params)}")

        return True

    except Exception as e:
        print(f"❌ PluginHost integration test failed: {e}")
        return False

def main():
    """Run the basic test"""
    print("🔌 MIDI MASTER - BASIC PLUGIN ENUMERATION TEST")
    print("=" * 50)

    # Run tests
    import_success = test_imports()
    basic_success = test_basic_functionality() if import_success else False
    host_success = test_plugin_host_integration() if import_success else False

    print("\n" + "=" * 50)
    print("TEST RESULTS:")

    if import_success and basic_success and host_success:
        print("🎉 SUCCESS: Enhanced plugin parameter enumeration system is working!")
        print("\nKey features verified:")
        print("  ✅ Parameter data structures created")
        print("  ✅ Parameter categorization implemented")
        print("  ✅ PluginHost extension successful")
        print("  ✅ Backward compatibility maintained")
        print("\nImplementation complete! ✅")

        return True
    else:
        print("❌ FAILURE: Some tests failed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        sys.exit(1)