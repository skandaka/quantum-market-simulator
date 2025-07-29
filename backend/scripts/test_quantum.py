#!/usr/bin/env python3

# Path: /Users/skandaa/Desktop/quantum-market-simulator/backend/scripts/test_quantum.py

"""
Enhanced Quantum Backend Test Script
Tests all quantum components with proper error handling and compatibility checks
"""

import sys
import traceback
from typing import Optional


def test_basic_imports():
    """Test basic imports with compatibility checks"""
    print("1. Testing basic imports...")

    try:
        import pydantic
        print(f"   ✅ Pydantic {pydantic.__version__} imported")

        # Test StringConstraints with fallback
        try:
            from pydantic import StringConstraints
            print("   ✅ StringConstraints imported from pydantic")
        except ImportError:
            try:
                from pydantic.types import StringConstraints
                print("   ✅ StringConstraints imported from pydantic.types")
            except ImportError:
                try:
                    from pydantic.v1 import StringConstraints
                    print("   ✅ StringConstraints imported from pydantic.v1")
                except ImportError:
                    print("   ⚠️  StringConstraints not available - using alternatives")

        import numpy as np
        print(f"   ✅ Numpy {np.__version__} imported")

        import classiq
        print("   ✅ Classiq imported")

        return True

    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False


def test_quantum_functions():
    """Test quantum function definitions"""
    print("2. Testing quantum function definitions...")

    try:
        # Import specific functions instead of using import *
        from classiq import qfunc, QArray, QBit, Output, allocate, hadamard, cnot

        # Define quantum functions with correct syntax
        @qfunc
        def simple_bell_state(q: QArray[QBit]):
            """Create a simple Bell state"""
            hadamard(q[0])
            cnot(q[0], q[1])

        @qfunc
        def main(q: Output[QArray[QBit]]):
            """Main quantum function - no inputs allowed"""
            allocate(2, q)
            simple_bell_state(q)

        print("   ✅ Quantum functions defined successfully")
        return True, main

    except Exception as e:
        print(f"   ❌ Quantum function definition failed: {e}")
        return False, None


def test_circuit_creation(main_func):
    """Test quantum circuit creation"""
    print("3. Testing circuit creation...")

    try:
        from classiq import create_model, write_qmod

        # Create model
        model = create_model(main_func)
        print("   ✅ Quantum model created successfully")

        # Try to write QMOD (this tests model validity)
        qmod_str = write_qmod(model)
        print("   ✅ QMOD generated successfully")
        print(f"   📄 QMOD length: {len(qmod_str)} characters")

        return True, model

    except Exception as e:
        print(f"   ❌ Circuit creation failed: {e}")
        if "API key" in str(e).lower() or "authentication" in str(e).lower():
            print("   💡 This might be due to missing API key - that's expected for now")
        return False, None


def test_classiq_client():
    """Test ClassiqClient integration"""
    print("4. Testing ClassiqClient integration...")

    try:
        # Import our custom ClassiqClient
        from app.services.quantum.classiq_auth import ClassiqClient
        print("   ✅ ClassiqClient imported successfully")

        # Create client instance
        client = ClassiqClient()
        print("   ✅ ClassiqClient instance created")

        # Test basic client methods (without requiring authentication)
        if hasattr(client, 'is_connected'):
            print(f"   📊 Connection status available")

        return True

    except ImportError as e:
        print(f"   ❌ ClassiqClient import failed: {e}")
        print("   💡 Make sure the quantum auth module exists")
        return False
    except Exception as e:
        print(f"   ❌ ClassiqClient test failed: {e}")
        return False


def test_environment_setup():
    """Test environment and configuration"""
    print("5. Testing environment setup...")

    try:
        import os
        from pathlib import Path

        # Check for .env file
        env_path = Path(".env")
        if env_path.exists():
            print("   ✅ .env file found")

            # Check for Classiq API key
            with open(env_path) as f:
                env_content = f.read()
                if "CLASSIQ_API_KEY" in env_content:
                    print("   ✅ CLASSIQ_API_KEY found in .env")
                else:
                    print("   ⚠️  CLASSIQ_API_KEY not found in .env")
        else:
            print("   ⚠️  .env file not found")

        # Check virtual environment
        if hasattr(sys, 'prefix') and hasattr(sys, 'base_prefix'):
            if sys.prefix != sys.base_prefix:
                print("   ✅ Virtual environment detected")
            else:
                print("   ⚠️  No virtual environment detected")

        return True

    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive quantum backend test"""
    print("🧪 Testing Quantum Backend Setup")
    print("=" * 50)

    all_passed = True

    # Test 1: Basic imports
    if not test_basic_imports():
        all_passed = False

    print()

    # Test 2: Quantum functions
    func_success, main_func = test_quantum_functions()
    if not func_success:
        all_passed = False

    print()

    # Test 3: Circuit creation (only if functions work)
    if func_success and main_func:
        circuit_success, model = test_circuit_creation(main_func)
        if not circuit_success:
            all_passed = False
    else:
        print("3. Skipping circuit creation (function test failed)")

    print()

    # Test 4: ClassiqClient
    if not test_classiq_client():
        all_passed = False

    print()

    # Test 5: Environment
    if not test_environment_setup():
        all_passed = False

    print()
    print("=" * 50)

    if all_passed:
        print("🎉 All tests passed! Quantum backend is ready.")
        print()
        print("Next steps:")
        print("1. Get your Classiq API key from https://platform.classiq.io/")
        print("2. Add it to your .env file: CLASSIQ_API_KEY=your_api_key")
        print("3. Run: python -m app.main")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print()
        print("Common fixes:")
        print("1. Run the fix script: ./scripts/fix_quantum.sh")
        print("2. Check your Python environment")
        print("3. Verify all dependencies are installed")

    return all_passed


if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)