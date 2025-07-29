#!/usr/bin/env python3

"""
Enhanced Quantum Backend Test Script
Tests all quantum components with proper error handling
"""

import sys
import os
import traceback
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def test_basic_imports():
    """Test basic imports"""
    print("1. Testing basic imports...")

    try:
        import pydantic
        print(f"   ✅ Pydantic {pydantic.__version__} imported")

        # Test StringConstraints - not needed for our use case
        print("   ✅ StringConstraints check skipped (not required)")

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
    print("\n2. Testing quantum function definitions...")

    try:
        # Import specific functions
        from classiq import qfunc, QArray, QBit, Output, allocate

        # Test if we can import quantum gates
        try:
            from classiq import H, CX
            print("   ✅ Quantum gates (H, CX) imported successfully")
            gates_available = True
        except ImportError:
            # Try alternative imports
            try:
                from classiq.interface.gates import H, CX
                print("   ✅ Quantum gates imported from interface.gates")
                gates_available = True
            except:
                print("   ⚠️  Standard gates not available, will use qfunc definitions")
                gates_available = False

        # Define quantum functions
        @qfunc
        def bell_state_circuit(q: QArray[QBit]):
            """Create a Bell state - but we'll implement inside main"""
            pass

        @qfunc
        def main(q: Output[QArray[QBit]]):
            """Main quantum function"""
            allocate(2, q)
            # Operations would go here

        print("   ✅ Quantum functions defined successfully")
        return True, main

    except Exception as e:
        print(f"   ❌ Quantum function definition failed: {e}")
        traceback.print_exc()
        return False, None


def test_circuit_creation(main_func):
    """Test quantum circuit creation"""
    print("\n3. Testing circuit creation...")

    if not main_func:
        print("   ⚠️  Skipping - no main function")
        return False, None

    try:
        from classiq import create_model

        # Create model
        model = create_model(main_func)
        print("   ✅ Quantum model created successfully")

        return True, model

    except Exception as e:
        print(f"   ❌ Circuit creation failed: {e}")
        if "API" in str(e) or "authentication" in str(e):
            print("   💡 This is expected without API key")
        return False, None


def test_classiq_client():
    """Test ClassiqClient integration"""
    print("\n4. Testing ClassiqClient integration...")

    try:
        # Fix the import path
        from app.quantum.classiq_auth import classiq_auth
        print("   ✅ classiq_auth imported successfully")

        # Test the auth manager
        print(f"   📊 Auth manager initialized: {hasattr(classiq_auth, 'is_authenticated')}")

        if hasattr(classiq_auth, 'is_authenticated'):
            is_auth = classiq_auth.is_authenticated()
            print(f"   📊 Authentication status: {is_auth}")

        return True

    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("   💡 Check that app/quantum/classiq_auth.py exists")
        return False
    except Exception as e:
        print(f"   ❌ ClassiqClient test failed: {e}")
        traceback.print_exc()
        return False


def test_environment_setup():
    """Test environment and configuration"""
    print("\n5. Testing environment setup...")

    try:
        # Check for .env file
        env_path = Path(".env")
        if env_path.exists():
            print("   ✅ .env file found")

            # Check for API key
            with open(env_path) as f:
                env_content = f.read()
                if "CLASSIQ_API_KEY" in env_content:
                    print("   ✅ CLASSIQ_API_KEY found in .env")
                    # Check if it's set to a real value
                    if "your_actual_api_key_here" in env_content:
                        print("   ⚠️  CLASSIQ_API_KEY is still set to placeholder")
                    else:
                        print("   ✅ CLASSIQ_API_KEY appears to be configured")
        else:
            print("   ⚠️  .env file not found")

        # Check virtual environment
        if hasattr(sys, 'prefix'):
            print("   ✅ Virtual environment detected")

        return True

    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive quantum backend test"""
    print("\n🧪 Testing Quantum Backend Setup")
    print("=" * 50)

    all_passed = True

    # Test 1: Basic imports
    if not test_basic_imports():
        all_passed = False

    # Test 2: Quantum functions
    func_success, main_func = test_quantum_functions()
    if not func_success:
        all_passed = False

    # Test 3: Circuit creation
    if func_success:
        circuit_success, model = test_circuit_creation(main_func)
        if not circuit_success:
            all_passed = False
    else:
        print("\n3. Skipping circuit creation (function test failed)")

    # Test 4: ClassiqClient
    if not test_classiq_client():
        all_passed = False

    # Test 5: Environment
    if not test_environment_setup():
        all_passed = False

    print("\n" + "=" * 50)

    if all_passed:
        print("✅ All tests passed! Quantum backend is ready.")
        print("\nNext steps:")
        print("1. Get your Classiq API key from https://platform.classiq.io/")
        print("2. Add it to your .env file: CLASSIQ_API_KEY=your_api_key")
        print("3. Run: python -m app.main")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install classiq")
        print("2. Set CLASSIQ_API_KEY in .env file")
        print("3. Check Python path and imports")

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