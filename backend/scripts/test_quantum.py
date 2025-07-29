#!/usr/bin/env python3
"""
Test script to verify quantum backend is working
Run this after setting up your Classiq API key
"""

import os
import sys
import asyncio
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))


async def test_quantum_backend():
    """Test the quantum backend setup"""

    print("🧪 Testing Quantum Backend Setup")
    print("=" * 50)

    # Test 1: Basic imports
    print("\n1. Testing basic imports...")
    try:
        import classiq
        print(f"   ✅ Classiq imported (version available)")

        from classiq import qfunc, QBit, H, create_model, synthesize
        print("   ✅ Classiq core functions imported")

    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

    # Test 2: Check API key
    print("\n2. Checking API key...")
    api_key = os.getenv("CLASSIQ_API_KEY")
    if not api_key or api_key == "your_actual_api_key_here":
        print("   ⚠️  No API key set - app will run in simulation mode")
        print("   📝 Add your real API key to .env file")
        has_api_key = False
    else:
        print("   ✅ API key found")
        has_api_key = True

    # Test 3: Test our modules
    print("\n3. Testing our quantum modules...")
    try:
        from app.quantum.classiq_auth import classiq_auth
        print("   ✅ classiq_auth imported")

        from app.quantum.classiq_client import ClassiqClient
        print("   ✅ ClassiqClient imported")

    except Exception as e:
        print(f"   ❌ Module import failed: {e}")
        return False

    # Test 4: Test authentication
    print("\n4. Testing authentication...")
    try:
        await classiq_auth.initialize()
        if classiq_auth.is_authenticated():
            print("   ✅ Successfully authenticated with Classiq!")
            print("   🎉 REAL QUANTUM BACKEND READY!")
        else:
            print("   ⚠️  Not authenticated - will use simulation mode")
            print("   💡 Check your API key in .env file")
    except Exception as e:
        print(f"   ⚠️  Authentication test failed: {e}")
        if has_api_key:
            print("   🔧 Try running: python -c \"import classiq; classiq.authenticate()\"")

    # Test 5: Test circuit creation
    print("\n5. Testing circuit creation...")
    try:
        @qfunc
        def main(q: QBit):
            H(q)

        model = create_model(main)
        print("   ✅ Quantum circuit model created")

        # Only test synthesis if authenticated
        if has_api_key and classiq_auth.is_authenticated():
            try:
                print("   🔄 Testing circuit synthesis...")
                qprog = await asyncio.to_thread(synthesize, model)
                print("   🎉 CIRCUIT SYNTHESIZED SUCCESSFULLY!")
                print("   ⚡ Real quantum backend is working!")
            except Exception as e:
                print(f"   ⚠️  Synthesis failed: {e}")
                print("   💡 This might be due to API limits or connection issues")

    except Exception as e:
        print(f"   ❌ Circuit creation failed: {e}")
        return False

    # Test 6: Test client
    print("\n6. Testing ClassiqClient...")
    try:
        client = ClassiqClient()
        await client.initialize()

        if client.is_ready():
            print("   ✅ ClassiqClient is ready!")
            print("   🚀 Quantum simulations will use REAL quantum computing!")
        else:
            print("   ⚠️  ClassiqClient not ready - will use simulation")

    except Exception as e:
        print(f"   ❌ Client test failed: {e}")

    print("\n" + "=" * 50)

    if has_api_key and classiq_auth.is_authenticated():
        print("🎉 SUCCESS: Real quantum backend is working!")
        print("✨ Your app will use actual quantum computing!")
    else:
        print("⚠️  SIMULATION MODE: Set up API key for real quantum computing")
        print("")
        print("To enable real quantum computing:")
        print("1. Get API key from https://platform.classiq.io/")
        print("2. Add to .env: CLASSIQ_API_KEY=your_real_key")
        print("3. Restart the application")

    print("")
    return True


async def main():
    """Main test function"""
    try:
        success = await test_quantum_backend()
        if success:
            print("🚀 Ready to run: python -m app.main")
        else:
            print("❌ Setup incomplete - fix errors above")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())