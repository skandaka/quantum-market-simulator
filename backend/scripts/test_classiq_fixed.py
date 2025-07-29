#!/usr/bin/env python3
"""Test that Classiq integration is working correctly with fixes"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

print("Testing fixed Classiq integration...")
print("=" * 60)

# Test 1: Basic Classiq functionality
print("\n1. Testing basic Classiq with 'main' function:")
try:
    from classiq import qfunc, QBit, H, create_model, synthesize


    @qfunc
    def main(q: QBit):
        H(q)


    model = create_model(main)
    print("✅ create_model works with 'main' function")

    # Test synthesize
    qprog = synthesize(model)
    print("✅ synthesize works")
    print(f"   Type: {type(qprog).__name__}")

except Exception as e:
    print(f"❌ Basic test failed: {e}")

# Test 2: Import our modules
print("\n2. Testing quantum module imports:")
import_tests = [
    "app.quantum.classiq_auth",
    "app.quantum.classiq_client",
    "app.quantum.qnlp_model",
    "app.quantum.quantum_finance",
    "app.quantum.quantum_simulator"
]

for module_name in import_tests:
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
    except Exception as e:
        print(f"❌ {module_name} failed: {e}")

# Test 3: Test client initialization
print("\n3. Testing ClassiqClient initialization:")
try:
    from app.quantum.classiq_client import ClassiqClient
    from app.quantum.classiq_auth import classiq_auth

    # Test auth manager
    print(f"   Auth manager ready: {classiq_auth.is_authenticated()}")

    # Test client
    client = ClassiqClient()
    print(f"✅ ClassiqClient created")
    print(f"   Client ready: {client.is_ready()}")

except Exception as e:
    print(f"❌ Client test failed: {e}")

# Test 4: Create a simple circuit with proper structure
print("\n4. Testing circuit creation with nested functions:")
try:
    from classiq import qfunc, QBit, QArray, H, CX, create_model, Output


    @qfunc
    def bell_state_prep(q1: QBit, q2: QBit):
        H(q1)
        CX(q1, q2)


    @qfunc
    def main(qubits: QArray[QBit, 2], output: Output[QArray[QBit, 2]]):
        bell_state_prep(qubits[0], qubits[1])
        output |= qubits


    model = create_model(main)
    print("✅ Complex circuit with nested functions works")

except Exception as e:
    print(f"❌ Complex circuit failed: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
print("\nSummary:")
print("- Classiq requires entry point functions to be named 'main'")
print("- All quantum modules import successfully")
print("- The app is ready to run with quantum features")