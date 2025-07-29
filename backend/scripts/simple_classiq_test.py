#!/usr/bin/env python3
"""Simple test to see what works with current Classiq version"""

import classiq

print("Testing basic Classiq functionality...")
print("=" * 60)

# Test 1: Basic imports
try:
    from classiq import qfunc, QBit, H, create_model

    print("✅ Basic imports work")


    # Test 2: Create a simple quantum function
    @qfunc
    def simple_circuit(q: QBit):
        H(q)


    print("✅ @qfunc decorator works")

    # Test 3: Create a model
    model = create_model(simple_circuit)
    print("✅ create_model works")
    print(f"   Model type: {type(model)}")

    # Test 4: Synthesize
    from classiq import synthesize

    qprog = synthesize(model)
    print("✅ synthesize works")
    print(f"   Quantum program type: {type(qprog)}")

except Exception as e:
    print(f"❌ Error in basic test: {e}")
    import traceback

    traceback.print_exc()

# Test what types are available
print("\nChecking available types:")
print("=" * 60)

type_checks = [
    ('Model', lambda: type(create_model(lambda q: H(q)))),
    ('QuantumProgram', lambda: type(synthesize(create_model(lambda q: H(q))))),
]

for name, getter in type_checks:
    try:
        t = getter()
        print(f"✅ {name} type: {t}")
    except Exception as e:
        print(f"❌ {name}: {e}")