#!/usr/bin/env python3
"""Check what's available in the Classiq API"""

import classiq

print("Classiq version:", classiq.__version__)
print("\nAvailable in classiq module:")
print("=" * 60)

# Get all attributes
attrs = dir(classiq)

# Filter and categorize
functions = []
classes = []
modules = []
other = []

for attr in attrs:
    if not attr.startswith('_'):
        obj = getattr(classiq, attr)
        if callable(obj):
            if hasattr(obj, '__module__') and 'classiq' in obj.__module__:
                if obj.__name__[0].isupper():
                    classes.append(attr)
                else:
                    functions.append(attr)
            else:
                functions.append(attr)
        elif hasattr(obj, '__module__'):
            modules.append(attr)
        else:
            other.append(attr)

print("\nFunctions:")
for f in sorted(functions):
    print(f"  - {f}")

print("\nClasses:")
for c in sorted(classes):
    print(f"  - {c}")

print("\nModules:")
for m in sorted(modules):
    print(f"  - {m}")

print("\nChecking specific imports:")
print("=" * 60)

# Check specific imports we need
needed_imports = [
    'qfunc', 'QBit', 'QArray', 'Output', 'create_model',
    'synthesize', 'execute', 'H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ',
    'CX', 'CZ', 'allocate', 'control', 'Model'
]

for imp in needed_imports:
    try:
        obj = getattr(classiq, imp, None)
        if obj is not None:
            print(f"✅ {imp} - available")
        else:
            print(f"❌ {imp} - NOT found")
    except Exception as e:
        print(f"❌ {imp} - error: {e}")

# Try alternative imports
print("\nTrying alternative import paths:")
print("=" * 60)

try:
    from classiq.interface.primitives.qfunc import qfunc
    print("✅ qfunc found via classiq.interface.primitives.qfunc")
except:
    print("❌ classiq.interface.primitives.qfunc not available")

try:
    from classiq.interface.generator.model import Model
    print("✅ Model found via classiq.interface.generator.model")
except:
    print("❌ classiq.interface.generator.model not available")

try:
    from classiq.model import Model
    print("✅ Model found via classiq.model")
except:
    print("❌ classiq.model not available")