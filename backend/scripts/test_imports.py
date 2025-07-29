#!/usr/bin/env python3
"""Test script to verify all imports are working correctly"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

print("Testing imports...")
print("=" * 60)

# Test core imports
try:
    from app.quantum.classiq_auth import classiq_auth

    print("✅ app.quantum.classiq_auth imported successfully")
except Exception as e:
    print(f"❌ Failed to import app.quantum.classiq_auth: {e}")

try:
    from app.quantum.classiq_client import ClassiqClient

    print("✅ app.quantum.classiq_client imported successfully")
except Exception as e:
    print(f"❌ Failed to import app.quantum.classiq_client: {e}")

try:
    from app.quantum.qnlp_model import QuantumNLPModel

    print("✅ app.quantum.qnlp_model imported successfully")
except Exception as e:
    print(f"❌ Failed to import app.quantum.qnlp_model: {e}")

try:
    from app.quantum.quantum_finance import QuantumFinanceAlgorithms

    print("✅ app.quantum.quantum_finance imported successfully")
except Exception as e:
    print(f"❌ Failed to import app.quantum.quantum_finance: {e}")

try:
    from app.quantum.quantum_simulator import QuantumSimulator

    print("✅ app.quantum.quantum_simulator imported successfully")
except Exception as e:
    print(f"❌ Failed to import app.quantum.quantum_simulator: {e}")

# Test Classiq availability
try:
    import classiq

    print("✅ Classiq is installed")

    # Test specific imports
    from classiq import qfunc, QBit, H, create_model

    print("✅ Classiq core functions imported successfully")
except ImportError:
    print("⚠️  Classiq is not installed - quantum features will be simulated")
    print("   To install: pip install classiq")
except Exception as e:
    print(f"❌ Classiq import error: {e}")

print("=" * 60)
print("Import test complete!")