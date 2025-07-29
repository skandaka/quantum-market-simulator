# Path: /Users/skandaa/Desktop/quantum-market-simulator/backend/examples/quantum_circuit_example.py

"""
Quantum Circuit Example with Correct Syntax
This demonstrates the proper way to create quantum circuits with Classiq
"""

from classiq import *


# ❌ INCORRECT: Main function cannot have quantum inputs
# @qfunc
# def main(q: QArray[QBit]):  # This causes the error!
#     hadamard(q[0])

# ✅ CORRECT: Main function should only have outputs and allocate qubits internally
@qfunc
def create_bell_state(q: QArray[QBit]):
    """Create a Bell state using two qubits"""
    hadamard(q[0])  # Put first qubit in superposition
    cnot(q[0], q[1])  # Entangle with second qubit


@qfunc
def main(q: Output[QArray[QBit]]):
    """
    Main quantum function - MUST only have Output parameters
    All qubits must be allocated inside the function
    """
    allocate(2, q)  # Allocate 2 qubits
    create_bell_state(q)  # Apply quantum operations


# Example of a more complex quantum circuit
@qfunc
def quantum_fourier_transform_2_qubits(q: QArray[QBit]):
    """2-qubit Quantum Fourier Transform"""
    hadamard(q[0])
    cphase(np.pi / 2, q[1], q[0])
    hadamard(q[1])
    swap(q[0], q[1])


@qfunc
def advanced_main(q: Output[QArray[QBit]]):
    """Main function for QFT example"""
    allocate(2, q)
    # Initialize some state
    hadamard(q[0])
    cnot(q[0], q[1])
    # Apply QFT
    quantum_fourier_transform_2_qubits(q)


def test_quantum_circuits():
    """Test the quantum circuits"""
    print("🧪 Testing Quantum Circuits")
    print("=" * 30)

    try:
        # Test simple Bell state
        print("1. Testing Bell state circuit...")
        model1 = create_model(main)
        qmod1 = write_qmod(model1)
        print(f"   ✅ Bell state circuit created (QMOD: {len(qmod1)} chars)")

        # Test advanced circuit
        print("2. Testing QFT circuit...")
        model2 = create_model(advanced_main)
        qmod2 = write_qmod(model2)
        print(f"   ✅ QFT circuit created (QMOD: {len(qmod2)} chars)")

        print("\n🎉 All quantum circuits created successfully!")
        return True

    except Exception as e:
        print(f"   ❌ Circuit creation failed: {e}")
        if "API key" in str(e).lower():
            print("   💡 Note: This error might be due to missing API key")
            print("   💡 The circuit syntax is correct - authentication needed for full testing")
        return False


if __name__ == "__main__":
    test_quantum_circuits()