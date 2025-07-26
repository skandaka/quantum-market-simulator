"""Quantum circuit construction utilities"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QuantumGate:
    """Representation of a quantum gate"""
    name: str
    qubits: List[int]
    parameters: Optional[List[float]] = None


class CircuitBuilder:
    """Helper class for building quantum circuits"""

    def __init__(self):
        self.gates = []
        self.num_qubits = 0

    def initialize_qubits(self, n: int):
        """Initialize number of qubits"""
        self.num_qubits = n
        self.gates = []

    def add_hadamard(self, qubit: int):
        """Add Hadamard gate"""
        self.gates.append(QuantumGate("H", [qubit]))

    def add_cnot(self, control: int, target: int):
        """Add CNOT gate"""
        self.gates.append(QuantumGate("CNOT", [control, target]))

    def add_rotation(self, gate_type: str, qubit: int, angle: float):
        """Add rotation gate (RX, RY, RZ)"""
        self.gates.append(QuantumGate(gate_type, [qubit], [angle]))

    def add_controlled_rotation(
            self,
            gate_type: str,
            control: int,
            target: int,
            angle: float
    ):
        """Add controlled rotation gate"""
        self.gates.append(
            QuantumGate(f"C{gate_type}", [control, target], [angle])
        )

    def add_state_preparation(
            self,
            amplitudes: np.ndarray,
            qubits: Optional[List[int]] = None
    ):
        """Add state preparation"""
        if qubits is None:
            qubits = list(range(self.num_qubits))

        # Validate amplitudes
        expected_dim = 2 ** len(qubits)
        if len(amplitudes) != expected_dim:
            raise ValueError(
                f"Amplitude vector size {len(amplitudes)} doesn't match "
                f"qubit count {len(qubits)} (expected {expected_dim})"
            )

        # Add state prep gate
        self.gates.append(
            QuantumGate(
                "StatePreparation",
                qubits,
                amplitudes.tolist()
            )
        )

    def add_variational_layer(
            self,
            parameters: np.ndarray,
            entanglement: str = "linear"
    ):
        """Add parameterized variational layer"""

        # Single qubit rotations
        for i in range(self.num_qubits):
            self.add_rotation("RY", i, parameters[i, 0])
            self.add_rotation("RZ", i, parameters[i, 1])

        # Entanglement
        if entanglement == "linear":
            for i in range(self.num_qubits - 1):
                self.add_cnot(i, i + 1)
        elif entanglement == "circular":
            for i in range(self.num_qubits):
                self.add_cnot(i, (i + 1) % self.num_qubits)
        elif entanglement == "all-to-all":
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    self.add_cnot(i, j)

        # Second rotation layer
        for i in range(self.num_qubits):
            self.add_rotation("RY", i, parameters[i, 2])

    def add_quantum_fourier_transform(self, qubits: Optional[List[int]] = None):
        """Add Quantum Fourier Transform"""
        if qubits is None:
            qubits = list(range(self.num_qubits))

        n = len(qubits)

        for j in range(n):
            # Hadamard on qubit j
            self.add_hadamard(qubits[j])

            # Controlled rotations
            for k in range(j + 1, n):
                angle = np.pi / (2 ** (k - j))
                self.add_controlled_rotation(
                    "RZ", qubits[k], qubits[j], angle
                )

        # Swap qubits
        for i in range(n // 2):
            self.gates.append(
                QuantumGate("SWAP", [qubits[i], qubits[n - i - 1]])
            )

    def add_amplitude_estimation(
            self,
            oracle_gates: List[QuantumGate],
            num_iterations: int
    ):
        """Add amplitude estimation circuit"""

        # Hadamard on all qubits
        for i in range(self.num_qubits):
            self.add_hadamard(i)

        # Grover iterations
        for _ in range(num_iterations):
            # Oracle
            self.gates.extend(oracle_gates)

            # Diffusion operator
            for i in range(self.num_qubits):
                self.add_hadamard(i)
            for i in range(self.num_qubits):
                self.add_rotation("RZ", i, np.pi)
            for i in range(self.num_qubits):
                self.add_hadamard(i)

        # QFT for phase estimation
        self.add_quantum_fourier_transform()

    def to_classiq_format(self) -> Dict[str, Any]:
        """Convert to Classiq circuit format"""

        layers = []

        # Group gates into layers
        current_layer = []
        used_qubits = set()

        for gate in self.gates:
            # Check if gate can be added to current layer
            gate_qubits = set(gate.qubits)

            if gate_qubits.intersection(used_qubits):
                # Conflict - start new layer
                if current_layer:
                    layers.append(self._format_layer(current_layer))
                current_layer = [gate]
                used_qubits = gate_qubits
            else:
                # Add to current layer
                current_layer.append(gate)
                used_qubits.update(gate_qubits)

        # Add final layer
        if current_layer:
            layers.append(self._format_layer(current_layer))

        return {
            "num_qubits": self.num_qubits,
            "layers": layers
        }

    def _format_layer(self, gates: List[QuantumGate]) -> Dict[str, Any]:
        """Format a layer of gates for Classiq"""

        formatted_gates = []

        for gate in gates:
            gate_dict = {
                "gate": gate.name,
                "qubits": gate.qubits
            }

            if gate.parameters:
                gate_dict["parameters"] = gate.parameters

            formatted_gates.append(gate_dict)

        return {
            "gates": formatted_gates,
            "parallel": True  # Gates in same layer can run in parallel
        }

    def get_circuit_depth(self) -> int:
        """Calculate circuit depth"""

        depth = 0
        current_depth_qubits = set()

        for gate in self.gates:
            gate_qubits = set(gate.qubits)

            if gate_qubits.intersection(current_depth_qubits):
                # New layer needed
                depth += 1
                current_depth_qubits = gate_qubits
            else:
                current_depth_qubits.update(gate_qubits)

        return depth + 1 if current_depth_qubits else 0

    def optimize_circuit(self):
        """Basic circuit optimization"""

        # Remove consecutive inverse operations
        optimized = []
        i = 0

        while i < len(self.gates):
            if i + 1 < len(self.gates):
                gate1 = self.gates[i]
                gate2 = self.gates[i + 1]

                # Check for inverse pairs
                if self._are_inverse_gates(gate1, gate2):
                    # Skip both gates
                    i += 2
                    continue

            optimized.append(self.gates[i])
            i += 1

        self.gates = optimized

    def _are_inverse_gates(self, gate1: QuantumGate, gate2: QuantumGate) -> bool:
        """Check if two gates are inverses"""

        # Same qubits
        if gate1.qubits != gate2.qubits:
            return False

        # Hadamard is self-inverse
        if gate1.name == "H" and gate2.name == "H":
            return True

        # CNOT is self-inverse
        if gate1.name == "CNOT" and gate2.name == "CNOT":
            return True

        # Check rotation gates
        rotation_gates = ["RX", "RY", "RZ"]
        if gate1.name in rotation_gates and gate2.name == gate1.name:
            if gate1.parameters and gate2.parameters:
                # Check if angles sum to 0 (mod 2π)
                angle_sum = gate1.parameters[0] + gate2.parameters[0]
                if abs(angle_sum % (2 * np.pi)) < 1e-10:
                    return True

        return False