"""Real Classiq platform integration client"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np
from datetime import datetime

from classiq import (
    synthesize, execute, Model, Output, QBit, QArray,
    allocate, apply_to_all, control, H, RY, RZ, X, Z,
    CX, CCX, suzuki_trotter, QFunc, create_model,
    show, VQE, optimize, Optimizer, EstimatorGradient
)
from classiq.interface.executor.execution_details import ExecutionDetails
from classiq.interface.backend.result import QuantumComputationResult
from classiq.synthesis import set_constraints, set_preferences

from app.quantum.classiq_auth import classiq_auth
from app.config import settings

logger = logging.getLogger(__name__)


class ClassiqClient:
    """Client for real Classiq quantum platform integration"""

    def __init__(self):
        self.auth_manager = classiq_auth
        self._models_cache = {}
        self._last_execution_details = None

    async def initialize(self):
        """Initialize Classiq connection"""
        await self.auth_manager.initialize()

    def is_ready(self) -> bool:
        """Check if client is ready for quantum operations"""
        return self.auth_manager.is_authenticated()

    async def create_amplitude_estimation_circuit(
        self,
        probabilities: List[float],
        num_qubits: int = 5
    ) -> Model:
        """Create quantum amplitude estimation circuit for market prediction"""

        @QFunc
        def oracle(target: QBit):
            """Oracle marking states based on probability distribution"""
            # This is a simplified oracle - in practice, would encode
            # the probability distribution more sophisticatedly
            X(target)

        @QFunc
        def grover_operator(qubits: QArray[QBit], oracle_workspace: QBit):
            """Grover operator for amplitude amplification"""
            oracle(oracle_workspace)
            apply_to_all(H, qubits)
            apply_to_all(Z, qubits)
            control(ctrl=qubits, operand=lambda: Z(oracle_workspace))
            apply_to_all(H, qubits)

        @QFunc
        def amplitude_estimation_circuit(
            qubits: QArray[QBit, num_qubits],
            oracle_workspace: QBit,
            measurement: Output[QArray[QBit, num_qubits]]
        ):
            """Main amplitude estimation circuit"""
            # Initialize superposition
            apply_to_all(H, qubits)

            # Apply Grover operator multiple times
            num_iterations = int(np.pi * np.sqrt(2**num_qubits) / 4)
            for _ in range(min(num_iterations, 10)):  # Limit iterations
                grover_operator(qubits, oracle_workspace)

            measurement |= qubits

        # Create and synthesize model
        model = create_model(amplitude_estimation_circuit)

        # Set constraints for Classiq synthesis
        constraints = {
            "max_width": min(num_qubits + 5, self.auth_manager.config.max_qubits),
            "max_depth": 1000,
            "optimization_level": self.auth_manager.config.optimization_level
        }

        model = set_constraints(model, **constraints)

        # Cache the model
        cache_key = f"amplitude_est_{num_qubits}_{hash(tuple(probabilities))}"
        self._models_cache[cache_key] = model

        return model

    async def create_vqe_circuit(
        self,
        num_qubits: int,
        num_layers: int = 3
    ) -> Model:
        """Create Variational Quantum Eigensolver circuit for optimization"""

        @QFunc
        def vqe_ansatz(
            qubits: QArray[QBit, num_qubits],
            parameters: QArray[float, num_layers * num_qubits * 2]
        ):
            """VQE ansatz with parameterized gates"""
            param_idx = 0

            for layer in range(num_layers):
                # Rotation layer
                for i in range(num_qubits):
                    RY(parameters[param_idx], qubits[i])
                    param_idx += 1
                    RZ(parameters[param_idx], qubits[i])
                    param_idx += 1

                # Entanglement layer
                for i in range(num_qubits - 1):
                    CX(qubits[i], qubits[i + 1])

                # Circular entanglement
                if num_qubits > 2:
                    CX(qubits[num_qubits - 1], qubits[0])

        # Create model
        model = create_model(vqe_ansatz)

        # Set optimization preferences
        preferences = {
            "backend_preferences": self.auth_manager.get_backend_preferences()
        }
        model = set_preferences(model, **preferences)

        return model

    async def create_quantum_ml_circuit(
        self,
        feature_vector: np.ndarray,
        num_classes: int = 5
    ) -> Model:
        """Create quantum machine learning circuit for sentiment classification"""

        num_features = len(feature_vector)
        num_qubits = int(np.ceil(np.log2(max(num_features, num_classes))))

        @QFunc
        def encode_features(
            qubits: QArray[QBit, num_qubits],
            features: QArray[float, num_features]
        ):
            """Encode classical features into quantum state"""
            # Amplitude encoding
            apply_to_all(H, qubits)

            # Encode features as rotation angles
            for i in range(min(num_features, num_qubits)):
                RY(features[i] * np.pi, qubits[i])

        @QFunc
        def variational_classifier(
            qubits: QArray[QBit, num_qubits],
            params: QArray[float, num_qubits * 4],
            features: QArray[float, num_features]
        ):
            """Variational quantum classifier"""
            # Encode features
            encode_features(qubits, features)

            # Variational layers
            for i in range(num_qubits):
                RY(params[i * 2], qubits[i])
                RZ(params[i * 2 + 1], qubits[i])

            # Entanglement
            for i in range(num_qubits - 1):
                CX(qubits[i], qubits[i + 1])

            # Final rotation layer
            for i in range(num_qubits):
                RY(params[num_qubits * 2 + i * 2], qubits[i])
                RZ(params[num_qubits * 2 + i * 2 + 1], qubits[i])

        model = create_model(variational_classifier)
        return model

    async def execute_circuit(
        self,
        model: Model,
        parameters: Optional[Dict[str, Any]] = None
    ) -> QuantumComputationResult:
        """Execute a quantum circuit on Classiq platform"""

        if not self.is_ready():
            raise RuntimeError("Classiq client not authenticated")

        try:
            # Synthesize the quantum program
            quantum_program = synthesize(model)

            # Show circuit if in debug mode
            if settings.debug:
                try:
                    show(quantum_program)
                except Exception as e:
                    logger.debug(f"Could not visualize circuit: {e}")

            # Execute with preferences
            execution_prefs = self.auth_manager.get_execution_preferences()

            # Run execution
            job = execute(
                quantum_program,
                execution_preferences=execution_prefs
            )

            # Wait for results
            results = job.result()

            # Store execution details
            self._last_execution_details = job.details()

            return results

        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            raise

    async def optimize_variational_circuit(
        self,
        model: Model,
        initial_params: np.ndarray,
        cost_function: callable,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """Optimize parameters for variational quantum circuits"""

        optimizer_config = {
            "optimizer": Optimizer.COBYLA,
            "max_iterations": max_iterations,
            "tolerance": 1e-6,
            "gradient_method": EstimatorGradient.FINITE_DIFF
        }

        # Create VQE instance
        vqe = VQE(
            model=model,
            optimizer_config=optimizer_config,
            cost_function=cost_function,
            initial_parameters=initial_params
        )

        # Run optimization
        result = optimize(vqe)

        return {
            "optimal_parameters": result.optimal_parameters,
            "optimal_value": result.optimal_value,
            "iterations": result.iterations,
            "convergence": result.converged
        }

    def get_last_execution_metrics(self) -> Dict[str, Any]:
        """Get metrics from the last execution"""

        if not self._last_execution_details:
            return {}

        details = self._last_execution_details

        return {
            "circuit_depth": details.get("depth", 0),
            "num_qubits": details.get("width", 0),
            "gate_count": details.get("gate_count", 0),
            "execution_time_ms": details.get("execution_time", 0),
            "queue_time_ms": details.get("queue_time", 0),
            "synthesis_time_ms": details.get("synthesis_time", 0),
            "backend_name": details.get("backend_name", "unknown"),
            "success_rate": details.get("success_rate", 1.0)
        }

    async def estimate_resources(
        self,
        model: Model
    ) -> Dict[str, Any]:
        """Estimate quantum resources needed for a circuit"""

        try:
            # Synthesize to get resource estimates
            quantum_program = synthesize(model)

            # Extract circuit properties
            circuit_properties = quantum_program.get_circuit_properties()

            return {
                "num_qubits": circuit_properties.width,
                "circuit_depth": circuit_properties.depth,
                "gate_count": circuit_properties.gate_count,
                "multi_qubit_gates": circuit_properties.multi_qubit_gate_count,
                "estimated_time_ms": circuit_properties.depth * 0.1,  # Rough estimate
                "feasible": circuit_properties.width <= self.auth_manager.config.max_qubits
            }

        except Exception as e:
            logger.error(f"Resource estimation failed: {e}")
            return {
                "error": str(e),
                "feasible": False
            }

    async def get_backend_status(self) -> Dict[str, Any]:
        """Get current backend status and queue information"""

        try:
            from classiq import get_backend_status

            backend_prefs = self.auth_manager.get_backend_preferences()
            status = get_backend_status(backend_prefs)

            return {
                "backend": backend_prefs.backend_name or "classiq_simulator",
                "status": "operational" if status.is_available else "offline",
                "queue_length": status.pending_jobs,
                "average_wait_time_seconds": status.average_queue_time,
                "available": status.is_available,
                "max_qubits": status.max_qubits,
                "provider": backend_prefs.backend_service_provider.name
            }

        except Exception as e:
            logger.error(f"Could not get backend status: {e}")
            return {
                "backend": "unknown",
                "status": "error",
                "error": str(e)
            }