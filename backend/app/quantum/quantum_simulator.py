"""Quantum market simulation engine"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import norm
import asyncio
from dataclasses import dataclass

from app.quantum.classiq_client import ClassiqClient
from app.quantum.circuit_builder import CircuitBuilder
from app.models.schemas import PriceScenario, MarketPrediction
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QuantumMarketState:
    """Quantum state representation of market conditions"""
    amplitude_vector: np.ndarray
    phase_vector: np.ndarray
    entanglement_map: Dict[str, float]
    coherence_time: float


class QuantumSimulator:
    """Quantum-enhanced market simulation"""

    def __init__(self, classiq_client: ClassiqClient):
        self.classiq_client = classiq_client
        self.circuit_builder = CircuitBuilder()

        # Simulation parameters
        self.num_qubits = 10
        self.num_scenarios = 1000
        self.decoherence_rate = 0.01

    async def simulate_market_scenarios(
            self,
            initial_state: Dict[str, Any],
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int = 1000
    ) -> List[PriceScenario]:
        """Generate quantum probability distribution of market scenarios"""

        # Encode market state into quantum state
        quantum_state = self._encode_market_state(initial_state, sentiment_impact)

        # Build quantum evolution circuit
        circuit_config = self._build_evolution_circuit(
            quantum_state,
            time_horizon
        )

        try:
            # Create and execute circuit
            circuit_id = await self.classiq_client.create_circuit(circuit_config)

            # Run multiple shots for scenario generation
            result = await self.classiq_client.execute_circuit(
                circuit_id,
                backend=settings.quantum_backend,
                shots=num_scenarios * 2  # Extra shots for better statistics
            )

            # Generate scenarios from quantum results
            scenarios = self._generate_scenarios_from_quantum(
                result,
                initial_state,
                time_horizon,
                num_scenarios
            )

            return scenarios

        except Exception as e:
            logger.error(f"Quantum simulation failed: {e}")
            # Fallback to quantum-inspired classical simulation
            return self._classical_quantum_inspired_scenarios(
                initial_state,
                sentiment_impact,
                time_horizon,
                num_scenarios
            )

    def _encode_market_state(
            self,
            initial_state: Dict[str, Any],
            sentiment_impact: float
    ) -> QuantumMarketState:
        """Encode market conditions into quantum state"""

        # Extract market features
        price = initial_state.get("price", 100)
        volatility = initial_state.get("volatility", 0.2)
        volume = initial_state.get("volume", 1000000)
        trend = initial_state.get("trend", 0)

        # Create amplitude encoding
        # Map market features to quantum amplitudes
        dimension = 2 ** self.num_qubits
        amplitudes = np.zeros(dimension)

        # Gaussian-like distribution centered on current state
        center = int(dimension * (price / (price + 100)))  # Normalize price
        width = int(dimension * volatility / 2)

        for i in range(dimension):
            distance = abs(i - center)
            amplitudes[i] = np.exp(-distance ** 2 / (2 * width ** 2))

        # Apply sentiment impact
        if sentiment_impact > 0:
            # Positive sentiment shifts distribution right
            amplitudes = np.roll(amplitudes, int(sentiment_impact * dimension / 10))
        else:
            # Negative sentiment shifts left
            amplitudes = np.roll(amplitudes, int(sentiment_impact * dimension / 10))

        # Normalize
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

        # Create phase encoding based on market dynamics
        phases = np.zeros(dimension)
        for i in range(dimension):
            # Phase encodes momentum and volatility
            phases[i] = (trend * np.pi / 2) + (volatility * np.sin(i * np.pi / dimension))

        # Create entanglement map (correlations)
        entanglement_map = {
            "price_volume": min(abs(np.corrcoef([price], [volume])[0, 1]), 1.0),
            "price_volatility": volatility / 0.5,  # Normalized
            "sentiment_strength": abs(sentiment_impact)
        }

        return QuantumMarketState(
            amplitude_vector=amplitudes,
            phase_vector=phases,
            entanglement_map=entanglement_map,
            coherence_time=1.0 / (1.0 + volatility)  # Higher volatility = faster decoherence
        )

    def _build_evolution_circuit(
            self,
            quantum_state: QuantumMarketState,
            time_steps: int
    ) -> Dict[str, Any]:
        """Build quantum circuit for market evolution"""

        return {
            "name": "quantum_market_evolution",
            "quantum_circuit": {
                "num_qubits": self.num_qubits,
                "layers": [
                    # Initial state preparation
                    {
                        "type": "state_preparation",
                        "amplitudes": quantum_state.amplitude_vector.tolist(),
                        "phases": quantum_state.phase_vector.tolist()
                    },
                    # Time evolution layers
                    *[
                        {
                            "type": "time_evolution",
                            "hamiltonian": self._market_hamiltonian(
                                quantum_state,
                                t / time_steps
                            ),
                            "time": 1.0 / time_steps,
                            "trotter_steps": 4
                        }
                        for t in range(time_steps)
                    ],
                    # Decoherence modeling
                    {
                        "type": "noise_channel",
                        "noise_model": "amplitude_damping",
                        "rate": self.decoherence_rate * time_steps
                    },
                    # Measurement
                    {
                        "type": "measurement",
                        "qubits": list(range(self.num_qubits)),
                        "basis": "computational"
                    }
                ]
            },
            "optimization_level": 1,
            "backend_properties": {
                "noise_aware": True,
                "optimization_goal": "fidelity"
            }
        }

    def _market_hamiltonian(
            self,
            quantum_state: QuantumMarketState,
            time_fraction: float
    ) -> List[Dict[str, Any]]:
        """Generate market Hamiltonian for quantum evolution"""

        # Hamiltonian terms representing market forces
        terms = []

        # Drift term (market trend)
        terms.append({
            "operator": "drift",
            "coefficient": 0.1 * (1 - time_fraction),  # Decreasing influence
            "qubits": list(range(self.num_qubits))
        })

        # Volatility term (random fluctuations)
        terms.append({
            "operator": "volatility",
            "coefficient": quantum_state.entanglement_map["price_volatility"],
            "qubits": list(range(self.num_qubits))
        })

        # Correlation terms (entanglement)
        for i in range(self.num_qubits - 1):
            terms.append({
                "operator": "correlation",
                "coefficient": quantum_state.entanglement_map.get(
                    "price_volume", 0.5
                ) * 0.1,
                "qubits": [i, i + 1]
            })

        return terms

    def _generate_scenarios_from_quantum(
            self,
            quantum_result: Dict[str, Any],
            initial_state: Dict[str, Any],
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Generate price scenarios from quantum measurement results"""

        counts = quantum_result.get("counts", {})
        total_shots = sum(counts.values())

        if total_shots == 0:
            # Fallback if no results
            return self._generate_classical_scenarios(
                initial_state, time_horizon, num_scenarios
            )

        scenarios = []
        scenario_id = 0

        # Convert measurement outcomes to price paths
        for bitstring, count in counts.items():
            if scenario_id >= num_scenarios:
                break

            # Probability of this outcome
            probability = count / total_shots

            # Convert bitstring to price path
            price_path = self._bitstring_to_price_path(
                bitstring,
                initial_state["price"],
                time_horizon
            )

            # Calculate returns and volatility
            returns_path = np.diff(price_path) / price_path[:-1]
            volatility_path = self._calculate_rolling_volatility(returns_path)

            scenarios.append(PriceScenario(
                scenario_id=scenario_id,
                price_path=price_path.tolist(),
                returns_path=returns_path.tolist(),
                volatility_path=volatility_path,
                probability_weight=probability
            ))

            scenario_id += 1

        # Ensure we have enough scenarios
        while len(scenarios) < num_scenarios:
            # Duplicate and perturb existing scenarios
            base_scenario = scenarios[len(scenarios) % len(scenarios)]
            perturbed = self._perturb_scenario(base_scenario)
            perturbed.scenario_id = len(scenarios)
            scenarios.append(perturbed)

        return scenarios[:num_scenarios]

    def _bitstring_to_price_path(
            self,
            bitstring: str,
            initial_price: float,
            time_steps: int
    ) -> np.ndarray:
        """Convert quantum measurement bitstring to price path"""

        # Interpret bitstring as sequence of price movements
        price_path = np.zeros(time_steps + 1)
        price_path[0] = initial_price

        # Use bits to determine up/down movements
        bits_per_step = max(1, len(bitstring) // time_steps)

        for t in range(time_steps):
            # Extract bits for this time step
            start_idx = t * bits_per_step
            end_idx = min((t + 1) * bits_per_step, len(bitstring))
            step_bits = bitstring[start_idx:end_idx]

            if step_bits:
                # Count 1s vs 0s to determine direction
                ones = step_bits.count('1')
                zeros = step_bits.count('0')

                # Calculate magnitude based on bit pattern
                magnitude = abs(ones - zeros) / len(step_bits)
                direction = 1 if ones > zeros else -1

                # Apply price change with some randomness
                change = direction * magnitude * 0.02  # 2% max change
                price_path[t + 1] = price_path[t] * (1 + change)
            else:
                price_path[t + 1] = price_path[t]

        return price_path

    def _calculate_rolling_volatility(
            self,
            returns: np.ndarray,
            window: int = 5
    ) -> List[float]:
        """Calculate rolling volatility from returns"""

        if len(returns) < window:
            # Not enough data for rolling calculation
            return [np.std(returns)] * len(returns)

        volatility = []
        for i in range(len(returns)):
            start = max(0, i - window + 1)
            end = i + 1
            vol = np.std(returns[start:end]) * np.sqrt(252)  # Annualized
            volatility.append(vol)

        return volatility

    def _perturb_scenario(self, scenario: PriceScenario) -> PriceScenario:
        """Create variation of existing scenario"""

        # Add small random perturbations
        price_array = np.array(scenario.price_path)
        noise = np.random.normal(0, 0.01, len(price_array))
        perturbed_prices = price_array * (1 + noise)

        # Recalculate returns and volatility
        returns = np.diff(perturbed_prices) / perturbed_prices[:-1]
        volatility = self._calculate_rolling_volatility(returns)

        return PriceScenario(
            scenario_id=scenario.scenario_id,
            price_path=perturbed_prices.tolist(),
            returns_path=returns.tolist(),
            volatility_path=volatility,
            probability_weight=scenario.probability_weight * 0.9  # Slightly lower weight
        )

    def _classical_quantum_inspired_scenarios(
            self,
            initial_state: Dict[str, Any],
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Quantum-inspired classical fallback"""

        scenarios = []

        # Parameters influenced by quantum principles
        base_volatility = initial_state.get("volatility", 0.2)
        drift = 0.05 + sentiment_impact * 0.1  # Sentiment affects drift

        for i in range(num_scenarios):
            # Generate path using quantum-inspired randomness
            price_path = [initial_state["price"]]
            returns_path = []

            for t in range(time_horizon):
                # Quantum-inspired volatility (changes over time)
                current_vol = base_volatility * (1 + 0.1 * np.sin(t * np.pi / time_horizon))

                # Generate return with fat tails (quantum jumps)
                if np.random.random() < 0.05:  # 5% chance of quantum jump
                    return_t = np.random.normal(drift / 252, current_vol * 3 / np.sqrt(252))
                else:
                    return_t = np.random.normal(drift / 252, current_vol / np.sqrt(252))

                returns_path.append(return_t)
                price_path.append(price_path[-1] * (1 + return_t))

            volatility_path = self._calculate_rolling_volatility(np.array(returns_path))

            scenarios.append(PriceScenario(
                scenario_id=i,
                price_path=price_path,
                returns_path=returns_path,
                volatility_path=volatility_path,
                probability_weight=1.0 / num_scenarios
            ))

        return scenarios

    def _generate_classical_scenarios(
            self,
            initial_state: Dict[str, Any],
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Generate classical scenarios as ultimate fallback"""

        scenarios = []
        price = initial_state["price"]
        volatility = initial_state.get("volatility", 0.2)

        for i in range(num_scenarios):
            # Simple geometric Brownian motion
            dt = 1 / 252  # Daily steps
            price_path = [price]
            returns_path = []

            for _ in range(time_horizon):
                return_t = np.random.normal(0.05 * dt, volatility * np.sqrt(dt))
                returns_path.append(return_t)
                price_path.append(price_path[-1] * (1 + return_t))

            scenarios.append(PriceScenario(
                scenario_id=i,
                price_path=price_path,
                returns_path=returns_path,
                volatility_path=[volatility] * time_horizon,
                probability_weight=1.0 / num_scenarios
            ))

        return scenarios