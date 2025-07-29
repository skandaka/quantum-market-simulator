"""Real quantum market simulation engine using Classiq"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import norm
import asyncio
from dataclasses import dataclass

try:
    from classiq import (
        qfunc, QBit, QArray, Output, allocate,
        H, RY, RZ, CX, X, Z, control,
        create_model, synthesize, execute
    )
    CLASSIQ_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Classiq import error: {e}")
    CLASSIQ_AVAILABLE = False

from app.quantum.classiq_client import ClassiqClient
from app.quantum.quantum_finance import QuantumFinanceAlgorithms, MarketScenario
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
    market_regime: str  # "bull", "bear", "volatile", "stable"


class QuantumSimulator:
    """Real quantum-enhanced market simulation using Classiq"""

    def __init__(self, classiq_client: ClassiqClient):
        self.client = classiq_client
        self.finance_algorithms = QuantumFinanceAlgorithms(classiq_client)

        # Simulation parameters
        self.num_qubits = 8  # Optimal for real quantum hardware
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

        # Encode market state into quantum representation
        quantum_state = self._encode_market_state(initial_state, sentiment_impact)

        # Check if quantum backend is available
        if not self.client.is_ready():
            logger.warning("Quantum backend not ready, using quantum-inspired classical")
            return self._quantum_inspired_scenarios(
                initial_state, sentiment_impact, time_horizon, num_scenarios
            )

        try:
            # Use quantum Monte Carlo for price simulation
            qmc_results = await self.finance_algorithms.quantum_monte_carlo_pricing(
                spot_price=initial_state["price"],
                volatility=initial_state.get("volatility", 0.2),
                drift=self._calculate_drift(sentiment_impact),
                time_horizon=time_horizon,
                num_paths=num_scenarios
            )

            # Convert QMC results to price scenarios
            scenarios = self._qmc_to_scenarios(qmc_results, initial_state)

            # Apply quantum enhancements
            enhanced_scenarios = await self._apply_quantum_effects(
                scenarios, quantum_state, time_horizon
            )

            return enhanced_scenarios

        except Exception as e:
            logger.error(f"Quantum simulation failed: {e}")
            return self._quantum_inspired_scenarios(
                initial_state, sentiment_impact, time_horizon, num_scenarios
            )

    def _encode_market_state(
            self,
            initial_state: Dict[str, Any],
            sentiment_impact: float
    ) -> QuantumMarketState:
        """Encode market conditions into quantum state representation"""

        price = initial_state.get("price", 100)
        volatility = initial_state.get("volatility", 0.2)
        volume = initial_state.get("volume", 1000000)
        trend = initial_state.get("trend", 0)

        # Determine market regime
        if sentiment_impact > 0.3 and trend > 0:
            market_regime = "bull"
        elif sentiment_impact < -0.3 and trend < 0:
            market_regime = "bear"
        elif volatility > 0.3:
            market_regime = "volatile"
        else:
            market_regime = "stable"

        # Create quantum amplitude encoding
        dimension = 2 ** self.num_qubits
        amplitudes = np.zeros(dimension)

        # Different distributions for different regimes
        if market_regime == "bull":
            # Right-skewed distribution
            center = int(dimension * 0.7)
            width = int(dimension * 0.2)
            for i in range(dimension):
                amplitudes[i] = np.exp(-((i - center) ** 2) / (2 * width ** 2))
                if i > center:
                    amplitudes[i] *= 1.5  # Boost upside

        elif market_regime == "bear":
            # Left-skewed distribution
            center = int(dimension * 0.3)
            width = int(dimension * 0.2)
            for i in range(dimension):
                amplitudes[i] = np.exp(-((i - center) ** 2) / (2 * width ** 2))
                if i < center:
                    amplitudes[i] *= 1.5  # Boost downside

        elif market_regime == "volatile":
            # Bimodal distribution
            center1 = int(dimension * 0.3)
            center2 = int(dimension * 0.7)
            width = int(dimension * 0.1)
            for i in range(dimension):
                amplitudes[i] = (
                        np.exp(-((i - center1) ** 2) / (2 * width ** 2)) +
                        np.exp(-((i - center2) ** 2) / (2 * width ** 2))
                )

        else:  # stable
            # Normal distribution
            center = int(dimension * 0.5)
            width = int(dimension * volatility)
            for i in range(dimension):
                amplitudes[i] = np.exp(-((i - center) ** 2) / (2 * width ** 2))

        # Normalize
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

        # Create phase encoding
        phases = np.zeros(dimension)
        for i in range(dimension):
            # Phase encodes momentum and correlation
            phases[i] = (
                    trend * np.pi +  # Trend component
                    volatility * np.sin(2 * np.pi * i / dimension) +  # Volatility cycles
                    sentiment_impact * np.pi / 2  # Sentiment impact
            )

        # Entanglement map (simplified)
        entanglement_map = {
            "price_volume": np.tanh(volume / 1e7),
            "sentiment_volatility": np.abs(sentiment_impact * volatility),
            "trend_momentum": np.tanh(trend)
        }

        # Coherence time (how long quantum effects persist)
        coherence_time = 1.0 / (1.0 + self.decoherence_rate * volatility)

        return QuantumMarketState(
            amplitude_vector=amplitudes,
            phase_vector=phases,
            entanglement_map=entanglement_map,
            coherence_time=coherence_time,
            market_regime=market_regime
        )

    def _calculate_drift(self, sentiment_impact: float) -> float:
        """Calculate drift based on sentiment impact"""
        # Base drift
        base_drift = 0.05  # 5% annual

        # Sentiment adjustment
        sentiment_adjustment = sentiment_impact * 0.1  # +/-10% max

        return base_drift + sentiment_adjustment

    def _qmc_to_scenarios(
            self,
            qmc_results: Dict[str, Any],
            initial_state: Dict[str, Any]
    ) -> List[PriceScenario]:
        """Convert quantum Monte Carlo results to price scenarios"""

        scenarios = []
        qmc_scenarios = qmc_results.get("scenarios", {})

        if "scenarios" in qmc_scenarios:
            for i, scenario in enumerate(qmc_scenarios["scenarios"]):
                price_path = scenario["path"]

                # Calculate returns
                returns_path = []
                for j in range(1, len(price_path)):
                    ret = np.log(price_path[j] / price_path[j-1])
                    returns_path.append(ret)

                # Calculate volatility path
                window = 5
                volatility_path = []
                for j in range(len(returns_path)):
                    start_idx = max(0, j - window + 1)
                    end_idx = j + 1

                    if end_idx - start_idx >= 2:
                        window_returns = returns_path[start_idx:end_idx]
                        vol = np.std(window_returns) * np.sqrt(252)
                    else:
                        vol = initial_state.get("volatility", 0.2)

                    volatility_path.append(vol)

                scenarios.append(PriceScenario(
                    scenario_id=i,
                    price_path=price_path,
                    returns_path=returns_path,
                    volatility_path=volatility_path,
                    probability_weight=scenario["weight"]
                ))

        return scenarios

    async def _apply_quantum_effects(
            self,
            scenarios: List[PriceScenario],
            quantum_state: QuantumMarketState,
            time_horizon: int
    ) -> List[PriceScenario]:
        """Apply quantum effects to enhance scenarios"""

        if not scenarios:
            return scenarios

        # Apply quantum interference patterns
        for scenario in scenarios:
            # Modulate probability weights based on quantum state
            phase_factor = np.cos(quantum_state.phase_vector[scenario.scenario_id % len(quantum_state.phase_vector)])
            scenario.probability_weight *= (1 + 0.1 * phase_factor)

            # Add quantum volatility component
            quantum_vol = quantum_state.entanglement_map["sentiment_volatility"]
            for i in range(len(scenario.volatility_path)):
                decay = np.exp(-i / (time_horizon * quantum_state.coherence_time))
                scenario.volatility_path[i] *= (1 + quantum_vol * decay * 0.1)

        # Normalize probability weights
        total_weight = sum(s.probability_weight for s in scenarios)
        for s in scenarios:
            s.probability_weight /= total_weight

        return scenarios

    def _quantum_inspired_scenarios(
            self,
            initial_state: Dict[str, Any],
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Generate quantum-inspired scenarios without real quantum hardware"""

        scenarios = []
        current_price = initial_state["price"]
        volatility = initial_state.get("volatility", 0.2)

        # Create quantum-inspired probability distribution
        quantum_state = self._encode_market_state(initial_state, sentiment_impact)

        for i in range(num_scenarios):
            # Use quantum state to influence path generation
            quantum_factor = quantum_state.amplitude_vector[i % len(quantum_state.amplitude_vector)]

            price_path = [current_price]
            returns_path = []
            volatility_path = []

            # Adjust drift based on quantum state
            drift = self._calculate_drift(sentiment_impact) * (1 + quantum_factor * 0.2)

            for t in range(time_horizon):
                # Quantum-influenced volatility
                t_factor = t / time_horizon
                quantum_vol_adjustment = np.sin(quantum_state.phase_vector[i % len(quantum_state.phase_vector)] + t_factor * np.pi)
                daily_vol = volatility * np.sqrt(1/252) * (1 + 0.1 * quantum_vol_adjustment)

                # Generate return with quantum effects
                random_component = np.random.normal(0, 1)
                daily_return = (drift - 0.5 * volatility ** 2) / 252 + daily_vol * random_component

                returns_path.append(daily_return)
                new_price = price_path[-1] * np.exp(daily_return)
                price_path.append(new_price)

                # Volatility path
                volatility_path.append(volatility * (1 + 0.1 * quantum_vol_adjustment))

            # Weight based on quantum probability
            weight = (1 + quantum_factor) / (num_scenarios * 1.5)

            scenarios.append(PriceScenario(
                scenario_id=i,
                price_path=price_path,
                returns_path=returns_path,
                volatility_path=volatility_path,
                probability_weight=weight
            ))

        # Normalize weights
        total_weight = sum(s.probability_weight for s in scenarios)
        for s in scenarios:
            s.probability_weight /= total_weight

        return scenarios