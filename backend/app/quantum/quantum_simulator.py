"""Real quantum market simulation engine using Classiq"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import norm
import asyncio
from dataclasses import dataclass

from classiq import (
    QFunc, QBit, QArray, Output, allocate, apply_to_all,
    H, RY, RZ, CX, X, Z, control, within_apply,
    create_model, synthesize, execute, QParam
)

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
                    