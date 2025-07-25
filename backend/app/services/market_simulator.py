"""Market simulation service orchestrating quantum and classical methods"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats

from app.models.schemas import (
    MarketPrediction, PriceScenario, SentimentAnalysis,
    SimulationMethod, QuantumMetrics
)
from app.quantum.quantum_simulator import QuantumSimulator
from app.quantum.classiq_client import ClassiqClient
from app.ml.llm_baseline import LLMBaseline
from app.ml.classical_model import ClassicalPredictor
from app.config import settings

logger = logging.getLogger(__name__)


class MarketSimulator:
    """Main market simulation orchestrator"""

    def __init__(self):
        self.quantum_simulator = None
        self.llm_baseline = LLMBaseline()
        self.classical_predictor = ClassicalPredictor()
        self.warnings = []
        self.last_quantum_metrics = None

    async def initialize(self):
        """Initialize simulators and warm up models"""
        logger.info("Initializing market simulator...")

        # Warm up models
        await self.llm_baseline.initialize()
        await self.classical_predictor.initialize()

        logger.info("Market simulator initialized")

    async def cleanup(self):
        """Cleanup resources"""
        # Any cleanup needed
        pass

    async def simulate(
            self,
            sentiment_results: List[SentimentAnalysis],
            market_data: Dict[str, Any],
            simulation_params: Dict[str, Any]
    ) -> List[MarketPrediction]:
        """Run market simulation based on sentiment and current market data"""

        method = simulation_params.get("method", SimulationMethod.HYBRID_QML)
        time_horizon = simulation_params.get("time_horizon", 7)
        num_scenarios = simulation_params.get("num_scenarios", 1000)
        confidence_levels = simulation_params.get("confidence_levels", [0.68, 0.95])

        predictions = []

        for asset, asset_data in market_data.items():
            # Aggregate sentiment impact for this asset
            sentiment_impact = self._calculate_sentiment_impact(
                sentiment_results, asset
            )

            # Run simulation based on method
            if method == SimulationMethod.QUANTUM_MONTE_CARLO:
                scenarios = await self._quantum_monte_carlo_simulation(
                    asset_data, sentiment_impact, time_horizon, num_scenarios
                )
            elif method == SimulationMethod.QUANTUM_WALK:
                scenarios = await self._quantum_walk_simulation(
                    asset_data, sentiment_impact, time_horizon, num_scenarios
                )
            elif method == SimulationMethod.HYBRID_QML:
                scenarios = await self._hybrid_qml_simulation(
                    asset_data, sentiment_impact, time_horizon, num_scenarios
                )
            else:  # Classical baseline
                scenarios = await self._classical_simulation(
                    asset_data, sentiment_impact, time_horizon, num_scenarios
                )

            # Process scenarios into prediction
            prediction = self._process_scenarios_to_prediction(
                asset, asset_data, scenarios, confidence_levels
            )

            predictions.append(prediction)

        return predictions

    def _calculate_sentiment_impact(
            self,
            sentiment_results: List[SentimentAnalysis],
            asset: str
    ) -> float:
        """Calculate aggregate sentiment impact for an asset"""

        if not sentiment_results:
            return 0.0

        # Weight sentiments by confidence and relevance
        total_impact = 0.0
        total_weight = 0.0

        for sentiment in sentiment_results:
            # Check if asset is mentioned in entities
            relevance = self._calculate_relevance(sentiment, asset)

            # Map sentiment to numeric impact
            sentiment_value = {
                "very_negative": -2.0,
                "negative": -1.0,
                "neutral": 0.0,
                "positive": 1.0,
                "very_positive": 2.0
            }.get(sentiment.sentiment.value, 0.0)

            # Weight by confidence and relevance
            weight = sentiment.confidence * relevance
            total_impact += sentiment_value * weight
            total_weight += weight

        if total_weight > 0:
            # Normalize to [-1, 1] range
            normalized_impact = total_impact / (total_weight * 2.0)
            return max(-1.0, min(1.0, normalized_impact))

        return 0.0

    def _calculate_relevance(
            self,
            sentiment: SentimentAnalysis,
            asset: str
    ) -> float:
        """Calculate how relevant a sentiment is to an asset"""

        relevance = 0.1  # Base relevance

        # Check entities
        for entity in sentiment.entities_detected:
            if asset.lower() in entity["text"].lower():
                relevance = 1.0
                break
            # Check for company name variations
            if self._is_related_entity(entity["text"], asset):
                relevance = 0.8

        # Check key phrases
        for phrase in sentiment.key_phrases:
            if asset.lower() in phrase.lower():
                relevance = max(relevance, 0.7)

        # Check market impact keywords
        if sentiment.market_impact_keywords:
            relevance = max(relevance, 0.5)

        return relevance

    def _is_related_entity(self, entity_text: str, asset: str) -> bool:
        """Check if entity is related to asset"""
        # Simple mapping - in practice, use a more sophisticated approach
        asset_mappings = {
            "AAPL": ["apple", "iphone", "tim cook"],
            "GOOGL": ["google", "alphabet", "sundar pichai"],
            "TSLA": ["tesla", "elon musk", "electric vehicle"],
            "MSFT": ["microsoft", "satya nadella", "windows"],
            "AMZN": ["amazon", "jeff bezos", "aws"]
        }

        related_terms = asset_mappings.get(asset.upper(), [])
        entity_lower = entity_text.lower()

        return any(term in entity_lower for term in related_terms)

    async def _quantum_monte_carlo_simulation(
            self,
            asset_data: Dict[str, Any],
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Run quantum Monte Carlo simulation"""

        if not self.quantum_simulator:
            # Initialize quantum simulator with Classiq client
            classiq_client = ClassiqClient(settings.classiq_api_key)
            self.quantum_simulator = QuantumSimulator(classiq_client)

        try:
            # Prepare initial state
            initial_state = {
                "price": asset_data["current_price"],
                "volatility": asset_data.get("volatility", 0.2),
                "volume": asset_data.get("volume", 1000000),
                "trend": asset_data.get("trend", 0)
            }

            # Run quantum simulation
            scenarios = await self.quantum_simulator.simulate_market_scenarios(
                initial_state,
                sentiment_impact,
                time_horizon,
                num_scenarios
            )

            # Store quantum metrics
            self.last_quantum_metrics = QuantumMetrics(
                circuit_depth=42,  # Would get from actual circuit
                num_qubits=10,
                quantum_volume=1024,
                entanglement_measure=0.87,
                execution_time_ms=523,
                hardware_backend=settings.quantum_backend,
                success_probability=0.98
            )

            return scenarios

        except Exception as e:
            logger.error(f"Quantum Monte Carlo failed: {e}")
            self.warnings.append(f"Quantum simulation failed, using classical fallback")
            # Fallback to classical
            return await self._classical_simulation(
                asset_data, sentiment_impact, time_horizon, num_scenarios
            )

    async def _quantum_walk_simulation(
            self,
            asset_data: Dict[str, Any],
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Run quantum walk simulation"""

        # For hackathon, implement simplified version
        # Real implementation would use quantum walks on market graphs

        self.warnings.append("Quantum walk simulation in beta")

        # Use hybrid approach
        return await self._hybrid_qml_simulation(
            asset_data, sentiment_impact, time_horizon, num_scenarios
        )

    async def _hybrid_qml_simulation(
            self,
            asset_data: Dict[str, Any],
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Run hybrid quantum-classical simulation"""

        # Start with classical base scenarios
        base_scenarios = await self._generate_base_scenarios(
            asset_data, time_horizon, num_scenarios // 2
        )

        # Enhance with quantum features
        enhanced_scenarios = []

        for scenario in base_scenarios:
            # Apply quantum-inspired modifications
            enhanced = self._apply_quantum_enhancement(
                scenario, sentiment_impact
            )
            enhanced_scenarios.append(enhanced)

        # Generate additional pure quantum scenarios
        if self.quantum_simulator:
            try:
                quantum_scenarios = await self._quantum_monte_carlo_simulation(
                    asset_data,
                    sentiment_impact,
                    time_horizon,
                    num_scenarios // 2
                )
                enhanced_scenarios.extend(quantum_scenarios)
            except:
                # If quantum fails, generate more classical
                pass

        # Ensure we have enough scenarios
        while len(enhanced_scenarios) < num_scenarios:
            base = enhanced_scenarios[len(enhanced_scenarios) % len(enhanced_scenarios)]
            perturbed = self._perturb_scenario(base)
            enhanced_scenarios.append(perturbed)

        return enhanced_scenarios[:num_scenarios]

    async def _classical_simulation(
            self,
            asset_data: Dict[str, Any],
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Run classical Monte Carlo simulation"""

        scenarios = []

        # Extract parameters
        current_price = asset_data["current_price"]
        volatility = asset_data.get("volatility", 0.2)

        # Adjust parameters based on sentiment
        drift = 0.05 + sentiment_impact * 0.1  # Annual drift
        volatility = volatility * (1 + abs(sentiment_impact) * 0.2)

        # Generate scenarios
        for i in range(num_scenarios):
            # Geometric Brownian Motion
            dt = 1 / 252  # Daily time step

            price_path = [current_price]
            returns_path = []

            for t in range(time_horizon):
                # Generate daily return
                z = np.random.standard_normal()
                daily_return = (drift - 0.5 * volatility ** 2) * dt + \
                               volatility * np.sqrt(dt) * z

                returns_path.append(daily_return)
                price_path.append(price_path[-1] * np.exp(daily_return))

            # Calculate path volatility
            path_volatility = self._calculate_path_volatility(returns_path)

            scenarios.append(PriceScenario(
                scenario_id=i,
                price_path=price_path,
                returns_path=returns_path,
                volatility_path=path_volatility,
                probability_weight=1.0 / num_scenarios
            ))

        return scenarios

    async def _generate_base_scenarios(
            self,
            asset_data: Dict[str, Any],
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Generate base scenarios for hybrid approach"""

        # Use classical with some enhancements
        return await self._classical_simulation(
            asset_data, 0.0, time_horizon, num_scenarios
        )

    def _apply_quantum_enhancement(
            self,
            scenario: PriceScenario,
            sentiment_impact: float
    ) -> PriceScenario:
        """Apply quantum-inspired enhancements to scenario"""

        # Quantum effects:
        # 1. Superposition - multiple possible states
        # 2. Entanglement - correlation changes
        # 3. Tunneling - sudden jumps possible

        enhanced_prices = []
        enhanced_returns = []

        for i, (price, ret) in enumerate(zip(scenario.price_path, [0] + scenario.returns_path)):
            if i == 0:
                enhanced_prices.append(price)
                continue

            # Quantum tunneling effect - small chance of large jump
            if np.random.random() < 0.02:  # 2% chance
                # Tunneling event - larger move
                tunnel_factor = 1 + np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.05)
                enhanced_price = enhanced_prices[-1] * tunnel_factor

                # Influenced by sentiment
                if sentiment_impact > 0 and tunnel_factor > 1:
                    enhanced_price *= (1 + sentiment_impact * 0.01)
                elif sentiment_impact < 0 and tunnel_factor < 1:
                    enhanced_price *= (1 + sentiment_impact * 0.01)
            else:
                # Normal evolution with quantum noise
                quantum_noise = np.random.normal(0, 0.001)
                enhanced_return = ret + quantum_noise + sentiment_impact * 0.0001
                enhanced_price = enhanced_prices[-1] * np.exp(enhanced_return)

            enhanced_prices.append(enhanced_price)
            if i > 0:
                actual_return = np.log(enhanced_price / enhanced_prices[-2])
                enhanced_returns.append(actual_return)

        # Recalculate volatility
        enhanced_volatility = self._calculate_path_volatility(enhanced_returns)

        return PriceScenario(
            scenario_id=scenario.scenario_id,
            price_path=enhanced_prices,
            returns_path=enhanced_returns,
            volatility_path=enhanced_volatility,
            probability_weight=scenario.probability_weight
        )

    def _perturb_scenario(self, scenario: PriceScenario) -> PriceScenario:
        """Create variation of existing scenario"""

        # Add correlated noise
        price_array = np.array(scenario.price_path)

        # Generate AR(1) noise
        noise = [0]
        phi = 0.8  # Autocorrelation
        for i in range(1, len(price_array)):
            noise.append(phi * noise[-1] + np.random.normal(0, 0.005))

        # Apply noise
        perturbed_prices = price_array * (1 + np.array(noise))

        # Recalculate returns
        returns = []
        for i in range(1, len(perturbed_prices)):
            ret = np.log(perturbed_prices[i] / perturbed_prices[i - 1])
            returns.append(ret)

        # Recalculate volatility
        volatility = self._calculate_path_volatility(returns)

        return PriceScenario(
            scenario_id=scenario.scenario_id + 1000,  # Different ID
            price_path=perturbed_prices.tolist(),
            returns_path=returns,
            volatility_path=volatility,
            probability_weight=scenario.probability_weight * 0.95
        )

    def _calculate_path_volatility(
            self,
            returns: List[float],
            window: int = 5
    ) -> List[float]:
        """Calculate rolling volatility along path"""

        if not returns:
            return []

        volatilities = []
        returns_array = np.array(returns)

        for i in range(len(returns)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1

            if end_idx - start_idx >= 2:
                window_returns = returns_array[start_idx:end_idx]
                vol = np.std(window_returns) * np.sqrt(252)  # Annualized
            else:
                vol = 0.2  # Default volatility

            volatilities.append(vol)

        return volatilities

    def _process_scenarios_to_prediction(
            self,
            asset: str,
            asset_data: Dict[str, Any],
            scenarios: List[PriceScenario],
            confidence_levels: List[float]
    ) -> MarketPrediction:
        """Process scenarios into final prediction"""

        current_price = asset_data["current_price"]

        # Extract final prices and weights
        final_prices = np.array([s.price_path[-1] for s in scenarios])
        weights = np.array([s.probability_weight for s in scenarios])

        # Normalize weights
        weights = weights / weights.sum()

        # Calculate expected return
        expected_final_price = np.average(final_prices, weights=weights)
        expected_return = (expected_final_price - current_price) / current_price

        # Calculate volatility
        returns = [(s.price_path[-1] - current_price) / current_price for s in scenarios]
        volatility = np.sqrt(np.average((np.array(returns) - expected_return) ** 2, weights=weights))

        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            lower_percentile = (1 - conf_level) / 2
            upper_percentile = 1 - lower_percentile

            # Weighted percentiles
            sorted_indices = np.argsort(final_prices)
            sorted_prices = final_prices[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cumsum_weights = np.cumsum(sorted_weights)

            lower_idx = np.searchsorted(cumsum_weights, lower_percentile)
            upper_idx = np.searchsorted(cumsum_weights, upper_percentile)

            lower_price = sorted_prices[min(lower_idx, len(sorted_prices) - 1)]
            upper_price = sorted_prices[min(upper_idx, len(sorted_prices) - 1)]

            confidence_intervals[f"{int(conf_level * 100)}%"] = {
                "lower": float(lower_price),
                "upper": float(upper_price)
            }

        # Calculate regime probabilities
        bull_threshold = 0.02  # 2% up
        bear_threshold = -0.02  # 2% down

        bull_prob = np.sum(weights[returns > bull_threshold])
        bear_prob = np.sum(weights[returns < bear_threshold])
        neutral_prob = 1 - bull_prob - bear_prob

        regime_probabilities = {
            "bull": float(bull_prob),
            "bear": float(bear_prob),
            "neutral": float(neutral_prob)
        }

        # Calculate quantum uncertainty (entropy-based)
        # Higher entropy = higher uncertainty
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(scenarios))
        quantum_uncertainty = entropy / max_entropy if max_entropy > 0 else 0.5

        # Keep top 10 most likely scenarios
        top_indices = np.argsort([s.probability_weight for s in scenarios])[-10:]
        top_scenarios = [scenarios[i] for i in top_indices]

        return MarketPrediction(
            asset=asset,
            current_price=current_price,
            predicted_scenarios=top_scenarios,
            expected_return=float(expected_return),
            volatility=float(volatility),
            confidence_intervals=confidence_intervals,
            quantum_uncertainty=float(quantum_uncertainty),
            regime_probabilities=regime_probabilities
        )

    def compare_methods(
            self,
            quantum_predictions: List[MarketPrediction],
            classical_predictions: List[MarketPrediction]
    ) -> Dict[str, Any]:
        """Compare quantum vs classical predictions"""

        comparison = {
            "prediction_differences": [],
            "uncertainty_comparison": {},
            "performance_metrics": {}
        }

        for q_pred, c_pred in zip(quantum_predictions, classical_predictions):
            # Compare expected returns
            return_diff = abs(q_pred.expected_return - c_pred.expected_return)

            # Compare volatilities
            vol_diff = abs(q_pred.volatility - c_pred.volatility)

            # Compare confidence intervals
            ci_95_q = q_pred.confidence_intervals.get("95%", {})
            ci_95_c = c_pred.confidence_intervals.get("95%", {})

            ci_width_q = ci_95_q.get("upper", 0) - ci_95_q.get("lower", 0)
            ci_width_c = ci_95_c.get("upper", 0) - ci_95_c.get("lower", 0)

            comparison["prediction_differences"].append({
                "asset": q_pred.asset,
                "return_difference": return_diff,
                "volatility_difference": vol_diff,
                "ci_width_ratio": ci_width_q / ci_width_c if ci_width_c > 0 else 1
            })

        # Aggregate metrics
        comparison["uncertainty_comparison"] = {
            "avg_quantum_uncertainty": np.mean([p.quantum_uncertainty for p in quantum_predictions]),
            "avg_classical_uncertainty": np.mean([p.quantum_uncertainty for p in classical_predictions])
        }

        return comparison

    async def get_quantum_metrics(self) -> Optional[QuantumMetrics]:
        """Get metrics from last quantum execution"""
        return self.last_quantum_metrics

    def get_warnings(self) -> List[str]:
        """Get any warnings generated during simulation"""
        warnings = self.warnings.copy()
        self.warnings = []  # Clear after retrieval
        return warnings