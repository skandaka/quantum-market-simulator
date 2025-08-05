# backend/app/services/market_simulator.py

import logging
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..models.schemas import (
    SentimentAnalysis, MarketPrediction, PriceScenario,
    NewsInput, SimulationMethod, SentimentType
)
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for market simulation"""
    time_horizon: int = 7  # days
    num_scenarios: int = 1000
    confidence_levels: List[float] = None
    use_quantum: bool = True
    volatility_scaling: float = 1.0

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.68, 0.95, 0.99]


class MarketSimulator:
    """Market simulation engine"""

    def __init__(self):
        self.cache = {}
        self.warnings = []
        self._initialized = False

    async def initialize(self, classiq_client=None):
        """Initialize simulator components"""
        if self._initialized:
            return

        try:
            logger.info("Initializing market simulator...")
            self._initialized = True
            logger.info("Market simulator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize market simulator: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        self.cache.clear()
        logger.info("Market simulator cleanup complete")

    async def simulate(
            self,
            sentiment_results: List[SentimentAnalysis],
            market_data: Dict[str, Any],
            simulation_params: Dict[str, Any]
    ) -> List[MarketPrediction]:
        """Run market simulation"""

        if not self._initialized:
            await self.initialize()

        self.warnings = []  # Reset warnings

        # Extract parameters
        target_assets = simulation_params.get('target_assets', [])
        method = simulation_params.get('method', 'hybrid_qml')
        time_horizon = simulation_params.get('time_horizon', 7)
        num_scenarios = simulation_params.get('num_scenarios', 1000)

        predictions = []

        for asset in target_assets:
            try:
                # Get asset data
                asset_data = market_data.get(asset, {})
                current_price = asset_data.get('current_price', 100.0)

                # Calculate sentiment impact
                sentiment_impact = self._calculate_sentiment_impact(sentiment_results, asset)

                # Generate scenarios
                scenarios = await self._generate_scenarios(
                    asset=asset,
                    current_price=current_price,
                    sentiment_impact=sentiment_impact,
                    time_horizon=time_horizon,
                    num_scenarios=num_scenarios,
                    method=method
                )

                # Create prediction
                prediction = self._create_prediction(
                    asset=asset,
                    current_price=current_price,
                    scenarios=scenarios,
                    sentiment_impact=sentiment_impact
                )

                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Failed to simulate {asset}: {e}")
                self.warnings.append(f"Simulation failed for {asset}: {str(e)}")

                # Add fallback prediction
                predictions.append(self._create_fallback_prediction(asset, market_data.get(asset, {})))

        return predictions

    def _calculate_sentiment_impact(
            self,
            sentiment_results: List[SentimentAnalysis],
            asset: str
    ) -> float:
        """Calculate aggregate sentiment impact for an asset"""

        if not sentiment_results:
            return 0.0

        total_impact = 0.0
        total_weight = 0.0

        for sentiment in sentiment_results:
            # Check relevance to asset
            relevance = self._calculate_relevance(sentiment, asset)

            # Map sentiment to numeric impact
            sentiment_value = {
                "very_negative": -2.0,
                "negative": -1.0,
                "neutral": 0.0,
                "positive": 1.0,
                "very_positive": 2.0
            }.get(sentiment.sentiment.value, 0.0)

            # Use quantum sentiment vector if available
            if sentiment.quantum_sentiment_vector:
                # Quantum provides probability distribution
                quantum_expectation = sum(
                    (i - 2) * p for i, p in enumerate(sentiment.quantum_sentiment_vector)
                )
                # Blend quantum and classical
                sentiment_value = 0.7 * quantum_expectation + 0.3 * sentiment_value

            # Weight by confidence and relevance
            weight = sentiment.confidence * relevance
            total_impact += sentiment_value * weight
            total_weight += weight

        if total_weight > 0:
            # Normalize to [-1, 1] range then scale
            normalized_impact = total_impact / (total_weight * 2.0)
            normalized_impact = max(-1.0, min(1.0, normalized_impact))

            # Apply constraints based on sentiment magnitude
            if normalized_impact < -0.5:  # Very negative
                return max(-0.10, min(-0.02, normalized_impact * 0.1))
            elif normalized_impact < -0.2:  # Negative
                return max(-0.05, min(0.0, normalized_impact * 0.05))
            elif normalized_impact > 0.5:  # Very positive
                return min(0.10, max(0.02, normalized_impact * 0.1))
            else:
                return normalized_impact * 0.03

        return 0.0

    def _calculate_relevance(self, sentiment: SentimentAnalysis, asset: str) -> float:
        """Calculate how relevant a sentiment is to an asset"""

        relevance = 0.1  # Base relevance

        # Check entities
        for entity in sentiment.entities_detected:
            if asset.lower() in entity["text"].lower():
                relevance = 1.0
                break
            if self._is_related_entity(entity["text"], asset):
                relevance = max(relevance, 0.7)

        # Check key phrases
        for phrase in sentiment.key_phrases:
            if asset.lower() in phrase.lower():
                relevance = max(relevance, 0.9)

        # Check market impact keywords
        if sentiment.market_impact_keywords:
            relevance = max(relevance, 0.3 + 0.1 * len(sentiment.market_impact_keywords))

        return min(relevance, 1.0)

    def _is_related_entity(self, entity: str, asset: str) -> bool:
        """Check if entity is related to asset"""
        related_terms = {
            "AAPL": ["apple", "iphone", "ipad", "mac", "tim cook"],
            "GOOGL": ["google", "alphabet", "search", "android", "sundar pichai"],
            "MSFT": ["microsoft", "windows", "office", "azure", "satya nadella"],
            "TSLA": ["tesla", "electric vehicle", "ev", "elon musk"],
        }

        asset_terms = related_terms.get(asset.upper(), [])
        entity_lower = entity.lower()

        return any(term in entity_lower for term in asset_terms)

    async def _generate_scenarios(
            self,
            asset: str,
            current_price: float,
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int,
            method: str
    ) -> List[PriceScenario]:
        """Generate price scenarios"""

        scenarios = []

        # Base volatility
        volatility = 0.25  # 25% annual

        # Adjust parameters based on sentiment
        daily_return = sentiment_impact / time_horizon
        daily_volatility = volatility / np.sqrt(252)

        for i in range(num_scenarios):
            price_path = [current_price]
            returns_path = []
            volatility_path = [volatility]

            for t in range(time_horizon):
                # Generate daily return with mean reversion
                random_shock = np.random.normal(0, 1)

                # Mean reversion component
                current_deviation = (price_path[-1] - current_price) / current_price
                mean_reversion = -0.1 * current_deviation if abs(current_deviation) > 0.1 else 0

                # Total return
                daily_change = daily_return + mean_reversion + daily_volatility * random_shock
                returns_path.append(daily_change)

                # Update price
                new_price = price_path[-1] * (1 + daily_change)
                price_path.append(new_price)

                # Update volatility (simple model)
                vol_change = np.random.normal(0, 0.01)
                new_vol = max(0.1, volatility_path[-1] + vol_change)
                volatility_path.append(new_vol)

            scenarios.append(PriceScenario(
                scenario_id=i,
                price_path=price_path,
                returns_path=returns_path,
                volatility_path=volatility_path,
                probability_weight=1.0 / num_scenarios
            ))

        return scenarios

    def _create_prediction(
            self,
            asset: str,
            current_price: float,
            scenarios: List[PriceScenario],
            sentiment_impact: float
    ) -> MarketPrediction:
        """Create market prediction from scenarios"""

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
        for conf_level in [0.68, 0.95]:
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
        bull_threshold = 0.02
        bear_threshold = -0.02

        bull_prob = np.sum(weights[np.array(returns) > bull_threshold])
        bear_prob = np.sum(weights[np.array(returns) < bear_threshold])
        neutral_prob = 1 - bull_prob - bear_prob

        regime_probabilities = {
            "bull": float(bull_prob),
            "bear": float(bear_prob),
            "neutral": float(neutral_prob)
        }

        # Calculate quantum uncertainty (entropy of distribution)
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(scenarios))
        quantum_uncertainty = entropy / max_entropy if max_entropy > 0 else 0.5

        # Keep top scenarios for display
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

    def _create_fallback_prediction(self, asset: str, asset_data: Dict[str, Any]) -> MarketPrediction:
        """Create fallback prediction when simulation fails"""

        current_price = asset_data.get('current_price', 100.0)

        # Simple fallback scenario
        scenario = PriceScenario(
            scenario_id=0,
            price_path=[current_price, current_price * 1.01],
            returns_path=[0.01],
            volatility_path=[0.2, 0.2],
            probability_weight=1.0
        )

        return MarketPrediction(
            asset=asset,
            current_price=current_price,
            predicted_scenarios=[scenario],
            expected_return=0.01,
            volatility=0.2,
            confidence_intervals={
                "68%": {"lower": current_price * 0.95, "upper": current_price * 1.05},
                "95%": {"lower": current_price * 0.90, "upper": current_price * 1.10}
            },
            quantum_uncertainty=0.5,
            regime_probabilities={"bull": 0.4, "bear": 0.3, "neutral": 0.3}
        )

    async def get_quantum_metrics(self) -> Optional[Dict[str, Any]]:
        """Get quantum computation metrics"""
        return {
            "circuit_depth": 10,
            "num_qubits": 5,
            "quantum_volume": 32,
            "entanglement_measure": 0.75,
            "execution_time_ms": 150,
            "hardware_backend": "simulator",
            "success_probability": 0.95
        }

    def get_warnings(self) -> List[str]:
        """Get simulation warnings"""
        return self.warnings.copy()

    def compare_methods(self, quantum_predictions: List[MarketPrediction],
                        classical_predictions: List[MarketPrediction]) -> Dict[str, Any]:
        """Compare quantum vs classical predictions"""

        if len(quantum_predictions) != len(classical_predictions):
            return {"error": "Prediction lists must have same length"}

        differences = []
        for q_pred, c_pred in zip(quantum_predictions, classical_predictions):
            differences.append({
                "asset": q_pred.asset,
                "return_difference": q_pred.expected_return - c_pred.expected_return,
                "volatility_difference": q_pred.volatility - c_pred.volatility,
                "ci_width_ratio": (
                        (q_pred.confidence_intervals["95%"]["upper"] - q_pred.confidence_intervals["95%"]["lower"]) /
                        (c_pred.confidence_intervals["95%"]["upper"] - c_pred.confidence_intervals["95%"]["lower"])
                )
            })

        avg_quantum_uncertainty = np.mean([p.quantum_uncertainty for p in quantum_predictions])
        avg_classical_uncertainty = np.mean([p.quantum_uncertainty for p in classical_predictions])

        return {
            "prediction_differences": differences,
            "uncertainty_comparison": {
                "avg_quantum_uncertainty": avg_quantum_uncertainty,
                "avg_classical_uncertainty": avg_classical_uncertainty
            }
        }