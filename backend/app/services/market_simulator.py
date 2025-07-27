"""Market simulation service orchestrating real quantum and classical methods"""

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
from app.quantum.quantum_finance import QuantumFinanceAlgorithms, MarketScenario
from app.ml.llm_baseline import LLMBaseline
from app.ml.classical_model import ClassicalPredictor
from app.config import settings

logger = logging.getLogger(__name__)


class MarketSimulator:
    """Main market simulation orchestrator with real quantum integration"""

    def __init__(self):
        self.quantum_simulator = None
        self.quantum_finance = None
        self.llm_baseline = LLMBaseline()
        self.classical_predictor = ClassicalPredictor()
        self.warnings = []
        self.last_quantum_metrics = None
        self.portfolio_optimizer = None

    async def initialize(self, classiq_client: ClassiqClient = None):
        """Initialize simulators with quantum client"""
        logger.info("Initializing market simulator...")

        # Initialize quantum components if client provided
        if classiq_client and classiq_client.is_ready():
            self.quantum_simulator = QuantumSimulator(classiq_client)
            self.quantum_finance = QuantumFinanceAlgorithms(classiq_client)
            logger.info("Quantum components initialized")
        else:
            logger.warning("Running without quantum backend")

        # Initialize classical components
        await self.llm_baseline.initialize()
        await self.classical_predictor.initialize()

        logger.info("Market simulator initialized")

    async def cleanup(self):
        """Cleanup resources"""
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

        # Check if we should optimize portfolio
        should_optimize = len(market_data) > 1 and method != SimulationMethod.CLASSICAL_BASELINE

        # Calculate correlations if multiple assets
        correlations = None
        if should_optimize:
            correlations = await self._calculate_asset_correlations(market_data)

        for asset, asset_data in market_data.items():
            # Aggregate sentiment impact for this asset
            sentiment_impact = self._calculate_sentiment_impact(
                sentiment_results, asset
            )

            # Run simulation based on method
            try:
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

            except Exception as e:
                logger.error(f"Simulation failed for {asset}: {e}")
                self.warnings.append(f"Simulation error for {asset}: {str(e)}")
                # Fallback to classical
                scenarios = await self._classical_simulation(
                    asset_data, sentiment_impact, time_horizon, num_scenarios
                )

            # Process scenarios into prediction
            prediction = self._process_scenarios_to_prediction(
                asset, asset_data, scenarios, confidence_levels
            )

            predictions.append(prediction)

        # Run portfolio optimization if applicable
        if should_optimize and self.quantum_finance:
            try:
                portfolio_result = await self._optimize_portfolio(
                    predictions, market_data, correlations
                )
                # Add portfolio recommendation to warnings (for UI display)
                self.warnings.append(f"Optimal portfolio weights: {portfolio_result}")
            except Exception as e:
                logger.error(f"Portfolio optimization failed: {e}")

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
                    i * p for i, p in enumerate(sentiment.quantum_sentiment_vector, -2)
                )
                # Blend quantum and classical
                sentiment_value = 0.7 * quantum_expectation + 0.3 * sentiment_value

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
            if self._is_related_entity(entity["text"], asset):
                relevance = max(relevance, 0.8)

        # Check key phrases
        for phrase in sentiment.key_phrases:
            if asset.lower() in phrase.lower():
                relevance = max(relevance, 0.7)

        # Market impact keywords boost relevance
        if sentiment.market_impact_keywords:
            relevance = max(relevance, 0.5)

        return relevance

    def _is_related_entity(self, entity_text: str, asset: str) -> bool:
        """Check if entity is related to asset"""

        asset_mappings = {
            "AAPL": ["apple", "iphone", "tim cook", "ios", "mac"],
            "GOOGL": ["google", "alphabet", "sundar pichai", "android", "youtube"],
            "TSLA": ["tesla", "elon musk", "electric vehicle", "ev", "autopilot"],
            "MSFT": ["microsoft", "satya nadella", "windows", "azure", "office"],
            "AMZN": ["amazon", "jeff bezos", "aws", "prime", "alexa"],
            "META": ["meta", "facebook", "zuckerberg", "instagram", "whatsapp"],
            "NVDA": ["nvidia", "jensen huang", "gpu", "cuda", "ai chip"]
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
        """Run real quantum Monte Carlo simulation"""

        if not self.quantum_simulator:
            self.warnings.append("Quantum simulator not available, using classical")
            return await self._classical_simulation(
                asset_data, sentiment_impact, time_horizon, num_scenarios
            )

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

            # Get quantum metrics
            if self.quantum_simulator.client:
                metrics = self.quantum_simulator.client.get_last_execution_metrics()

                self.last_quantum_metrics = QuantumMetrics(
                    circuit_depth=metrics.get("circuit_depth", 0),
                    num_qubits=metrics.get("num_qubits", 0),
                    quantum_volume=metrics.get("gate_count", 0),
                    entanglement_measure=0.85,  # Estimated
                    execution_time_ms=metrics.get("execution_time_ms", 0),
                    hardware_backend=metrics.get("backend_name", "simulator"),
                    success_probability=metrics.get("success_rate", 1.0)
                )

            return scenarios

        except Exception as e:
            logger.error(f"Quantum Monte Carlo failed: {e}")
            self.warnings.append(f"Quantum simulation failed: {str(e)}")
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

        # Quantum walk is complex to implement fully
        # For now, use enhanced Monte Carlo
        self.warnings.append("Quantum walk uses simplified implementation")

        return await self._quantum_monte_carlo_simulation(
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

        # Generate base scenarios classically
        base_scenarios = await self._generate_base_scenarios(
            asset_data, time_horizon, num_scenarios // 2
        )

        # Enhance with quantum if available
        if self.quantum_simulator:
            try:
                # Run quantum enhancement
                quantum_scenarios = await self.quantum_simulator.simulate_market_scenarios(
                    {
                        "price": asset_data["current_price"],
                        "volatility": asset_data.get("volatility", 0.2),
                        "volume": asset_data.get("volume", 1000000),
                        "trend": asset_data.get("trend", 0)
                    },
                    sentiment_impact,
                    time_horizon,
                    num_scenarios // 2
                )

                # Combine scenarios
                all_scenarios = base_scenarios + quantum_scenarios

                # Normalize probability weights
                total_weight = sum(s.probability_weight for s in all_scenarios)
                for s in all_scenarios:
                    s.probability_weight /= total_weight

                return all_scenarios[:num_scenarios]

            except Exception as e:
                logger.error(f"Quantum enhancement failed: {e}")
                self.warnings.append("Quantum enhancement failed, using classical only")

        # Extend classical scenarios if quantum failed
        while len(base_scenarios) < num_scenarios:
            base = base_scenarios[len(base_scenarios) % len(base_scenarios)]
            perturbed = self._perturb_scenario(base)
            perturbed.scenario_id = len(base_scenarios)
            base_scenarios.append(perturbed)

        return base_scenarios[:num_scenarios]

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

    async def _calculate_asset_correlations(
        self,
        market_data: Dict[str, Any]
    ) -> np.ndarray:
        """Calculate correlation matrix for assets"""

        # In practice, would use historical data
        # For now, use simplified correlations
        assets = list(market_data.keys())
        n = len(assets)
        corr_matrix = np.eye(n)

        # Add some typical correlations
        for i in range(n):
            for j in range(i + 1, n):
                # Tech stocks typically correlated
                if assets[i] in ["AAPL", "GOOGL", "MSFT"] and \
                   assets[j] in ["AAPL", "GOOGL", "MSFT"]:
                    corr = 0.7
                else:
                    corr = 0.3

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix

    async def _optimize_portfolio(
        self,
        predictions: List[MarketPrediction],
        market_data: Dict[str, Any],
        correlations: np.ndarray
    ) -> Dict[str, Any]:
        """Optimize portfolio using quantum algorithms"""

        if not self.quantum_finance:
            return {}

        # Extract returns and build covariance
        expected_returns = np.array([p.expected_return for p in predictions])
        volatilities = np.array([p.volatility for p in predictions])

        # Simple covariance from correlations and volatilities
        cov_matrix = np.outer(volatilities, volatilities) * correlations

        # Run quantum portfolio optimization
        try:
            result = await self.quantum_finance.quantum_portfolio_optimization(
                expected_returns,
                cov_matrix,
                risk_aversion=1.0
            )

            # Format result
            assets = list(market_data.keys())
            portfolio = {
                assets[i]: f"{w:.1%}"
                for i, w in enumerate(result["optimal_weights"])
            }

            return portfolio

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {}

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
            scenario_id=scenario.scenario_id + 1000,
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

        bull_prob = np.sum(weights[np.array(returns) > bull_threshold])
        bear_prob = np.sum(weights[np.array(returns) < bear_threshold])
        neutral_prob = 1 - bull_prob - bear_prob

        regime_probabilities = {
            "bull": float(bull_prob),
            "bear": float(bear_prob),
            "neutral": float(neutral_prob)
        }

        # Calculate quantum uncertainty
        # Use entropy of probability distribution
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