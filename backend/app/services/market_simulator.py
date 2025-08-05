# backend/app/services/market_simulator.py

import logging
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from app.models.schemas import (
    SentimentAnalysis, MarketPrediction, PriceScenario,
    NewsInput, SimulationMethod, SentimentType
)
from app.ml.classical_model import ClassicalPredictor
from app.ml.market_predictor import EnhancedMarketPredictor
from app.quantum.quantum_finance import QuantumFinanceAlgorithms
from app.config import settings

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
    """Quantum-enhanced market simulation engine"""

    def __init__(self):
        self.classical_predictor = ClassicalPredictor()
        self.enhanced_predictor = None  # Will be initialized later
        self.quantum_finance = None
        self.cache = {}
        self.warnings = []
        self._initialized = False

    async def initialize(self, classiq_client=None):
        """Initialize simulator components"""
        if self._initialized:
            return

        try:
            # Initialize classical predictor
            await self.classical_predictor.initialize()

            # Initialize enhanced market predictor with classical predictor as base
            self.enhanced_predictor = EnhancedMarketPredictor(self.classical_predictor)
            logger.info("Enhanced market predictor initialized")

            # Initialize quantum finance if available
            if settings.enable_quantum:
                self.quantum_finance = QuantumFinanceAlgorithms()
                await self.quantum_finance.initialize(classiq_client)
                logger.info("Quantum finance module initialized")
            else:
                logger.info("Running in classical mode")

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize market simulator: {e}")
            raise

    async def simulate_market_impact(
        self,
        sentiment_results: List[SentimentAnalysis],
        market_data: Dict[str, Any],
        mode: SimulationMethod = SimulationMethod.HYBRID_QML,
        config: Optional[SimulationConfig] = None
    ) -> List[MarketPrediction]:
        """Simulate market impact of news sentiment"""

        if not self._initialized:
            await self.initialize()

        if config is None:
            config = SimulationConfig()

        self.warnings = []  # Reset warnings

        # Extract assets from market data
        assets = list(market_data.keys())

        # Generate predictions for each asset
        predictions = []
        for asset in assets:
            try:
                prediction = await self._simulate_asset(
                    asset,
                    sentiment_results,
                    market_data.get(asset, {}),
                    mode,
                    config
                )
                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Failed to simulate {asset}: {e}")
                self.warnings.append(f"Simulation failed for {asset}: {str(e)}")

        # Portfolio optimization if multiple assets
        if len(predictions) > 1 and self.quantum_finance:
            try:
                correlations = await self._calculate_asset_correlations(market_data)
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
            normalized_impact = max(-1.0, min(1.0, normalized_impact))
            
            # CRITICAL: Apply hard constraints based on sentiment
            if normalized_impact < -0.5:  # Very negative sentiment
                # Force negative return between -10% and -2%
                constrained_impact = max(-0.10, min(-0.02, normalized_impact * 0.1))
            elif normalized_impact < -0.2:  # Negative sentiment  
                # Force negative return between -5% and 0%
                constrained_impact = max(-0.05, min(0.0, normalized_impact * 0.05))
            elif normalized_impact > 0.5:  # Very positive sentiment
                # Force positive return between 2% and 10%
                constrained_impact = min(0.10, max(0.02, normalized_impact * 0.1))
            else:
                # Moderate the impact for neutral sentiments
                constrained_impact = normalized_impact * 0.03
            
            return constrained_impact

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
                relevance = max(relevance, 0.7)

        # Check key phrases
        for phrase in sentiment.key_phrases:
            if asset.lower() in phrase.lower():
                relevance = max(relevance, 0.9)

        # Check market impact keywords
        if sentiment.market_impact_keywords:
            # High impact keywords increase relevance
            relevance = max(relevance, 0.3 + 0.1 * len(sentiment.market_impact_keywords))

        return min(relevance, 1.0)

    def _is_related_entity(self, entity: str, asset: str) -> bool:
        """Check if entity is related to asset"""
        # Simple implementation - could be enhanced with knowledge graph
        related_terms = {
            "AAPL": ["apple", "iphone", "ipad", "mac", "tim cook"],
            "GOOGL": ["google", "alphabet", "search", "android", "sundar pichai"],
            "MSFT": ["microsoft", "windows", "office", "azure", "satya nadella"],
            "TSLA": ["tesla", "electric vehicle", "ev", "elon musk"],
        }

        asset_terms = related_terms.get(asset.upper(), [])
        entity_lower = entity.lower()

        return any(term in entity_lower for term in asset_terms)

    async def _simulate_asset(
        self,
        asset: str,
        sentiment_results: List[SentimentAnalysis],
        asset_data: Dict[str, Any],
        mode: SimulationMethod,
        config: SimulationConfig
    ) -> MarketPrediction:
        """Simulate market impact for a single asset"""

        # Use the enhanced predictor if available, otherwise fall back to scenarios
        if self.enhanced_predictor:
            try:
                prediction = await self.enhanced_predictor.predict_with_constraints(
                    sentiment_results,
                    asset_data,
                    asset
                )
                return prediction
            except Exception as e:
                logger.warning(f"Enhanced predictor failed: {e}, falling back to scenario-based simulation")
                # Continue with traditional approach

        # Calculate sentiment impact
        sentiment_impact = self._calculate_sentiment_impact(sentiment_results, asset)

        # Get current price
        current_price = asset_data.get("current_price", 100.0)

        # Generate scenarios based on mode
        if mode == SimulationMethod.QUANTUM and self.quantum_finance:
            scenarios = await self._quantum_simulation(
                asset_data, sentiment_impact, config.time_horizon, config.num_scenarios
            )
        elif mode == SimulationMethod.CLASSICAL:
            scenarios = await self._classical_simulation(
                asset_data, sentiment_impact, config.time_horizon, config.num_scenarios
            )
        else:  # HYBRID_QML
            scenarios = await self._hybrid_simulation(
                asset_data, sentiment_impact, config.time_horizon, config.num_scenarios
            )

        # Process scenarios into prediction
        prediction = self._process_scenarios_to_prediction(
            asset, asset_data, scenarios, config.confidence_levels
        )

        return prediction

    async def _quantum_simulation(
        self,
        asset_data: Dict[str, Any],
        sentiment_impact: float,
        time_horizon: int,
        num_scenarios: int
    ) -> List[PriceScenario]:
        """Run quantum-enhanced simulation"""

        current_price = asset_data.get("current_price", 100.0)
        historical_volatility = self._calculate_historical_volatility(asset_data)

        # Adjust parameters based on sentiment
        drift = sentiment_impact / time_horizon  # Daily drift
        volatility = historical_volatility * (1 + abs(sentiment_impact))

        try:
            # Run quantum Monte Carlo
            qmc_result = await self.quantum_finance.quantum_monte_carlo(
                spot_price=current_price,
                volatility=volatility,
                drift=drift,
                time_horizon=time_horizon,
                num_paths=num_scenarios
            )

            # Convert to scenarios
            scenarios = []
            for i, path in enumerate(qmc_result["scenarios"]["paths"]):
                scenario = PriceScenario(
                    scenario_id=i,
                    price_path=path,
                    returns_path=self._calculate_returns(path),
                    volatility_path=[volatility] * len(path),
                    probability_weight=qmc_result["scenarios"]["weights"][i]
                )
                scenarios.append(scenario)

            return scenarios

        except Exception as e:
            logger.error(f"Quantum simulation failed: {e}")
            # Fallback to classical
            return await self._classical_simulation(
                asset_data, sentiment_impact, time_horizon, num_scenarios
            )

    async def _classical_simulation(
        self,
        asset_data: Dict[str, Any],
        sentiment_impact: float,
        time_horizon: int,
        num_scenarios: int
    ) -> List[PriceScenario]:
        """Run classical Monte Carlo simulation"""

        current_price = asset_data.get("current_price", 100.0)
        historical_volatility = self._calculate_historical_volatility(asset_data)

        # Adjust parameters based on sentiment
        daily_return = sentiment_impact / time_horizon
        daily_volatility = historical_volatility / np.sqrt(252)

        scenarios = []
        for i in range(num_scenarios):
            price_path = [current_price]
            returns_path = []

            for t in range(time_horizon):
                # Generate daily return
                random_shock = np.random.normal(0, 1)
                daily_change = daily_return + daily_volatility * random_shock

                # Apply mean reversion for extreme moves
                if abs(price_path[-1] / current_price - 1) > 0.1:
                    mean_reversion = -0.1 * (price_path[-1] / current_price - 1)
                    daily_change += mean_reversion

                # Calculate new price
                new_price = price_path[-1] * (1 + daily_change)
                price_path.append(new_price)
                returns_path.append(daily_change)

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

    async def _hybrid_simulation(
        self,
        asset_data: Dict[str, Any],
        sentiment_impact: float,
        time_horizon: int,
        num_scenarios: int
    ) -> List[PriceScenario]:
        """Run hybrid quantum-classical simulation"""

        # Use quantum for subset of scenarios
        quantum_scenarios = int(num_scenarios * 0.3)
        classical_scenarios = num_scenarios - quantum_scenarios

        scenarios = []

        # Quantum scenarios (if available)
        if self.quantum_finance and quantum_scenarios > 0:
            try:
                quantum_results = await self._quantum_simulation(
                    asset_data, sentiment_impact, time_horizon, quantum_scenarios
                )
                scenarios.extend(quantum_results)
            except Exception as e:
                logger.warning(f"Quantum simulation failed in hybrid mode: {e}")
                classical_scenarios = num_scenarios

        # Classical scenarios
        if classical_scenarios > 0:
            classical_results = await self._classical_simulation(
                asset_data, sentiment_impact, time_horizon, classical_scenarios
            )
            scenarios.extend(classical_results)

        # Reweight scenarios
        total_weight = sum(s.probability_weight for s in scenarios)
        for scenario in scenarios:
            scenario.probability_weight /= total_weight

        return scenarios

    def _calculate_historical_volatility(
        self,
        asset_data: Dict[str, Any]
    ) -> float:
        """Calculate historical volatility from price data"""

        prices = asset_data.get("historical_prices", [])
        if len(prices) < 2:
            return 0.2  # Default 20% annual volatility

        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = np.log(prices[i] / prices[i-1])
                returns.append(ret)

        if not returns:
            return 0.2

        # Calculate annualized volatility
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)

        # Cap volatility
        return min(max(annual_vol, 0.1), 0.8)

    def _calculate_returns(self, price_path: List[float]) -> List[float]:
        """Calculate returns from price path"""
        returns = []
        for i in range(1, len(price_path)):
            if price_path[i-1] > 0:
                ret = (price_path[i] - price_path[i-1]) / price_path[i-1]
                returns.append(ret)
        return returns

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

        # Calculate overall confidence
        # Higher confidence if: low volatility, clear regime, quantum advantages
        confidence = 0.7  # Base confidence
        if volatility < 0.15:
            confidence += 0.1
        if max(regime_probabilities.values()) > 0.6:
            confidence += 0.1
        if self.quantum_finance and quantum_uncertainty < 0.7:
            confidence += 0.05

        confidence = min(confidence, 0.95)

        return MarketPrediction(
            asset=asset,
            current_price=current_price,
            expected_return=float(expected_return),
            volatility=float(volatility),
            confidence=float(confidence),
            confidence_intervals=confidence_intervals,
            predicted_scenarios=top_scenarios,
            regime_probabilities=regime_probabilities,
            quantum_uncertainty=float(quantum_uncertainty),
            time_horizon_days=len(scenarios[0].price_path) - 1 if scenarios else 7
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
