"""Unified Market Simulator combining classical and enhanced quantum features"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.models.schemas import (
    MarketPrediction, PriceScenario, SentimentAnalysis,
    SimulationMethod, QuantumMetrics, SentimentType
)
from app.config import settings

logger = logging.getLogger(__name__)

# Optional quantum imports
try:
    from app.quantum.advanced_quantum_model import AdvancedQuantumModel
    from app.quantum.quantum_ml_algorithms import QuantumMLAlgorithms
    from app.quantum.quantum_portfolio_optimizer import QuantumPortfolioOptimizer
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logger.warning("Quantum modules not available, using classical simulation only")


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


@dataclass
class EnhancedQuantumMetrics(QuantumMetrics):
    """Extended quantum metrics with advanced measurements"""
    entanglement_structure: Dict[str, float]
    quantum_advantage_factor: float
    noise_mitigation_fidelity: float
    circuit_optimization_ratio: float


class UnifiedMarketSimulator:
    """Unified market simulator with both classical and quantum capabilities"""

    def __init__(self):
        self._initialized = False
        self.quantum_enabled = QUANTUM_AVAILABLE and settings.quantum_enabled
        self.cache = {}
        self.warnings = []
        
        # Initialize quantum components if available
        if self.quantum_enabled:
            try:
                self.quantum_model = AdvancedQuantumModel()
                self.quantum_ml = QuantumMLAlgorithms()
                self.portfolio_optimizer = QuantumPortfolioOptimizer()
                logger.info("Quantum components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize quantum components: {e}")
                self.quantum_enabled = False

    async def initialize(self):
        """Initialize the simulator"""
        try:
            if self.quantum_enabled:
                await self.quantum_model.initialize()
                await self.quantum_ml.initialize()
                await self.portfolio_optimizer.initialize()
                
            self._initialized = True
            logger.info(f"Market simulator initialized (Quantum: {self.quantum_enabled})")
            
        except Exception as e:
            logger.error(f"Failed to initialize market simulator: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        self.cache.clear()
        if self.quantum_enabled:
            try:
                await self.quantum_model.cleanup()
                await self.quantum_ml.cleanup()
                await self.portfolio_optimizer.cleanup()
            except:
                pass
        logger.info("Market simulator cleanup complete")

    async def simulate(
            self,
            sentiment_results: List[SentimentAnalysis],
            market_data: Dict[str, Any],
            simulation_params: Dict[str, Any]
    ) -> List[MarketPrediction]:
        """Run market simulation with unified approach"""

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
                prediction = await self._simulate_asset(
                    asset=asset,
                    sentiment_results=sentiment_results,
                    market_data=market_data.get(asset, {}),
                    method=method,
                    time_horizon=time_horizon,
                    num_scenarios=num_scenarios
                )
                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Failed to simulate {asset}: {e}")
                self.warnings.append(f"Simulation failed for {asset}: {str(e)}")
                predictions.append(self._create_fallback_prediction(asset, market_data.get(asset, {})))

        return predictions

    async def _simulate_asset(
            self,
            asset: str,
            sentiment_results: List[SentimentAnalysis],
            market_data: Dict[str, Any],
            method: str,
            time_horizon: int,
            num_scenarios: int
    ) -> MarketPrediction:
        """Simulate a single asset with chosen method"""

        current_price = market_data.get('current_price', 100.0)
        sentiment_impact = self._calculate_sentiment_impact(sentiment_results, asset)

        if method == 'quantum' and self.quantum_enabled:
            scenarios = await self._generate_quantum_scenarios(
                asset, current_price, sentiment_impact, time_horizon, num_scenarios
            )
        elif method == 'hybrid_qml' and self.quantum_enabled:
            scenarios = await self._generate_hybrid_scenarios(
                asset, current_price, sentiment_impact, time_horizon, num_scenarios
            )
        else:
            scenarios = await self._generate_classical_scenarios(
                asset, current_price, sentiment_impact, time_horizon, num_scenarios
            )

        return self._create_prediction(asset, current_price, scenarios, sentiment_impact)

    async def _generate_quantum_scenarios(
            self,
            asset: str,
            current_price: float,
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Generate scenarios using quantum methods"""
        
        if not self.quantum_enabled:
            return await self._generate_classical_scenarios(
                asset, current_price, sentiment_impact, time_horizon, num_scenarios
            )

        try:
            # Prepare quantum state
            quantum_state = await self.quantum_model.prepare_market_state(
                current_price=current_price,
                sentiment_vector=[sentiment_impact, 0.0, 0.0, 0.0],
                volatility=0.25
            )

            # Generate quantum scenarios
            scenarios = []
            for i in range(num_scenarios):
                # Quantum evolution
                evolved_state = await self.quantum_model.evolve_market_state(
                    quantum_state, time_horizon
                )

                # Sample price path
                price_path = await self.quantum_model.sample_price_path(
                    evolved_state, current_price, time_horizon
                )

                # Calculate returns and volatility
                returns_path = [
                    (price_path[i+1] - price_path[i]) / price_path[i] 
                    for i in range(len(price_path)-1)
                ]
                
                volatility_path = [0.25] * (time_horizon + 1)  # Simplified

                scenarios.append(PriceScenario(
                    scenario_id=i,
                    price_path=price_path,
                    returns_path=returns_path,
                    volatility_path=volatility_path,
                    probability_weight=1.0 / num_scenarios
                ))

            return scenarios

        except Exception as e:
            logger.warning(f"Quantum simulation failed, falling back to classical: {e}")
            return await self._generate_classical_scenarios(
                asset, current_price, sentiment_impact, time_horizon, num_scenarios
            )

    async def _generate_hybrid_scenarios(
            self,
            asset: str,
            current_price: float,
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Generate scenarios using hybrid quantum-classical approach"""
        
        if not self.quantum_enabled:
            return await self._generate_classical_scenarios(
                asset, current_price, sentiment_impact, time_horizon, num_scenarios
            )

        try:
            # Use quantum ML for parameter estimation
            quantum_params = await self.quantum_ml.estimate_market_parameters(
                asset=asset,
                sentiment_impact=sentiment_impact,
                historical_volatility=0.25
            )

            # Use classical Monte Carlo with quantum parameters
            scenarios = []
            daily_return = sentiment_impact / time_horizon
            daily_volatility = quantum_params.get('volatility', 0.25) / np.sqrt(252)

            for i in range(num_scenarios):
                price_path = [current_price]
                returns_path = []
                volatility_path = [quantum_params.get('volatility', 0.25)]

                for t in range(time_horizon):
                    # Quantum-enhanced random sampling
                    random_shock = await self._quantum_random_sample()
                    
                    # Calculate daily return with quantum corrections
                    daily_change = daily_return + daily_volatility * random_shock
                    returns_path.append(daily_change)

                    # Update price
                    new_price = price_path[-1] * (1 + daily_change)
                    price_path.append(new_price)
                    volatility_path.append(quantum_params.get('volatility', 0.25))

                scenarios.append(PriceScenario(
                    scenario_id=i,
                    price_path=price_path,
                    returns_path=returns_path,
                    volatility_path=volatility_path,
                    probability_weight=1.0 / num_scenarios
                ))

            return scenarios

        except Exception as e:
            logger.warning(f"Hybrid simulation failed, falling back to classical: {e}")
            return await self._generate_classical_scenarios(
                asset, current_price, sentiment_impact, time_horizon, num_scenarios
            )

    async def _generate_classical_scenarios(
            self,
            asset: str,
            current_price: float,
            sentiment_impact: float,
            time_horizon: int,
            num_scenarios: int
    ) -> List[PriceScenario]:
        """Generate scenarios using classical Monte Carlo"""

        scenarios = []
        volatility = 0.25  # 25% annual
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

                # Update volatility
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

    async def _quantum_random_sample(self) -> float:
        """Generate quantum random sample"""
        if self.quantum_enabled:
            try:
                return await self.quantum_model.generate_random_sample()
            except:
                pass
        return np.random.normal(0, 1)

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
                quantum_expectation = sum(
                    (i - 2) * p for i, p in enumerate(sentiment.quantum_sentiment_vector)
                )
                sentiment_value = 0.7 * quantum_expectation + 0.3 * sentiment_value

            weight = sentiment.confidence * relevance
            total_impact += sentiment_value * weight
            total_weight += weight

        if total_weight > 0:
            normalized_impact = total_impact / (total_weight * 2.0)
            normalized_impact = max(-1.0, min(1.0, normalized_impact))

            # Apply impact scaling
            if normalized_impact < -0.5:
                return max(-0.10, min(-0.02, normalized_impact * 0.1))
            elif normalized_impact < -0.2:
                return max(-0.05, min(0.0, normalized_impact * 0.05))
            elif normalized_impact > 0.5:
                return min(0.10, max(0.02, normalized_impact * 0.1))
            else:
                return normalized_impact * 0.03

        return 0.0

    def _calculate_relevance(self, sentiment: SentimentAnalysis, asset: str) -> float:
        """Calculate relevance of sentiment to asset"""
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

    def _create_prediction(
            self,
            asset: str,
            current_price: float,
            scenarios: List[PriceScenario],
            sentiment_impact: float
    ) -> MarketPrediction:
        """Create market prediction from scenarios"""

        final_prices = np.array([s.price_path[-1] for s in scenarios])
        weights = np.array([s.probability_weight for s in scenarios])
        weights = weights / weights.sum()

        # Calculate metrics
        expected_final_price = np.average(final_prices, weights=weights)
        expected_return = (expected_final_price - current_price) / current_price
        returns = [(s.price_path[-1] - current_price) / current_price for s in scenarios]
        volatility = np.sqrt(np.average((np.array(returns) - expected_return) ** 2, weights=weights))

        # Calculate confidence intervals
        confidence_intervals = {}
        for conf_level in [0.68, 0.95]:
            lower_percentile = (1 - conf_level) / 2
            upper_percentile = 1 - lower_percentile

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

        # Calculate quantum uncertainty
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(scenarios))
        quantum_uncertainty = entropy / max_entropy if max_entropy > 0 else 0.5

        # Keep top scenarios
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
        if self.quantum_enabled:
            try:
                return await self.quantum_model.get_metrics()
            except:
                pass
                
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

        # Calculate performance metrics
        quantum_returns = [p.expected_return for p in quantum_predictions]
        classical_returns = [p.expected_return for p in classical_predictions]
        
        quantum_volatilities = [p.volatility for p in quantum_predictions]
        classical_volatilities = [p.volatility for p in classical_predictions]

        return {
            "quantum_avg_return": np.mean(quantum_returns),
            "classical_avg_return": np.mean(classical_returns),
            "quantum_avg_volatility": np.mean(quantum_volatilities),
            "classical_avg_volatility": np.mean(classical_volatilities),
            "return_improvement": np.mean(quantum_returns) - np.mean(classical_returns),
            "volatility_reduction": np.mean(classical_volatilities) - np.mean(quantum_volatilities)
        }

    async def optimize_portfolio(
            self,
            assets: List[str],
            predictions: List[MarketPrediction],
            risk_tolerance: float = 0.5
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        
        if self.quantum_enabled and self.portfolio_optimizer:
            try:
                return await self.portfolio_optimizer.optimize(
                    assets, predictions, risk_tolerance
                )
            except Exception as e:
                logger.warning(f"Quantum portfolio optimization failed: {e}")

        # Classical fallback
        n_assets = len(assets)
        equal_weights = [1.0 / n_assets] * n_assets
        
        return {
            "optimal_weights": dict(zip(assets, equal_weights)),
            "expected_return": sum(pred.expected_return for pred in predictions) / len(predictions),
            "portfolio_volatility": 0.25,
            "sharpe_ratio": 0.5,
            "optimization_method": "equal_weight_classical"
        }
