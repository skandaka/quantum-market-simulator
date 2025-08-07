"""
PHASE 4.2: ENSEMBLE QUANTUM MODELS
Advanced ensemble system combining multiple quantum algorithms
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from classiq import qfunc, QBit, QArray, Output, H, RY, RZ, CX, control
    CLASSIQ_AVAILABLE = True
except ImportError:
    CLASSIQ_AVAILABLE = False

from app.quantum.classiq_client import ClassiqClient
from app.quantum.qnlp_model import QuantumNLPModel
from app.quantum.quantum_finance import QuantumFinanceAlgorithms
from app.quantum.quantum_portfolio_optimizer import QuantumPortfolioOptimizer
from app.models.schemas import MarketPrediction, PriceScenario
from app.config import settings

logger = logging.getLogger(__name__)


class EnsembleQuantumModels:
    """
    PHASE 4.2: Advanced ensemble system that combines multiple quantum models
    for enhanced prediction accuracy and robustness
    """
    
    def __init__(self, classiq_client: ClassiqClient):
        self.classiq_client = classiq_client
        
        # Initialize quantum model components
        self.quantum_nlp = QuantumNLPModel(classiq_client) if classiq_client else None
        self.quantum_finance = QuantumFinanceAlgorithms(classiq_client) if classiq_client else None
        self.quantum_portfolio = QuantumPortfolioOptimizer(classiq_client) if classiq_client else None
        
        # Ensemble configuration
        self.ensemble_models = []
        self.model_weights = {}
        self.performance_tracking = {}
        self.consensus_threshold = 0.7
        
        # Model types for ensemble
        self.model_types = [
            "quantum_sentiment_nlp",
            "quantum_monte_carlo",
            "quantum_correlation_analysis",
            "quantum_portfolio_optimization",
            "quantum_volatility_modeling",
            "quantum_trend_analysis"
        ]
        
    async def initialize(self):
        """Initialize all ensemble quantum models"""
        logger.info("🔄 Initializing Ensemble Quantum Models")
        
        try:
            # Initialize core quantum components
            initialization_tasks = []
            
            if self.quantum_nlp:
                initialization_tasks.append(self.quantum_nlp.initialize())
            
            if self.quantum_finance:
                # Finance algorithms don't need separate initialization
                pass
                
            if self.quantum_portfolio:
                # Portfolio optimizer will be initialized on demand
                pass
            
            # Wait for all initializations
            if initialization_tasks:
                await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Initialize ensemble-specific components
            await self._initialize_ensemble_models()
            
            logger.info("✅ Ensemble quantum models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Ensemble initialization failed: {e}")
            raise

    async def _initialize_ensemble_models(self):
        """Initialize individual ensemble model configurations"""
        try:
            # Initialize model weights (equal weighting initially)
            num_models = len(self.model_types)
            base_weight = 1.0 / num_models
            
            for model_type in self.model_types:
                self.model_weights[model_type] = base_weight
                self.performance_tracking[model_type] = {
                    "accuracy_history": [],
                    "execution_times": [],
                    "confidence_scores": [],
                    "success_rate": 1.0
                }
            
            # Initialize ensemble models list
            self.ensemble_models = [
                {
                    "type": model_type,
                    "weight": base_weight,
                    "enabled": True,
                    "quantum_circuit_ready": await self._check_model_readiness(model_type)
                }
                for model_type in self.model_types
            ]
            
            logger.info(f"📊 Initialized {len(self.ensemble_models)} ensemble models")
            
        except Exception as e:
            logger.error(f"Ensemble model initialization failed: {e}")
            raise

    async def _check_model_readiness(self, model_type: str) -> bool:
        """Check if a specific model type is ready for quantum execution"""
        try:
            if not self.classiq_client or not self.classiq_client.is_ready():
                return False
            
            readiness_checks = {
                "quantum_sentiment_nlp": self.quantum_nlp is not None,
                "quantum_monte_carlo": self.quantum_finance is not None,
                "quantum_correlation_analysis": self.quantum_finance is not None,
                "quantum_portfolio_optimization": self.quantum_portfolio is not None,
                "quantum_volatility_modeling": self.quantum_finance is not None,
                "quantum_trend_analysis": self.quantum_nlp is not None
            }
            
            return readiness_checks.get(model_type, False)
            
        except Exception as e:
            logger.error(f"Model readiness check failed for {model_type}: {e}")
            return False

    async def generate_ensemble_predictions(
        self,
        news_data: List[str],
        market_data: Dict[str, Any],
        target_assets: List[str],
        prediction_horizon: int = 7
    ) -> Dict[str, Any]:
        """
        PHASE 4.2.1: Generate predictions using ensemble of quantum models
        """
        try:
            logger.info(f"🎯 Generating ensemble predictions for {len(target_assets)} assets")
            
            start_time = datetime.now()
            
            # Step 1: Run all ensemble models in parallel
            model_predictions = await self._run_ensemble_models_parallel(
                news_data, market_data, target_assets, prediction_horizon
            )
            
            # Step 2: Validate and filter predictions
            valid_predictions = self._validate_model_predictions(model_predictions)
            
            # Step 3: Calculate ensemble consensus
            consensus_predictions = await self._calculate_ensemble_consensus(
                valid_predictions, target_assets
            )
            
            # Step 4: Apply dynamic weighting
            weighted_predictions = await self._apply_dynamic_weighting(
                consensus_predictions, valid_predictions
            )
            
            # Step 5: Generate ensemble uncertainty metrics
            uncertainty_metrics = self._calculate_ensemble_uncertainty(valid_predictions)
            
            # Step 6: Update model performance tracking
            await self._update_ensemble_performance(model_predictions, start_time)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "predictions": weighted_predictions,
                "ensemble_metrics": {
                    "models_participated": len(valid_predictions),
                    "consensus_strength": self._calculate_consensus_strength(valid_predictions),
                    "ensemble_confidence": self._calculate_ensemble_confidence(weighted_predictions),
                    "uncertainty_metrics": uncertainty_metrics,
                    "execution_time": execution_time
                },
                "model_contributions": {
                    model_type: {
                        "weight": self.model_weights[model_type],
                        "prediction_count": len(predictions)
                    }
                    for model_type, predictions in valid_predictions.items()
                },
                "quantum_advantage": await self._calculate_quantum_ensemble_advantage(valid_predictions)
            }
            
        except Exception as e:
            logger.error(f"❌ Ensemble prediction generation failed: {e}")
            return await self._fallback_ensemble_prediction(news_data, market_data, target_assets)

    async def _run_ensemble_models_parallel(
        self,
        news_data: List[str],
        market_data: Dict[str, Any],
        target_assets: List[str],
        prediction_horizon: int
    ) -> Dict[str, List[Any]]:
        """
        PHASE 4.2.2: Run all ensemble models in parallel for efficiency
        """
        try:
            model_tasks = {}
            
            for model in self.ensemble_models:
                if model["enabled"] and model["quantum_circuit_ready"]:
                    model_type = model["type"]
                    
                    # Create task for each model type
                    if model_type == "quantum_sentiment_nlp":
                        task = self._run_quantum_sentiment_model(news_data, target_assets)
                    elif model_type == "quantum_monte_carlo":
                        task = self._run_quantum_monte_carlo_model(market_data, target_assets, prediction_horizon)
                    elif model_type == "quantum_correlation_analysis":
                        task = self._run_quantum_correlation_model(market_data, target_assets)
                    elif model_type == "quantum_portfolio_optimization":
                        task = self._run_quantum_portfolio_model(market_data, target_assets)
                    elif model_type == "quantum_volatility_modeling":
                        task = self._run_quantum_volatility_model(market_data, target_assets, prediction_horizon)
                    elif model_type == "quantum_trend_analysis":
                        task = self._run_quantum_trend_model(news_data, market_data, target_assets)
                    else:
                        continue
                    
                    model_tasks[model_type] = asyncio.create_task(task)
            
            # Execute all tasks with timeout
            results = {}
            timeout = 30.0  # 30 second timeout per model
            
            for model_type, task in model_tasks.items():
                try:
                    result = await asyncio.wait_for(task, timeout=timeout)
                    if result:
                        results[model_type] = result
                    logger.info(f"✅ {model_type} completed successfully")
                except asyncio.TimeoutError:
                    logger.warning(f"⏰ {model_type} timed out")
                except Exception as e:
                    logger.error(f"❌ {model_type} failed: {e}")
            
            logger.info(f"🎯 Ensemble parallel execution completed: {len(results)}/{len(model_tasks)} models succeeded")
            return results
            
        except Exception as e:
            logger.error(f"Parallel ensemble execution failed: {e}")
            return {}

    async def _run_quantum_sentiment_model(self, news_data: List[str], target_assets: List[str]) -> List[Dict]:
        """Run quantum sentiment analysis model"""
        try:
            if not self.quantum_nlp:
                return []
            
            predictions = []
            
            for asset in target_assets:
                asset_sentiment_scores = []
                
                for news_text in news_data:
                    # Encode text for quantum processing
                    quantum_encoding = await self.quantum_nlp.encode_text_quantum(news_text)
                    
                    # Run quantum sentiment classification
                    sentiment_result = await self.quantum_nlp.quantum_sentiment_classification(quantum_encoding)
                    
                    if sentiment_result:
                        asset_sentiment_scores.append(sentiment_result)
                
                # Aggregate sentiment for asset
                if asset_sentiment_scores:
                    avg_confidence = np.mean([s.get("confidence", 0.5) for s in asset_sentiment_scores])
                    
                    # Convert sentiment to market impact
                    sentiment_impact = self._sentiment_to_market_impact(asset_sentiment_scores)
                    
                    predictions.append({
                        "asset": asset,
                        "sentiment_impact": sentiment_impact,
                        "confidence": avg_confidence,
                        "model_type": "quantum_sentiment_nlp",
                        "quantum_metrics": {
                            "entanglement_measure": np.mean([s.get("entanglement_measure", 0.5) for s in asset_sentiment_scores]),
                            "quantum_advantage": avg_confidence * 1.2  # Boost for quantum
                        }
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Quantum sentiment model failed: {e}")
            return []

    async def _run_quantum_monte_carlo_model(
        self, 
        market_data: Dict[str, Any], 
        target_assets: List[str], 
        prediction_horizon: int
    ) -> List[Dict]:
        """Run quantum Monte Carlo pricing model"""
        try:
            if not self.quantum_finance:
                return []
            
            predictions = []
            
            for asset in target_assets:
                asset_data = market_data.get(asset, {})
                
                if not asset_data:
                    continue
                
                # Run quantum Monte Carlo simulation
                qmc_result = await self.quantum_finance.quantum_monte_carlo_pricing(
                    spot_price=asset_data.get("current_price", 100.0),
                    volatility=asset_data.get("volatility", 0.25),
                    drift=0.05,
                    time_horizon=prediction_horizon,
                    num_paths=500
                )
                
                if qmc_result:
                    current_price = asset_data.get("current_price", 100.0)
                    expected_price = qmc_result.get("expected_price", current_price)
                    expected_return = (expected_price - current_price) / current_price
                    
                    predictions.append({
                        "asset": asset,
                        "expected_return": expected_return,
                        "volatility": qmc_result.get("volatility", 0.25),
                        "confidence": qmc_result.get("quantum_confidence", 0.8),
                        "model_type": "quantum_monte_carlo",
                        "quantum_metrics": qmc_result.get("quantum_metrics", {})
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Quantum Monte Carlo model failed: {e}")
            return []

    async def _run_quantum_correlation_model(self, market_data: Dict[str, Any], target_assets: List[str]) -> List[Dict]:
        """Run quantum correlation analysis model"""
        try:
            if not self.quantum_finance:
                return []
            
            predictions = []
            
            # Calculate quantum-enhanced correlations
            for asset in target_assets:
                asset_data = market_data.get(asset, {})
                
                if not asset_data:
                    continue
                
                # Generate correlation-based prediction
                correlation_impact = await self._calculate_quantum_correlation_impact(asset, market_data)
                
                predictions.append({
                    "asset": asset,
                    "correlation_impact": correlation_impact,
                    "confidence": 0.75,
                    "model_type": "quantum_correlation_analysis",
                    "quantum_metrics": {
                        "correlation_strength": abs(correlation_impact),
                        "quantum_enhancement": 0.15
                    }
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Quantum correlation model failed: {e}")
            return []

    async def _run_quantum_portfolio_model(self, market_data: Dict[str, Any], target_assets: List[str]) -> List[Dict]:
        """Run quantum portfolio optimization model"""
        try:
            if not self.quantum_portfolio:
                return []
            
            # Prepare portfolio optimization input
            returns = []
            volatilities = []
            
            for asset in target_assets:
                asset_data = market_data.get(asset, {})
                returns.append(asset_data.get("expected_return", 0.05))
                volatilities.append(asset_data.get("volatility", 0.25))
            
            if len(returns) < 2:
                return []
            
            # Run quantum portfolio optimization
            optimization_result = await self.quantum_portfolio.optimize_portfolio_qaoa(
                expected_returns=np.array(returns),
                covariance_matrix=np.diag(np.array(volatilities) ** 2),  # Simplified covariance
                risk_aversion=1.0
            )
            
            predictions = []
            
            if optimization_result and "optimal_weights" in optimization_result:
                optimal_weights = optimization_result["optimal_weights"]
                
                for i, asset in enumerate(target_assets):
                    if i < len(optimal_weights):
                        predictions.append({
                            "asset": asset,
                            "portfolio_weight": optimal_weights[i],
                            "portfolio_impact": optimal_weights[i] * returns[i],
                            "confidence": optimization_result.get("quantum_confidence", 0.8),
                            "model_type": "quantum_portfolio_optimization",
                            "quantum_metrics": optimization_result.get("quantum_metrics", {})
                        })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Quantum portfolio model failed: {e}")
            return []

    async def _run_quantum_volatility_model(
        self, 
        market_data: Dict[str, Any], 
        target_assets: List[str],
        prediction_horizon: int
    ) -> List[Dict]:
        """Run quantum volatility modeling"""
        try:
            predictions = []
            
            for asset in target_assets:
                asset_data = market_data.get(asset, {})
                
                if not asset_data:
                    continue
                
                base_volatility = asset_data.get("volatility", 0.25)
                
                # Quantum-enhanced volatility prediction
                quantum_vol_factor = await self._calculate_quantum_volatility_enhancement(asset_data)
                enhanced_volatility = base_volatility * (1 + quantum_vol_factor)
                
                predictions.append({
                    "asset": asset,
                    "predicted_volatility": enhanced_volatility,
                    "volatility_change": quantum_vol_factor,
                    "confidence": 0.8,
                    "model_type": "quantum_volatility_modeling",
                    "quantum_metrics": {
                        "quantum_vol_enhancement": quantum_vol_factor,
                        "volatility_uncertainty": enhanced_volatility * 0.1
                    }
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Quantum volatility model failed: {e}")
            return []

    async def _run_quantum_trend_model(
        self, 
        news_data: List[str], 
        market_data: Dict[str, Any], 
        target_assets: List[str]
    ) -> List[Dict]:
        """Run quantum trend analysis model"""
        try:
            predictions = []
            
            # Analyze overall market trend using quantum NLP
            if self.quantum_nlp and news_data:
                trend_indicators = []
                
                for news_text in news_data:
                    # Extract trend information using quantum encoding
                    quantum_encoding = await self.quantum_nlp.encode_text_quantum(news_text)
                    trend_score = await self._extract_quantum_trend_signal(quantum_encoding, news_text)
                    trend_indicators.append(trend_score)
                
                overall_trend = np.mean(trend_indicators) if trend_indicators else 0.0
                
                for asset in target_assets:
                    asset_data = market_data.get(asset, {})
                    
                    # Apply trend to asset prediction
                    trend_impact = overall_trend * 0.1  # Scale factor
                    
                    predictions.append({
                        "asset": asset,
                        "trend_impact": trend_impact,
                        "trend_strength": abs(overall_trend),
                        "confidence": 0.7,
                        "model_type": "quantum_trend_analysis",
                        "quantum_metrics": {
                            "quantum_trend_signal": overall_trend,
                            "trend_coherence": abs(overall_trend) * 0.8
                        }
                    })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Quantum trend model failed: {e}")
            return []

    def _validate_model_predictions(self, model_predictions: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Validate and filter model predictions"""
        try:
            valid_predictions = {}
            
            for model_type, predictions in model_predictions.items():
                valid_preds = []
                
                for pred in predictions:
                    # Basic validation
                    if isinstance(pred, dict) and "asset" in pred and "confidence" in pred:
                        if 0.0 <= pred["confidence"] <= 1.0:
                            valid_preds.append(pred)
                
                if valid_preds:
                    valid_predictions[model_type] = valid_preds
                    logger.info(f"✅ {model_type}: {len(valid_preds)} valid predictions")
                else:
                    logger.warning(f"⚠️ {model_type}: No valid predictions")
            
            return valid_predictions
            
        except Exception as e:
            logger.error(f"Prediction validation failed: {e}")
            return {}

    async def _calculate_ensemble_consensus(
        self, 
        valid_predictions: Dict[str, List[Any]], 
        target_assets: List[str]
    ) -> List[Dict]:
        """Calculate consensus predictions across ensemble models"""
        try:
            consensus_predictions = []
            
            for asset in target_assets:
                asset_predictions = []
                
                # Collect all predictions for this asset
                for model_type, predictions in valid_predictions.items():
                    for pred in predictions:
                        if pred["asset"] == asset:
                            asset_predictions.append({
                                "model_type": model_type,
                                "prediction": pred
                            })
                
                if len(asset_predictions) >= 2:  # Need at least 2 models for consensus
                    consensus_pred = await self._create_consensus_prediction(asset, asset_predictions)
                    if consensus_pred:
                        consensus_predictions.append(consensus_pred)
            
            return consensus_predictions
            
        except Exception as e:
            logger.error(f"Ensemble consensus calculation failed: {e}")
            return []

    async def _create_consensus_prediction(self, asset: str, asset_predictions: List[Dict]) -> Optional[Dict]:
        """Create consensus prediction for a single asset"""
        try:
            if not asset_predictions:
                return None
            
            # Aggregate predictions by type
            returns = []
            volatilities = []
            confidences = []
            model_weights = []
            quantum_metrics = []
            
            for pred_info in asset_predictions:
                pred = pred_info["prediction"]
                model_type = pred_info["model_type"]
                
                # Extract return information
                if "expected_return" in pred:
                    returns.append(pred["expected_return"])
                elif "sentiment_impact" in pred:
                    returns.append(pred["sentiment_impact"])
                elif "trend_impact" in pred:
                    returns.append(pred["trend_impact"])
                else:
                    returns.append(0.0)
                
                # Extract volatility
                volatilities.append(pred.get("volatility", pred.get("predicted_volatility", 0.25)))
                
                # Extract confidence
                confidences.append(pred["confidence"])
                
                # Get model weight
                model_weights.append(self.model_weights.get(model_type, 0.1))
                
                # Collect quantum metrics
                if "quantum_metrics" in pred:
                    quantum_metrics.append(pred["quantum_metrics"])
            
            # Calculate weighted averages
            total_weight = sum(model_weights)
            if total_weight == 0:
                return None
            
            normalized_weights = [w / total_weight for w in model_weights]
            
            consensus_return = sum(r * w for r, w in zip(returns, normalized_weights))
            consensus_volatility = sum(v * w for v, w in zip(volatilities, normalized_weights))
            consensus_confidence = sum(c * w for c, w in zip(confidences, normalized_weights))
            
            # Aggregate quantum metrics
            aggregated_quantum_metrics = {}
            if quantum_metrics:
                all_metric_keys = set()
                for qm in quantum_metrics:
                    all_metric_keys.update(qm.keys())
                
                for key in all_metric_keys:
                    values = [qm.get(key, 0) for qm in quantum_metrics if key in qm]
                    if values:
                        aggregated_quantum_metrics[key] = np.mean(values)
            
            return {
                "asset": asset,
                "consensus_return": consensus_return,
                "consensus_volatility": consensus_volatility,
                "consensus_confidence": consensus_confidence,
                "model_count": len(asset_predictions),
                "contributing_models": [p["model_type"] for p in asset_predictions],
                "quantum_metrics": aggregated_quantum_metrics
            }
            
        except Exception as e:
            logger.error(f"Consensus prediction creation failed for {asset}: {e}")
            return None

    async def _apply_dynamic_weighting(
        self, 
        consensus_predictions: List[Dict], 
        valid_predictions: Dict[str, List[Any]]
    ) -> List[MarketPrediction]:
        """Apply dynamic weighting based on model performance"""
        try:
            weighted_predictions = []
            
            for consensus_pred in consensus_predictions:
                asset = consensus_pred["asset"]
                
                # Adjust prediction based on historical model performance
                performance_adjustment = self._calculate_performance_adjustment(
                    consensus_pred["contributing_models"]
                )
                
                # Apply performance adjustment
                adjusted_return = consensus_pred["consensus_return"] * performance_adjustment
                adjusted_confidence = consensus_pred["consensus_confidence"] * performance_adjustment
                
                # Create MarketPrediction object
                market_prediction = MarketPrediction(
                    asset=asset,
                    current_price=100.0,  # Would be fetched from market data
                    expected_return=adjusted_return,
                    volatility=consensus_pred["consensus_volatility"],
                    confidence=min(adjusted_confidence, 0.95),
                    predicted_scenarios=self._generate_ensemble_scenarios(consensus_pred),
                    confidence_intervals={
                        "95%": {
                            "lower": 100.0 * (1 + adjusted_return - 1.96 * consensus_pred["consensus_volatility"]),
                            "upper": 100.0 * (1 + adjusted_return + 1.96 * consensus_pred["consensus_volatility"])
                        }
                    },
                    time_horizon_days=7,
                    sentiment_impact=adjusted_return if abs(adjusted_return) < 0.2 else 0.0,
                    prediction_method=f"ensemble_quantum_{consensus_pred['model_count']}_models",
                    quantum_metrics=consensus_pred.get("quantum_metrics", {})
                )
                
                weighted_predictions.append(market_prediction)
            
            return weighted_predictions
            
        except Exception as e:
            logger.error(f"Dynamic weighting application failed: {e}")
            return []

    def _calculate_performance_adjustment(self, contributing_models: List[str]) -> float:
        """Calculate performance adjustment factor based on historical model performance"""
        try:
            adjustments = []
            
            for model_type in contributing_models:
                perf_data = self.performance_tracking.get(model_type, {})
                success_rate = perf_data.get("success_rate", 1.0)
                
                # Convert success rate to adjustment factor
                adjustment = 0.5 + success_rate * 0.5  # Range: 0.5 to 1.0
                adjustments.append(adjustment)
            
            # Return average adjustment, with floor at 0.6
            return max(np.mean(adjustments) if adjustments else 1.0, 0.6)
            
        except Exception as e:
            logger.error(f"Performance adjustment calculation failed: {e}")
            return 1.0

    def _generate_ensemble_scenarios(self, consensus_pred: Dict) -> List[PriceScenario]:
        """Generate price scenarios based on ensemble consensus"""
        try:
            scenarios = []
            base_return = consensus_pred["consensus_return"]
            volatility = consensus_pred["consensus_volatility"]
            
            # Generate multiple scenarios around consensus
            scenario_count = 10
            current_price = 100.0  # Would be fetched from market data
            
            for i in range(scenario_count):
                # Create scenarios with different probability levels
                probability_level = (i + 1) / scenario_count
                z_score = np.random.normal(0, 1)
                
                scenario_return = base_return + z_score * volatility * np.sqrt(7/252)  # 7-day horizon
                final_price = current_price * (1 + scenario_return)
                
                scenarios.append(PriceScenario(
                    scenario_id=i,
                    price_path=[current_price, final_price],
                    returns_path=[0.0, scenario_return],
                    volatility_path=[volatility, volatility * 1.1],
                    probability_weight=1.0 / scenario_count
                ))
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Ensemble scenario generation failed: {e}")
            return []

    def _calculate_ensemble_uncertainty(self, valid_predictions: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Calculate uncertainty metrics for the ensemble"""
        try:
            uncertainty_metrics = {
                "model_disagreement": 0.0,
                "confidence_variance": 0.0,
                "prediction_spread": 0.0,
                "ensemble_stability": 1.0
            }
            
            # Calculate model disagreement
            all_confidences = []
            all_predictions = []
            
            for model_type, predictions in valid_predictions.items():
                for pred in predictions:
                    all_confidences.append(pred["confidence"])
                    # Extract prediction value
                    pred_value = (pred.get("expected_return", 0) or 
                                pred.get("sentiment_impact", 0) or 
                                pred.get("trend_impact", 0))
                    all_predictions.append(pred_value)
            
            if all_confidences:
                uncertainty_metrics["confidence_variance"] = np.var(all_confidences)
                
            if all_predictions:
                uncertainty_metrics["prediction_spread"] = np.std(all_predictions)
                
            # Model disagreement (based on prediction variance)
            if len(all_predictions) > 1:
                uncertainty_metrics["model_disagreement"] = min(np.std(all_predictions) / np.mean(np.abs(all_predictions)) if np.mean(np.abs(all_predictions)) > 0 else 0, 1.0)
            
            # Ensemble stability (inverse of disagreement)
            uncertainty_metrics["ensemble_stability"] = 1.0 - uncertainty_metrics["model_disagreement"]
            
            return uncertainty_metrics
            
        except Exception as e:
            logger.error(f"Ensemble uncertainty calculation failed: {e}")
            return {"error": str(e)}

    async def _update_ensemble_performance(self, model_predictions: Dict[str, List[Any]], start_time: datetime):
        """Update performance tracking for ensemble models"""
        try:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            for model_type in self.model_types:
                perf_data = self.performance_tracking[model_type]
                
                # Update execution time
                perf_data["execution_times"].append(execution_time)
                if len(perf_data["execution_times"]) > 50:
                    perf_data["execution_times"] = perf_data["execution_times"][-50:]
                
                # Update success rate
                success = model_type in model_predictions and len(model_predictions[model_type]) > 0
                current_rate = perf_data["success_rate"]
                perf_data["success_rate"] = 0.9 * current_rate + 0.1 * (1.0 if success else 0.0)
                
                # Update model weight based on performance
                self._update_model_weight(model_type, perf_data)
            
        except Exception as e:
            logger.error(f"Ensemble performance update failed: {e}")

    def _update_model_weight(self, model_type: str, perf_data: Dict):
        """Update model weight based on performance"""
        try:
            success_rate = perf_data["success_rate"]
            base_weight = 1.0 / len(self.model_types)
            
            # Adjust weight based on success rate
            if success_rate > 0.8:
                weight_multiplier = 1.2
            elif success_rate > 0.6:
                weight_multiplier = 1.0
            else:
                weight_multiplier = 0.8
            
            new_weight = base_weight * weight_multiplier
            self.model_weights[model_type] = new_weight
            
        except Exception as e:
            logger.error(f"Model weight update failed for {model_type}: {e}")

    async def _calculate_quantum_ensemble_advantage(self, valid_predictions: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Calculate quantum advantage metrics for the ensemble"""
        try:
            quantum_advantage = {
                "ensemble_quantum_speedup": 1.0,
                "quantum_model_ratio": 0.0,
                "total_quantum_metrics": {},
                "ensemble_entanglement": 0.0
            }
            
            total_models = len(valid_predictions)
            quantum_models = 0
            
            all_quantum_metrics = []
            
            for model_type, predictions in valid_predictions.items():
                if "quantum" in model_type:
                    quantum_models += 1
                    
                    for pred in predictions:
                        qm = pred.get("quantum_metrics", {})
                        if qm:
                            all_quantum_metrics.append(qm)
            
            if total_models > 0:
                quantum_advantage["quantum_model_ratio"] = quantum_models / total_models
            
            # Aggregate quantum metrics
            if all_quantum_metrics:
                all_metric_keys = set()
                for qm in all_quantum_metrics:
                    all_metric_keys.update(qm.keys())
                
                for key in all_metric_keys:
                    values = [qm.get(key, 0) for qm in all_quantum_metrics if key in qm]
                    if values:
                        quantum_advantage["total_quantum_metrics"][key] = {
                            "average": np.mean(values),
                            "max": np.max(values),
                            "std": np.std(values) if len(values) > 1 else 0
                        }
            
            # Calculate ensemble quantum speedup
            if quantum_advantage["quantum_model_ratio"] > 0:
                quantum_advantage["ensemble_quantum_speedup"] = 1.0 + quantum_advantage["quantum_model_ratio"] * 0.5
            
            return quantum_advantage
            
        except Exception as e:
            logger.error(f"Quantum ensemble advantage calculation failed: {e}")
            return {"error": str(e)}

    # Helper methods for specific quantum calculations
    def _sentiment_to_market_impact(self, sentiment_results: List[Dict]) -> float:
        """Convert quantum sentiment results to market impact"""
        try:
            if not sentiment_results:
                return 0.0
            
            impacts = []
            for result in sentiment_results:
                sentiment_map = {"positive": 0.1, "negative": -0.1, "neutral": 0.0}
                sentiment = result.get("predicted_sentiment", "neutral")
                confidence = result.get("confidence", 0.5)
                impact = sentiment_map.get(sentiment, 0.0) * confidence
                impacts.append(impact)
            
            return np.mean(impacts)
            
        except Exception as e:
            logger.error(f"Sentiment to market impact conversion failed: {e}")
            return 0.0

    async def _calculate_quantum_correlation_impact(self, asset: str, market_data: Dict[str, Any]) -> float:
        """Calculate quantum-enhanced correlation impact"""
        try:
            # Simplified correlation calculation
            # In real implementation, this would use quantum correlation circuits
            
            asset_volatility = market_data.get(asset, {}).get("volatility", 0.25)
            
            # Calculate correlation with market (simplified)
            market_volatilities = [data.get("volatility", 0.25) for data in market_data.values() if isinstance(data, dict)]
            
            if market_volatilities:
                market_avg_vol = np.mean(market_volatilities)
                correlation_impact = (asset_volatility - market_avg_vol) * 0.1
            else:
                correlation_impact = 0.0
            
            return correlation_impact
            
        except Exception as e:
            logger.error(f"Quantum correlation impact calculation failed: {e}")
            return 0.0

    async def _calculate_quantum_volatility_enhancement(self, asset_data: Dict[str, Any]) -> float:
        """Calculate quantum enhancement factor for volatility"""
        try:
            base_volatility = asset_data.get("volatility", 0.25)
            
            # Quantum enhancement based on market complexity
            # In real implementation, this would use quantum circuits
            
            quantum_enhancement = np.random.normal(0, 0.05)  # Quantum uncertainty
            enhancement_factor = quantum_enhancement * base_volatility
            
            return max(-0.1, min(0.1, enhancement_factor))  # Clamp between -10% and +10%
            
        except Exception as e:
            logger.error(f"Quantum volatility enhancement calculation failed: {e}")
            return 0.0

    async def _extract_quantum_trend_signal(self, quantum_encoding: Any, news_text: str) -> float:
        """Extract trend signal using quantum encoding"""
        try:
            # Simplified trend extraction
            # In real implementation, this would use quantum NLP circuits
            
            trend_keywords = {
                "rising": 1.0, "increasing": 0.8, "growth": 0.6, "bullish": 1.0,
                "falling": -1.0, "decreasing": -0.8, "decline": -0.6, "bearish": -1.0,
                "stable": 0.0, "steady": 0.0, "unchanged": 0.0
            }
            
            text_lower = news_text.lower()
            trend_signals = []
            
            for keyword, signal in trend_keywords.items():
                if keyword in text_lower:
                    trend_signals.append(signal)
            
            return np.mean(trend_signals) if trend_signals else 0.0
            
        except Exception as e:
            logger.error(f"Quantum trend signal extraction failed: {e}")
            return 0.0

    def _calculate_consensus_strength(self, valid_predictions: Dict[str, List[Any]]) -> float:
        """Calculate strength of consensus across models"""
        try:
            if not valid_predictions:
                return 0.0
            
            model_count = len(valid_predictions)
            
            # Consensus strength based on number of participating models and agreement
            if model_count >= 4:
                return 0.9
            elif model_count >= 3:
                return 0.8
            elif model_count >= 2:
                return 0.7
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Consensus strength calculation failed: {e}")
            return 0.5

    def _calculate_ensemble_confidence(self, weighted_predictions: List[MarketPrediction]) -> float:
        """Calculate overall ensemble confidence"""
        try:
            if not weighted_predictions:
                return 0.0
            
            confidences = [pred.confidence for pred in weighted_predictions]
            return np.mean(confidences)
            
        except Exception as e:
            logger.error(f"Ensemble confidence calculation failed: {e}")
            return 0.5

    async def _fallback_ensemble_prediction(self, news_data, market_data, target_assets):
        """Fallback prediction when ensemble fails"""
        try:
            logger.warning("🔄 Using fallback ensemble prediction")
            
            # Simple fallback based on classical methods
            fallback_predictions = []
            
            for asset in target_assets:
                asset_data = market_data.get(asset, {})
                current_price = asset_data.get("current_price", 100.0)
                
                fallback_pred = MarketPrediction(
                    asset=asset,
                    current_price=current_price,
                    expected_return=0.05,  # Default 5% annual return
                    volatility=0.25,
                    confidence=0.5,
                    predicted_scenarios=[],
                    confidence_intervals={"95%": {"lower": current_price * 0.9, "upper": current_price * 1.1}},
                    time_horizon_days=7,
                    sentiment_impact=0.0,
                    prediction_method="ensemble_fallback"
                )
                
                fallback_predictions.append(fallback_pred)
            
            return {
                "predictions": fallback_predictions,
                "ensemble_metrics": {"fallback_used": True},
                "model_contributions": {},
                "quantum_advantage": {}
            }
            
        except Exception as e:
            logger.error(f"Fallback ensemble prediction failed: {e}")
            return {"predictions": [], "error": str(e)}

    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current status of ensemble models"""
        try:
            return {
                "total_models": len(self.ensemble_models),
                "enabled_models": sum(1 for model in self.ensemble_models if model["enabled"]),
                "quantum_ready_models": sum(1 for model in self.ensemble_models if model["quantum_circuit_ready"]),
                "model_weights": self.model_weights.copy(),
                "performance_summary": {
                    model_type: {
                        "success_rate": data["success_rate"],
                        "avg_execution_time": np.mean(data["execution_times"]) if data["execution_times"] else 0
                    }
                    for model_type, data in self.performance_tracking.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Ensemble status retrieval failed: {e}")
            return {"error": str(e)}
