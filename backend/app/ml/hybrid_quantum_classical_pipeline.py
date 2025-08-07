"""
PHASE 4.1: HYBRID QUANTUM-CLASSICAL PIPELINE
Advanced model that combines quantum advantage with classical reliability
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json

try:
    from classiq import qfunc, QBit, QArray, Output, H, RY, RZ, CX, control
    CLASSIQ_AVAILABLE = True
except ImportError:
    CLASSIQ_AVAILABLE = False

from app.quantum.classiq_client import ClassiqClient
from app.quantum.qnlp_model import QuantumNLPModel
from app.quantum.quantum_finance import QuantumFinanceAlgorithms
from app.models.schemas import SentimentAnalysis, MarketPrediction, PriceScenario
from app.ml.market_predictor import EnhancedMarketPredictor
from app.config import settings

logger = logging.getLogger(__name__)


class HybridQuantumClassicalPipeline:
    """
    PHASE 4.1: Advanced hybrid pipeline that intelligently combines
    quantum and classical approaches for optimal accuracy
    """
    
    def __init__(self, classiq_client: ClassiqClient):
        self.classiq_client = classiq_client
        self.quantum_nlp = QuantumNLPModel(classiq_client) if classiq_client else None
        self.quantum_finance = QuantumFinanceAlgorithms(classiq_client) if classiq_client else None
        self.classical_predictor = EnhancedMarketPredictor()
        
        # Pipeline configuration
        self.quantum_confidence_threshold = 0.85
        self.classical_backup_threshold = 0.6
        self.hybrid_weight_quantum = 0.7
        self.hybrid_weight_classical = 0.3
        
        # Performance tracking
        self.performance_history = []
        self.model_selection_history = []
        
    async def initialize(self):
        """Initialize the hybrid pipeline components"""
        logger.info("🔄 Initializing Hybrid Quantum-Classical Pipeline")
        
        try:
            if self.quantum_nlp:
                await self.quantum_nlp.initialize()
                logger.info("✅ Quantum NLP initialized")
            
            # Initialize classical components
            await self.classical_predictor.initialize()
            logger.info("✅ Classical predictor initialized")
            
            logger.info("🎯 Hybrid pipeline ready for intelligent model selection")
            
        except Exception as e:
            logger.error(f"❌ Hybrid pipeline initialization failed: {e}")
            raise

    async def predict_with_hybrid_intelligence(
        self,
        news_data: List[str],
        market_data: Dict[str, Any],
        target_assets: List[str],
        prediction_horizon: int = 7
    ) -> Dict[str, Any]:
        """
        PHASE 4.1.1: Intelligent hybrid prediction with dynamic model selection
        """
        try:
            logger.info(f"🧠 Starting hybrid prediction for {len(target_assets)} assets")
            
            # Step 1: Analyze data characteristics to determine optimal approach
            data_analysis = await self._analyze_data_characteristics(news_data, market_data)
            
            # Step 2: Generate predictions using multiple approaches
            prediction_results = {}
            
            # Quantum predictions (if suitable)
            if data_analysis["quantum_suitable"]:
                quantum_results = await self._generate_quantum_predictions(
                    news_data, market_data, target_assets, prediction_horizon
                )
                prediction_results["quantum"] = quantum_results
            
            # Classical predictions (always generated)
            classical_results = await self._generate_classical_predictions(
                news_data, market_data, target_assets, prediction_horizon
            )
            prediction_results["classical"] = classical_results
            
            # Step 3: Ensemble/hybrid combination
            if data_analysis["quantum_suitable"] and "quantum" in prediction_results:
                hybrid_results = await self._create_intelligent_ensemble(
                    prediction_results["quantum"],
                    prediction_results["classical"],
                    data_analysis
                )
                prediction_results["hybrid"] = hybrid_results
            else:
                # Use classical with quantum-inspired enhancements
                prediction_results["hybrid"] = await self._enhance_classical_with_quantum_insights(
                    prediction_results["classical"], data_analysis
                )
            
            # Step 4: Model selection and confidence assessment
            final_predictions = await self._select_optimal_predictions(
                prediction_results, data_analysis
            )
            
            # Step 5: Performance tracking and learning
            self._update_performance_tracking(prediction_results, data_analysis)
            
            return {
                "predictions": final_predictions,
                "model_selection": data_analysis["selected_model"],
                "confidence_metrics": data_analysis["confidence_assessment"],
                "quantum_advantage": prediction_results.get("quantum", {}).get("quantum_metrics", {}),
                "hybrid_performance": self._calculate_hybrid_performance_metrics(prediction_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Hybrid prediction failed: {e}")
            # Fallback to classical prediction
            return await self._fallback_classical_prediction(news_data, market_data, target_assets)

    async def _analyze_data_characteristics(
        self, 
        news_data: List[str], 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        PHASE 4.1.2: Analyze data to determine optimal modeling approach
        """
        try:
            analysis = {
                "quantum_suitable": False,
                "complexity_score": 0.0,
                "uncertainty_level": 0.0,
                "correlation_strength": 0.0,
                "sentiment_complexity": 0.0,
                "selected_model": "classical",
                "confidence_assessment": {}
            }
            
            # Analyze news sentiment complexity
            if news_data:
                # Check for complex sentiment patterns
                total_text = " ".join(news_data)
                sentiment_indicators = [
                    "uncertainty", "volatile", "complex", "unprecedented", 
                    "conflicting", "mixed signals", "unclear", "ambiguous"
                ]
                complexity_keywords = sum(1 for indicator in sentiment_indicators if indicator in total_text.lower())
                analysis["sentiment_complexity"] = min(complexity_keywords / len(sentiment_indicators), 1.0)
                
                # Text length and diversity analysis
                text_diversity = len(set(total_text.split())) / len(total_text.split()) if total_text else 0
                analysis["text_diversity"] = text_diversity
            
            # Analyze market data complexity
            if market_data:
                # Calculate volatility and correlation patterns
                volatilities = []
                for asset, data in market_data.items():
                    if isinstance(data, dict) and "volatility" in data:
                        volatilities.append(data["volatility"])
                
                if volatilities:
                    avg_volatility = np.mean(volatilities)
                    volatility_std = np.std(volatilities) if len(volatilities) > 1 else 0
                    analysis["uncertainty_level"] = min(avg_volatility, 1.0)
                    analysis["volatility_dispersion"] = volatility_std
            
            # Calculate overall complexity score
            analysis["complexity_score"] = (
                analysis["sentiment_complexity"] * 0.4 +
                analysis["uncertainty_level"] * 0.3 +
                analysis.get("text_diversity", 0) * 0.3
            )
            
            # Determine quantum suitability
            quantum_threshold = 0.6
            analysis["quantum_suitable"] = (
                analysis["complexity_score"] > quantum_threshold and
                self.classiq_client and self.classiq_client.is_ready() and
                len(news_data) >= 3  # Minimum data for quantum advantage
            )
            
            # Model selection logic
            if analysis["quantum_suitable"]:
                if analysis["complexity_score"] > 0.8:
                    analysis["selected_model"] = "quantum"
                else:
                    analysis["selected_model"] = "hybrid"
            else:
                analysis["selected_model"] = "classical_enhanced"
            
            # Confidence assessment
            analysis["confidence_assessment"] = {
                "data_quality": min(len(news_data) / 5.0, 1.0),  # Normalized by ideal count
                "market_stability": 1.0 - analysis["uncertainty_level"],
                "model_suitability": 0.9 if analysis["quantum_suitable"] else 0.7,
                "overall_confidence": 0.0  # Will be calculated later
            }
            
            conf_metrics = analysis["confidence_assessment"]
            conf_metrics["overall_confidence"] = (
                conf_metrics["data_quality"] * 0.3 +
                conf_metrics["market_stability"] * 0.4 +
                conf_metrics["model_suitability"] * 0.3
            )
            
            logger.info(f"📊 Data analysis: {analysis['selected_model']} model selected "
                       f"(complexity: {analysis['complexity_score']:.2f}, "
                       f"confidence: {conf_metrics['overall_confidence']:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return {
                "quantum_suitable": False,
                "selected_model": "classical",
                "complexity_score": 0.0,
                "confidence_assessment": {"overall_confidence": 0.5}
            }

    async def _generate_quantum_predictions(
        self,
        news_data: List[str],
        market_data: Dict[str, Any],
        target_assets: List[str],
        prediction_horizon: int
    ) -> Dict[str, Any]:
        """
        PHASE 4.1.3: Generate quantum-enhanced predictions
        """
        try:
            logger.info("🔬 Generating quantum predictions")
            
            quantum_predictions = []
            quantum_metrics = {}
            
            for asset in target_assets:
                asset_data = market_data.get(asset, {})
                
                # Quantum sentiment analysis for this asset
                quantum_sentiment_results = []
                if self.quantum_nlp:
                    for news_text in news_data:
                        sentiment_result = await self.quantum_nlp.quantum_sentiment_classification(
                            await self.quantum_nlp.encode_text_quantum(news_text)
                        )
                        if sentiment_result:
                            quantum_sentiment_results.append(sentiment_result)
                
                # Quantum financial modeling
                if self.quantum_finance and asset_data:
                    qmc_result = await self.quantum_finance.quantum_monte_carlo_pricing(
                        spot_price=asset_data.get("current_price", 100.0),
                        volatility=asset_data.get("volatility", 0.25),
                        drift=self._calculate_quantum_drift(quantum_sentiment_results),
                        time_horizon=prediction_horizon,
                        num_paths=500
                    )
                    
                    # Convert to prediction format
                    prediction = self._quantum_result_to_prediction(
                        asset, qmc_result, quantum_sentiment_results, asset_data, prediction_horizon
                    )
                    quantum_predictions.append(prediction)
                    
                    # Collect quantum metrics
                    if qmc_result and "quantum_metrics" in qmc_result:
                        quantum_metrics[asset] = qmc_result["quantum_metrics"]
            
            return {
                "predictions": quantum_predictions,
                "quantum_metrics": quantum_metrics,
                "model_type": "quantum",
                "generation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Quantum prediction generation failed: {e}")
            return {"predictions": [], "quantum_metrics": {}, "error": str(e)}

    async def _generate_classical_predictions(
        self,
        news_data: List[str],
        market_data: Dict[str, Any],
        target_assets: List[str],
        prediction_horizon: int
    ) -> Dict[str, Any]:
        """
        PHASE 4.1.4: Generate enhanced classical predictions
        """
        try:
            logger.info("📈 Generating classical predictions")
            
            classical_predictions = []
            
            for asset in target_assets:
                asset_data = market_data.get(asset, {})
                
                # Classical sentiment analysis
                sentiment_scores = []
                for news_text in news_data:
                    # Simple sentiment scoring (can be enhanced with transformers)
                    sentiment_score = self._calculate_classical_sentiment(news_text)
                    sentiment_scores.append(sentiment_score)
                
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
                
                # Classical financial modeling
                prediction = await self._classical_financial_model(
                    asset, asset_data, avg_sentiment, prediction_horizon
                )
                classical_predictions.append(prediction)
            
            return {
                "predictions": classical_predictions,
                "model_type": "classical",
                "generation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Classical prediction generation failed: {e}")
            return {"predictions": [], "error": str(e)}

    async def _create_intelligent_ensemble(
        self,
        quantum_results: Dict[str, Any],
        classical_results: Dict[str, Any],
        data_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        PHASE 4.1.5: Create intelligent ensemble combining quantum and classical predictions
        """
        try:
            logger.info("🤝 Creating intelligent quantum-classical ensemble")
            
            quantum_preds = quantum_results.get("predictions", [])
            classical_preds = classical_results.get("predictions", [])
            
            if not quantum_preds or not classical_preds:
                logger.warning("Insufficient predictions for ensemble, using available predictions")
                return quantum_results if quantum_preds else classical_results
            
            ensemble_predictions = []
            
            # Dynamic weight calculation based on data analysis
            complexity_score = data_analysis.get("complexity_score", 0.5)
            uncertainty_level = data_analysis.get("uncertainty_level", 0.5)
            
            # Adjust weights based on conditions
            quantum_weight = self.hybrid_weight_quantum
            classical_weight = self.hybrid_weight_classical
            
            if complexity_score > 0.8:
                # High complexity favors quantum
                quantum_weight = 0.8
                classical_weight = 0.2
            elif uncertainty_level < 0.3:
                # Low uncertainty favors classical
                quantum_weight = 0.4
                classical_weight = 0.6
            
            # Ensemble predictions asset by asset
            for i, (q_pred, c_pred) in enumerate(zip(quantum_preds, classical_preds)):
                if hasattr(q_pred, 'asset') and hasattr(c_pred, 'asset'):
                    ensemble_pred = self._combine_predictions(q_pred, c_pred, quantum_weight, classical_weight)
                    ensemble_predictions.append(ensemble_pred)
            
            return {
                "predictions": ensemble_predictions,
                "model_type": "hybrid_ensemble",
                "quantum_weight": quantum_weight,
                "classical_weight": classical_weight,
                "ensemble_metrics": {
                    "component_count": 2,
                    "confidence_boost": min(quantum_weight * classical_weight * 4, 0.2),
                    "ensemble_advantage": (quantum_weight * complexity_score + 
                                        classical_weight * (1 - uncertainty_level))
                },
                "generation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return classical_results  # Fallback to classical

    def _combine_predictions(self, quantum_pred, classical_pred, q_weight, c_weight):
        """Combine quantum and classical predictions intelligently"""
        try:
            # Weighted combination of key metrics
            combined_return = (quantum_pred.expected_return * q_weight + 
                             classical_pred.expected_return * c_weight)
            
            combined_volatility = (quantum_pred.volatility * q_weight + 
                                 classical_pred.volatility * c_weight)
            
            combined_confidence = (quantum_pred.confidence * q_weight + 
                                 classical_pred.confidence * c_weight)
            
            # Combine scenarios (take best from both)
            combined_scenarios = []
            q_scenarios = quantum_pred.predicted_scenarios[:len(quantum_pred.predicted_scenarios)//2]
            c_scenarios = classical_pred.predicted_scenarios[:len(classical_pred.predicted_scenarios)//2]
            
            combined_scenarios.extend(q_scenarios)
            combined_scenarios.extend(c_scenarios)
            
            # Renumber scenario IDs
            for i, scenario in enumerate(combined_scenarios):
                scenario.scenario_id = i
            
            # Create combined prediction
            return MarketPrediction(
                asset=quantum_pred.asset,
                current_price=quantum_pred.current_price,
                expected_return=combined_return,
                volatility=combined_volatility,
                confidence=combined_confidence,
                predicted_scenarios=combined_scenarios,
                confidence_intervals=quantum_pred.confidence_intervals,  # Use quantum CI
                time_horizon_days=quantum_pred.time_horizon_days,
                sentiment_impact=(quantum_pred.sentiment_impact * q_weight + 
                               classical_pred.sentiment_impact * c_weight),
                prediction_method=f"hybrid_ensemble_q{q_weight:.1f}_c{c_weight:.1f}",
                quantum_metrics=getattr(quantum_pred, 'quantum_metrics', None)
            )
            
        except Exception as e:
            logger.error(f"Prediction combination failed: {e}")
            return quantum_pred  # Fallback to quantum prediction

    def _calculate_quantum_drift(self, quantum_sentiment_results: List[Dict]) -> float:
        """Calculate drift parameter from quantum sentiment analysis"""
        if not quantum_sentiment_results:
            return 0.05  # Default drift
        
        sentiment_values = []
        for result in quantum_sentiment_results:
            if "confidence" in result and "predicted_sentiment" in result:
                sentiment_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                sentiment_val = sentiment_map.get(result["predicted_sentiment"], 0.0)
                weighted_sentiment = sentiment_val * result["confidence"]
                sentiment_values.append(weighted_sentiment)
        
        avg_sentiment = np.mean(sentiment_values) if sentiment_values else 0.0
        return 0.05 + avg_sentiment * 0.1  # Base drift + sentiment adjustment

    def _quantum_result_to_prediction(self, asset, qmc_result, sentiment_results, asset_data, horizon):
        """Convert quantum Monte Carlo result to MarketPrediction"""
        try:
            current_price = asset_data.get("current_price", 100.0)
            expected_price = qmc_result.get("expected_price", current_price)
            expected_return = (expected_price - current_price) / current_price
            
            # Generate scenarios from quantum result
            scenarios = []
            if "scenarios" in qmc_result:
                for i, scenario_data in enumerate(qmc_result["scenarios"][:10]):  # Limit scenarios
                    scenarios.append(PriceScenario(
                        scenario_id=i,
                        price_path=[current_price, scenario_data.get("final_price", expected_price)],
                        returns_path=[0.0, expected_return],
                        volatility_path=[0.2, 0.25],
                        probability_weight=1.0/10
                    ))
            
            return MarketPrediction(
                asset=asset,
                current_price=current_price,
                expected_return=expected_return,
                volatility=qmc_result.get("volatility", 0.25),
                confidence=qmc_result.get("quantum_confidence", 0.8),
                predicted_scenarios=scenarios,
                confidence_intervals={
                    "95%": {"lower": current_price * 0.9, "upper": current_price * 1.1}
                },
                time_horizon_days=horizon,
                sentiment_impact=self._calculate_sentiment_impact(sentiment_results),
                prediction_method="quantum_enhanced",
                quantum_metrics=qmc_result.get("quantum_metrics", {})
            )
            
        except Exception as e:
            logger.error(f"Quantum result conversion failed: {e}")
            return None

    def _calculate_classical_sentiment(self, text: str) -> float:
        """Simple classical sentiment calculation"""
        positive_words = ["good", "great", "excellent", "positive", "strong", "growth", "profit", "gain"]
        negative_words = ["bad", "terrible", "negative", "weak", "loss", "decline", "fall", "drop"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)

    def _calculate_sentiment_impact(self, sentiment_results: List[Dict]) -> float:
        """Calculate overall sentiment impact from quantum sentiment results"""
        if not sentiment_results:
            return 0.0
        
        impacts = []
        for result in sentiment_results:
            sentiment_map = {"positive": 0.1, "negative": -0.1, "neutral": 0.0}
            sentiment = result.get("predicted_sentiment", "neutral")
            confidence = result.get("confidence", 0.5)
            impact = sentiment_map.get(sentiment, 0.0) * confidence
            impacts.append(impact)
        
        return np.mean(impacts) if impacts else 0.0

    async def _classical_financial_model(self, asset, asset_data, sentiment, horizon):
        """Enhanced classical financial model"""
        try:
            current_price = asset_data.get("current_price", 100.0)
            base_volatility = asset_data.get("volatility", 0.25)
            
            # Enhanced return calculation with sentiment
            base_return = 0.05 / 252 * horizon  # Annualized base return
            sentiment_adjustment = sentiment * 0.1
            expected_return = base_return + sentiment_adjustment
            
            # Generate classical scenarios
            scenarios = []
            for i in range(10):
                random_factor = np.random.normal(0, 1)
                scenario_return = expected_return + random_factor * base_volatility / np.sqrt(252) * np.sqrt(horizon)
                final_price = current_price * (1 + scenario_return)
                
                scenarios.append(PriceScenario(
                    scenario_id=i,
                    price_path=[current_price, final_price],
                    returns_path=[0.0, scenario_return],
                    volatility_path=[base_volatility, base_volatility * 1.1],
                    probability_weight=0.1
                ))
            
            return MarketPrediction(
                asset=asset,
                current_price=current_price,
                expected_return=expected_return,
                volatility=base_volatility,
                confidence=0.75,
                predicted_scenarios=scenarios,
                confidence_intervals={
                    "95%": {"lower": current_price * (1 + expected_return - 1.96 * base_volatility),
                           "upper": current_price * (1 + expected_return + 1.96 * base_volatility)}
                },
                time_horizon_days=horizon,
                sentiment_impact=sentiment,
                prediction_method="enhanced_classical"
            )
            
        except Exception as e:
            logger.error(f"Classical financial model failed: {e}")
            return None

    async def _select_optimal_predictions(self, prediction_results, data_analysis):
        """Select the optimal predictions based on analysis and performance"""
        try:
            selected_model = data_analysis.get("selected_model", "classical")
            
            if selected_model == "quantum" and "quantum" in prediction_results:
                return prediction_results["quantum"]["predictions"]
            elif selected_model == "hybrid" and "hybrid" in prediction_results:
                return prediction_results["hybrid"]["predictions"]
            else:
                return prediction_results["classical"]["predictions"]
                
        except Exception as e:
            logger.error(f"Optimal prediction selection failed: {e}")
            return prediction_results.get("classical", {}).get("predictions", [])

    def _update_performance_tracking(self, prediction_results, data_analysis):
        """Update performance tracking for continuous learning"""
        try:
            performance_entry = {
                "timestamp": datetime.now().isoformat(),
                "selected_model": data_analysis.get("selected_model"),
                "complexity_score": data_analysis.get("complexity_score"),
                "confidence": data_analysis.get("confidence_assessment", {}).get("overall_confidence"),
                "quantum_available": "quantum" in prediction_results,
                "hybrid_created": "hybrid" in prediction_results
            }
            
            self.performance_history.append(performance_entry)
            
            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                
        except Exception as e:
            logger.error(f"Performance tracking update failed: {e}")

    def _calculate_hybrid_performance_metrics(self, prediction_results):
        """Calculate comprehensive performance metrics for the hybrid approach"""
        try:
            metrics = {
                "models_used": list(prediction_results.keys()),
                "total_predictions": sum(len(result.get("predictions", [])) 
                                       for result in prediction_results.values()),
                "quantum_coverage": len(prediction_results.get("quantum", {}).get("predictions", [])),
                "classical_coverage": len(prediction_results.get("classical", {}).get("predictions", [])),
                "hybrid_advantage": 0.0
            }
            
            # Calculate hybrid advantage if applicable
            if "hybrid" in prediction_results and "classical" in prediction_results:
                hybrid_conf = np.mean([p.confidence for p in prediction_results["hybrid"]["predictions"]])
                classical_conf = np.mean([p.confidence for p in prediction_results["classical"]["predictions"]])
                metrics["hybrid_advantage"] = hybrid_conf - classical_conf
            
            return metrics
            
        except Exception as e:
            logger.error(f"Hybrid performance metrics calculation failed: {e}")
            return {"error": str(e)}

    async def _fallback_classical_prediction(self, news_data, market_data, target_assets):
        """Fallback to pure classical prediction in case of errors"""
        try:
            logger.warning("🔄 Using fallback classical prediction")
            
            classical_results = await self._generate_classical_predictions(
                news_data, market_data, target_assets, 7
            )
            
            return {
                "predictions": classical_results.get("predictions", []),
                "model_selection": "classical_fallback",
                "confidence_metrics": {"overall_confidence": 0.6},
                "fallback_used": True
            }
            
        except Exception as e:
            logger.error(f"Fallback classical prediction failed: {e}")
            return {"predictions": [], "error": str(e)}

    async def _enhance_classical_with_quantum_insights(self, classical_results, data_analysis):
        """Enhance classical predictions with quantum-inspired insights"""
        try:
            logger.info("✨ Enhancing classical predictions with quantum insights")
            
            enhanced_predictions = []
            
            for prediction in classical_results.get("predictions", []):
                # Apply quantum-inspired uncertainty bounds
                quantum_uncertainty_factor = data_analysis.get("complexity_score", 0.5) * 0.1
                
                # Enhance confidence with quantum insights
                enhanced_confidence = prediction.confidence * (1 + quantum_uncertainty_factor)
                enhanced_confidence = min(enhanced_confidence, 0.95)  # Cap at 95%
                
                # Enhance volatility with quantum effects
                enhanced_volatility = prediction.volatility * (1 + quantum_uncertainty_factor)
                
                # Create enhanced prediction
                enhanced_pred = MarketPrediction(
                    asset=prediction.asset,
                    current_price=prediction.current_price,
                    expected_return=prediction.expected_return,
                    volatility=enhanced_volatility,
                    confidence=enhanced_confidence,
                    predicted_scenarios=prediction.predicted_scenarios,
                    confidence_intervals=prediction.confidence_intervals,
                    time_horizon_days=prediction.time_horizon_days,
                    sentiment_impact=prediction.sentiment_impact,
                    prediction_method="classical_quantum_enhanced"
                )
                
                enhanced_predictions.append(enhanced_pred)
            
            return {
                "predictions": enhanced_predictions,
                "model_type": "classical_quantum_enhanced",
                "enhancement_factor": quantum_uncertainty_factor,
                "generation_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Classical enhancement with quantum insights failed: {e}")
            return classical_results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary of the hybrid pipeline"""
        try:
            if not self.performance_history:
                return {"status": "No performance data available"}
            
            recent_history = self.performance_history[-20:]  # Last 20 predictions
            
            model_usage = {}
            for entry in recent_history:
                model = entry.get("selected_model", "unknown")
                model_usage[model] = model_usage.get(model, 0) + 1
            
            avg_confidence = np.mean([entry.get("confidence", 0) for entry in recent_history])
            avg_complexity = np.mean([entry.get("complexity_score", 0) for entry in recent_history])
            
            quantum_usage_rate = sum(1 for entry in recent_history 
                                   if entry.get("quantum_available", False)) / len(recent_history)
            
            return {
                "total_predictions": len(self.performance_history),
                "recent_performance": {
                    "avg_confidence": avg_confidence,
                    "avg_complexity": avg_complexity,
                    "quantum_usage_rate": quantum_usage_rate,
                    "model_usage_distribution": model_usage
                },
                "quantum_availability": quantum_usage_rate,
                "hybrid_effectiveness": avg_confidence if avg_confidence > 0.7 else 0.6
            }
            
        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {"error": str(e)}
