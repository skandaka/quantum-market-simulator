# backend/app/ml/market_predictor.py

import logging
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

from app.models.schemas import SentimentType, MarketPrediction

logger = logging.getLogger(__name__)


@dataclass
class PredictionConstraints:
    """Enhanced constraints for market predictions based on sentiment"""
    min_return: float
    max_return: float
    confidence_multiplier: float
    volatility_multiplier: float
    crisis_override: bool = False


# Enhanced sentiment-based constraints with stronger negative impacts
SENTIMENT_CONSTRAINTS = {
    SentimentType.VERY_NEGATIVE: PredictionConstraints(
        min_return=-0.25,  # -25% minimum for extreme negative news
        max_return=-0.08,  # -8% maximum (must be strongly negative)
        confidence_multiplier=0.95,
        volatility_multiplier=3.0,
        crisis_override=True
    ),
    SentimentType.NEGATIVE: PredictionConstraints(
        min_return=-0.15,  # -15% minimum
        max_return=-0.03,  # -3% maximum
        confidence_multiplier=0.90,
        volatility_multiplier=2.0
    ),
    SentimentType.NEUTRAL: PredictionConstraints(
        min_return=-0.02,
        max_return=0.02,
        confidence_multiplier=0.8,
        volatility_multiplier=1.0
    ),
    SentimentType.POSITIVE: PredictionConstraints(
        min_return=0.01,
        max_return=0.08,
        confidence_multiplier=0.85,
        volatility_multiplier=1.3
    ),
    SentimentType.VERY_POSITIVE: PredictionConstraints(
        min_return=0.03,
        max_return=0.15,
        confidence_multiplier=0.92,
        volatility_multiplier=1.8
    )
}


class EnhancedMarketPredictor:
    """Market predictor with massively enhanced accuracy and impact"""

    def __init__(self, base_predictor=None):
        self.base_predictor = base_predictor
        self.explanation_builder = PredictionExplanationBuilder()

    async def predict_with_constraints(
            self,
            sentiment_results: List[Any],
            market_data: Dict[str, Any],
            asset: str
    ) -> Dict[str, Any]:
        """Generate prediction with enhanced sentiment-based constraints"""

        # Calculate aggregate sentiment with crisis detection
        agg_sentiment = self._calculate_aggregate_sentiment_enhanced(sentiment_results)

        # Get base prediction if available
        if self.base_predictor:
            base_prediction = await self.base_predictor.predict(
                sentiment_results, market_data, asset
            )
        else:
            base_prediction = self._generate_base_prediction(agg_sentiment, market_data)

        # Apply enhanced constraints with crisis overrides
        constrained_prediction = self._apply_enhanced_constraints(
            base_prediction, agg_sentiment
        )

        # Add detailed explanation
        explanation = self.explanation_builder.build_enhanced_explanation(
            sentiment_results,
            agg_sentiment,
            base_prediction,
            constrained_prediction
        )

        constrained_prediction["explanation"] = explanation

        # Add warnings for extreme predictions
        warnings = self._enhanced_sanity_check(
            base_prediction, constrained_prediction, agg_sentiment
        )
        if warnings:
            constrained_prediction["warnings"] = warnings

        return constrained_prediction

    def _calculate_aggregate_sentiment_enhanced(
            self,
            sentiment_results: List[Any]
    ) -> Dict[str, Any]:
        """Enhanced aggregate sentiment calculation with crisis detection"""

        if not sentiment_results:
            return {
                "type": SentimentType.NEUTRAL,
                "score": 0.0,
                "confidence": 0.5,
                "is_crisis": False,
                "crisis_severity": 0.0,
                "num_crisis_indicators": 0
            }

        total_weight = 0.0
        weighted_score = 0.0
        crisis_detected = False
        max_crisis_severity = 0.0
        total_crisis_indicators = 0

        sentiment_counts = {s: 0 for s in SentimentType}

        for result in sentiment_results:
            weight = result.confidence
            sentiment_counts[result.sentiment] += 1

            # Enhanced crisis detection
            if hasattr(result, 'crisis_indicators'):
                crisis = result.crisis_indicators
                if crisis and crisis.get("is_crisis"):
                    crisis_detected = True
                    max_crisis_severity = max(max_crisis_severity, crisis.get("severity", 0.8))
                    total_crisis_indicators += crisis.get("num_triggers", 1)

            # Enhanced score mapping with stronger negative weights
            score = {
                SentimentType.VERY_NEGATIVE: -1.0,
                SentimentType.NEGATIVE: -0.6,
                SentimentType.NEUTRAL: 0.0,
                SentimentType.POSITIVE: 0.5,
                SentimentType.VERY_POSITIVE: 0.8
            }.get(result.sentiment, 0.0)

            # Amplify negative scores if multiple negative sentiments
            if result.sentiment in [SentimentType.VERY_NEGATIVE, SentimentType.NEGATIVE]:
                score *= 1.2  # Amplify negative impact

            weighted_score += score * weight
            total_weight += weight

        avg_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Enhanced sentiment determination with bias toward negative
        dominant_sentiment = SentimentType.NEUTRAL
        if avg_score <= -0.6 or (crisis_detected and max_crisis_severity >= 0.8):
            dominant_sentiment = SentimentType.VERY_NEGATIVE
        elif avg_score <= -0.2 or (crisis_detected and max_crisis_severity >= 0.6):
            dominant_sentiment = SentimentType.NEGATIVE
        elif avg_score >= 0.6:
            dominant_sentiment = SentimentType.VERY_POSITIVE
        elif avg_score >= 0.2:
            dominant_sentiment = SentimentType.POSITIVE

        # Override to very negative if multiple crisis indicators
        if total_crisis_indicators >= 3:
            dominant_sentiment = SentimentType.VERY_NEGATIVE
            avg_score = min(avg_score, -0.8)
        elif total_crisis_indicators >= 2:
            if dominant_sentiment not in [SentimentType.VERY_NEGATIVE]:
                dominant_sentiment = SentimentType.NEGATIVE
            avg_score = min(avg_score, -0.5)

        return {
            "type": dominant_sentiment,
            "score": avg_score,
            "confidence": min(0.98, total_weight / len(sentiment_results) * 1.1),
            "is_crisis": crisis_detected,
            "crisis_severity": max_crisis_severity,
            "num_crisis_indicators": total_crisis_indicators,
            "sentiment_distribution": sentiment_counts
        }

    def _apply_enhanced_constraints(
            self,
            prediction: Dict[str, Any],
            agg_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply enhanced sentiment-based constraints with crisis overrides"""

        constraints = SENTIMENT_CONSTRAINTS[agg_sentiment["type"]]

        # Get original return
        original_return = prediction.get("expected_return", 0.0)

        # Crisis override - apply severe negative impact
        if agg_sentiment["is_crisis"]:
            crisis_severity = agg_sentiment["crisis_severity"]
            num_indicators = agg_sentiment["num_crisis_indicators"]

            # Calculate crisis-adjusted return
            if num_indicators >= 3:
                # Multiple severe crisis indicators
                crisis_return = -0.20 - (crisis_severity * 0.10)  # -20% to -30%
            elif num_indicators >= 2:
                # Multiple crisis indicators
                crisis_return = -0.12 - (crisis_severity * 0.08)  # -12% to -20%
            else:
                # Single crisis indicator
                crisis_return = -0.08 - (crisis_severity * 0.07)  # -8% to -15%

            constrained_return = min(original_return, crisis_return)

            # Increase volatility significantly
            volatility_multiplier = 2.5 + (crisis_severity * 1.5)
            confidence_multiplier = 0.95  # High confidence in crisis impact

        else:
            # Apply normal constraints with enhanced negative bias
            if agg_sentiment["type"] in [SentimentType.VERY_NEGATIVE, SentimentType.NEGATIVE]:
                # For negative sentiment, ensure strong negative return
                min_negative = constraints.min_return
                max_negative = constraints.max_return

                # Calculate based on sentiment score
                sentiment_score = agg_sentiment["score"]
                range_size = max_negative - min_negative

                # Map sentiment score to return (more negative score = more negative return)
                constrained_return = max_negative + (1 + sentiment_score) * range_size
                constrained_return = np.clip(constrained_return, min_negative, max_negative)

            else:
                # For positive/neutral sentiment, use normal constraints
                constrained_return = np.clip(
                    original_return,
                    constraints.min_return,
                    constraints.max_return
                )

            volatility_multiplier = constraints.volatility_multiplier
            confidence_multiplier = constraints.confidence_multiplier

        # Build constrained prediction
        constrained_prediction = prediction.copy()
        constrained_prediction["expected_return"] = constrained_return
        constrained_prediction["volatility"] = prediction.get("volatility", 0.25) * volatility_multiplier
        constrained_prediction["confidence"] = min(
            0.98,
            prediction.get("confidence", 0.7) * confidence_multiplier
        )

        # Add sentiment metadata
        constrained_prediction["sentiment_type"] = agg_sentiment["type"].value
        constrained_prediction["sentiment_score"] = agg_sentiment["score"]
        constrained_prediction["is_crisis"] = agg_sentiment["is_crisis"]

        if agg_sentiment["is_crisis"]:
            constrained_prediction["crisis_severity"] = agg_sentiment["crisis_severity"]
            constrained_prediction["crisis_indicators_count"] = agg_sentiment["num_crisis_indicators"]

        return constrained_prediction

    def _generate_base_prediction(
            self,
            agg_sentiment: Dict[str, Any],
            market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate base prediction when no base predictor is available"""

        # Base prediction from sentiment
        sentiment_to_return = {
            SentimentType.VERY_NEGATIVE: -0.15,
            SentimentType.NEGATIVE: -0.08,
            SentimentType.NEUTRAL: 0.0,
            SentimentType.POSITIVE: 0.05,
            SentimentType.VERY_POSITIVE: 0.10
        }

        base_return = sentiment_to_return.get(agg_sentiment["type"], 0.0)

        # Adjust for crisis
        if agg_sentiment["is_crisis"]:
            base_return = min(base_return, -0.12)

        return {
            "expected_return": base_return,
            "volatility": 0.25,
            "confidence": 0.7,
            "method": "sentiment_based"
        }

    def _enhanced_sanity_check(
            self,
            base_prediction: Dict[str, Any],
            final_prediction: Dict[str, Any],
            agg_sentiment: Dict[str, Any]
    ) -> List[str]:
        """Enhanced sanity checks for predictions"""

        warnings = []

        # Check 1: Extreme prediction warning
        if abs(final_prediction["expected_return"]) > 0.20:
            warnings.append(
                f"Extreme price movement predicted: {final_prediction['expected_return'] * 100:.1f}%"
            )

        # Check 2: Crisis volatility warning
        if agg_sentiment["is_crisis"] and final_prediction["volatility"] > 0.5:
            warnings.append(
                "Crisis detected - expect extreme volatility and uncertain outcomes"
            )

        # Check 3: Multiple crisis indicators
        if agg_sentiment.get("num_crisis_indicators", 0) >= 3:
            warnings.append(
                f"Multiple crisis indicators ({agg_sentiment['num_crisis_indicators']}) detected - prediction confidence may be affected"
            )

        # Check 4: Large adjustment from base
        if base_prediction:
            return_diff = abs(
                final_prediction["expected_return"] - base_prediction.get("expected_return", 0)
            )
            if return_diff > 0.10:
                warnings.append(
                    f"Large sentiment-based adjustment applied: {return_diff * 100:.1f}% change"
                )

        return warnings


class PredictionExplanationBuilder:
    """Build enhanced human-readable explanations for predictions"""

    def build_enhanced_explanation(
            self,
            sentiment_results: List[Any],
            agg_sentiment: Dict[str, Any],
            base_prediction: Dict[str, Any],
            final_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build comprehensive enhanced explanation"""

        explanation = {
            "summary": self._build_enhanced_summary(agg_sentiment, final_prediction),
            "sentiment_analysis": self._explain_sentiment_enhanced(sentiment_results, agg_sentiment),
            "prediction_factors": self._explain_factors_enhanced(base_prediction, final_prediction, agg_sentiment),
            "confidence_reasoning": self._explain_confidence_enhanced(final_prediction, agg_sentiment),
            "key_drivers": self._identify_key_drivers_enhanced(sentiment_results, agg_sentiment),
            "risk_assessment": self._assess_risks(agg_sentiment, final_prediction)
        }

        return explanation

    def _build_enhanced_summary(
            self,
            agg_sentiment: Dict[str, Any],
            prediction: Dict[str, Any]
    ) -> str:
        """Build enhanced summary explanation"""

        sentiment_str = agg_sentiment["type"].value.replace("_", " ").title()
        direction = "decrease" if prediction["expected_return"] < 0 else "increase"
        magnitude = abs(prediction["expected_return"]) * 100

        # Enhanced summary with crisis information
        if agg_sentiment["is_crisis"]:
            severity_str = "severe" if agg_sentiment["crisis_severity"] >= 0.8 else "significant"
            summary = f"⚠️ CRISIS ALERT: {severity_str.title()} negative news detected. "
            summary += f"Based on {agg_sentiment['num_crisis_indicators']} crisis indicators, "
            summary += f"we predict the stock will {direction} by approximately {magnitude:.1f}% "
            summary += f"with {prediction['confidence'] * 100:.0f}% confidence. "
            summary += "Expect extreme volatility and potential for larger moves."
        else:
            summary = f"Based on {sentiment_str} sentiment analysis, "
            summary += f"we predict the stock will {direction} by approximately {magnitude:.1f}% "
            summary += f"with {prediction['confidence'] * 100:.0f}% confidence."

            if magnitude > 10:
                summary += " This represents a significant price movement."

        return summary

    def _explain_sentiment_enhanced(
            self,
            sentiment_results: List[Any],
            agg_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced sentiment analysis explanation"""

        keywords = []
        crisis_keywords = []

        for result in sentiment_results:
            if hasattr(result, 'market_impact_keywords'):
                keywords.extend(result.market_impact_keywords)
            if hasattr(result, 'crisis_indicators'):
                crisis = result.crisis_indicators
                if crisis and crisis.get("keywords"):
                    crisis_keywords.extend(crisis["keywords"])

        explanation = {
            "dominant_sentiment": agg_sentiment["type"].value,
            "sentiment_score": f"{agg_sentiment['score']:.2f} (-1 to +1 scale)",
            "confidence_level": f"{agg_sentiment['confidence'] * 100:.0f}%",
            "key_keywords": list(set(keywords))[:15],
            "news_items_analyzed": len(sentiment_results),
            "sentiment_distribution": {
                k.value: v for k, v in agg_sentiment["sentiment_distribution"].items()
            }
        }

        if crisis_keywords:
            explanation["crisis_triggers"] = list(set(crisis_keywords))[:10]
            explanation["crisis_severity"] = f"{agg_sentiment['crisis_severity'] * 100:.0f}%"

        return explanation

    def _explain_factors_enhanced(
            self,
            base_prediction: Dict[str, Any],
            final_prediction: Dict[str, Any],
            agg_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Explain enhanced prediction factors"""

        factors = {
            "sentiment_impact": "Primary driver",
            "expected_return": f"{final_prediction['expected_return'] * 100:.2f}%",
            "volatility": f"{final_prediction['volatility'] * 100:.1f}%",
            "prediction_method": final_prediction.get("method", "enhanced_sentiment_based")
        }

        if agg_sentiment["is_crisis"]:
            factors[
                "crisis_adjustment"] = f"Applied {agg_sentiment['crisis_severity'] * 100:.0f}% crisis severity adjustment"
            factors["crisis_indicators"] = f"{agg_sentiment['num_crisis_indicators']} crisis indicators detected"

        if base_prediction:
            adjustment = abs(final_prediction["expected_return"] - base_prediction.get("expected_return", 0))
            factors["sentiment_adjustment"] = f"{adjustment * 100:.2f}% adjustment from base model"

        return factors

    def _explain_confidence_enhanced(
            self,
            prediction: Dict[str, Any],
            agg_sentiment: Dict[str, Any]
    ) -> str:
        """Enhanced confidence explanation"""

        confidence = prediction["confidence"]

        if confidence >= 0.9:
            confidence_str = "Very high confidence"
            reason = "Strong consensus in sentiment analysis"
        elif confidence >= 0.8:
            confidence_str = "High confidence"
            reason = "Clear sentiment signals detected"
        elif confidence >= 0.7:
            confidence_str = "Moderate confidence"
            reason = "Mixed but leaning sentiment signals"
        else:
            confidence_str = "Low confidence"
            reason = "Conflicting or weak sentiment signals"

        if agg_sentiment["is_crisis"]:
            reason += f" with {agg_sentiment['num_crisis_indicators']} crisis indicators reinforcing the prediction"

        return f"{confidence_str} ({confidence * 100:.0f}%) - {reason}"

    def _identify_key_drivers_enhanced(
            self,
            sentiment_results: List[Any],
            agg_sentiment: Dict[str, Any]
    ) -> List[str]:
        """Identify enhanced key factors driving the prediction"""

        drivers = []

        # Crisis drivers (highest priority)
        if agg_sentiment["is_crisis"]:
            severity_pct = agg_sentiment["crisis_severity"] * 100
            drivers.append(f"🚨 Crisis detected with {severity_pct:.0f}% severity level")

            if agg_sentiment["num_crisis_indicators"] >= 3:
                drivers.append(
                    f"⚠️ Multiple crisis indicators ({agg_sentiment['num_crisis_indicators']}) compounding negative impact")

        # Sentiment consensus
        sentiment_dist = agg_sentiment["sentiment_distribution"]
        very_negative_count = sentiment_dist.get(SentimentType.VERY_NEGATIVE, 0)
        negative_count = sentiment_dist.get(SentimentType.NEGATIVE, 0)

        if very_negative_count >= 2:
            drivers.append(f"📉 Multiple very negative sentiment signals ({very_negative_count})")
        elif very_negative_count + negative_count >= 3:
            drivers.append(
                f"📊 Strong negative sentiment consensus ({very_negative_count + negative_count} negative items)")

        # Individual high-impact items
        for result in sentiment_results:
            if result.confidence > 0.9:
                sentiment_type = result.sentiment.value.replace("_", " ")
                drivers.append(f"High confidence {sentiment_type} signal ({result.confidence * 100:.0f}%)")

            if hasattr(result, 'crisis_indicators'):
                crisis = result.crisis_indicators
                if crisis and crisis.get("is_crisis") and crisis.get("keywords"):
                    keywords_str = ", ".join(crisis["keywords"][:3])
                    drivers.append(f"Critical keywords detected: {keywords_str}")

        return drivers[:8]  # Top 8 drivers

    def _assess_risks(
            self,
            agg_sentiment: Dict[str, Any],
            prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks associated with the prediction"""

        risks = {
            "volatility_risk": "Low",
            "confidence_risk": "Low",
            "black_swan_risk": "Low"
        }

        # Volatility risk
        if prediction["volatility"] > 0.5:
            risks["volatility_risk"] = "Very High"
        elif prediction["volatility"] > 0.35:
            risks["volatility_risk"] = "High"
        elif prediction["volatility"] > 0.25:
            risks["volatility_risk"] = "Moderate"

        # Confidence risk
        if prediction["confidence"] < 0.6:
            risks["confidence_risk"] = "High"
        elif prediction["confidence"] < 0.75:
            risks["confidence_risk"] = "Moderate"

        # Black swan risk (extreme events)
        if agg_sentiment["is_crisis"] and agg_sentiment["crisis_severity"] >= 0.8:
            risks["black_swan_risk"] = "Very High"
        elif agg_sentiment["is_crisis"]:
            risks["black_swan_risk"] = "High"
        elif abs(prediction["expected_return"]) > 0.15:
            risks["black_swan_risk"] = "Moderate"

        return risks


# Export the enhanced predictor
MarketPredictor = EnhancedMarketPredictor