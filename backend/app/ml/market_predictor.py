# backend/app/ml/market_predictor.py

import logging
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

from app.models.schemas import SentimentType, MarketPrediction

logger = logging.getLogger(__name__)


@dataclass
class PredictionConstraints:
    """Constraints for market predictions based on sentiment"""
    min_return: float
    max_return: float
    confidence_multiplier: float
    volatility_multiplier: float


# Sentiment-based constraints
SENTIMENT_CONSTRAINTS = {
    SentimentType.VERY_NEGATIVE: PredictionConstraints(
        min_return=-0.10,  # -10% minimum
        max_return=-0.02,  # -2% maximum (must be negative)
        confidence_multiplier=0.9,
        volatility_multiplier=2.0
    ),
    SentimentType.NEGATIVE: PredictionConstraints(
        min_return=-0.05,
        max_return=0.0,
        confidence_multiplier=0.85,
        volatility_multiplier=1.5
    ),
    SentimentType.NEUTRAL: PredictionConstraints(
        min_return=-0.02,
        max_return=0.02,
        confidence_multiplier=0.8,
        volatility_multiplier=1.0
    ),
    SentimentType.POSITIVE: PredictionConstraints(
        min_return=0.0,
        max_return=0.05,
        confidence_multiplier=0.85,
        volatility_multiplier=1.2
    ),
    SentimentType.VERY_POSITIVE: PredictionConstraints(
        min_return=0.02,
        max_return=0.10,
        confidence_multiplier=0.9,
        volatility_multiplier=1.5
    )
}


class EnhancedMarketPredictor:
    """Market predictor with sanity checks and constraints"""

    def __init__(self, base_predictor):
        self.base_predictor = base_predictor
        self.explanation_builder = PredictionExplanationBuilder()

    async def predict_with_constraints(
            self,
            sentiment_results: List[Any],
            market_data: Dict[str, Any],
            asset: str
    ) -> Dict[str, Any]:
        """Generate prediction with sentiment-based constraints"""

        # Get base prediction from original model
        base_prediction = await self.base_predictor.predict(
            sentiment_results, market_data, asset
        )

        # Calculate aggregate sentiment
        agg_sentiment = self._calculate_aggregate_sentiment(sentiment_results)

        # Apply constraints
        constrained_prediction = self._apply_constraints(
            base_prediction, agg_sentiment
        )

        # Add explanation
        explanation = self.explanation_builder.build_explanation(
            sentiment_results,
            agg_sentiment,
            base_prediction,
            constrained_prediction
        )

        constrained_prediction["explanation"] = explanation

        # Add sanity check warnings
        warnings = self._sanity_check(
            base_prediction, constrained_prediction, agg_sentiment
        )
        if warnings:
            constrained_prediction["warnings"] = warnings

        return constrained_prediction

    def _calculate_aggregate_sentiment(
            self,
            sentiment_results: List[Any]
    ) -> Dict[str, Any]:
        """Calculate weighted aggregate sentiment"""

        if not sentiment_results:
            return {
                "type": SentimentType.NEUTRAL,
                "score": 0.0,
                "confidence": 0.5,
                "is_crisis": False
            }

        total_weight = 0.0
        weighted_score = 0.0
        crisis_detected = False

        sentiment_counts = {s: 0 for s in SentimentType}

        for result in sentiment_results:
            weight = result.confidence
            sentiment_counts[result.sentiment] += 1

            # Check for crisis
            if hasattr(result, 'crisis_indicators'):
                if result.crisis_indicators and result.crisis_indicators.get("is_crisis"):
                    crisis_detected = True

            # Score mapping
            score = {
                SentimentType.VERY_NEGATIVE: -1.0,
                SentimentType.NEGATIVE: -0.5,
                SentimentType.NEUTRAL: 0.0,
                SentimentType.POSITIVE: 0.5,
                SentimentType.VERY_POSITIVE: 1.0
            }.get(result.sentiment, 0.0)

            weighted_score += score * weight
            total_weight += weight

        avg_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Determine dominant sentiment
        dominant_sentiment = SentimentType.NEUTRAL
        if avg_score <= -0.7:
            dominant_sentiment = SentimentType.VERY_NEGATIVE
        elif avg_score <= -0.3:
            dominant_sentiment = SentimentType.NEGATIVE
        elif avg_score >= 0.7:
            dominant_sentiment = SentimentType.VERY_POSITIVE
        elif avg_score >= 0.3:
            dominant_sentiment = SentimentType.POSITIVE

        return {
            "type": dominant_sentiment,
            "score": avg_score,
            "confidence": total_weight / len(sentiment_results),
            "is_crisis": crisis_detected,
            "sentiment_distribution": sentiment_counts
        }

    def _apply_constraints(
            self,
            prediction: Dict[str, Any],
            agg_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply sentiment-based constraints to prediction"""

        constraints = SENTIMENT_CONSTRAINTS[agg_sentiment["type"]]

        # Constrain expected return
        original_return = prediction.get("expected_return", 0.0)
        constrained_return = np.clip(
            original_return,
            constraints.min_return,
            constraints.max_return
        )

        # Crisis override
        if agg_sentiment["is_crisis"]:
            # Force negative prediction for crisis
            constrained_return = min(constrained_return, -0.03)

        # Adjust confidence based on constraint application
        confidence_adjustment = 1.0
        if abs(original_return - constrained_return) > 0.01:
            # Significant adjustment needed - reduce confidence
            confidence_adjustment = 0.8

        # Update prediction
        constrained_pred = prediction.copy()
        constrained_pred["expected_return"] = constrained_return
        constrained_pred["volatility"] = prediction.get("volatility", 0.02) * constraints.volatility_multiplier
        constrained_pred["confidence"] = prediction.get("confidence",
                                                        0.7) * constraints.confidence_multiplier * confidence_adjustment
        constrained_pred["sentiment_constraint_applied"] = True
        constrained_pred["original_return"] = original_return

        return constrained_pred

    def _sanity_check(
            self,
            base_prediction: Dict[str, Any],
            constrained_prediction: Dict[str, Any],
            agg_sentiment: Dict[str, Any]
    ) -> List[str]:
        """Perform sanity checks on predictions"""

        warnings = []

        # Check 1: Severe sentiment mismatch
        if agg_sentiment["score"] <= -0.7 and constrained_prediction["expected_return"] > 0:
            warnings.append(
                "Warning: Very negative sentiment but positive prediction - likely error"
            )

        if agg_sentiment["score"] >= 0.7 and constrained_prediction["expected_return"] < 0:
            warnings.append(
                "Warning: Very positive sentiment but negative prediction - unusual"
            )

        # Check 2: Large constraint adjustment
        return_diff = abs(
            base_prediction.get("expected_return", 0) -
            constrained_prediction["expected_return"]
        )
        if return_diff > 0.05:
            warnings.append(
                f"Large adjustment applied: {return_diff * 100:.1f}% change from base model"
            )

        # Check 3: Crisis situation
        if agg_sentiment["is_crisis"]:
            warnings.append(
                "Crisis situation detected - predictions may be volatile"
            )

        return warnings


class PredictionExplanationBuilder:
    """Build human-readable explanations for predictions"""

    def build_explanation(
            self,
            sentiment_results: List[Any],
            agg_sentiment: Dict[str, Any],
            base_prediction: Dict[str, Any],
            final_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build comprehensive explanation"""

        explanation = {
            "summary": self._build_summary(agg_sentiment, final_prediction),
            "sentiment_analysis": self._explain_sentiment(sentiment_results, agg_sentiment),
            "prediction_factors": self._explain_factors(base_prediction, final_prediction),
            "confidence_reasoning": self._explain_confidence(final_prediction),
            "key_drivers": self._identify_key_drivers(sentiment_results)
        }

        return explanation

    def _build_summary(
            self,
            agg_sentiment: Dict[str, Any],
            prediction: Dict[str, Any]
    ) -> str:
        """Build summary explanation"""

        sentiment_str = agg_sentiment["type"].value.replace("_", " ").title()
        direction = "increase" if prediction["expected_return"] > 0 else "decrease"
        magnitude = abs(prediction["expected_return"]) * 100

        summary = f"Based on {sentiment_str} sentiment, we predict the stock will {direction} "
        summary += f"by approximately {magnitude:.2f}% with {prediction['confidence'] * 100:.0f}% confidence."

        if agg_sentiment["is_crisis"]:
            summary += " Crisis indicators detected - expect high volatility."

        return summary

    def _explain_sentiment(
            self,
            sentiment_results: List[Any],
            agg_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Explain sentiment analysis"""

        keywords = []
        for result in sentiment_results:
            if hasattr(result, 'market_impact_keywords'):
                keywords.extend(result.market_impact_keywords)

        return {
            "dominant_sentiment": agg_sentiment["type"].value,
            "sentiment_score": f"{agg_sentiment['score']:.2f} (-1 to +1 scale)",
            "key_keywords": list(set(keywords))[:10],
            "news_items_analyzed": len(sentiment_results)
        }

    def _identify_key_drivers(
            self,
            sentiment_results: List[Any]
    ) -> List[str]:
        """Identify key factors driving the prediction"""

        drivers = []

        for result in sentiment_results:
            if hasattr(result, 'crisis_indicators'):
                crisis = result.crisis_indicators
                if crisis and crisis.get("is_crisis"):
                    drivers.append(f"Crisis detected: {', '.join(crisis['triggered_keywords'][:3])}")

            if result.confidence > 0.9:
                drivers.append(f"High confidence {result.sentiment.value} sentiment")

        return drivers[:5]  # Top 5 drivers