"""LLM baseline model for comparison"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
import openai
from transformers import pipeline
import torch

from app.config import settings
from app.models.schemas import NewsInput, SentimentType

logger = logging.getLogger(__name__)


class LLMBaseline:
    """LLM-based baseline for sentiment analysis and market prediction"""

    def __init__(self):
        self.openai_client = None
        self.local_model = None
        self.initialized = False

    async def initialize(self):
        """Initialize LLM models"""

        # Setup OpenAI if available
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            self.openai_client = openai
            logger.info("OpenAI client initialized")

        # Load local model as fallback
        try:
            self.local_model = pipeline(
                "text-generation",
                model="gpt2",  # Small model for hackathon
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Local LLM model loaded")
        except Exception as e:
            logger.warning(f"Failed to load local LLM: {e}")

        self.initialized = True

    async def analyze_sentiment_llm(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using LLM"""

        if self.openai_client:
            return await self._openai_sentiment(text)
        elif self.local_model:
            return await self._local_model_sentiment(text)
        else:
            # Fallback to rule-based
            return self._rule_based_sentiment(text)

    async def _openai_sentiment(self, text: str) -> Dict[str, Any]:
        """Use OpenAI for sentiment analysis"""

        try:
            prompt = f"""Analyze the financial sentiment of this text. 
            Return a JSON with:
            - sentiment: one of [very_negative, negative, neutral, positive, very_positive]
            - confidence: float between 0 and 1
            - key_factors: list of key phrases affecting sentiment
            - market_impact: expected market impact [high, medium, low]

            Text: {text[:500]}

            JSON Response:"""

            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            # Parse response
            result_text = response.choices[0].message.content

            # Simple parsing - in production use proper JSON parsing
            sentiment = "neutral"
            confidence = 0.7

            if "very_positive" in result_text.lower():
                sentiment = SentimentType.VERY_POSITIVE
                confidence = 0.9
            elif "positive" in result_text.lower():
                sentiment = SentimentType.POSITIVE
                confidence = 0.8
            elif "very_negative" in result_text.lower():
                sentiment = SentimentType.VERY_NEGATIVE
                confidence = 0.9
            elif "negative" in result_text.lower():
                sentiment = SentimentType.NEGATIVE
                confidence = 0.8

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "method": "openai_gpt"
            }

        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._rule_based_sentiment(text)

    async def _local_model_sentiment(self, text: str) -> Dict[str, Any]:
        """Use local model for sentiment"""

        # For hackathon, use simple approach
        # In production, fine-tune model for financial sentiment

        return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment as fallback"""

        text_lower = text.lower()

        # Define sentiment words
        very_positive_words = ["surge", "soar", "breakthrough", "record high"]
        positive_words = ["rise", "gain", "beat", "profit", "growth"]
        negative_words = ["fall", "drop", "loss", "decline", "miss"]
        very_negative_words = ["crash", "plunge", "collapse", "bankruptcy"]

        # Count occurrences
        very_pos_count = sum(1 for word in very_positive_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        very_neg_count = sum(1 for word in very_negative_words if word in text_lower)

        # Determine sentiment
        if very_pos_count > 0:
            sentiment = SentimentType.VERY_POSITIVE
            confidence = min(0.9, 0.7 + very_pos_count * 0.1)
        elif very_neg_count > 0:
            sentiment = SentimentType.VERY_NEGATIVE
            confidence = min(0.9, 0.7 + very_neg_count * 0.1)
        elif pos_count > neg_count:
            sentiment = SentimentType.POSITIVE
            confidence = min(0.85, 0.6 + (pos_count - neg_count) * 0.05)
        elif neg_count > pos_count:
            sentiment = SentimentType.NEGATIVE
            confidence = min(0.85, 0.6 + (neg_count - pos_count) * 0.05)
        else:
            sentiment = SentimentType.NEUTRAL
            confidence = 0.6

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "method": "rule_based"
        }

    async def predict_market_impact(
            self,
            news: str,
            asset: str,
            current_price: float
    ) -> Dict[str, Any]:
        """Predict market impact using LLM"""

        if self.openai_client:
            return await self._openai_market_prediction(news, asset, current_price)
        else:
            return self._simple_market_prediction(news, asset, current_price)

    async def _openai_market_prediction(
            self,
            news: str,
            asset: str,
            current_price: float
    ) -> Dict[str, Any]:
        """Use OpenAI for market prediction"""

        try:
            prompt = f"""Given this news about {asset}, predict the likely market impact.
            Current price: ${current_price}

            News: {news[:500]}

            Provide:
            1. Direction: up/down/neutral
            2. Magnitude: percentage change expected
            3. Confidence: how certain (0-1)
            4. Time frame: when impact will occur

            Be realistic and conservative in predictions."""

            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial market analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )

            # Parse response (simplified for hackathon)
            result_text = response.choices[0].message.content.lower()

            direction = "neutral"
            if "up" in result_text or "increase" in result_text:
                direction = "up"
            elif "down" in result_text or "decrease" in result_text:
                direction = "down"

            # Extract magnitude (look for percentages)
            import re
            percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', result_text)
            magnitude = float(percentages[0]) / 100 if percentages else 0.02

            return {
                "direction": direction,
                "magnitude": magnitude,
                "confidence": 0.7,
                "timeframe": "1-3 days",
                "method": "llm_prediction"
            }

        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            return self._simple_market_prediction(news, asset, current_price)

    def _simple_market_prediction(
            self,
            news: str,
            asset: str,
            current_price: float
    ) -> Dict[str, Any]:
        """Simple market prediction as fallback"""

        sentiment_result = self._rule_based_sentiment(news)
        sentiment = sentiment_result["sentiment"]
        confidence = sentiment_result["confidence"]

        # Map sentiment to market impact
        impact_map = {
            SentimentType.VERY_POSITIVE: ("up", 0.03),
            SentimentType.POSITIVE: ("up", 0.015),
            SentimentType.NEUTRAL: ("neutral", 0.0),
            SentimentType.NEGATIVE: ("down", 0.015),
            SentimentType.VERY_NEGATIVE: ("down", 0.03)
        }

        direction, magnitude = impact_map.get(sentiment, ("neutral", 0.0))

        return {
            "direction": direction,
            "magnitude": magnitude,
            "confidence": confidence * 0.8,  # Reduce confidence
            "timeframe": "1-2 days",
            "method": "rule_based_prediction"
        }