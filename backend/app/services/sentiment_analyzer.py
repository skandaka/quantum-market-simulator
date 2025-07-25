"""Sentiment analysis service with quantum enhancement"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import spacy

from app.models.schemas import SentimentAnalysis, SentimentType
from app.quantum.qnlp_model import QuantumNLPModel
from app.quantum.classiq_client import ClassiqClient
from app.config import settings

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Hybrid classical-quantum sentiment analyzer"""

    def __init__(self, quantum_client: ClassiqClient):
        self.quantum_client = quantum_client
        self.qnlp_model = QuantumNLPModel(quantum_client)

        # Initialize classical models
        self._init_classical_models()

        # Load spaCy for entity recognition
        self.nlp = spacy.load("en_core_web_sm")

    def _init_classical_models(self):
        """Initialize classical NLP models"""
        # FinBERT for financial sentiment
        self.finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1
        )

        # General sentiment model as backup
        self.general_sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    async def analyze_batch(
            self,
            processed_news: List[Dict[str, Any]],
            use_quantum: bool = True
    ) -> List[SentimentAnalysis]:
        """Analyze sentiment for multiple news items"""
        tasks = [
            self.analyze_single(news, use_quantum)
            for news in processed_news
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any failures
        analyzed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Sentiment analysis failed for item {i}: {result}")
                # Fallback to neutral sentiment
                analyzed.append(self._create_neutral_sentiment())
            else:
                analyzed.append(result)

        return analyzed

    async def analyze_single(
            self,
            processed_news: Dict[str, Any],
            use_quantum: bool = True
    ) -> SentimentAnalysis:
        """Analyze sentiment for a single news item"""
        text = processed_news["cleaned_text"]

        # Run classical analysis
        classical_result = await self._classical_analysis(text)

        # Run quantum analysis if enabled
        quantum_result = None
        if use_quantum and len(text) < 500:  # Quantum has length limits
            try:
                quantum_result = await self._quantum_analysis(text)
            except Exception as e:
                logger.error(f"Quantum analysis failed: {e}")

        # Combine results
        combined_sentiment = self._combine_sentiments(
            classical_result,
            quantum_result
        )

        # Extract additional features
        entities = self._extract_entities(text)
        key_phrases = self._extract_key_phrases(processed_news)
        market_keywords = processed_news.get("financial_keywords", {})

        return SentimentAnalysis(
            sentiment=combined_sentiment["sentiment"],
            confidence=combined_sentiment["confidence"],
            quantum_sentiment_vector=combined_sentiment.get("quantum_vector", []),
            classical_sentiment_score=classical_result["score"],
            entities_detected=entities,
            key_phrases=key_phrases,
            market_impact_keywords=self._get_impact_keywords(market_keywords)
        )

    async def _classical_analysis(self, text: str) -> Dict[str, Any]:
        """Run classical sentiment analysis"""
        # Try FinBERT first
        try:
            finbert_result = self.finbert(text[:512])[0]  # BERT has 512 token limit

            # Map FinBERT labels to our sentiment types
            label_map = {
                "positive": SentimentType.POSITIVE,
                "negative": SentimentType.NEGATIVE,
                "neutral": SentimentType.NEUTRAL
            }

            sentiment = label_map.get(
                finbert_result["label"].lower(),
                SentimentType.NEUTRAL
            )

            score = finbert_result["score"]

        except Exception as e:
            logger.error(f"FinBERT failed: {e}")
            # Fallback to TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                sentiment = SentimentType.POSITIVE
            elif polarity < -0.1:
                sentiment = SentimentType.NEGATIVE
            else:
                sentiment = SentimentType.NEUTRAL

            score = abs(polarity)

        # Add intensity modifiers
        if score > 0.9:
            if sentiment == SentimentType.POSITIVE:
                sentiment = SentimentType.VERY_POSITIVE
            elif sentiment == SentimentType.NEGATIVE:
                sentiment = SentimentType.VERY_NEGATIVE

        return {
            "sentiment": sentiment,
            "score": score,
            "method": "classical"
        }

    async def _quantum_analysis(self, text: str) -> Dict[str, Any]:
        """Run quantum-enhanced sentiment analysis"""
        # Prepare text for quantum circuit
        quantum_features = await self.qnlp_model.encode_text(text)

        # Run quantum classification
        quantum_result = await self.qnlp_model.classify_sentiment(
            quantum_features
        )

        # Map quantum output to sentiment
        quantum_probs = quantum_result["probabilities"]

        # Find dominant sentiment
        sentiment_idx = np.argmax(quantum_probs)
        sentiments = [
            SentimentType.VERY_NEGATIVE,
            SentimentType.NEGATIVE,
            SentimentType.NEUTRAL,
            SentimentType.POSITIVE,
            SentimentType.VERY_POSITIVE
        ]

        return {
            "sentiment": sentiments[sentiment_idx],
            "score": quantum_probs[sentiment_idx],
            "quantum_vector": quantum_probs.tolist(),
            "circuit_depth": quantum_result.get("circuit_depth", 0),
            "method": "quantum"
        }

    def _combine_sentiments(
            self,
            classical: Dict[str, Any],
            quantum: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine classical and quantum sentiment results"""
        if not quantum:
            # Only classical available
            return {
                "sentiment": classical["sentiment"],
                "confidence": classical["score"],
                "quantum_vector": []
            }

        # Weight combination (can be tuned)
        classical_weight = 0.6
        quantum_weight = 0.4

        # Map sentiments to numeric values
        sentiment_values = {
            SentimentType.VERY_NEGATIVE: -2,
            SentimentType.NEGATIVE: -1,
            SentimentType.NEUTRAL: 0,
            SentimentType.POSITIVE: 1,
            SentimentType.VERY_POSITIVE: 2
        }

        classical_value = sentiment_values[classical["sentiment"]]
        quantum_value = sentiment_values[quantum["sentiment"]]

        # Weighted average
        combined_value = (
                classical_value * classical_weight +
                quantum_value * quantum_weight
        )

        # Map back to sentiment
        if combined_value <= -1.5:
            final_sentiment = SentimentType.VERY_NEGATIVE
        elif combined_value <= -0.5:
            final_sentiment = SentimentType.NEGATIVE
        elif combined_value <= 0.5:
            final_sentiment = SentimentType.NEUTRAL
        elif combined_value <= 1.5:
            final_sentiment = SentimentType.POSITIVE
        else:
            final_sentiment = SentimentType.VERY_POSITIVE

        # Combined confidence
        confidence = (
                classical["score"] * classical_weight +
                quantum["score"] * quantum_weight
        )

        return {
            "sentiment": final_sentiment,
            "confidence": min(confidence, 1.0),
            "quantum_vector": quantum.get("quantum_vector", [])
        }

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        doc = self.nlp(text[:1000])  # Limit length for performance

        entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE", "MONEY", "PERCENT", "DATE"]:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        return entities

    def _extract_key_phrases(self, processed_news: Dict[str, Any]) -> List[str]:
        """Extract key phrases from processed news"""
        # Combine various extracted features
        key_phrases = []

        # Add detected entities
        entities = processed_news.get("entities", {})
        if "organizations" in entities:
            key_phrases.extend(entities["organizations"][:3])

        # Add significant financial keywords
        keywords = processed_news.get("financial_keywords", {})
        for category in ["bullish", "bearish", "actions"]:
            if category in keywords:
                key_phrases.extend(keywords[category][:2])

        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in key_phrases:
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                unique_phrases.append(phrase)

        return unique_phrases[:10]  # Limit to top 10

    def _get_impact_keywords(self, financial_keywords: Dict[str, List[str]]) -> List[str]:
        """Get keywords likely to impact markets"""
        high_impact = []

        # Prioritize certain keyword categories
        priority_keywords = {
            "bullish": ["surge", "breakthrough", "record high", "beat expectations"],
            "bearish": ["crash", "plunge", "recession", "miss expectations"],
            "actions": ["upgrade", "downgrade", "merger", "acquisition"]
        }

        for category, priority_list in priority_keywords.items():
            category_keywords = financial_keywords.get(category, [])
            for keyword in category_keywords:
                if keyword in priority_list:
                    high_impact.append(keyword)

        return high_impact

    def _create_neutral_sentiment(self) -> SentimentAnalysis:
        """Create a neutral sentiment result as fallback"""
        return SentimentAnalysis(
            sentiment=SentimentType.NEUTRAL,
            confidence=0.5,
            quantum_sentiment_vector=[],
            classical_sentiment_score=0.5,
            entities_detected=[],
            key_phrases=[],
            market_impact_keywords=[]
        )