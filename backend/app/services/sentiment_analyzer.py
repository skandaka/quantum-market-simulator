"""Sentiment analysis service with real quantum enhancement"""

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
    """Hybrid classical-quantum sentiment analyzer with real quantum backend"""

    def __init__(self, quantum_client: ClassiqClient):
        self.quantum_client = quantum_client
        self.qnlp_model = QuantumNLPModel(quantum_client)

        # Initialize classical models
        self._init_classical_models()

        # Load spaCy for entity recognition
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("SpaCy model not found, entity extraction will be limited")
            self.nlp = None

        # Track quantum usage for optimization
        self.quantum_call_count = 0
        self.quantum_success_rate = 1.0

    def _init_classical_models(self):
        """Initialize classical NLP models"""
        try:
            # FinBERT for financial sentiment
            self.finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {e}")
            self.finbert = None

        # General sentiment model as backup
        try:
            self.general_sentiment = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        except:
            logger.warning("Could not load general sentiment model")
            self.general_sentiment = None

    async def initialize(self):
        """Initialize the sentiment analyzer"""
        await self.qnlp_model.initialize()
        logger.info("Sentiment analyzer initialized")

    async def analyze_batch(
        self,
        processed_news: List[Dict[str, Any]],
        use_quantum: bool = True
    ) -> List[SentimentAnalysis]:
        """Analyze sentiment for multiple news items"""

        # Determine which items should use quantum
        quantum_candidates = []
        classical_items = []

        if use_quantum and await self._should_use_quantum():
            # Select high-impact items for quantum processing
            for i, news in enumerate(processed_news):
                if self._is_high_impact(news):
                    quantum_candidates.append((i, news))
                else:
                    classical_items.append((i, news))
        else:
            classical_items = [(i, news) for i, news in enumerate(processed_news)]

        # Process in parallel
        tasks = []

        # Quantum processing
        for idx, news in quantum_candidates[:3]:  # Limit quantum calls
            task = self._analyze_with_quantum(news, idx)
            tasks.append(task)

        # Classical processing
        for idx, news in classical_items:
            task = self._analyze_with_classical(news, idx)
            tasks.append(task)

        # Additional quantum items processed classically
        for idx, news in quantum_candidates[3:]:
            task = self._analyze_with_classical(news, idx)
            tasks.append(task)

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort by original index and handle errors
        analyzed = [None] * len(processed_news)
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                idx, analysis = result
                if not isinstance(analysis, Exception):
                    analyzed[idx] = analysis
                else:
                    logger.error(f"Analysis failed for item {idx}: {analysis}")
                    analyzed[idx] = self._create_neutral_sentiment()
            else:
                logger.error(f"Unexpected result format: {result}")

        # Fill any remaining None values
        for i in range(len(analyzed)):
            if analyzed[i] is None:
                analyzed[i] = self._create_neutral_sentiment()

        return analyzed

    async def analyze_single(
        self,
        processed_news: Dict[str, Any],
        use_quantum: bool = True
    ) -> SentimentAnalysis:
        """Analyze sentiment for a single news item"""

        if use_quantum and await self._should_use_quantum() and self._is_high_impact(processed_news):
            try:
                _, analysis = await self._analyze_with_quantum(processed_news, 0)
                return analysis
            except Exception as e:
                logger.error(f"Quantum analysis failed: {e}")

        # Fallback to classical
        _, analysis = await self._analyze_with_classical(processed_news, 0)
        return analysis

    async def _should_use_quantum(self) -> bool:
        """Determine if quantum backend should be used"""

        # Check if quantum client is ready
        if not self.quantum_client.is_ready():
            return False

        # Adaptive quantum usage based on success rate
        if self.quantum_success_rate < 0.5:
            return False

        # Rate limiting
        if self.quantum_call_count > 100:  # Daily limit
            return False

        return True

    def _is_high_impact(self, processed_news: Dict[str, Any]) -> bool:
        """Determine if news is high-impact and worth quantum processing"""

        # Check for high-impact keywords
        high_impact_keywords = {
            "crash", "surge", "plunge", "soar", "bankruptcy", "merger",
            "acquisition", "scandal", "breakthrough", "record", "unprecedented"
        }

        text_lower = processed_news.get("cleaned_text", "").lower()
        if any(keyword in text_lower for keyword in high_impact_keywords):
            return True

        # Check for multiple financial entities
        entities = processed_news.get("entities", {})
        if len(entities.get("organizations", [])) > 2:
            return True

        # Check text metrics
        metrics = processed_news.get("text_metrics", {})
        if metrics.get("exclamation_count", 0) > 2:
            return True

        return False

    async def _analyze_with_quantum(
        self,
        processed_news: Dict[str, Any],
        index: int
    ) -> Tuple[int, SentimentAnalysis]:
        """Analyze using quantum-enhanced processing"""

        text = processed_news["cleaned_text"]
        self.quantum_call_count += 1

        try:
            # Encode text for quantum processing
            quantum_features = await self.qnlp_model.encode_text_quantum(text)

            # Run quantum classification
            quantum_result = await self.qnlp_model.quantum_sentiment_classification(
                quantum_features
            )

            # Run classical analysis in parallel for comparison
            classical_result = await self._classical_analysis(text)

            # Combine quantum and classical results
            combined_result = self._combine_quantum_classical(
                quantum_result, classical_result, weight_quantum=0.7
            )

            # Extract additional features
            entities = self._extract_entities(text)
            key_phrases = self._extract_key_phrases(processed_news)
            market_keywords = self._get_impact_keywords(
                processed_news.get("financial_keywords", {})
            )

            analysis = SentimentAnalysis(
                sentiment=combined_result["sentiment"],
                confidence=combined_result["confidence"],
                quantum_sentiment_vector=quantum_result["probabilities"].tolist(),
                classical_sentiment_score=classical_result["score"],
                entities_detected=entities,
                key_phrases=key_phrases,
                market_impact_keywords=market_keywords
            )

            # Update success rate
            self.quantum_success_rate = 0.95 * self.quantum_success_rate + 0.05

            return (index, analysis)

        except Exception as e:
            logger.error(f"Quantum sentiment analysis failed: {e}")
            self.quantum_success_rate = 0.95 * self.quantum_success_rate
            # Fallback to classical
            return await self._analyze_with_classical(processed_news, index)

    async def _analyze_with_classical(
        self,
        processed_news: Dict[str, Any],
        index: int
    ) -> Tuple[int, SentimentAnalysis]:
        """Analyze using only classical methods"""

        text = processed_news["cleaned_text"]

        # Run classical analysis
        classical_result = await self._classical_analysis(text)

        # Extract additional features
        entities = self._extract_entities(text)
        key_phrases = self._extract_key_phrases(processed_news)
        market_keywords = self._get_impact_keywords(
            processed_news.get("financial_keywords", {})
        )

        analysis = SentimentAnalysis(
            sentiment=classical_result["sentiment"],
            confidence=classical_result["score"],
            quantum_sentiment_vector=[],  # No quantum vector
            classical_sentiment_score=classical_result["score"],
            entities_detected=entities,
            key_phrases=key_phrases,
            market_impact_keywords=market_keywords
        )

        return (index, analysis)

    async def _classical_analysis(self, text: str) -> Dict[str, Any]:
        """Run classical sentiment analysis"""

        # Try FinBERT first
        if self.finbert:
            try:
                result = self.finbert(text[:512])[0]  # BERT limit

                # Map labels
                label_map = {
                    "positive": SentimentType.POSITIVE,
                    "negative": SentimentType.NEGATIVE,
                    "neutral": SentimentType.NEUTRAL
                }

                sentiment = label_map.get(
                    result["label"].lower(),
                    SentimentType.NEUTRAL
                )

                score = result["score"]

                # Add intensity for very positive/negative
                if score > 0.9:
                    if sentiment == SentimentType.POSITIVE:
                        sentiment = SentimentType.VERY_POSITIVE
                    elif sentiment == SentimentType.NEGATIVE:
                        sentiment = SentimentType.VERY_NEGATIVE

                return {
                    "sentiment": sentiment,
                    "score": score,
                    "method": "finbert"
                }

            except Exception as e:
                logger.debug(f"FinBERT failed: {e}")

        # Fallback to TextBlob
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.5:
                sentiment = SentimentType.VERY_POSITIVE
            elif polarity > 0.1:
                sentiment = SentimentType.POSITIVE
            elif polarity < -0.5:
                sentiment = SentimentType.VERY_NEGATIVE
            elif polarity < -0.1:
                sentiment = SentimentType.NEGATIVE
            else:
                sentiment = SentimentType.NEUTRAL

            return {
                "sentiment": sentiment,
                "score": min(abs(polarity) + 0.5, 1.0),
                "method": "textblob"
            }

        except:
            return {
                "sentiment": SentimentType.NEUTRAL,
                "score": 0.5,
                "method": "fallback"
            }

    def _combine_quantum_classical(
        self,
        quantum_result: Dict[str, Any],
        classical_result: Dict[str, Any],
        weight_quantum: float = 0.6
    ) -> Dict[str, Any]:
        """Intelligently combine quantum and classical results"""

        # Get sentiment indices
        sentiment_map = {
            SentimentType.VERY_NEGATIVE: 0,
            SentimentType.NEGATIVE: 1,
            SentimentType.NEUTRAL: 2,
            SentimentType.POSITIVE: 3,
            SentimentType.VERY_POSITIVE: 4
        }

        classical_idx = sentiment_map[classical_result["sentiment"]]
        quantum_sentiment = quantum_result["predicted_sentiment"]
        quantum_idx = sentiment_map[SentimentType(quantum_sentiment)]

        # Check agreement
        if classical_idx == quantum_idx:
            # Strong agreement - high confidence
            return {
                "sentiment": classical_result["sentiment"],
                "confidence": min(
                    weight_quantum * quantum_result["confidence"] +
                    (1 - weight_quantum) * classical_result["score"] +
                    0.1,  # Agreement bonus
                    0.95
                )
            }
        elif abs(classical_idx - quantum_idx) == 1:
            # Close agreement - weighted average
            avg_idx = weight_quantum * quantum_idx + (1 - weight_quantum) * classical_idx
            final_idx = int(round(avg_idx))

            sentiments = list(sentiment_map.keys())
            return {
                "sentiment": sentiments[final_idx],
                "confidence": (
                    weight_quantum * quantum_result["confidence"] +
                    (1 - weight_quantum) * classical_result["score"]
                ) * 0.9  # Small penalty for minor disagreement
            }
        else:
            # Disagreement - prefer classical but reduce confidence
            return {
                "sentiment": classical_result["sentiment"],
                "confidence": classical_result["score"] * 0.7
            }

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""

        if not self.nlp:
            return []

        try:
            doc = self.nlp(text[:1000])  # Limit length

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

        except Exception as e:
            logger.debug(f"Entity extraction failed: {e}")
            return []

    def _extract_key_phrases(self, processed_news: Dict[str, Any]) -> List[str]:
        """Extract key phrases from processed news"""

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

        return unique_phrases[:10]

    def _get_impact_keywords(self, financial_keywords: Dict[str, List[str]]) -> List[str]:
        """Get high-impact market keywords"""

        high_impact = []

        # Priority keywords
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