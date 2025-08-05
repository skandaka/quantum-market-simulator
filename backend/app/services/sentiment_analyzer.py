# backend/app/services/sentiment_analyzer.py

import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from datetime import datetime

from app.models.schemas import SentimentAnalysis, SentimentType
from app.quantum.qnlp_model import QuantumNLPModel
from app.config import settings

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available")

# Crisis keywords with severity scores
CRISIS_KEYWORDS = {
    "health_severe": {
        "keywords": ["cancer", "radiation", "toxic", "poison", "deadly", "fatal", "death", "kills", "carcinogen"],
        "severity": 0.9
    },
    "health_moderate": {
        "keywords": ["illness", "sick", "disease", "outbreak", "contamination", "harmful", "hazard"],
        "severity": 0.7
    },
    "legal_severe": {
        "keywords": ["fraud", "criminal", "indicted", "arrested", "guilty", "convicted", "jail", "prison"],
        "severity": 0.8
    },
    "financial_severe": {
        "keywords": ["bankruptcy", "collapse", "crash", "default", "insolvent", "liquidation"],
        "severity": 0.85
    },
    "regulatory_severe": {
        "keywords": ["banned", "illegal", "shutdown", "investigation", "probe", "violation"],
        "severity": 0.75
    }
}


class SentimentAnalyzer:
    """Advanced sentiment analyzer with quantum capabilities"""

    def __init__(self, classiq_client=None):
        self.classiq_client = classiq_client
        self.qnlp_model = QuantumNLPModel(classiq_client) if classiq_client else QuantumNLPModel(None)
        self.classical_model = None
        self.finbert = None
        self.nlp = None
        self._initialized = False
        self._finbert_initialized = False
        self._spacy_initialized = False
        self.quantum_success_rate = 0.0  # Track success rate

    async def initialize(self):
        """Initialize sentiment analyzer components"""
        if self._initialized:
            return

        try:
            # Initialize quantum model
            await self.qnlp_model.initialize()
            logger.info("Quantum NLP model initialized")
        except Exception as e:
            logger.warning(f"Quantum NLP initialization failed: {e}")

        self._initialized = True

    def _init_finbert(self):
        """Lazy initialization of FinBERT model"""
        if self._finbert_initialized or not TRANSFORMERS_AVAILABLE:
            return

        try:
            self.finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT initialization failed: {e}")
            self.finbert = None
        finally:
            self._finbert_initialized = True

    def _init_spacy(self):
        """Lazy initialization of spaCy model"""
        if self._spacy_initialized or not SPACY_AVAILABLE:
            return

        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"SpaCy model not found: {e}")
            self.nlp = None
        finally:
            self._spacy_initialized = True

    def _check_crisis(self, text: str) -> Dict[str, Any]:
        """Check for crisis keywords in text"""
        text_lower = text.lower()
        triggered_keywords = []
        triggered_categories = []
        max_severity = 0.0

        for category, config in CRISIS_KEYWORDS.items():
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    triggered_keywords.append(keyword)
                    triggered_categories.append(category)
                    max_severity = max(max_severity, config["severity"])

        # Amplify severity for multiple keywords or certain contexts
        if len(triggered_keywords) > 2:
            max_severity = min(1.0, max_severity * 1.2)

        # Check for amplifying words
        amplifiers = ["millions", "widespread", "everyone", "massive", "all"]
        if any(word in text_lower for word in amplifiers):
            max_severity = min(1.0, max_severity * 1.15)

        return {
            "is_crisis": len(triggered_keywords) > 0,
            "severity": max_severity,
            "keywords": list(set(triggered_keywords)),
            "categories": list(set(triggered_categories))
        }

    async def analyze_batch(
            self,
            processed_news: List[Dict[str, Any]],
            use_quantum: bool = True
    ) -> List[SentimentAnalysis]:
        """Analyze sentiment for multiple news items"""

        # Initialize models on first use
        if not self._finbert_initialized:
            self._init_finbert()

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

        # Quantum processing (limited)
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
                    logger.error(f"Analysis error: {analysis}")
                    analyzed[idx] = self._create_neutral_sentiment()
            else:
                logger.error(f"Unexpected result format: {result}")

        # Replace None with neutral sentiment
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

        text = processed_news.get("cleaned_text", "")

        # First, check for crisis
        crisis_check = self._check_crisis(text)

        if crisis_check["is_crisis"]:
            # Crisis detected - return strong negative sentiment
            severity = crisis_check["severity"]

            if severity >= 0.8:
                sentiment = SentimentType.VERY_NEGATIVE
                confidence = 0.95
                quantum_vector = [0.0, 0.0, 0.0, 0.05, 0.95]
            else:
                sentiment = SentimentType.NEGATIVE
                confidence = 0.9
                quantum_vector = [0.0, 0.0, 0.1, 0.3, 0.6]

            return SentimentAnalysis(
                sentiment=sentiment,
                confidence=confidence,
                quantum_sentiment_vector=quantum_vector,
                classical_sentiment_score=1 - severity,
                entities_detected=self._extract_entities(text),
                key_phrases=crisis_check["keywords"],
                market_impact_keywords=crisis_check["keywords"]
            )

        # No crisis - proceed with normal analysis
        result = await self.analyze_batch([processed_news], use_quantum)
        return result[0] if result else self._create_neutral_sentiment()

    async def _should_use_quantum(self) -> bool:
        """Determine if quantum processing should be used"""
        # Use quantum if available and success rate is good
        if not self.qnlp_model.is_ready():
            return False

        # Start with 50% chance, adjust based on success rate
        base_probability = 0.5
        adjusted_probability = base_probability + (self.quantum_success_rate * 0.3)

        return np.random.random() < adjusted_probability

    def _is_high_impact(self, news: Dict[str, Any]) -> bool:
        """Determine if news is high impact and worth quantum processing"""
        # Check for high-impact keywords
        text = news.get("cleaned_text", "").lower()

        high_impact_words = [
            "breakthrough", "crash", "surge", "plunge", "scandal",
            "merger", "acquisition", "bankruptcy", "investigation",
            "earnings", "revenue", "profit", "loss"
        ]

        return any(word in text for word in high_impact_words)

    async def _analyze_with_quantum(
            self,
            processed_news: Dict[str, Any],
            index: int
    ) -> Tuple[int, SentimentAnalysis]:
        """Analyze using quantum NLP model"""
        try:
            text = processed_news["cleaned_text"]

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

        # Check for crisis first
        crisis_check = self._check_crisis(text)
        if crisis_check["is_crisis"]:
            # Return crisis sentiment
            severity = crisis_check["severity"]
            sentiment = SentimentType.VERY_NEGATIVE if severity >= 0.8 else SentimentType.NEGATIVE

            return (index, SentimentAnalysis(
                sentiment=sentiment,
                confidence=0.9,
                quantum_sentiment_vector=[],
                classical_sentiment_score=1 - severity,
                entities_detected=self._extract_entities(text),
                key_phrases=crisis_check["keywords"],
                market_impact_keywords=crisis_check["keywords"]
            ))

        # Run classical analysis
        classical_result = await self._classical_analysis(text)

        # Extract additional features
        entities = self._extract_entities(text)
        key_phrases = self._extract_key_phrases(processed_news)
        market_keywords = self._get_impact_keywords(
            processed_news.get("financial_keywords", {})
        )

        analysis = SentimentAnalysis(
            sentiment=self._score_to_sentiment(classical_result["score"]),
            confidence=classical_result["confidence"],
            quantum_sentiment_vector=[],
            classical_sentiment_score=classical_result["score"],
            entities_detected=entities,
            key_phrases=key_phrases,
            market_impact_keywords=market_keywords
        )

        return (index, analysis)

    async def _classical_analysis(self, text: str) -> Dict[str, Any]:
        """Perform classical sentiment analysis"""
        # Initialize FinBERT if needed
        if not self._finbert_initialized:
            self._init_finbert()

        if self.finbert:
            try:
                # Run FinBERT
                result = await self._run_finbert(text)
                return result
            except Exception as e:
                logger.error(f"FinBERT analysis failed: {e}")

        # Fallback to rule-based
        return self._rule_based_sentiment(text)

    async def _run_finbert(self, text: str) -> Dict[str, Any]:
        """Run FinBERT sentiment analysis"""
        # Truncate text for BERT
        truncated_text = text[:500]

        # Run in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.finbert(truncated_text)[0]
        )

        # Map FinBERT output
        label = result["label"].lower()
        score = result["score"]

        # Convert to normalized score
        if label == "positive":
            normalized_score = 0.5 + (score * 0.5)
        elif label == "negative":
            normalized_score = 0.5 - (score * 0.5)
        else:  # neutral
            normalized_score = 0.5

        return {
            "score": normalized_score,
            "confidence": score,
            "method": "finbert"
        }

    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment as fallback"""
        text_lower = text.lower()

        positive_words = [
            "profit", "gain", "growth", "surge", "breakthrough",
            "success", "positive", "beat", "exceed", "upgrade"
        ]

        negative_words = [
            "loss", "decline", "fall", "crash", "negative",
            "miss", "downgrade", "concern", "risk", "threat"
        ]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            score = 0.5 + (0.1 * min(pos_count - neg_count, 5))
        elif neg_count > pos_count:
            score = 0.5 - (0.1 * min(neg_count - pos_count, 5))
        else:
            score = 0.5

        confidence = min(0.8, 0.5 + 0.1 * (pos_count + neg_count))

        return {
            "score": score,
            "confidence": confidence,
            "method": "rule_based"
        }

    def _combine_quantum_classical(
            self,
            quantum_result: Dict[str, Any],
            classical_result: Dict[str, Any],
            weight_quantum: float = 0.7
    ) -> Dict[str, Any]:
        """Combine quantum and classical results"""
        # Get quantum sentiment from probability distribution
        quantum_probs = quantum_result["probabilities"]
        quantum_sentiment = self._probs_to_sentiment(quantum_probs)
        quantum_confidence = quantum_result.get("confidence", 0.8)

        # Get classical sentiment
        classical_sentiment = self._score_to_sentiment(classical_result["score"])
        classical_confidence = classical_result["confidence"]

        # Weighted combination
        if quantum_sentiment == classical_sentiment:
            # Agreement - high confidence
            final_sentiment = quantum_sentiment
            final_confidence = min(0.95,
                                   weight_quantum * quantum_confidence +
                                   (1 - weight_quantum) * classical_confidence + 0.1)
        else:
            # Disagreement - use weighted approach
            if quantum_confidence > classical_confidence * 1.5:
                final_sentiment = quantum_sentiment
            elif classical_confidence > quantum_confidence * 1.5:
                final_sentiment = classical_sentiment
            else:
                # Close confidence - use quantum with reduced confidence
                final_sentiment = quantum_sentiment

            final_confidence = min(0.85,
                                   weight_quantum * quantum_confidence +
                                   (1 - weight_quantum) * classical_confidence)

        return {
            "sentiment": final_sentiment,
            "confidence": final_confidence
        }

    def _probs_to_sentiment(self, probabilities: np.ndarray) -> SentimentType:
        """Convert probability distribution to sentiment type"""
        # Assuming 5 classes: very_negative, negative, neutral, positive, very_positive
        sentiment_idx = np.argmax(probabilities)

        mapping = {
            0: SentimentType.VERY_NEGATIVE,
            1: SentimentType.NEGATIVE,
            2: SentimentType.NEUTRAL,
            3: SentimentType.POSITIVE,
            4: SentimentType.VERY_POSITIVE
        }

        return mapping.get(sentiment_idx, SentimentType.NEUTRAL)

    def _score_to_sentiment(self, score: float) -> SentimentType:
        """Convert numerical score to sentiment type"""
        if score < 0.2:
            return SentimentType.VERY_NEGATIVE
        elif score < 0.4:
            return SentimentType.NEGATIVE
        elif score < 0.6:
            return SentimentType.NEUTRAL
        elif score < 0.8:
            return SentimentType.POSITIVE
        else:
            return SentimentType.VERY_POSITIVE

    def _map_finbert_sentiment(self, label: str) -> SentimentType:
        """Map FinBERT labels to our sentiment types"""
        mapping = {
            "positive": SentimentType.POSITIVE,
            "negative": SentimentType.NEGATIVE,
            "neutral": SentimentType.NEUTRAL
        }
        return mapping.get(label.lower(), SentimentType.NEUTRAL)

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        if not self._spacy_initialized:
            self._init_spacy()

        if not self.nlp:
            return []

        try:
            doc = self.nlp(text[:1000])  # Limit text length
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