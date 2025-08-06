# backend/app/services/sentiment_analyzer.py

import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import spacy
from datetime import datetime
import re

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

# Enhanced crisis keywords with severity scores
CRISIS_KEYWORDS = {
    "health_extreme": {
        "keywords": ["cancer", "radiation", "toxic", "poison", "deadly", "fatal", "death", "kills",
                     "carcinogen", "lethal", "terminal", "epidemic", "pandemic", "shortens lifespan"],
        "severity": 0.95
    },
    "health_severe": {
        "keywords": ["illness", "sick", "disease", "outbreak", "contamination", "harmful", "hazard",
                     "hospitalized", "emergency", "headaches", "symptoms", "diagnosis"],
        "severity": 0.85
    },
    "legal_extreme": {
        "keywords": ["serial killer", "murder", "homicide", "terrorist", "war crimes", "genocide"],
        "severity": 0.98
    },
    "legal_severe": {
        "keywords": ["fraud", "criminal", "indicted", "arrested", "guilty", "convicted", "jail",
                     "prison", "lawsuit", "sued", "illegal", "felony", "prosecution"],
        "severity": 0.9
    },
    "labor_severe": {
        "keywords": ["whips", "slave", "abuse", "exploitation", "doesn't pay", "unpaid", "sweatshop",
                     "forced labor", "child labor", "human rights", "torture", "violence"],
        "severity": 0.92
    },
    "financial_severe": {
        "keywords": ["bankruptcy", "collapse", "crash", "default", "insolvent", "liquidation",
                     "worthless", "ponzi", "scam", "scandal"],
        "severity": 0.88
    },
    "regulatory_severe": {
        "keywords": ["banned", "shutdown", "investigation", "probe", "violation", "recalled",
                     "defective", "dangerous", "unsafe", "subpoena"],
        "severity": 0.8
    },
    "reputation_severe": {
        "keywords": ["boycott", "protest", "outrage", "backlash", "controversy", "unethical",
                     "immoral", "corrupt", "dishonest", "lying"],
        "severity": 0.75
    }
}

# Enhanced sentiment impact multipliers
SENTIMENT_IMPACT_MULTIPLIERS = {
    "extreme_negative": {
        "keywords": ["worst", "terrible", "horrific", "catastrophic", "disaster", "nightmare"],
        "multiplier": 2.5
    },
    "very_negative": {
        "keywords": ["bad", "poor", "weak", "declining", "struggling", "failing"],
        "multiplier": 1.8
    },
    "negative": {
        "keywords": ["concern", "worry", "issue", "problem", "challenge", "difficult"],
        "multiplier": 1.3
    },
    "positive": {
        "keywords": ["good", "strong", "growth", "improvement", "success", "profit"],
        "multiplier": 1.2
    },
    "very_positive": {
        "keywords": ["excellent", "outstanding", "record", "breakthrough", "revolutionary"],
        "multiplier": 1.6
    }
}


class EnhancedSentimentAnalyzer:
    """Massively enhanced sentiment analyzer with improved accuracy"""

    def __init__(self, classiq_client=None):
        self.classiq_client = classiq_client
        self.qnlp_model = QuantumNLPModel(classiq_client) if classiq_client else QuantumNLPModel(None)
        self.classical_model = None
        self.finbert = None
        self.nlp = None
        self._initialized = False
        self._finbert_initialized = False
        self._spacy_initialized = False
        self.quantum_success_rate = 0.0

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

    def _check_crisis_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced crisis detection with better severity scoring"""
        text_lower = text.lower()
        triggered_keywords = []
        triggered_categories = []
        max_severity = 0.0
        severity_scores = []

        for category, config in CRISIS_KEYWORDS.items():
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    triggered_keywords.append(keyword)
                    triggered_categories.append(category)
                    severity_scores.append(config["severity"])

        if severity_scores:
            # Calculate compound severity
            max_severity = max(severity_scores)
            avg_severity = sum(severity_scores) / len(severity_scores)

            # Amplify for multiple crisis indicators
            if len(triggered_keywords) >= 3:
                max_severity = min(1.0, max_severity * 1.5)
            elif len(triggered_keywords) >= 2:
                max_severity = min(1.0, max_severity * 1.25)

            # Check for company name mentions (increases impact)
            company_names = ["apple", "google", "microsoft", "tesla", "amazon", "meta", "nvidia"]
            company_mentioned = any(name in text_lower for name in company_names)
            if company_mentioned:
                max_severity = min(1.0, max_severity * 1.2)

            # Check for CEO/executive mentions
            executive_keywords = ["ceo", "chief executive", "president", "founder", "board", "executive"]
            executive_mentioned = any(keyword in text_lower for keyword in executive_keywords)
            if executive_mentioned:
                max_severity = min(1.0, max_severity * 1.15)

            # Check for amplifying words
            amplifiers = ["millions", "widespread", "everyone", "massive", "all", "entire", "global"]
            if any(word in text_lower for word in amplifiers):
                max_severity = min(1.0, max_severity * 1.1)

        return {
            "is_crisis": len(triggered_keywords) > 0,
            "severity": max_severity,
            "keywords": list(set(triggered_keywords)),
            "categories": list(set(triggered_categories)),
            "num_triggers": len(triggered_keywords)
        }

    def _calculate_sentiment_intensity(self, text: str, base_sentiment: SentimentType) -> float:
        """Calculate intensity multiplier based on language strength"""
        text_lower = text.lower()
        max_multiplier = 1.0

        for intensity_level, config in SENTIMENT_IMPACT_MULTIPLIERS.items():
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    max_multiplier = max(max_multiplier, config["multiplier"])

        # Check for exclamation marks (increase intensity)
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            max_multiplier *= (1 + min(exclamation_count * 0.1, 0.3))

        # Check for all caps words (increase intensity)
        words = text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if len(caps_words) > 0:
            max_multiplier *= (1 + min(len(caps_words) * 0.05, 0.2))

        return max_multiplier

    async def analyze_single(
            self,
            processed_news: Dict[str, Any],
            use_quantum: bool = True
    ) -> SentimentAnalysis:
        """Analyze sentiment for a single news item with enhanced accuracy"""

        text = processed_news.get("cleaned_text", "")

        # Enhanced crisis check
        crisis_check = self._check_crisis_enhanced(text)

        if crisis_check["is_crisis"]:
            # Crisis detected - calculate severe negative sentiment
            severity = crisis_check["severity"]
            num_triggers = crisis_check["num_triggers"]

            # Enhanced severity mapping
            if severity >= 0.9 or num_triggers >= 3:
                sentiment = SentimentType.VERY_NEGATIVE
                confidence = 0.98
                quantum_vector = [0.0, 0.0, 0.0, 0.02, 0.98]  # Heavily weighted to very negative
                impact_score = -0.95  # Near maximum negative impact
            elif severity >= 0.8 or num_triggers >= 2:
                sentiment = SentimentType.VERY_NEGATIVE
                confidence = 0.95
                quantum_vector = [0.0, 0.0, 0.05, 0.15, 0.80]
                impact_score = -0.85
            elif severity >= 0.7:
                sentiment = SentimentType.NEGATIVE
                confidence = 0.92
                quantum_vector = [0.0, 0.05, 0.10, 0.35, 0.50]
                impact_score = -0.75
            else:
                sentiment = SentimentType.NEGATIVE
                confidence = 0.88
                quantum_vector = [0.0, 0.10, 0.20, 0.40, 0.30]
                impact_score = -0.65

            # Apply intensity multiplier
            intensity = self._calculate_sentiment_intensity(text, sentiment)
            impact_score = max(-1.0, impact_score * intensity)

            return SentimentAnalysis(
                sentiment=sentiment,
                confidence=confidence,
                quantum_sentiment_vector=quantum_vector,
                classical_sentiment_score=impact_score,
                entities_detected=self._extract_entities(text),
                key_phrases=crisis_check["keywords"],
                market_impact_keywords=crisis_check["keywords"],
                crisis_indicators=crisis_check
            )

        # No crisis - proceed with enhanced normal analysis
        result = await self.analyze_batch([processed_news], use_quantum)
        return result[0] if result else self._create_neutral_sentiment()

    async def analyze_batch(
            self,
            processed_news_list: List[Dict[str, Any]],
            use_quantum: bool = True
    ) -> List[SentimentAnalysis]:
        """Batch analyze multiple news items with enhanced processing"""

        if not self._initialized:
            await self.initialize()

        results = []

        for idx, news in enumerate(processed_news_list):
            # Determine analysis method
            if use_quantum and await self._should_use_quantum() and self._is_high_impact(news):
                index, analysis = await self._analyze_with_quantum(news, idx)
            else:
                index, analysis = await self._analyze_with_classical(news, idx)

            results.append(analysis)

        return results

    async def _analyze_with_classical(
            self,
            processed_news: Dict[str, Any],
            index: int
    ) -> Tuple[int, SentimentAnalysis]:
        """Enhanced classical sentiment analysis"""
        text = processed_news["cleaned_text"]

        # Check for crisis first
        crisis_check = self._check_crisis_enhanced(text)
        if crisis_check["is_crisis"]:
            # Return crisis sentiment with enhanced severity
            severity = crisis_check["severity"]
            num_triggers = crisis_check["num_triggers"]

            if severity >= 0.85 or num_triggers >= 3:
                sentiment = SentimentType.VERY_NEGATIVE
                impact_score = -0.9
            elif severity >= 0.7 or num_triggers >= 2:
                sentiment = SentimentType.NEGATIVE
                impact_score = -0.7
            else:
                sentiment = SentimentType.NEGATIVE
                impact_score = -0.5

            intensity = self._calculate_sentiment_intensity(text, sentiment)
            impact_score = max(-1.0, impact_score * intensity)

            return (index, SentimentAnalysis(
                sentiment=sentiment,
                confidence=0.95,
                quantum_sentiment_vector=[],
                classical_sentiment_score=impact_score,
                entities_detected=self._extract_entities(text),
                key_phrases=crisis_check["keywords"],
                market_impact_keywords=crisis_check["keywords"],
                crisis_indicators=crisis_check
            ))

        # Run enhanced classical analysis
        classical_result = await self._classical_analysis_enhanced(text)

        # Extract additional features
        entities = self._extract_entities(text)
        key_phrases = self._extract_key_phrases(processed_news)
        market_keywords = self._get_impact_keywords(
            processed_news.get("financial_keywords", {})
        )

        analysis = SentimentAnalysis(
            sentiment=self._score_to_sentiment_enhanced(classical_result["score"]),
            confidence=classical_result["confidence"],
            quantum_sentiment_vector=[],
            classical_sentiment_score=classical_result["score"],
            entities_detected=entities,
            key_phrases=key_phrases,
            market_impact_keywords=market_keywords
        )

        return (index, analysis)

    async def _classical_analysis_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced classical sentiment analysis"""
        # Initialize FinBERT if needed
        if not self._finbert_initialized:
            self._init_finbert()

        if self.finbert:
            try:
                # Run FinBERT
                result = await self._run_finbert(text)

                # Apply intensity multiplier
                intensity = self._calculate_sentiment_intensity(text, SentimentType.NEUTRAL)
                result["score"] = max(-1.0, min(1.0, result["score"] * intensity))
                result["confidence"] = min(0.99, result["confidence"] * (1 + (intensity - 1) * 0.1))

                return result
            except Exception as e:
                logger.error(f"FinBERT analysis failed: {e}")

        # Fallback to enhanced rule-based
        return self._rule_based_sentiment_enhanced(text)

    def _rule_based_sentiment_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced rule-based sentiment analysis"""
        text_lower = text.lower()

        # Enhanced positive and negative word lists
        very_positive_words = [
            "breakthrough", "revolutionary", "outstanding", "exceptional", "remarkable",
            "record-breaking", "unprecedented", "phenomenal", "extraordinary", "spectacular"
        ]

        positive_words = [
            "profit", "gain", "growth", "increase", "improve", "success", "beat",
            "exceed", "surge", "rally", "boost", "advance", "rise", "upgrade"
        ]

        negative_words = [
            "loss", "decline", "decrease", "fall", "drop", "miss", "fail",
            "weak", "poor", "concern", "worry", "risk", "threat", "warning"
        ]

        very_negative_words = [
            "crash", "plunge", "collapse", "disaster", "crisis", "bankruptcy",
            "scandal", "fraud", "investigation", "lawsuit", "recall", "violation"
        ]

        # Count occurrences with weights
        very_pos_count = sum(2.5 for word in very_positive_words if word in text_lower)
        pos_count = sum(1.5 for word in positive_words if word in text_lower)
        neg_count = sum(1.5 for word in negative_words if word in text_lower)
        very_neg_count = sum(2.5 for word in very_negative_words if word in text_lower)

        # Calculate weighted score
        total_positive = very_pos_count + pos_count
        total_negative = very_neg_count + neg_count

        if total_positive + total_negative == 0:
            return {"score": 0.0, "confidence": 0.5}

        # Enhanced scoring with stronger negative bias for bad news
        score = (total_positive - total_negative * 1.3) / (total_positive + total_negative)
        score = max(-1.0, min(1.0, score))

        # Apply intensity multiplier
        intensity = self._calculate_sentiment_intensity(text, SentimentType.NEUTRAL)
        score = score * intensity

        # Calculate confidence based on word count
        word_count = total_positive + total_negative
        confidence = min(0.95, 0.6 + (word_count * 0.05))

        return {"score": score, "confidence": confidence}

    def _score_to_sentiment_enhanced(self, score: float) -> SentimentType:
        """Enhanced score to sentiment mapping with better thresholds"""
        if score <= -0.6:
            return SentimentType.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentType.NEGATIVE
        elif score >= 0.6:
            return SentimentType.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentType.POSITIVE
        else:
            return SentimentType.NEUTRAL

    async def _run_finbert(self, text: str) -> Dict[str, Any]:
        """Run FinBERT sentiment analysis"""
        # Truncate text for BERT
        truncated_text = text[:500]

        # Run in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.finbert(truncated_text)[0]
        )

        # Map FinBERT output with enhanced scoring
        label = result["label"].lower()
        score = result["score"]

        # Enhanced score conversion
        if label == "positive":
            normalized_score = 0.3 + (score * 0.7)
        elif label == "negative":
            normalized_score = -0.3 - (score * 0.7)
        else:
            normalized_score = (score - 0.5) * 0.4

        return {
            "score": normalized_score,
            "confidence": score * 0.9
        }

    async def _should_use_quantum(self) -> bool:
        """Determine if quantum processing should be used"""
        if not self.qnlp_model.is_ready():
            return False

        # Use quantum with adjusted probability
        base_probability = 0.4
        adjusted_probability = base_probability + (self.quantum_success_rate * 0.4)

        return np.random.random() < adjusted_probability

    def _is_high_impact(self, news: Dict[str, Any]) -> bool:
        """Determine if news is high impact"""
        text = news.get("cleaned_text", "").lower()

        high_impact_words = [
            "breakthrough", "crash", "surge", "plunge", "scandal",
            "merger", "acquisition", "bankruptcy", "investigation",
            "earnings", "revenue", "profit", "loss", "recall",
            "lawsuit", "fraud", "crisis", "emergency"
        ]

        return any(word in text for word in high_impact_words)

    async def _analyze_with_quantum(
            self,
            processed_news: Dict[str, Any],
            index: int
    ) -> Tuple[int, SentimentAnalysis]:
        """Quantum-enhanced sentiment analysis"""
        try:
            text = processed_news["cleaned_text"]

            # Encode text for quantum processing
            quantum_features = await self.qnlp_model.encode_text_quantum(text)

            # Run quantum classification
            quantum_result = await self.qnlp_model.quantum_sentiment_classification(
                quantum_features
            )

            # Run classical analysis in parallel
            classical_result = await self._classical_analysis_enhanced(text)

            # Combine results with enhanced weighting
            combined_result = self._combine_quantum_classical_enhanced(
                quantum_result, classical_result, weight_quantum=0.6
            )

            # Extract features
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
            return await self._analyze_with_classical(processed_news, index)

    def _combine_quantum_classical_enhanced(
            self,
            quantum_result: Dict[str, Any],
            classical_result: Dict[str, Any],
            weight_quantum: float = 0.6
    ) -> Dict[str, Any]:
        """Enhanced combination of quantum and classical results"""

        # Get quantum sentiment from probability distribution
        quantum_probs = quantum_result["probabilities"]
        quantum_sentiment_idx = np.argmax(quantum_probs)
        quantum_confidence = quantum_probs[quantum_sentiment_idx]

        sentiments = [
            SentimentType.VERY_NEGATIVE,
            SentimentType.NEGATIVE,
            SentimentType.NEUTRAL,
            SentimentType.POSITIVE,
            SentimentType.VERY_POSITIVE
        ]

        quantum_sentiment = sentiments[quantum_sentiment_idx]
        classical_sentiment = self._score_to_sentiment_enhanced(classical_result["score"])

        # Enhanced combination logic
        if quantum_sentiment == classical_sentiment:
            # Agreement - high confidence
            final_sentiment = quantum_sentiment
            final_confidence = min(0.98, (quantum_confidence + classical_result["confidence"]) / 2 * 1.1)
        else:
            # Disagreement - weighted average with bias toward negative
            if quantum_sentiment in [SentimentType.VERY_NEGATIVE, SentimentType.NEGATIVE] or \
                    classical_sentiment in [SentimentType.VERY_NEGATIVE, SentimentType.NEGATIVE]:
                # If either is negative, bias toward negative
                weight_quantum = 0.7 if quantum_sentiment in [SentimentType.VERY_NEGATIVE,
                                                              SentimentType.NEGATIVE] else 0.3

            if weight_quantum > 0.5:
                final_sentiment = quantum_sentiment
            else:
                final_sentiment = classical_sentiment

            final_confidence = (quantum_confidence * weight_quantum +
                                classical_result["confidence"] * (1 - weight_quantum))

        return {
            "sentiment": final_sentiment,
            "confidence": final_confidence
        }

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if not self._spacy_initialized:
            self._init_spacy()

        if not self.nlp:
            return []

        try:
            doc = self.nlp(text[:1000])  # Limit text length
            entities = []

            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE", "MONEY", "PERCENT"]:
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

        # Remove duplicates
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

        # Enhanced priority keywords
        priority_keywords = {
            "bullish": ["surge", "breakthrough", "record high", "beat expectations", "soar", "rally"],
            "bearish": ["crash", "plunge", "recession", "miss expectations", "collapse", "crisis"],
            "actions": ["upgrade", "downgrade", "merger", "acquisition", "buyout", "bankruptcy"]
        }

        for category, priority_list in priority_keywords.items():
            category_keywords = financial_keywords.get(category, [])
            for keyword in category_keywords:
                if keyword in priority_list:
                    high_impact.append(keyword)

        return high_impact

    def _create_neutral_sentiment(self) -> SentimentAnalysis:
        """Create a neutral sentiment result"""
        return SentimentAnalysis(
            sentiment=SentimentType.NEUTRAL,
            confidence=0.5,
            quantum_sentiment_vector=[],
            classical_sentiment_score=0.0,
            entities_detected=[],
            key_phrases=[],
            market_impact_keywords=[]
        )


# Export the enhanced analyzer
SentimentAnalyzer = EnhancedSentimentAnalyzer