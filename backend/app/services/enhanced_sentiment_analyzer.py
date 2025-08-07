"""Enhanced sentiment analyzer with improved crisis detection"""

import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import asyncio

from app.models.schemas import SentimentAnalysis, SentimentType
from app.quantum.qnlp_model import QuantumNLPModel
from app.config import settings

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from transformers import pipeline
    import torch
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

# ENHANCED CRISIS KEYWORDS WITH CATEGORIES
CRISIS_KEYWORDS = {
    "legal_critical": {
        "keywords": ["investigation", "criminal", "charges", "lawsuit", "fraud", "guilty", 
                     "arrested", "indicted", "prosecute", "court", "illegal", "violation",
                     "SEC investigation", "DOJ", "FBI", "regulatory action", "subpoena"],
        "severity": 0.95,
        "impact_multiplier": -0.4
    },
    "health_critical": {
        "keywords": ["death", "deadly", "fatal", "cancer", "toxic", "poison", "disease",
                     "headaches", "lifespan", "medical study", "health risk", "illness",
                     "hospitalized", "CDC", "FDA warning", "recall", "contaminated"],
        "severity": 0.92,
        "impact_multiplier": -0.35
    },
    "labor_critical": {
        "keywords": ["forced labor", "slave", "child labor", "unpaid", "exploitation",
                     "human rights", "abuse", "sweatshop", "unsafe conditions", "strike",
                     "boycott", "workers protest", "union action", "OSHA violation"],
        "severity": 0.88,
        "impact_multiplier": -0.3
    },
    "financial_severe": {
        "keywords": ["bankruptcy", "collapse", "crash", "default", "insolvent", 
                     "liquidation", "worthless", "ponzi", "scam", "scandal", "losses",
                     "miss earnings", "revenue decline", "profit warning"],
        "severity": 0.9,
        "impact_multiplier": -0.45
    },
    "reputation_severe": {
        "keywords": ["boycott", "protest", "outrage", "backlash", "controversy",
                     "unethical", "immoral", "corrupt", "dishonest", "lying", "cover-up",
                     "whistleblower", "leaked", "exposed", "scandal"],
        "severity": 0.8,
        "impact_multiplier": -0.25
    },
    "operational_severe": {
        "keywords": ["shutdown", "halted", "suspended", "banned", "blocked", "failed",
                     "defective", "malfunction", "breach", "hacked", "cyberattack",
                     "data leak", "outage", "disruption"],
        "severity": 0.85,
        "impact_multiplier": -0.3
    }
}

# POSITIVE KEYWORDS FOR BALANCE
POSITIVE_KEYWORDS = {
    "breakthrough": ["breakthrough", "innovation", "revolutionary", "first", "leading"],
    "financial_positive": ["record earnings", "beat expectations", "profit surge", "revenue growth"],
    "strategic": ["acquisition", "partnership", "expansion", "market leader", "dominant"]
}

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer with improved accuracy"""

    def __init__(self, classiq_client=None):
        self.classiq_client = classiq_client
        self.qnlp_model = QuantumNLPModel(classiq_client) if classiq_client else None
        self.finbert = None
        self.nlp = None
        self._initialized = False
        
        # Initialize FinBERT for financial sentiment
        if TRANSFORMERS_AVAILABLE:
            try:
                self.finbert = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("FinBERT initialized successfully")
            except Exception as e:
                logger.warning(f"FinBERT initialization failed: {e}")
                self.finbert = None
        
        # Initialize spaCy for entity recognition
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.warning("spaCy model not available")

    async def initialize(self):
        """Initialize the sentiment analyzer"""
        self._initialized = True
        if self.qnlp_model:
            await self.qnlp_model.initialize()
        logger.info("Sentiment analyzer initialized")

    async def analyze_batch(self, news_texts: List[str]) -> List[SentimentAnalysis]:
        """Analyze sentiment for multiple news items"""
        results = []
        for text in news_texts:
            result = await self.analyze_single(text)
            results.append(result)
        return results

    async def analyze_single(self, text: str) -> SentimentAnalysis:
        """Enhanced single text sentiment analysis"""
        
        # Step 1: Crisis Detection First (CRITICAL)
        crisis_result = self._detect_crisis_keywords(text)
        
        # Step 2: Get FinBERT sentiment if available
        finbert_sentiment = self._get_finbert_sentiment(text)
        
        # Step 3: Calculate combined sentiment score
        base_score = finbert_sentiment.get("score", 0.0)
        
        # Apply crisis impact
        if crisis_result["is_crisis"]:
            # Override with negative sentiment for crisis
            crisis_impact = crisis_result["severity"] * crisis_result["impact_multiplier"]
            final_score = min(base_score + crisis_impact, -0.3)  # Ensure negative
            
            # Map to sentiment type
            if final_score < -0.6:
                sentiment = SentimentType.VERY_NEGATIVE
            elif final_score < -0.2:
                sentiment = SentimentType.NEGATIVE
            else:
                sentiment = SentimentType.NEGATIVE  # At minimum, negative for crisis
                
            confidence = min(0.95, crisis_result["severity"])
        else:
            # Non-crisis sentiment mapping
            final_score = base_score
            
            if final_score > 0.3:
                sentiment = SentimentType.POSITIVE
            elif final_score > 0.6:
                sentiment = SentimentType.VERY_POSITIVE
            elif final_score < -0.3:
                sentiment = SentimentType.NEGATIVE
            elif final_score < -0.6:
                sentiment = SentimentType.VERY_NEGATIVE
            else:
                sentiment = SentimentType.NEUTRAL
                
            confidence = finbert_sentiment.get("confidence", 0.75)
        
        # Step 4: Extract entities and keywords
        entities = self._extract_entities(text)
        key_phrases = self._extract_key_phrases(text)
        
        # Step 5: Get market impact keywords
        market_keywords = self._get_market_impact_keywords(text, crisis_result)
        
        # Step 6: Quantum enhancement if available
        quantum_vector = []
        if self.qnlp_model and self.classiq_client and hasattr(self.classiq_client, 'is_ready') and self.classiq_client.is_ready():
            try:
                quantum_vector = await self._get_quantum_sentiment_vector(text)
            except:
                quantum_vector = []
        
        return SentimentAnalysis(
            sentiment=sentiment,
            confidence=confidence,
            quantum_sentiment_vector=quantum_vector,
            classical_sentiment_score=final_score,
            entities_detected=entities,
            key_phrases=key_phrases,
            market_impact_keywords=market_keywords,
            crisis_indicators=crisis_result if crisis_result["is_crisis"] else None
        )

    def _detect_crisis_keywords(self, text: str) -> Dict[str, Any]:
        """Detect crisis keywords with severity scoring"""
        text_lower = text.lower()
        detected_crises = []
        total_severity = 0.0
        impact_multiplier = 0.0
        
        for category, details in CRISIS_KEYWORDS.items():
            for keyword in details["keywords"]:
                if keyword.lower() in text_lower:
                    detected_crises.append({
                        "category": category,
                        "keyword": keyword,
                        "severity": details["severity"]
                    })
                    total_severity = max(total_severity, details["severity"])
                    impact_multiplier = min(impact_multiplier, details["impact_multiplier"])
        
        is_crisis = len(detected_crises) > 0
        
        return {
            "is_crisis": is_crisis,
            "detected_keywords": [d["keyword"] for d in detected_crises],
            "categories": list(set(d["category"] for d in detected_crises)),
            "severity": total_severity,
            "impact_multiplier": impact_multiplier,
            "crisis_count": len(detected_crises)
        }

    def _get_finbert_sentiment(self, text: str) -> Dict[str, float]:
        """Get FinBERT sentiment analysis"""
        if not self.finbert:
            return {"score": 0.0, "confidence": 0.5}
        
        try:
            result = self.finbert(text[:512])[0]  # Truncate for BERT limit
            
            # Map FinBERT labels to scores
            label_map = {
                "positive": 0.7,
                "negative": -0.7,
                "neutral": 0.0
            }
            
            score = label_map.get(result["label"].lower(), 0.0)
            confidence = result["score"]
            
            return {"score": score, "confidence": confidence}
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {"score": 0.0, "confidence": 0.5}

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:1000])  # Limit text length
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT"]:
                    entities.append({
                        "text": ent.text,
                        "type": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
            
            return entities[:10]  # Limit number of entities
        except:
            return []

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple keyword extraction
        important_words = []
        
        # Check for crisis keywords
        text_lower = text.lower()
        for category, details in CRISIS_KEYWORDS.items():
            for keyword in details["keywords"]:
                if keyword in text_lower:
                    important_words.append(keyword)
        
        # Check for positive keywords
        for category, keywords in POSITIVE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    important_words.append(keyword)
        
        return list(set(important_words))[:10]

    def _get_market_impact_keywords(self, text: str, crisis_result: Dict) -> List[str]:
        """Get keywords that impact market"""
        impact_keywords = []
        
        if crisis_result["is_crisis"]:
            impact_keywords.extend(crisis_result["detected_keywords"][:5])
        
        # Add financial terms
        financial_terms = ["earnings", "revenue", "profit", "loss", "growth", "decline"]
        text_lower = text.lower()
        
        for term in financial_terms:
            if term in text_lower:
                impact_keywords.append(term)
        
        return list(set(impact_keywords))[:10]

    async def _get_quantum_sentiment_vector(self, text: str) -> List[float]:
        """Get quantum-enhanced sentiment vector"""
        try:
            if self.qnlp_model:
                quantum_encoding = await self.qnlp_model.encode_text_quantum(text)
                return quantum_encoding.get("quantum_features", [])
        except:
            return []
        
        return []


# Export the enhanced analyzer (maintain compatibility)
SentimentAnalyzer = EnhancedSentimentAnalyzer

# Keep all other methods from the original class for backward compatibility
class CrisisDetector:
    """Dedicated crisis detection component"""
    
    def __init__(self):
        self.crisis_keywords = CRISIS_KEYWORDS
        
    def detect_crisis_level(self, text: str) -> float:
        """Detect crisis level from 0.0 to 1.0"""
        crisis_result = self._detect_crisis_keywords(text)
        return crisis_result["severity"] if crisis_result["is_crisis"] else 0.0
    
    def _detect_crisis_keywords(self, text: str) -> Dict[str, Any]:
        """Detect crisis keywords with severity scoring"""
        text_lower = text.lower()
        detected_crises = []
        total_severity = 0.0
        
        for category, details in self.crisis_keywords.items():
            for keyword in details["keywords"]:
                if keyword.lower() in text_lower:
                    detected_crises.append({
                        "category": category,
                        "keyword": keyword,
                        "severity": details["severity"]
                    })
                    total_severity = max(total_severity, details["severity"])
        
        is_crisis = len(detected_crises) > 0
        
        return {
            "is_crisis": is_crisis,
            "detected_keywords": [d["keyword"] for d in detected_crises],
            "categories": list(set(d["category"] for d in detected_crises)),
            "severity": total_severity,
            "crisis_count": len(detected_crises)
        }
