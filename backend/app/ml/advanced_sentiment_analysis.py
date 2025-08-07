"""
PHASE 4.3: ADVANCED SENTIMENT ANALYSIS WITH MARKET CONTEXT
Context-aware sentiment analysis that considers market conditions and temporal dynamics
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass

try:
    from classiq import qfunc, QBit, QArray, Output, H, RY, RZ, CX, control
    CLASSIQ_AVAILABLE = True
except ImportError:
    CLASSIQ_AVAILABLE = False

from app.quantum.classiq_client import ClassiqClient
from app.quantum.qnlp_model import QuantumNLPModel
from app.models.schemas import SentimentAnalysis
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Market context information for sentiment analysis"""
    volatility_regime: str  # "low", "medium", "high"
    market_trend: str  # "bullish", "bearish", "neutral"
    sector_performance: Dict[str, float]
    economic_indicators: Dict[str, float]
    market_stress_level: float  # 0.0 to 1.0
    trading_volume_ratio: float  # relative to average
    time_of_day: str  # "pre-market", "market-hours", "post-market"
    

@dataclass
class ContextualSentiment:
    """Enhanced sentiment analysis with market context"""
    raw_sentiment: str
    context_adjusted_sentiment: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    market_impact_score: float  # -1.0 to 1.0
    temporal_decay_factor: float  # 0.0 to 1.0
    sector_relevance: Dict[str, float]
    urgency_level: str  # "low", "medium", "high", "critical"
    quantum_coherence: Optional[float] = None
    

class AdvancedSentimentAnalyzer:
    """
    PHASE 4.3: Advanced sentiment analysis that incorporates market context,
    temporal dynamics, and quantum-enhanced processing
    """
    
    def __init__(self, classiq_client: ClassiqClient):
        self.classiq_client = classiq_client
        self.quantum_nlp = QuantumNLPModel(classiq_client) if classiq_client else None
        
        # Sentiment lexicons and patterns
        self.market_sentiment_lexicon = self._initialize_market_lexicon()
        self.sector_keywords = self._initialize_sector_keywords()
        self.temporal_patterns = self._initialize_temporal_patterns()
        self.market_regime_patterns = self._initialize_market_regime_patterns()
        
        # Context analysis parameters
        self.context_weights = {
            "market_volatility": 0.25,
            "sector_performance": 0.20,
            "temporal_proximity": 0.15,
            "market_stress": 0.20,
            "volume_impact": 0.10,
            "quantum_enhancement": 0.10
        }
        
        # Historical context memory
        self.sentiment_history = []
        self.market_context_history = []
        
    async def initialize(self):
        """Initialize the advanced sentiment analyzer"""
        logger.info("🔄 Initializing Advanced Sentiment Analyzer")
        
        try:
            if self.quantum_nlp:
                await self.quantum_nlp.initialize()
                logger.info("✅ Quantum NLP component initialized")
            
            # Initialize context analyzers
            await self._initialize_context_analyzers()
            
            logger.info("🎯 Advanced sentiment analyzer ready")
            
        except Exception as e:
            logger.error(f"❌ Advanced sentiment analyzer initialization failed: {e}")
            raise

    async def _initialize_context_analyzers(self):
        """Initialize context analysis components"""
        try:
            # Initialize pattern matchers
            self.urgency_patterns = [
                (r'\b(breaking|urgent|alert|immediate|critical)\b', 'critical'),
                (r'\b(significant|major|important|substantial)\b', 'high'),
                (r'\b(notable|moderate|some)\b', 'medium'),
                (r'\b(minor|slight|small)\b', 'low')
            ]
            
            # Initialize sector impact calculators
            self.sector_impact_weights = {
                "technology": 1.2,
                "finance": 1.1,
                "healthcare": 1.0,
                "energy": 1.3,
                "consumer": 0.9,
                "industrial": 1.0
            }
            
            logger.info("Context analyzers initialized")
            
        except Exception as e:
            logger.error(f"Context analyzer initialization failed: {e}")
            raise

    async def analyze_contextual_sentiment(
        self,
        news_texts: List[str],
        market_context: MarketContext,
        target_assets: List[str],
        timestamp: Optional[datetime] = None
    ) -> List[ContextualSentiment]:
        """
        PHASE 4.3.1: Analyze sentiment with full market context awareness
        """
        try:
            logger.info(f"🧠 Analyzing contextual sentiment for {len(news_texts)} news items")
            
            if timestamp is None:
                timestamp = datetime.now()
            
            contextual_sentiments = []
            
            for i, news_text in enumerate(news_texts):
                try:
                    # Step 1: Extract raw sentiment
                    raw_sentiment = await self._extract_raw_sentiment(news_text)
                    
                    # Step 2: Apply quantum enhancement if available
                    quantum_enhanced = await self._apply_quantum_sentiment_enhancement(news_text, raw_sentiment)
                    
                    # Step 3: Apply market context
                    context_adjusted = await self._apply_market_context(
                        quantum_enhanced, market_context, target_assets
                    )
                    
                    # Step 4: Calculate temporal dynamics
                    temporal_adjusted = await self._apply_temporal_dynamics(
                        context_adjusted, timestamp, i
                    )
                    
                    # Step 5: Determine sector relevance
                    sector_relevance = await self._calculate_sector_relevance(
                        news_text, target_assets
                    )
                    
                    # Step 6: Calculate urgency and impact
                    urgency_impact = await self._calculate_urgency_and_impact(
                        news_text, temporal_adjusted, market_context
                    )
                    
                    # Step 7: Create contextual sentiment object
                    contextual_sentiment = ContextualSentiment(
                        raw_sentiment=raw_sentiment["sentiment"],
                        context_adjusted_sentiment=temporal_adjusted["adjusted_sentiment"],
                        sentiment_score=temporal_adjusted["adjusted_score"],
                        confidence=temporal_adjusted["confidence"],
                        market_impact_score=urgency_impact["market_impact"],
                        temporal_decay_factor=temporal_adjusted["decay_factor"],
                        sector_relevance=sector_relevance,
                        urgency_level=urgency_impact["urgency_level"],
                        quantum_coherence=quantum_enhanced.get("quantum_coherence")
                    )
                    
                    contextual_sentiments.append(contextual_sentiment)
                    
                except Exception as e:
                    logger.error(f"Failed to analyze sentiment for news item {i}: {e}")
                    # Add fallback sentiment
                    contextual_sentiments.append(self._create_fallback_sentiment())
            
            # Step 8: Apply cross-news correlation analysis
            correlated_sentiments = await self._apply_cross_news_correlation(
                contextual_sentiments, market_context
            )
            
            # Step 9: Update historical context
            self._update_sentiment_history(correlated_sentiments, market_context, timestamp)
            
            logger.info(f"✅ Contextual sentiment analysis completed: {len(correlated_sentiments)} sentiments processed")
            return correlated_sentiments
            
        except Exception as e:
            logger.error(f"❌ Contextual sentiment analysis failed: {e}")
            return [self._create_fallback_sentiment() for _ in news_texts]

    async def _extract_raw_sentiment(self, news_text: str) -> Dict[str, Any]:
        """
        PHASE 4.3.2: Extract raw sentiment from text
        """
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(news_text)
            
            # Apply market-specific sentiment lexicon
            lexicon_sentiment = self._calculate_lexicon_sentiment(cleaned_text)
            
            # Extract key financial entities and events
            financial_entities = self._extract_financial_entities(cleaned_text)
            
            # Calculate base sentiment score
            base_score = lexicon_sentiment["score"]
            base_sentiment = self._score_to_sentiment(base_score)
            
            return {
                "sentiment": base_sentiment,
                "score": base_score,
                "confidence": lexicon_sentiment["confidence"],
                "financial_entities": financial_entities,
                "key_phrases": lexicon_sentiment["key_phrases"]
            }
            
        except Exception as e:
            logger.error(f"Raw sentiment extraction failed: {e}")
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.5}

    async def _apply_quantum_sentiment_enhancement(
        self, 
        news_text: str, 
        raw_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        PHASE 4.3.3: Apply quantum enhancement to sentiment analysis
        """
        try:
            enhanced_sentiment = raw_sentiment.copy()
            
            if self.quantum_nlp:
                # Encode text for quantum processing
                quantum_encoding = await self.quantum_nlp.encode_text_quantum(news_text)
                
                # Apply quantum sentiment classification
                quantum_result = await self.quantum_nlp.quantum_sentiment_classification(quantum_encoding)
                
                if quantum_result:
                    # Combine quantum and classical results
                    quantum_score = self._sentiment_to_score(quantum_result.get("predicted_sentiment", "neutral"))
                    quantum_confidence = quantum_result.get("confidence", 0.5)
                    
                    # Weighted combination
                    classical_weight = 0.6
                    quantum_weight = 0.4
                    
                    combined_score = (classical_weight * raw_sentiment["score"] + 
                                    quantum_weight * quantum_score)
                    
                    combined_confidence = (classical_weight * raw_sentiment["confidence"] + 
                                         quantum_weight * quantum_confidence)
                    
                    enhanced_sentiment.update({
                        "score": combined_score,
                        "confidence": combined_confidence,
                        "sentiment": self._score_to_sentiment(combined_score),
                        "quantum_coherence": quantum_result.get("entanglement_measure", 0.5),
                        "quantum_enhancement": True
                    })
                    
                    logger.info(f"🔬 Quantum enhancement applied: {raw_sentiment['score']:.3f} → {combined_score:.3f}")
            
            return enhanced_sentiment
            
        except Exception as e:
            logger.error(f"Quantum sentiment enhancement failed: {e}")
            return raw_sentiment

    async def _apply_market_context(
        self,
        sentiment_data: Dict[str, Any],
        market_context: MarketContext,
        target_assets: List[str]
    ) -> Dict[str, Any]:
        """
        PHASE 4.3.4: Apply market context to adjust sentiment
        """
        try:
            context_adjusted = sentiment_data.copy()
            
            base_score = sentiment_data["score"]
            base_confidence = sentiment_data["confidence"]
            
            # Market volatility adjustment
            volatility_adjustment = self._calculate_volatility_adjustment(
                base_score, market_context.volatility_regime
            )
            
            # Market trend adjustment
            trend_adjustment = self._calculate_trend_adjustment(
                base_score, market_context.market_trend
            )
            
            # Market stress adjustment
            stress_adjustment = self._calculate_stress_adjustment(
                base_score, market_context.market_stress_level
            )
            
            # Volume adjustment
            volume_adjustment = self._calculate_volume_adjustment(
                base_score, market_context.trading_volume_ratio
            )
            
            # Time-of-day adjustment
            temporal_adjustment = self._calculate_temporal_market_adjustment(
                base_score, market_context.time_of_day
            )
            
            # Combine all adjustments
            total_adjustment = (
                volatility_adjustment * self.context_weights["market_volatility"] +
                trend_adjustment * self.context_weights["sector_performance"] +
                stress_adjustment * self.context_weights["market_stress"] +
                volume_adjustment * self.context_weights["volume_impact"] +
                temporal_adjustment * self.context_weights["temporal_proximity"]
            )
            
            # Apply adjustment with bounds
            adjusted_score = np.clip(base_score + total_adjustment, -1.0, 1.0)
            
            # Adjust confidence based on market certainty
            market_uncertainty = market_context.market_stress_level + (
                0.3 if market_context.volatility_regime == "high" else 
                0.1 if market_context.volatility_regime == "medium" else 0.0
            )
            
            adjusted_confidence = base_confidence * (1.0 - market_uncertainty * 0.3)
            
            context_adjusted.update({
                "score": adjusted_score,
                "confidence": adjusted_confidence,
                "sentiment": self._score_to_sentiment(adjusted_score),
                "market_adjustments": {
                    "volatility": volatility_adjustment,
                    "trend": trend_adjustment,
                    "stress": stress_adjustment,
                    "volume": volume_adjustment,
                    "temporal": temporal_adjustment,
                    "total": total_adjustment
                }
            })
            
            return context_adjusted
            
        except Exception as e:
            logger.error(f"Market context application failed: {e}")
            return sentiment_data

    async def _apply_temporal_dynamics(
        self,
        sentiment_data: Dict[str, Any],
        timestamp: datetime,
        news_index: int
    ) -> Dict[str, Any]:
        """
        PHASE 4.3.5: Apply temporal dynamics and decay factors
        """
        try:
            temporal_adjusted = sentiment_data.copy()
            
            # Calculate time-based decay factor
            now = datetime.now()
            time_diff = (now - timestamp).total_seconds() / 3600  # hours
            
            # Exponential decay with half-life of 6 hours for financial news
            half_life = 6.0
            decay_factor = np.exp(-np.log(2) * time_diff / half_life)
            
            # News ordering impact (more recent news in the list is more impactful)
            ordering_factor = 1.0 - (news_index * 0.05)  # 5% reduction per position
            ordering_factor = max(ordering_factor, 0.5)  # Minimum 50% impact
            
            # Market hours amplification
            market_hours_factor = self._calculate_market_hours_factor(timestamp)
            
            # Combined temporal factor
            total_temporal_factor = decay_factor * ordering_factor * market_hours_factor
            
            # Apply temporal adjustment
            adjusted_score = sentiment_data["score"] * total_temporal_factor
            
            # Confidence adjustment (recent news is more reliable)
            confidence_boost = min(decay_factor, 1.0) * 0.1
            adjusted_confidence = min(sentiment_data["confidence"] + confidence_boost, 1.0)
            
            temporal_adjusted.update({
                "adjusted_score": adjusted_score,
                "adjusted_sentiment": self._score_to_sentiment(adjusted_score),
                "confidence": adjusted_confidence,
                "decay_factor": total_temporal_factor,
                "temporal_factors": {
                    "time_decay": decay_factor,
                    "ordering": ordering_factor,
                    "market_hours": market_hours_factor
                }
            })
            
            return temporal_adjusted
            
        except Exception as e:
            logger.error(f"Temporal dynamics application failed: {e}")
            return sentiment_data

    async def _calculate_sector_relevance(
        self,
        news_text: str,
        target_assets: List[str]
    ) -> Dict[str, float]:
        """
        PHASE 4.3.6: Calculate sector relevance scores
        """
        try:
            sector_relevance = {}
            
            # Extract sectors from target assets (simplified)
            asset_sectors = self._map_assets_to_sectors(target_assets)
            
            # Calculate relevance for each sector
            for sector in asset_sectors:
                relevance_score = 0.0
                
                # Keyword matching
                sector_keywords = self.sector_keywords.get(sector, [])
                text_lower = news_text.lower()
                
                keyword_matches = sum(1 for keyword in sector_keywords if keyword in text_lower)
                keyword_relevance = min(keyword_matches / max(len(sector_keywords), 1) * 2, 1.0)
                
                # Company/asset name matching
                asset_names = [asset for asset in target_assets if self._get_asset_sector(asset) == sector]
                asset_matches = sum(1 for asset in asset_names if asset.lower() in text_lower)
                asset_relevance = min(asset_matches / max(len(asset_names), 1), 1.0)
                
                # Combined relevance
                relevance_score = 0.6 * keyword_relevance + 0.4 * asset_relevance
                sector_relevance[sector] = relevance_score
            
            return sector_relevance
            
        except Exception as e:
            logger.error(f"Sector relevance calculation failed: {e}")
            return {sector: 0.5 for sector in ["technology", "finance", "healthcare"]}

    async def _calculate_urgency_and_impact(
        self,
        news_text: str,
        sentiment_data: Dict[str, Any],
        market_context: MarketContext
    ) -> Dict[str, Any]:
        """
        PHASE 4.3.7: Calculate urgency level and market impact
        """
        try:
            # Detect urgency from text patterns
            urgency_level = "low"
            urgency_score = 0.0
            
            for pattern, level in self.urgency_patterns:
                if re.search(pattern, news_text, re.IGNORECASE):
                    level_scores = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
                    if level_scores[level] > urgency_score:
                        urgency_score = level_scores[level]
                        urgency_level = level
            
            # Calculate market impact based on sentiment strength and urgency
            sentiment_magnitude = abs(sentiment_data.get("adjusted_score", 0))
            confidence = sentiment_data.get("confidence", 0.5)
            
            base_impact = sentiment_magnitude * confidence
            urgency_multiplier = 1.0 + urgency_score
            
            # Market context multipliers
            volatility_multiplier = {
                "low": 0.8,
                "medium": 1.0,
                "high": 1.3
            }.get(market_context.volatility_regime, 1.0)
            
            stress_multiplier = 1.0 + market_context.market_stress_level * 0.5
            
            # Final market impact score
            market_impact = base_impact * urgency_multiplier * volatility_multiplier * stress_multiplier
            market_impact = np.clip(market_impact, -1.0, 1.0)
            
            # Apply sentiment direction
            if sentiment_data.get("adjusted_score", 0) < 0:
                market_impact = -abs(market_impact)
            else:
                market_impact = abs(market_impact)
            
            return {
                "urgency_level": urgency_level,
                "urgency_score": urgency_score,
                "market_impact": market_impact,
                "impact_factors": {
                    "sentiment_magnitude": sentiment_magnitude,
                    "confidence": confidence,
                    "urgency_multiplier": urgency_multiplier,
                    "volatility_multiplier": volatility_multiplier,
                    "stress_multiplier": stress_multiplier
                }
            }
            
        except Exception as e:
            logger.error(f"Urgency and impact calculation failed: {e}")
            return {"urgency_level": "medium", "market_impact": 0.0}

    async def _apply_cross_news_correlation(
        self,
        sentiments: List[ContextualSentiment],
        market_context: MarketContext
    ) -> List[ContextualSentiment]:
        """
        PHASE 4.3.8: Apply cross-news correlation analysis
        """
        try:
            if len(sentiments) < 2:
                return sentiments
            
            correlated_sentiments = []
            
            for i, sentiment in enumerate(sentiments):
                # Calculate correlation with other news items
                correlations = []
                
                for j, other_sentiment in enumerate(sentiments):
                    if i != j:
                        # Sector overlap correlation
                        sector_overlap = self._calculate_sector_overlap(
                            sentiment.sector_relevance, 
                            other_sentiment.sector_relevance
                        )
                        
                        # Sentiment direction correlation
                        sentiment_correlation = np.sign(sentiment.sentiment_score) * np.sign(other_sentiment.sentiment_score)
                        
                        # Urgency correlation
                        urgency_scores = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
                        urgency_correlation = abs(
                            urgency_scores[sentiment.urgency_level] - 
                            urgency_scores[other_sentiment.urgency_level]
                        )
                        
                        # Combined correlation
                        total_correlation = (
                            sector_overlap * 0.4 +
                            (sentiment_correlation + 1) / 2 * 0.4 +  # Normalize to 0-1
                            (1 - urgency_correlation) * 0.2  # Invert urgency difference
                        )
                        
                        correlations.append(total_correlation)
                
                # Apply correlation adjustment
                if correlations:
                    avg_correlation = np.mean(correlations)
                    correlation_boost = avg_correlation * 0.1  # Max 10% boost
                    
                    # Amplify sentiment if highly correlated with other news
                    adjusted_score = sentiment.sentiment_score * (1 + correlation_boost)
                    adjusted_score = np.clip(adjusted_score, -1.0, 1.0)
                    
                    # Boost confidence if correlated
                    adjusted_confidence = min(sentiment.confidence + correlation_boost * 0.5, 1.0)
                    
                    # Create new sentiment with correlation adjustments
                    correlated_sentiment = ContextualSentiment(
                        raw_sentiment=sentiment.raw_sentiment,
                        context_adjusted_sentiment=self._score_to_sentiment(adjusted_score),
                        sentiment_score=adjusted_score,
                        confidence=adjusted_confidence,
                        market_impact_score=sentiment.market_impact_score * (1 + correlation_boost),
                        temporal_decay_factor=sentiment.temporal_decay_factor,
                        sector_relevance=sentiment.sector_relevance,
                        urgency_level=sentiment.urgency_level,
                        quantum_coherence=sentiment.quantum_coherence
                    )
                    
                    correlated_sentiments.append(correlated_sentiment)
                else:
                    correlated_sentiments.append(sentiment)
            
            return correlated_sentiments
            
        except Exception as e:
            logger.error(f"Cross-news correlation analysis failed: {e}")
            return sentiments

    # Helper methods for context analysis
    def _initialize_market_lexicon(self) -> Dict[str, float]:
        """Initialize market-specific sentiment lexicon"""
        return {
            # Positive financial terms
            "profit": 0.8, "gain": 0.7, "growth": 0.6, "bullish": 0.9,
            "rally": 0.8, "surge": 0.7, "outperform": 0.6, "beat": 0.7,
            "strong": 0.6, "positive": 0.5, "optimistic": 0.6, "upgrade": 0.7,
            
            # Negative financial terms
            "loss": -0.8, "decline": -0.7, "bearish": -0.9, "crash": -1.0,
            "fall": -0.6, "drop": -0.6, "weak": -0.5, "negative": -0.5,
            "pessimistic": -0.6, "downgrade": -0.7, "concern": -0.4, "risk": -0.3,
            
            # Neutral/context terms
            "stable": 0.0, "neutral": 0.0, "unchanged": 0.0, "flat": 0.0,
            "mixed": 0.0, "uncertain": -0.1, "volatile": -0.2
        }

    def _initialize_sector_keywords(self) -> Dict[str, List[str]]:
        """Initialize sector-specific keywords"""
        return {
            "technology": ["software", "tech", "AI", "cloud", "digital", "innovation", "startup"],
            "finance": ["bank", "credit", "loan", "mortgage", "financial", "investment", "insurance"],
            "healthcare": ["medical", "pharma", "drug", "treatment", "healthcare", "biotech", "clinical"],
            "energy": ["oil", "gas", "renewable", "energy", "power", "electricity", "solar", "wind"],
            "consumer": ["retail", "consumer", "shopping", "brand", "sales", "demand", "spending"],
            "industrial": ["manufacturing", "industrial", "construction", "infrastructure", "logistics"]
        }

    def _initialize_temporal_patterns(self) -> Dict[str, float]:
        """Initialize temporal pattern weights"""
        return {
            "earnings_season": 1.3,
            "market_open": 1.2,
            "market_close": 1.1,
            "after_hours": 0.8,
            "weekend": 0.6,
            "holiday": 0.5
        }

    def _initialize_market_regime_patterns(self) -> Dict[str, Dict[str, float]]:
        """Initialize market regime adjustment patterns"""
        return {
            "bull_market": {"positive": 1.1, "negative": 0.9},
            "bear_market": {"positive": 0.9, "negative": 1.2},
            "volatile_market": {"positive": 1.0, "negative": 1.0},
            "stable_market": {"positive": 1.0, "negative": 1.0}
        }

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Basic text cleaning
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip().lower()

    def _calculate_lexicon_sentiment(self, text: str) -> Dict[str, Any]:
        """Calculate sentiment using market lexicon"""
        words = text.split()
        sentiment_scores = []
        key_phrases = []
        
        for word in words:
            if word in self.market_sentiment_lexicon:
                score = self.market_sentiment_lexicon[word]
                sentiment_scores.append(score)
                key_phrases.append(word)
        
        if sentiment_scores:
            avg_score = np.mean(sentiment_scores)
            confidence = min(len(sentiment_scores) / 10.0, 1.0)  # More matches = higher confidence
        else:
            avg_score = 0.0
            confidence = 0.3  # Low confidence for unknown text
        
        return {
            "score": avg_score,
            "confidence": confidence,
            "key_phrases": key_phrases
        }

    def _extract_financial_entities(self, text: str) -> List[str]:
        """Extract financial entities from text"""
        entities = []
        
        # Simple pattern matching for financial entities
        patterns = [
            r'\b[A-Z]{2,5}\b',  # Stock symbols
            r'\$\d+(?:\.\d+)?[BMK]?',  # Dollar amounts
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\bQ[1-4]\b',  # Quarters
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return entities

    def _score_to_sentiment(self, score: float) -> str:
        """Convert numerical score to sentiment label"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        else:
            return "neutral"

    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment label to numerical score"""
        sentiment_map = {"positive": 0.7, "negative": -0.7, "neutral": 0.0}
        return sentiment_map.get(sentiment, 0.0)

    def _calculate_volatility_adjustment(self, base_score: float, volatility_regime: str) -> float:
        """Calculate volatility-based adjustment"""
        adjustments = {"low": 0.0, "medium": 0.1, "high": 0.2}
        adjustment = adjustments.get(volatility_regime, 0.1)
        return np.sign(base_score) * adjustment

    def _calculate_trend_adjustment(self, base_score: float, market_trend: str) -> float:
        """Calculate trend-based adjustment"""
        if market_trend == "bullish":
            return 0.1 if base_score > 0 else -0.05
        elif market_trend == "bearish":
            return -0.1 if base_score < 0 else 0.05
        else:
            return 0.0

    def _calculate_stress_adjustment(self, base_score: float, stress_level: float) -> float:
        """Calculate stress-based adjustment"""
        # High stress amplifies negative sentiment
        if base_score < 0:
            return -stress_level * 0.2
        else:
            return -stress_level * 0.1  # Slight dampening of positive sentiment

    def _calculate_volume_adjustment(self, base_score: float, volume_ratio: float) -> float:
        """Calculate volume-based adjustment"""
        # High volume amplifies sentiment
        volume_effect = (volume_ratio - 1.0) * 0.1
        return np.sign(base_score) * volume_effect

    def _calculate_temporal_market_adjustment(self, base_score: float, time_of_day: str) -> float:
        """Calculate time-of-day adjustment"""
        adjustments = {
            "pre-market": 0.05,
            "market-hours": 0.0,
            "post-market": 0.02
        }
        return adjustments.get(time_of_day, 0.0)

    def _calculate_market_hours_factor(self, timestamp: datetime) -> float:
        """Calculate market hours amplification factor"""
        hour = timestamp.hour
        
        # Market hours (9:30 AM - 4:00 PM EST) get higher weight
        if 9 <= hour <= 16:
            return 1.2
        elif 17 <= hour <= 20:  # After hours
            return 1.0
        else:  # Pre-market and overnight
            return 0.8

    def _map_assets_to_sectors(self, assets: List[str]) -> List[str]:
        """Map assets to their sectors"""
        # Simplified mapping - in reality would use asset database
        sector_mapping = {
            "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
            "JPM": "finance", "BAC": "finance", "WFC": "finance",
            "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare"
        }
        
        sectors = []
        for asset in assets:
            sector = sector_mapping.get(asset, "technology")  # Default to tech
            if sector not in sectors:
                sectors.append(sector)
        
        return sectors

    def _get_asset_sector(self, asset: str) -> str:
        """Get sector for specific asset"""
        sector_mapping = {
            "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
            "JPM": "finance", "BAC": "finance", "WFC": "finance",
            "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare"
        }
        return sector_mapping.get(asset, "technology")

    def _calculate_sector_overlap(self, sector_relevance1: Dict[str, float], sector_relevance2: Dict[str, float]) -> float:
        """Calculate overlap between sector relevance scores"""
        common_sectors = set(sector_relevance1.keys()) & set(sector_relevance2.keys())
        
        if not common_sectors:
            return 0.0
        
        overlaps = []
        for sector in common_sectors:
            overlap = min(sector_relevance1[sector], sector_relevance2[sector])
            overlaps.append(overlap)
        
        return np.mean(overlaps)

    def _create_fallback_sentiment(self) -> ContextualSentiment:
        """Create fallback sentiment for error cases"""
        return ContextualSentiment(
            raw_sentiment="neutral",
            context_adjusted_sentiment="neutral",
            sentiment_score=0.0,
            confidence=0.3,
            market_impact_score=0.0,
            temporal_decay_factor=1.0,
            sector_relevance={"technology": 0.5, "finance": 0.5},
            urgency_level="low"
        )

    def _update_sentiment_history(
        self, 
        sentiments: List[ContextualSentiment], 
        market_context: MarketContext, 
        timestamp: datetime
    ):
        """Update historical sentiment and context data"""
        try:
            # Add to sentiment history
            for sentiment in sentiments:
                self.sentiment_history.append({
                    "timestamp": timestamp,
                    "sentiment": sentiment,
                    "market_context": market_context
                })
            
            # Keep only recent history (last 100 items)
            if len(self.sentiment_history) > 100:
                self.sentiment_history = self.sentiment_history[-100:]
                
        except Exception as e:
            logger.error(f"Sentiment history update failed: {e}")

    def get_sentiment_insights(self) -> Dict[str, Any]:
        """Get insights from historical sentiment analysis"""
        try:
            if not self.sentiment_history:
                return {"status": "No sentiment history available"}
            
            recent_sentiments = self.sentiment_history[-20:]  # Last 20 sentiments
            
            # Calculate averages
            avg_score = np.mean([item["sentiment"].sentiment_score for item in recent_sentiments])
            avg_confidence = np.mean([item["sentiment"].confidence for item in recent_sentiments])
            avg_impact = np.mean([item["sentiment"].market_impact_score for item in recent_sentiments])
            
            # Sentiment distribution
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for item in recent_sentiments:
                sentiment = item["sentiment"].context_adjusted_sentiment
                sentiment_counts[sentiment] += 1
            
            # Urgency distribution
            urgency_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
            for item in recent_sentiments:
                urgency = item["sentiment"].urgency_level
                urgency_counts[urgency] += 1
            
            return {
                "analysis_summary": {
                    "avg_sentiment_score": avg_score,
                    "avg_confidence": avg_confidence,
                    "avg_market_impact": avg_impact,
                    "total_analyzed": len(self.sentiment_history)
                },
                "sentiment_distribution": sentiment_counts,
                "urgency_distribution": urgency_counts,
                "quantum_enhancement_rate": sum(1 for item in recent_sentiments 
                                               if item["sentiment"].quantum_coherence is not None) / len(recent_sentiments)
            }
            
        except Exception as e:
            logger.error(f"Sentiment insights generation failed: {e}")
            return {"error": str(e)}
