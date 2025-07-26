"""Pydantic models for request/response schemas"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
import numpy as np


class NewsSourceType(str, Enum):
    """Supported news source types"""
    HEADLINE = "headline"
    TWEET = "tweet"
    ARTICLE = "article"
    SEC_FILING = "sec_filing"
    EARNINGS_CALL = "earnings_call"
    PRESS_RELEASE = "press_release"


class SentimentType(str, Enum):
    """Sentiment classification types"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class MarketAsset(str, Enum):
    """Supported market assets"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"


class SimulationMethod(str, Enum):
    """Simulation methods available"""
    QUANTUM_MONTE_CARLO = "quantum_monte_carlo"
    QUANTUM_WALK = "quantum_walk"
    HYBRID_QML = "hybrid_qml"
    CLASSICAL_BASELINE = "classical_baseline"


# Request Models
class NewsInput(BaseModel):
    """News input for analysis"""
    content: str = Field(..., min_length=10, max_length=10000)
    source_type: NewsSourceType = NewsSourceType.HEADLINE
    source_url: Optional[HttpUrl] = None
    published_at: Optional[datetime] = None
    author: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

    @validator('content')
    def clean_content(cls, v):
        # Remove excessive whitespace
        return ' '.join(v.split())


class SimulationRequest(BaseModel):
    """Market simulation request"""
    news_inputs: List[NewsInput]
    target_assets: List[str] = Field(..., min_items=1, max_items=10)
    asset_type: MarketAsset = MarketAsset.STOCK
    simulation_method: SimulationMethod = SimulationMethod.HYBRID_QML
    time_horizon_days: int = Field(default=7, ge=1, le=30)
    num_scenarios: int = Field(default=1000, ge=100, le=10000)
    include_quantum_metrics: bool = True
    compare_with_classical: bool = True

    @validator('target_assets')
    def validate_assets(cls, v):
        # Ensure unique assets
        return list(set(v))


class BacktestRequest(BaseModel):
    """Backtesting request"""
    historical_news: List[NewsInput]
    asset: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(default=100000, gt=0)
    position_size: float = Field(default=0.1, gt=0, le=1)


# Response Models
class QuantumMetrics(BaseModel):
    """Quantum computation metrics"""
    circuit_depth: int
    num_qubits: int
    quantum_volume: float
    entanglement_measure: float
    execution_time_ms: float
    hardware_backend: str
    success_probability: float


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results"""
    sentiment: SentimentType
    confidence: float = Field(..., ge=0, le=1)
    quantum_sentiment_vector: List[float]
    classical_sentiment_score: float
    entities_detected: List[Dict[str, str]]
    key_phrases: List[str]
    market_impact_keywords: List[str]


class PriceScenario(BaseModel):
    """Single price scenario"""
    scenario_id: int
    price_path: List[float]
    returns_path: List[float]
    volatility_path: List[float]
    probability_weight: float


class MarketPrediction(BaseModel):
    """Market prediction for an asset"""
    asset: str
    current_price: float
    predicted_scenarios: List[PriceScenario]
    expected_return: float
    volatility: float
    confidence_intervals: Dict[str, Dict[str, float]]  # {confidence_level: {lower, upper}}
    quantum_uncertainty: float
    regime_probabilities: Dict[str, float]  # bull/bear/neutral probabilities


class SimulationResponse(BaseModel):
    """Complete simulation response"""
    request_id: str
    timestamp: datetime
    news_analysis: List[SentimentAnalysis]
    market_predictions: List[MarketPrediction]
    quantum_metrics: Optional[QuantumMetrics]
    classical_comparison: Optional[Dict[str, Any]]
    execution_time_seconds: float
    warnings: List[str] = []


class BacktestResult(BaseModel):
    """Backtesting results"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    performance_metrics: Dict[str, float]


# WebSocket Models
class WSMessage(BaseModel):
    """WebSocket message format"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SimulationProgress(BaseModel):
    """Progress update for long-running simulations"""
    request_id: str
    stage: str
    progress: float = Field(..., ge=0, le=1)
    message: str
    estimated_time_remaining: Optional[float] = None