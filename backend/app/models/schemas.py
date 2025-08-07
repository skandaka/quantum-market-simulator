# backend/app/models/schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class SentimentType(str, Enum):
    """Enhanced sentiment types"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class NewsSourceType(str, Enum):
    """News source types"""
    HEADLINE = "headline"
    ARTICLE = "article"
    TWEET = "tweet"
    SEC_FILING = "sec_filing"
    PDF = "pdf"
    URL = "url"


class NewsInput(BaseModel):
    """Enhanced news input with multiple sources"""
    content: str = Field(..., description="News content text")
    source_type: str = Field(default="headline", description="Type: headline, url, pdf")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    file_name: Optional[str] = Field(None, description="File name if PDF")


class SentimentAnalysis(BaseModel):
    """Enhanced sentiment analysis result"""
    sentiment: SentimentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    quantum_sentiment_vector: List[float] = Field(default_factory=list)
    classical_sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    entities_detected: List[Dict[str, Any]] = Field(default_factory=list)
    key_phrases: List[str] = Field(default_factory=list)
    market_impact_keywords: List[str] = Field(default_factory=list)
    crisis_indicators: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True


class PriceScenario(BaseModel):
    """Price scenario for simulation with quantum support"""
    scenario_id: int
    price_path: List[float]
    returns_path: List[float]
    volatility_path: List[float]
    probability_weight: float = Field(..., ge=0.0, le=1.0)
    quantum_amplitude: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quantum amplitude for this scenario")
    quantum_phase: Optional[float] = Field(None, description="Quantum phase in radians")
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 6) if v is not None else None
        }


class MarketPrediction(BaseModel):
    """Enhanced market prediction with detailed analysis and quantum support"""
    asset: str
    current_price: float
    expected_return: float = Field(..., ge=-1.0, le=1.0)
    volatility: float = Field(..., ge=0.0, le=2.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_scenarios: List[PriceScenario]
    confidence_intervals: Dict[str, Dict[str, float]]
    time_horizon_days: int
    sentiment_impact: float = Field(default=0.0, ge=-1.0, le=1.0)
    prediction_method: str
    explanation: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    is_crisis: Optional[bool] = None
    crisis_severity: Optional[float] = None
    quantum_metrics: Optional[Dict[str, Any]] = Field(None, description="Quantum algorithm performance metrics")
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 6) if v is not None else None
        }


class QuantumMetrics(BaseModel):
    """Quantum computation metrics"""
    quantum_advantage: float = Field(default=1.0, description="Speedup vs classical")
    entanglement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence_time: float = Field(default=0.0, description="Microseconds")
    gate_fidelity: float = Field(default=0.99, ge=0.0, le=1.0)
    circuit_depth: int = Field(default=1, ge=1)
    num_qubits: int = Field(default=1, ge=1)
    execution_time: float = Field(default=0.0, description="Seconds")
    classical_comparison: Optional[Dict[str, float]] = None


class SimulationRequest(BaseModel):
    """Enhanced simulation request"""
    news_inputs: List[NewsInput] = Field(..., min_items=1, max_items=10)
    target_assets: List[str] = Field(..., min_items=1, max_items=10)
    simulation_method: str = Field(
        default="hybrid_qml",
        description="Method: hybrid_qml, quantum_monte_carlo, quantum_walk, classical_baseline"
    )
    time_horizon_days: int = Field(default=7, ge=1, le=30)
    num_scenarios: int = Field(default=1000, ge=100, le=5000)
    compare_with_classical: bool = Field(default=True)
    include_portfolio_optimization: bool = Field(default=False)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "news_inputs": [
                    {
                        "content": "Apple reports record breaking quarterly earnings",
                        "source_type": "headline"
                    }
                ],
                "target_assets": ["AAPL"],
                "simulation_method": "hybrid_qml",
                "time_horizon_days": 7,
                "num_scenarios": 1000
            }
        }


class SimulationResponse(BaseModel):
    """Enhanced simulation response"""
    request_id: str
    timestamp: datetime
    news_analysis: List[Dict[str, Any]]
    market_predictions: List[MarketPrediction]
    quantum_metrics: Optional[QuantumMetrics] = None
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            float: lambda v: round(v, 6) if v is not None else None
        }


class PortfolioPosition(BaseModel):
    """Portfolio position"""
    symbol: str
    shares: float
    avg_cost: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    weight: Optional[float] = None


class PortfolioOptimizationRequest(BaseModel):
    """Portfolio optimization request"""
    positions: List[PortfolioPosition]
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    optimization_method: str = Field(default="quantum_vqe")
    constraints: Optional[Dict[str, Any]] = None


class PortfolioOptimizationResponse(BaseModel):
    """Portfolio optimization response"""
    original_portfolio: Dict[str, Any]
    optimized_portfolio: Dict[str, Any]
    recommended_adjustments: List[Dict[str, Any]]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    quantum_metrics: Optional[QuantumMetrics] = None


class MarketDataPoint(BaseModel):
    """Market data point"""
    timestamp: datetime
    price: float
    volume: float
    volatility: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HistoricalDataRequest(BaseModel):
    """Historical data request"""
    asset: str
    start_date: datetime
    end_date: datetime
    interval: str = Field(default="1d", description="1m, 5m, 1h, 1d")


class HistoricalDataResponse(BaseModel):
    """Historical data response"""
    asset: str
    data_points: List[MarketDataPoint]
    statistics: Dict[str, float]

