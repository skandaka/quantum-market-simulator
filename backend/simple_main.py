# Simple working version of the app
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Quantum Market Simulator",
    description="Quantum computing for financial market prediction",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple schemas
class NewsInput(BaseModel):
    content: str
    source_type: str = "headline"

class SimulationRequest(BaseModel):
    news_inputs: List[NewsInput]
    target_assets: List[str]
    simulation_method: str = "hybrid_qml"
    time_horizon_days: int = 7
    num_scenarios: int = 1000

class MarketPrediction(BaseModel):
    asset: str
    expected_return: float
    volatility: float
    confidence: float
    time_horizon_days: int
    scenarios_count: int

class SimulationResponse(BaseModel):
    request_id: str
    timestamp: datetime
    news_analysis: List[Dict[str, Any]]
    market_predictions: List[MarketPrediction]
    quantum_metrics: Optional[Dict[str, Any]] = None
    warnings: List[str] = []

# Routes
@app.get("/")
async def root():
    return {
        "application": "Quantum Market Simulator",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "simulate": "/api/v1/simulate"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "operational",
            "quantum_simulator": "ready"
        }
    }

@app.post("/api/v1/simulate", response_model=SimulationResponse)
async def simulate_market(request: SimulationRequest):
    """Market simulation endpoint"""
    try:
        logger.info(f"Processing simulation for {len(request.news_inputs)} news items")

        # Mock sentiment analysis
        news_analysis = []
        for news in request.news_inputs:
            sentiment_score = random.uniform(-1, 1)
            if sentiment_score > 0.3:
                sentiment = "positive"
            elif sentiment_score < -0.3:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            news_analysis.append({
                "headline": news.content[:200],
                "sentiment": sentiment,
                "confidence": random.uniform(0.7, 0.95),
                "sentiment_score": sentiment_score,
                "entities": ["market", "finance"],
                "key_phrases": ["trading", "market movement"]
            })

        # Mock market predictions
        predictions = []
        for asset in request.target_assets:
            # Generate realistic predictions based on sentiment
            avg_sentiment = sum(a.get("sentiment_score", 0) for a in news_analysis) / len(news_analysis) if news_analysis else 0

            base_return = avg_sentiment * 0.1  # Sentiment impact
            noise = random.uniform(-0.05, 0.05)
            expected_return = base_return + noise

            prediction = MarketPrediction(
                asset=asset,
                expected_return=expected_return,
                volatility=random.uniform(0.15, 0.35),
                confidence=random.uniform(0.6, 0.9),
                time_horizon_days=request.time_horizon_days,
                scenarios_count=request.num_scenarios
            )
            predictions.append(prediction)

        # Mock quantum metrics
        quantum_metrics = {
            "quantum_advantage": round(random.uniform(1.1, 2.5), 2),
            "execution_time": round(random.uniform(0.1, 1.0), 2),
            "qubits_used": random.randint(4, 12),
            "circuit_depth": random.randint(10, 50),
            "method": request.simulation_method
        }

        # Generate warnings
        warnings = []
        extreme_predictions = [p for p in predictions if abs(p.expected_return) > 0.1]
        if extreme_predictions:
            warnings.append(f"Extreme price movements predicted for {len(extreme_predictions)} assets")

        if any(p.confidence < 0.7 for p in predictions):
            warnings.append("Low confidence predictions detected")

        response = SimulationResponse(
            request_id=f"sim_{int(datetime.utcnow().timestamp())}",
            timestamp=datetime.utcnow(),
            news_analysis=news_analysis,
            market_predictions=predictions,
            quantum_metrics=quantum_metrics,
            warnings=warnings
        )

        logger.info("Simulation completed successfully")
        return response

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

@app.get("/api/v1/supported-assets")
async def get_supported_assets():
    """Get supported trading assets"""
    return {
        "stocks": [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
            {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "E-Commerce"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive"},
            {"symbol": "META", "name": "Meta Platforms", "sector": "Technology"},
            {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Semiconductors"}
        ],
        "crypto": [
            {"symbol": "BTC-USD", "name": "Bitcoin", "sector": "Cryptocurrency"},
            {"symbol": "ETH-USD", "name": "Ethereum", "sector": "Cryptocurrency"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Quantum Market Simulator...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
