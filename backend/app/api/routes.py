# backend/app/api/routes.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import random
import numpy as np
from datetime import datetime
import asyncio
import json
import aiohttp
from bs4 import BeautifulSoup

# Optional imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyPDF2 not available - PDF upload will be disabled")

from app.models.schemas import (
    SimulationRequest,
    SimulationResponse,
    SentimentAnalysis,
    SentimentType,
    PriceScenario,
    MarketPrediction,
    NewsInput
)
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Services will be initialized when needed to avoid circular imports


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_market(request: SimulationRequest):
    """
    Enhanced market simulation endpoint with improved sentiment analysis
    """
    try:
        logger.info(f"Starting simulation for {len(request.news_inputs)} news items")

        # For now, return a mock response until services are properly implemented
        mock_sentiment_results = []
        mock_predictions = []

        for i, news_input in enumerate(request.news_inputs):
            # Mock sentiment analysis
            mock_sentiment_results.append({
                "headline": news_input.content[:200],
                "sentiment": "neutral",
                "confidence": 0.75,
                "entities": [],
                "key_phrases": [],
                "market_keywords": []
            })

        for asset in request.target_assets:
            # Mock prediction with all required fields
            from app.models.schemas import MarketPrediction, PriceScenario
            
            # Generate mock current price
            current_price = 150.0 if asset == "AAPL" else random.uniform(50, 500)
            
            # Generate mock price scenarios
            mock_scenarios = []
            for i in range(min(10, request.num_scenarios)):  # Limit scenarios for performance
                scenario_price = current_price * (1 + random.uniform(-0.1, 0.1))
                mock_scenarios.append(PriceScenario(
                    scenario_id=i,
                    price_path=[current_price, scenario_price],
                    returns_path=[0.0, (scenario_price - current_price) / current_price],
                    volatility_path=[0.2, 0.25],
                    probability_weight=1.0 / min(10, request.num_scenarios)
                ))
            
            # Generate confidence intervals
            confidence_intervals = {
                "95%": {"lower": current_price * 0.9, "upper": current_price * 1.1},
                "90%": {"lower": current_price * 0.92, "upper": current_price * 1.08},
                "68%": {"lower": current_price * 0.95, "upper": current_price * 1.05}
            }
            
            mock_predictions.append(MarketPrediction(
                asset=asset,
                current_price=current_price,
                expected_return=random.uniform(-0.05, 0.05),
                volatility=random.uniform(0.15, 0.35),
                confidence=random.uniform(0.6, 0.9),
                predicted_scenarios=mock_scenarios,
                confidence_intervals=confidence_intervals,
                time_horizon_days=request.time_horizon_days,
                prediction_method=request.simulation_method,
                sentiment_impact=random.uniform(-0.2, 0.2)
            ))

        # Format response
        response = SimulationResponse(
            request_id=f"sim_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            news_analysis=mock_sentiment_results,
            market_predictions=mock_predictions,
            quantum_metrics={"quantum_advantage": 1.2, "execution_time": 0.5} if settings.enable_quantum else None,
            warnings=["This is a mock response - full implementation pending"]
        )

        logger.info(f"Simulation completed successfully")
        return response

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Extract text from uploaded PDF for analysis
    """
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        if not PDF_AVAILABLE:
            raise HTTPException(status_code=503, detail="PDF processing not available")

        # Read PDF content
        content = await file.read()

        # Extract text using PyPDF2
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))

        extracted_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            extracted_text += page.extract_text() + "\n"

        # Clean and truncate text
        extracted_text = extracted_text.strip()[:5000]  # Limit to 5000 chars

        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        return {
            "success": True,
            "filename": file.filename,
            "extracted_text": extracted_text,
            "char_count": len(extracted_text)
        }

    except Exception as e:
        logger.error(f"PDF upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fetch-url")
async def fetch_url_content(url: str = Form(...)):
    """
    Fetch and extract content from a URL
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {response.status}")

                html = await response.text()

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text
        text = soup.get_text()

        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Truncate to reasonable length
        extracted_text = text[:5000]

        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from URL")

        return {
            "success": True,
            "url": url,
            "extracted_text": extracted_text,
            "char_count": len(extracted_text)
        }

    except aiohttp.ClientTimeout:
        raise HTTPException(status_code=408, detail="URL fetch timeout")
    except Exception as e:
        logger.error(f"URL fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-data/{asset}")
async def get_market_data(asset: str):
    """
    Get real-time market data for an asset
    """
    try:
        # Return mock data for now
        return {
            "asset": asset,
            "current_price": _get_default_price(asset),
            "volume": random.randint(10000000, 100000000),
            "volatility": random.uniform(0.15, 0.35),
            "market_cap": random.randint(100000000, 10000000000),
            "change_24h": random.uniform(-5, 5),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to fetch market data for {asset}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "sentiment_analyzer": "ready",
            "market_simulator": "ready",
            "quantum_enabled": settings.enable_quantum,
            "pdf_available": PDF_AVAILABLE
        }
    }


def _format_sentiment(sentiment: SentimentAnalysis, news_input: NewsInput) -> Dict[str, Any]:
    """Format sentiment analysis for response"""
    result = {
        "headline": news_input.content[:200],
        "sentiment": sentiment.sentiment.value,
        "confidence": sentiment.confidence,
        "entities": sentiment.entities_detected,
        "key_phrases": sentiment.key_phrases,
        "market_keywords": sentiment.market_impact_keywords
    }

    # Add crisis information if present
    if hasattr(sentiment, 'crisis_indicators'):
        crisis = sentiment.crisis_indicators
        if crisis and crisis.get("is_crisis"):
            result["crisis_detected"] = True
            result["crisis_severity"] = crisis.get("severity", 0.5)
            result["crisis_keywords"] = crisis.get("keywords", [])

    return result


def _generate_warnings(
        sentiment_results: List[SentimentAnalysis],
        simulation_results: Dict[str, Any]
) -> List[str]:
    """Generate warnings based on analysis"""
    warnings = []

    # Check for crisis conditions
    crisis_count = sum(
        1 for s in sentiment_results
        if hasattr(s, 'crisis_indicators') and s.crisis_indicators.get("is_crisis")
    )

    if crisis_count > 0:
        warnings.append(f"⚠️ {crisis_count} crisis indicator(s) detected - expect high volatility")

    # Check for extreme predictions
    predictions = simulation_results.get("predictions", [])
    extreme_predictions = [
        p for p in predictions
        if abs(p.expected_return) > 0.15
    ]

    if extreme_predictions:
        assets = ", ".join([p.asset for p in extreme_predictions])
        warnings.append(f"Extreme price movements predicted for: {assets}")

    # Check for low confidence
    low_confidence = [
        p for p in predictions
        if p.confidence < 0.6
    ]

    if low_confidence:
        warnings.append("Low confidence in some predictions due to conflicting signals")

    # Check sentiment consensus
    sentiment_types = [s.sentiment for s in sentiment_results]
    if len(set(sentiment_types)) >= 3:
        warnings.append("Mixed sentiment signals detected across news items")

    return warnings


def _get_default_price(asset: str) -> float:
    """Get default price for common assets"""
    default_prices = {
        "AAPL": 195.0,
        "GOOGL": 155.0,
        "MSFT": 420.0,
        "AMZN": 170.0,
        "TSLA": 250.0,
        "META": 500.0,
        "NVDA": 850.0,
        "BTC-USD": 65000.0,
        "ETH-USD": 3500.0
    }
    return default_prices.get(asset, 100.0)


# Additional utility endpoints

@router.post("/analyze-sentiment")
async def analyze_sentiment(content: str = Form(...)):
    """
    Quick sentiment analysis endpoint
    """
    try:
        # Mock sentiment analysis for now
        import random
        sentiments = ["very_positive", "positive", "neutral", "negative", "very_negative"]

        return {
            "sentiment": random.choice(sentiments),
            "confidence": random.uniform(0.6, 0.95),
            "score": random.uniform(-1, 1),
            "keywords": ["market", "analysis"],
            "crisis_detected": False
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-assets")
async def get_supported_assets():
    """
    Get list of supported assets
    """
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

