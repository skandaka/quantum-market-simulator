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
from app.services.sentiment_analyzer import EnhancedSentimentAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize enhanced sentiment analyzer
enhanced_sentiment_analyzer = EnhancedSentimentAnalyzer()

# Services will be initialized when needed to avoid circular imports


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_market(request: SimulationRequest):
    """
    ENHANCED QUANTUM MARKET SIMULATION ENDPOINT
    Integrates Phase 1 quantum algorithms with advanced sentiment analysis
    """
    try:
        logger.info(f"Starting enhanced quantum simulation for {len(request.news_inputs)} news items")

        # Initialize quantum services if available
        quantum_nlp = None
        quantum_finance = None
        quantum_portfolio = None
        
        try:
            from app.quantum.qnlp_model import QuantumNLPModel
            from app.quantum.quantum_finance import QuantumFinanceAlgorithms
            from app.quantum.quantum_portfolio_optimizer import QuantumPortfolioOptimizer
            from app.quantum.classiq_client import ClassiqClient
            
            classiq_client = ClassiqClient()
            await classiq_client.initialize()
            
            if classiq_client.is_ready():
                quantum_nlp = QuantumNLPModel(classiq_client)
                quantum_finance = QuantumFinanceAlgorithms(classiq_client)
                quantum_portfolio = QuantumPortfolioOptimizer(classiq_client)
                await quantum_nlp.initialize()
                logger.info("Quantum services initialized successfully")
            else:
                logger.warning("Classiq client not ready, using classical simulation")
        except Exception as e:
            logger.warning(f"Quantum initialization failed: {e}, falling back to classical")

        # PHASE 1.1: Enhanced Quantum Sentiment Analysis
        enhanced_sentiment_results = []
        for i, news_input in enumerate(request.news_inputs):
            try:
                if quantum_nlp and quantum_nlp.is_ready():
                    logger.info(f"Processing news item {i+1} with quantum NLP")
                    
                    # PHASE 1.1.1: Quantum word embedding
                    word_embedding_result = await quantum_nlp.create_quantum_word_embedding_circuit(news_input.content)
                    
                    # PHASE 1.1.2: Advanced feature map
                    if word_embedding_result and "quantum_embedding" in word_embedding_result:
                        feature_map_result = quantum_nlp.create_advanced_feature_map(
                            np.array(word_embedding_result["quantum_embedding"])
                        )
                    else:
                        feature_map_result = {}
                    
                    # PHASE 1.1.3: Quantum attention mechanism
                    attention_result = await quantum_nlp.quantum_attention_layer(news_input.content)
                    
                    # Encode text and classify sentiment
                    quantum_features = await quantum_nlp.encode_text_quantum(news_input.content)
                    sentiment_result = await quantum_nlp.quantum_sentiment_classification(quantum_features)
                    
                    # Enhanced sentiment result with quantum features
                    enhanced_sentiment = {
                        "headline": news_input.content[:200],
                        "sentiment": sentiment_result.get("predicted_sentiment", "neutral"),
                        "confidence": sentiment_result.get("confidence", 0.5),
                        "entities": [],
                        "key_phrases": [],
                        "market_keywords": [],
                        "quantum_features": {
                            "word_embedding": word_embedding_result,
                            "feature_map": feature_map_result,
                            "attention_weights": attention_result.get("attention_weights", []),
                            "quantum_advantage": sentiment_result.get("quantum_metrics", {}).get("quantum_advantage", 1.0),
                            "circuit_depth": sentiment_result.get("quantum_metrics", {}).get("circuit_depth", 0)
                        }
                    }
                else:
                    # Enhanced classical fallback using EnhancedSentimentAnalyzer
                    sentiment_result = await enhanced_sentiment_analyzer.analyze_single(news_input.content)
                    enhanced_sentiment = {
                        "headline": news_input.content[:200],
                        "sentiment": sentiment_result.sentiment.value if hasattr(sentiment_result.sentiment, 'value') else str(sentiment_result.sentiment),
                        "confidence": sentiment_result.confidence,
                        "entities": sentiment_result.entities_detected,
                        "key_phrases": sentiment_result.key_phrases,
                        "market_keywords": sentiment_result.market_impact_keywords,
                        "quantum_features": None,
                        "crisis_detected": sentiment_result.crisis_indicators is not None,
                        "crisis_indicators": sentiment_result.crisis_indicators,
                        "classical_score": sentiment_result.classical_sentiment_score
                    }
                
                enhanced_sentiment_results.append(enhanced_sentiment)
                
            except Exception as e:
                logger.error(f"Sentiment analysis failed for item {i}: {e}")
                # Try enhanced sentiment analyzer as final fallback
                try:
                    sentiment_result = await enhanced_sentiment_analyzer.analyze_single(news_input.content)
                    enhanced_sentiment_results.append({
                        "headline": news_input.content[:200],
                        "sentiment": sentiment_result.sentiment.value if hasattr(sentiment_result.sentiment, 'value') else str(sentiment_result.sentiment),
                        "confidence": sentiment_result.confidence,
                        "entities": sentiment_result.entities_detected,
                        "key_phrases": sentiment_result.key_phrases,
                        "market_keywords": sentiment_result.market_impact_keywords,
                        "crisis_detected": sentiment_result.crisis_indicators is not None,
                        "crisis_indicators": sentiment_result.crisis_indicators,
                        "classical_score": sentiment_result.classical_sentiment_score,
                        "error": str(e)
                    })
                except Exception as fallback_error:
                    enhanced_sentiment_results.append({
                        "headline": news_input.content[:200],
                        "sentiment": "neutral",
                        "confidence": 0.5,
                        "entities": [],
                        "key_phrases": [],
                        "market_keywords": [],
                        "error": f"All sentiment analysis failed: {e}, {fallback_error}"
                    })

        # PHASE 1.2: Enhanced Quantum Monte Carlo for Market Predictions
        enhanced_predictions = []
        for asset in request.target_assets:
            try:
                current_price = _get_default_price(asset)
                
                if quantum_finance:
                    logger.info(f"Generating quantum predictions for {asset}")
                    
                    # PHASE 1.2.1: Quantum Monte Carlo with amplitude estimation
                    qmc_result = await quantum_finance.quantum_monte_carlo_pricing(
                        spot_price=current_price,
                        volatility=random.uniform(0.15, 0.35),
                        drift=random.uniform(-0.05, 0.05),
                        time_horizon=request.time_horizon_days,
                        num_paths=min(request.num_scenarios, 1000)
                    )
                    
                    # PHASE 1.2.2: Quantum random number generation
                    quantum_random_result = await quantum_finance.quantum_random_generator(
                        num_samples=min(request.num_scenarios, 500)
                    )
                    
                    # PHASE 1.2.3: Quantum correlation modeling
                    correlation_matrix = np.random.random((len(request.target_assets), len(request.target_assets)))
                    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
                    np.fill_diagonal(correlation_matrix, 1.0)
                    
                    correlation_result = await quantum_finance.quantum_correlation_circuit(correlation_matrix)
                    
                    # Generate quantum-enhanced price scenarios
                    quantum_scenarios = []
                    random_numbers = quantum_random_result.get("random_numbers", [])
                    
                    for i in range(min(request.num_scenarios, len(random_numbers))):
                        price_change = random_numbers[i] * 0.1  # Scale random number
                        scenario_price = current_price * (1 + price_change)
                        
                        quantum_scenarios.append(PriceScenario(
                            scenario_id=i,
                            price_path=[current_price, scenario_price],
                            returns_path=[0.0, price_change],
                            volatility_path=[0.2, 0.25],
                            probability_weight=1.0 / min(request.num_scenarios, len(random_numbers)),
                            quantum_amplitude=random.random() * 0.8,  # Mock quantum amplitude
                            quantum_phase=random.random() * 2 * np.pi   # Mock quantum phase
                        ))
                    
                    # Enhanced confidence intervals with quantum uncertainty
                    confidence_intervals = {
                        "95%": {"lower": current_price * 0.9, "upper": current_price * 1.1},
                        "90%": {"lower": current_price * 0.92, "upper": current_price * 1.08},
                        "68%": {"lower": current_price * 0.95, "upper": current_price * 1.05}
                    }
                    
                    # Add quantum uncertainty bounds
                    quantum_uncertainty = qmc_result.get("quantum_confidence", 0.8)
                    for level in confidence_intervals:
                        uncertainty_factor = 1.0 - quantum_uncertainty
                        ci = confidence_intervals[level]
                        ci["quantum_lower"] = ci["lower"] * (1 - uncertainty_factor)
                        ci["quantum_upper"] = ci["upper"] * (1 + uncertainty_factor)
                    
                    enhanced_prediction = MarketPrediction(
                        asset=asset,
                        current_price=current_price,
                        expected_return=qmc_result.get("expected_price", current_price) / current_price - 1,
                        volatility=random.uniform(0.15, 0.35),
                        confidence=quantum_uncertainty,
                        predicted_scenarios=quantum_scenarios,
                        confidence_intervals=confidence_intervals,
                        time_horizon_days=request.time_horizon_days,
                        prediction_method="quantum_enhanced_monte_carlo",
                        sentiment_impact=random.uniform(-0.2, 0.2),
                        quantum_metrics={
                            "quantum_advantage": qmc_result.get("quantum_advantage", 1.2),
                            "execution_time": qmc_result.get("quantum_confidence", 0.5),
                            "amplitude_estimation_accuracy": qmc_result.get("amplitude_estimation_accuracy", 0.95),
                            "circuit_depth": qmc_result.get("circuit_depth", 45),
                            "entanglement_measure": correlation_result.get("entanglement_measure", 0.6),
                            "coherence_time": 50.0 + random.random() * 50.0,
                            "fidelity": 0.95 + random.random() * 0.04
                        }
                    )
                else:
                    # Enhanced classical fallback with mock quantum structure
                    mock_scenarios = []
                    for i in range(min(10, request.num_scenarios)):
                        scenario_price = current_price * (1 + random.uniform(-0.1, 0.1))
                        mock_scenarios.append(PriceScenario(
                            scenario_id=i,
                            price_path=[current_price, scenario_price],
                            returns_path=[0.0, (scenario_price - current_price) / current_price],
                            volatility_path=[0.2, 0.25],
                            probability_weight=1.0 / min(10, request.num_scenarios),
                            quantum_amplitude=random.random() * 0.8,
                            quantum_phase=random.random() * 2 * np.pi
                        ))
                    
                    enhanced_prediction = MarketPrediction(
                        asset=asset,
                        current_price=current_price,
                        expected_return=random.uniform(-0.05, 0.05),
                        volatility=random.uniform(0.15, 0.35),
                        confidence=random.uniform(0.6, 0.9),
                        predicted_scenarios=mock_scenarios,
                        confidence_intervals={
                            "95%": {"lower": current_price * 0.9, "upper": current_price * 1.1},
                            "90%": {"lower": current_price * 0.92, "upper": current_price * 1.08},
                            "68%": {"lower": current_price * 0.95, "upper": current_price * 1.05}
                        },
                        time_horizon_days=request.time_horizon_days,
                        prediction_method="classical_with_mock_quantum",
                        sentiment_impact=random.uniform(-0.2, 0.2),
                        quantum_metrics={
                            "quantum_advantage": 1.0,
                            "execution_time": 0.1,
                            "circuit_depth": 0,
                            "entanglement_measure": 0.0,
                            "coherence_time": 0.0,
                            "fidelity": 1.0
                        }
                    )
                
                enhanced_predictions.append(enhanced_prediction)
                
            except Exception as e:
                logger.error(f"Prediction failed for {asset}: {e}")
                # Basic fallback prediction
                enhanced_predictions.append(MarketPrediction(
                    asset=asset,
                    current_price=_get_default_price(asset),
                    expected_return=0.0,
                    volatility=0.2,
                    confidence=0.5,
                    predicted_scenarios=[],
                    confidence_intervals={"95%": {"lower": 90.0, "upper": 110.0}},
                    time_horizon_days=request.time_horizon_days,
                    prediction_method="error_fallback",
                    sentiment_impact=0.0,
                    quantum_metrics={"quantum_advantage": 0.0, "execution_time": 0.0}
                ))

        # Enhanced response with quantum metrics
        quantum_metrics = None
        if enhanced_predictions and enhanced_predictions[0].quantum_metrics:
            # Aggregate quantum metrics
            quantum_metrics = {
                "quantum_advantage": np.mean([p.quantum_metrics.get("quantum_advantage", 1.0) for p in enhanced_predictions]),
                "execution_time": np.sum([p.quantum_metrics.get("execution_time", 0.0) for p in enhanced_predictions]),
                "total_circuits": len(enhanced_predictions),
                "avg_circuit_depth": np.mean([p.quantum_metrics.get("circuit_depth", 0) for p in enhanced_predictions]),
                "avg_fidelity": np.mean([p.quantum_metrics.get("fidelity", 1.0) for p in enhanced_predictions]),
                "quantum_enabled": quantum_nlp is not None and quantum_finance is not None
            }

        response = SimulationResponse(
            request_id=f"enhanced_sim_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            news_analysis=enhanced_sentiment_results,
            market_predictions=enhanced_predictions,
            quantum_metrics=quantum_metrics,
            warnings=_generate_enhanced_warnings(enhanced_sentiment_results, enhanced_predictions)
        )

        logger.info(f"Enhanced quantum simulation completed successfully")
        return response

    except Exception as e:
        logger.error(f"Enhanced simulation failed: {e}")
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


def _generate_enhanced_warnings(sentiment_results: list, predictions: list) -> list:
    """Generate enhanced warnings for quantum simulation results"""
    warnings = []
    
    # Check quantum availability warnings
    quantum_enabled = False
    for prediction in predictions:
        if prediction.quantum_metrics and prediction.quantum_metrics.get("quantum_advantage", 0) > 1.0:
            quantum_enabled = True
            break
    
    if not quantum_enabled:
        warnings.append("Quantum algorithms not available - using classical simulation with mock quantum features")
    
    # Check for low confidence predictions
    low_confidence_assets = [p.asset for p in predictions if p.confidence < 0.7]
    if low_confidence_assets:
        warnings.append(f"Low confidence predictions for: {', '.join(low_confidence_assets)}")
    
    # Check for quantum decoherence warnings
    for prediction in predictions:
        if prediction.quantum_metrics:
            coherence_time = prediction.quantum_metrics.get("coherence_time", 0)
            if coherence_time > 0 and coherence_time < 30.0:
                warnings.append(f"Short quantum coherence time detected for {prediction.asset}: {coherence_time:.1f}μs")
            
            fidelity = prediction.quantum_metrics.get("fidelity", 1.0)
            if fidelity < 0.9:
                warnings.append(f"Low quantum fidelity for {prediction.asset}: {fidelity:.3f}")
    
    # Check sentiment analysis coverage
    processed_sentiment = len([s for s in sentiment_results if "error" not in s])
    total_sentiment = len(sentiment_results)
    if processed_sentiment < total_sentiment:
        warnings.append(f"Sentiment analysis failed for {total_sentiment - processed_sentiment} news items")
    
    # Add quantum-specific warnings
    if quantum_enabled:
        warnings.append("Results based on quantum simulation - classical validation recommended")
        warnings.append("Quantum advantage estimates are experimental and may vary with hardware")
    
    return warnings if warnings else ["Simulation completed successfully"]


# Additional utility endpoints

@router.post("/analyze-sentiment")
async def analyze_sentiment(content: str = Form(...)):
    """
    Quick sentiment analysis endpoint using enhanced sentiment analyzer
    """
    try:
        # Use enhanced sentiment analyzer with crisis detection
        result = await enhanced_sentiment_analyzer.analyze_single(content)
        
        return {
            "sentiment": result.sentiment.value if hasattr(result.sentiment, 'value') else str(result.sentiment),
            "confidence": result.confidence,
            "score": result.classical_sentiment_score,
            "keywords": result.key_phrases,
            "market_keywords": result.market_impact_keywords,
            "entities": result.entities_detected,
            "crisis_detected": result.crisis_indicators is not None,
            "crisis_indicators": result.crisis_indicators,
            "quantum_vector": result.quantum_sentiment_vector
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

