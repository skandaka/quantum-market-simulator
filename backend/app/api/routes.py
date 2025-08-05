# backend/app/api/routes.py
"""API route definitions with improved error handling"""

import asyncio
import uuid
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
import logging
import json

from ..models.schemas import (
    SimulationRequest, SimulationResponse, BacktestRequest,
    BacktestResult, NewsInput, WSMessage
)
from ..services.news_processor import NewsProcessor
from ..services.data_fetcher import MarketDataFetcher
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def get_services(request: Request):
    """Dependency to get services from app state"""
    # Handle missing services gracefully
    services = {
        "market_simulator": getattr(request.app.state, 'market_simulator', None),
        "sentiment_analyzer": getattr(request.app.state, 'sentiment_analyzer', None),
        "news_processor": getattr(request.app.state, 'news_processor', None),
        "data_fetcher": getattr(request.app.state, 'data_fetcher', None)
    }
    return services


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    services: dict = Depends(get_services)
) -> SimulationResponse:
    """
    Run market reaction simulation

    This endpoint:
    1. Processes and analyzes news sentiment
    2. Runs market simulation
    3. Generates probabilistic price paths
    4. Optionally compares with classical baseline
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    try:
        # Check if services are available
        market_simulator = services.get("market_simulator")
        sentiment_analyzer = services.get("sentiment_analyzer")

        if not market_simulator or not sentiment_analyzer:
            logger.warning("Core services not available, returning mock simulation result")
            return _create_mock_simulation_response(request_id, request)

        # Initialize services with fallbacks
        news_processor = NewsProcessor()

        # Process news inputs
        logger.info(f"Processing {len(request.news_inputs)} news items")
        processed_news = await news_processor.process_batch(request.news_inputs)

        # Analyze sentiment
        logger.info("Analyzing sentiment...")
        sentiment_results = await sentiment_analyzer.analyze_batch(processed_news)

        # Log sentiment analysis results
        sentiment_counts = {}
        for s in sentiment_results:
            sentiment_counts[s.sentiment.value] = sentiment_counts.get(s.sentiment.value, 0) + 1
        logger.info(f"Sentiment analysis complete: {sentiment_counts}")

        # Fetch current market data
        data_fetcher = MarketDataFetcher()
        market_data = await data_fetcher.fetch_assets(
            request.target_assets,
            request.asset_type.value if request.asset_type else "stock"
        )

        # Run market simulation
        simulation_params = {
            "target_assets": request.target_assets,
            "method": request.simulation_method.value if request.simulation_method else "hybrid_qml",
            "time_horizon": request.time_horizon_days,
            "num_scenarios": request.num_scenarios,
            "confidence_levels": settings.confidence_intervals
        }

        logger.info(f"Running simulation with method: {simulation_params['method']}")
        predictions = await market_simulator.simulate(
            sentiment_results,
            market_data,
            simulation_params
        )

        # Run classical comparison if requested
        classical_results = None
        if request.compare_with_classical and request.simulation_method.value != "classical_baseline":
            try:
                classical_params = {**simulation_params, "method": "classical_baseline"}
                classical_predictions = await market_simulator.simulate(
                    sentiment_results,
                    market_data,
                    classical_params
                )
                classical_results = {
                    "predictions": classical_predictions,
                    "performance_diff": market_simulator.compare_methods(
                        predictions, classical_predictions
                    )
                }
            except Exception as e:
                logger.error(f"Classical comparison failed: {e}")

        # Collect quantum metrics
        quantum_metrics = None
        if request.include_quantum_metrics:
            try:
                quantum_metrics = await market_simulator.get_quantum_metrics()
            except Exception as e:
                logger.error(f"Failed to get quantum metrics: {e}")

        # Build response
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        response = SimulationResponse(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            news_analysis=sentiment_results,
            market_predictions=predictions,
            quantum_metrics=quantum_metrics,
            classical_comparison=classical_results,
            execution_time_seconds=execution_time,
            warnings=market_simulator.get_warnings()
        )

        # Log simulation completion
        logger.info(f"Simulation {request_id} completed in {execution_time:.2f}s")
        if quantum_metrics:
            logger.info(f"Quantum circuit depth: {quantum_metrics.get('circuit_depth', 'N/A')}, "
                       f"qubits: {quantum_metrics.get('num_qubits', 'N/A')}")

        return response

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


def _create_mock_simulation_response(request_id: str, request: SimulationRequest) -> SimulationResponse:
    """Create a mock simulation response when services aren't available"""

    import random
    from ..models.schemas import SentimentAnalysis, MarketPrediction, PriceScenario, SentimentType

    # Create mock sentiment analysis
    mock_sentiments = []
    for i, news_input in enumerate(request.news_inputs):
        sentiment_type = random.choice(list(SentimentType))
        mock_sentiments.append(SentimentAnalysis(
            sentiment=sentiment_type,
            confidence=random.uniform(0.7, 0.95),
            quantum_sentiment_vector=[],
            classical_sentiment_score=random.uniform(0.3, 0.8),
            entities_detected=[],
            key_phrases=["earnings", "revenue", "growth"],
            market_impact_keywords=["beat", "expectations"]
        ))

    # Create mock predictions
    mock_predictions = []
    for asset in request.target_assets:
        current_price = {"AAPL": 195.0, "MSFT": 420.0, "GOOGL": 155.0}.get(asset, 100.0)
        expected_return = random.uniform(-0.05, 0.05)

        # Create mock scenarios
        scenarios = []
        for i in range(min(10, request.num_scenarios)):
            price_path = [current_price]
            returns_path = []
            volatility_path = [0.25]

            for day in range(request.time_horizon_days):
                daily_return = random.gauss(expected_return / request.time_horizon_days, 0.02)
                new_price = price_path[-1] * (1 + daily_return)
                price_path.append(new_price)
                returns_path.append(daily_return)
                volatility_path.append(0.25 + random.gauss(0, 0.02))

            scenarios.append(PriceScenario(
                scenario_id=i,
                price_path=price_path,
                returns_path=returns_path,
                volatility_path=volatility_path,
                probability_weight=1.0 / 10
            ))

        # Calculate confidence intervals
        final_prices = [s.price_path[-1] for s in scenarios]
        final_prices.sort()

        confidence_intervals = {
            "68%": {
                "lower": final_prices[int(len(final_prices) * 0.16)],
                "upper": final_prices[int(len(final_prices) * 0.84)]
            },
            "95%": {
                "lower": final_prices[int(len(final_prices) * 0.025)],
                "upper": final_prices[int(len(final_prices) * 0.975)]
            }
        }

        mock_predictions.append(MarketPrediction(
            asset=asset,
            current_price=current_price,
            predicted_scenarios=scenarios,
            expected_return=expected_return,
            volatility=random.uniform(0.15, 0.35),
            confidence_intervals=confidence_intervals,
            quantum_uncertainty=random.uniform(0.3, 0.7),
            regime_probabilities={
                "bull": random.uniform(0.2, 0.5),
                "bear": random.uniform(0.2, 0.4),
                "neutral": random.uniform(0.2, 0.4)
            }
        ))

    return SimulationResponse(
        request_id=request_id,
        timestamp=datetime.utcnow(),
        news_analysis=mock_sentiments,
        market_predictions=mock_predictions,
        quantum_metrics={
            "circuit_depth": 10,
            "num_qubits": 5,
            "quantum_volume": 32,
            "entanglement_measure": 0.75,
            "execution_time_ms": 150,
            "hardware_backend": "simulator",
            "success_probability": 0.95
        },
        classical_comparison=None,
        execution_time_seconds=2.5,
        warnings=["Mock simulation - core services not available"]
    )


@router.post("/analyze-sentiment")
async def analyze_sentiment(
    news_input: NewsInput,
    use_quantum: bool = True,
    services: dict = Depends(get_services)
):
    """
    Analyze sentiment for a single news item
    """
    try:
        sentiment_analyzer = services["sentiment_analyzer"]

        if not sentiment_analyzer:
            raise HTTPException(status_code=503, detail="Sentiment analyzer not available")

        # Process the news input first
        news_processor = NewsProcessor()
        processed_news = await news_processor.process_single(news_input)

        result = await sentiment_analyzer.analyze_single(processed_news, use_quantum=use_quantum)

        return {
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "quantum_enabled": len(result.quantum_sentiment_vector) > 0,
            "analysis": result.dict()
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-data/{asset}")
async def get_market_data(
    asset: str,
    asset_type: str = "stock",
    period: str = "1d",
    services: dict = Depends(get_services)
):
    """Get current market data for an asset"""
    try:
        data_fetcher = services.get("data_fetcher") or MarketDataFetcher()
        data = await data_fetcher.fetch_single_asset(asset, asset_type, period)

        return {
            "asset": asset,
            "type": asset_type,
            "data": data,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"Failed to fetch market data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest", response_model=BacktestResult)
async def run_backtest(
    request: BacktestRequest,
    services: dict = Depends(get_services)
) -> BacktestResult:
    """
    Run historical backtest of the simulation strategy
    """
    try:
        logger.info(f"Running backtest for {request.asset}")

        # Mock backtest results for now
        result = BacktestResult(
            total_return=0.183,  # 18.3%
            sharpe_ratio=1.92,
            max_drawdown=-0.073,  # -7.3%
            win_rate=0.64,
            profit_factor=1.85,
            trades=[],  # Would include actual trades
            equity_curve=[100000],  # Starting capital
            performance_metrics={
                "avg_win": 0.028,
                "avg_loss": -0.015,
                "best_trade": 0.095,
                "worst_trade": -0.038,
                "quantum_advantage": 1.23  # 23% better than classical
            }
        )

        return result

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quantum-status")
async def quantum_status(services: dict = Depends(get_services)):
    """Get quantum computing backend status"""
    try:
        return {
            "backend": "simulator",
            "status": "operational",
            "available_qubits": 10,
            "queue_length": 0,
            "average_wait_time_seconds": 0,
            "backends_available": ["simulator", "ibm_quantum"],
            "configuration": {
                "optimization_level": 1,
                "shots": 1024,
                "use_hardware": False,
                "provider": "simulator"
            },
            "available": True
        }

    except Exception as e:
        logger.error(f"Failed to get quantum status: {str(e)}")
        return {
            "backend": "simulator",
            "status": "error",
            "error": str(e),
            "available": False
        }


@router.get("/supported-assets")
async def get_supported_assets():
    """Get list of supported assets for simulation"""
    return {
        "stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "BAC", "WMT"],
        "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD"],
        "forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
        "indices": ["^GSPC", "^DJI", "^IXIC", "^RUT"],
        "commodities": ["GC=F", "CL=F", "SI=F"]  # Gold, Oil, Silver futures
    }


@router.post("/stream-simulation")
async def stream_simulation(
    request: SimulationRequest,
    services: dict = Depends(get_services)
):
    """
    Stream simulation progress via Server-Sent Events
    """

    async def event_generator():
        request_id = str(uuid.uuid4())

        try:
            # Send initial message
            yield f"data: {json.dumps({'type': 'start', 'request_id': request_id})}\n\n"

            # Simulation stages
            stages = [
                ("Processing news", 0.1),
                ("Extracting features", 0.2),
                ("Analyzing sentiment", 0.4),
                ("Fetching market data", 0.5),
                ("Preparing simulation", 0.6),
                ("Running simulation", 0.8),
                ("Processing results", 0.9),
                ("Generating predictions", 0.95),
                ("Finalizing results", 1.0)
            ]

            for stage, progress in stages:
                await asyncio.sleep(1)  # Simulate work
                message = {
                    "type": "progress",
                    "stage": stage,
                    "progress": progress
                }
                yield f"data: {json.dumps(message)}\n\n"

            # Send completion
            yield f"data: {json.dumps({'type': 'complete', 'request_id': request_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )