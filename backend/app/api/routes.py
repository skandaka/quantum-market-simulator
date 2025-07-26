"""API route definitions"""

import asyncio
import uuid
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
import logging

from app.models.schemas import (
    SimulationRequest, SimulationResponse, BacktestRequest,
    BacktestResult, NewsInput, WSMessage
)
from app.services.news_processor import NewsProcessor
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.market_simulator import MarketSimulator
from app.services.data_fetcher import MarketDataFetcher
from app.quantum.quantum_simulator import QuantumSimulator
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def get_services(request: Request):
    """Dependency to get services from app state"""
    return {
        "market_simulator": request.app.state.market_simulator,
        "classiq_client": request.app.state.classiq_client
    }


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(
        request: SimulationRequest,
        background_tasks: BackgroundTasks,
        services: dict = Depends(get_services)
) -> SimulationResponse:
    """
    Run market reaction simulation for given news inputs

    This endpoint:
    1. Processes and analyzes news sentiment
    2. Runs quantum-enhanced market simulation
    3. Generates probabilistic price paths
    4. Compares with classical baseline if requested
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    try:
        # Initialize services
        news_processor = NewsProcessor()
        sentiment_analyzer = SentimentAnalyzer(
            quantum_client=services["classiq_client"]
        )
        market_simulator = services["market_simulator"]

        # Process news inputs
        logger.info(f"Processing {len(request.news_inputs)} news items")
        processed_news = await news_processor.process_batch(request.news_inputs)

        # Analyze sentiment with quantum enhancement
        sentiment_results = await sentiment_analyzer.analyze_batch(
            processed_news,
            use_quantum=request.simulation_method != "classical_baseline"
        )

        # Fetch current market data
        data_fetcher = MarketDataFetcher()
        market_data = await data_fetcher.fetch_assets(
            request.target_assets,
            request.asset_type
        )

        # Run market simulation
        simulation_params = {
            "method": request.simulation_method,
            "time_horizon": request.time_horizon_days,
            "num_scenarios": request.num_scenarios,
            "confidence_levels": settings.confidence_intervals
        }

        predictions = await market_simulator.simulate(
            sentiment_results,
            market_data,
            simulation_params
        )

        # Run classical comparison if requested
        classical_results = None
        if request.compare_with_classical:
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

        # Collect quantum metrics
        quantum_metrics = None
        if request.include_quantum_metrics:
            quantum_metrics = await market_simulator.get_quantum_metrics()

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

        return response

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.post("/analyze-sentiment")
async def analyze_sentiment(
        news_input: NewsInput,
        use_quantum: bool = True,
        services: dict = Depends(get_services)
):
    """
    Analyze sentiment for a single news item

    Quick endpoint for testing sentiment analysis
    """
    try:
        sentiment_analyzer = SentimentAnalyzer(
            quantum_client=services["classiq_client"]
        )

        result = await sentiment_analyzer.analyze_single(
            news_input,
            use_quantum=use_quantum
        )

        return {
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "quantum_enabled": use_quantum,
            "analysis": result.dict()
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-data/{asset}")
async def get_market_data(
        asset: str,
        asset_type: str = "stock",
        period: str = "1d"
):
    """Get current market data for an asset"""
    try:
        fetcher = MarketDataFetcher()
        data = await fetcher.fetch_single_asset(asset, asset_type, period)

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

    Tests how the quantum predictions would have performed historically
    """
    try:
        # This would be implemented with historical data
        # For now, return mock results
        logger.info(f"Running backtest for {request.asset}")

        # TODO: Implement actual backtesting logic
        result = BacktestResult(
            total_return=0.157,  # 15.7%
            sharpe_ratio=1.84,
            max_drawdown=-0.089,  # -8.9%
            win_rate=0.62,
            profit_factor=1.73,
            trades=[],  # Would include actual trades
            equity_curve=[100000],  # Starting capital
            performance_metrics={
                "avg_win": 0.023,
                "avg_loss": -0.013,
                "best_trade": 0.087,
                "worst_trade": -0.042
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
        classiq_client = services["classiq_client"]
        status = await classiq_client.get_backend_status()

        return {
            "backend": settings.quantum_backend,
            "status": "operational",
            "available_qubits": settings.max_qubits,
            "queue_length": status.get("queue_length", 0),
            "average_wait_time": status.get("avg_wait_time", 0),
            "backends_available": status.get("backends", [])
        }

    except Exception as e:
        logger.error(f"Failed to get quantum status: {str(e)}")
        return {
            "backend": settings.quantum_backend,
            "status": "error",
            "error": str(e)
        }


@router.get("/supported-assets")
async def get_supported_assets():
    """Get list of supported assets for simulation"""
    return {
        "stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"],
        "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "forex": ["EURUSD", "GBPUSD", "USDJPY"],
        "indices": ["^GSPC", "^DJI", "^IXIC"]
    }


@router.post("/stream-simulation")
async def stream_simulation(
        request: SimulationRequest,
        services: dict = Depends(get_services)
):
    """
    Stream simulation progress via Server-Sent Events

    Useful for long-running simulations
    """

    async def event_generator():
        request_id = str(uuid.uuid4())

        try:
            # Send initial message
            yield f"data: {{'type': 'start', 'request_id': '{request_id}'}}\n\n"

            # Simulate progress updates
            stages = [
                ("Processing news", 0.2),
                ("Analyzing sentiment", 0.4),
                ("Fetching market data", 0.5),
                ("Running quantum simulation", 0.8),
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
                yield f"data: {message}\n\n"

            # Send completion
            yield f"data: {{'type': 'complete', 'request_id': '{request_id}'}}\n\n"

        except Exception as e:
            yield f"data: {{'type': 'error', 'error': '{str(e)}'}}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )