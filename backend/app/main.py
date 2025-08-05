# backend/app/main.py
"""
Enhanced Quantum Market Simulator - Main Application
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .api import routes
from .services.sentiment_analyzer import SentimentAnalyzer
from .services.unified_market_simulator import UnifiedMarketSimulator
from .services.news_processor import NewsProcessor
from .services.data_fetcher import MarketDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track initialization state
_initialization_complete = False
_initialization_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global _initialization_complete

    async with _initialization_lock:
        if not _initialization_complete:
            logger.info("=" * 50)
            logger.info("🚀 Starting Quantum Market Simulator...")
            logger.info("=" * 50)

            # Initialize standard components
            logger.info("📈 Initializing core components...")
            app.state.market_simulator = UnifiedMarketSimulator()
            app.state.sentiment_analyzer = SentimentAnalyzer()
            app.state.news_processor = NewsProcessor()
            app.state.data_fetcher = MarketDataFetcher()

            try:
                await app.state.market_simulator.initialize()
                await app.state.sentiment_analyzer.initialize()
                logger.info("✅ Core components ready")
            except Exception as e:
                logger.error(f"Component initialization warning: {e}")

            # Initialize performance monitoring
            logger.info("📊 Setting up performance monitoring...")
            app.state.performance_metrics = {
                "total_simulations": 0,
                "quantum_advantage_demonstrated": 0,
                "average_execution_time": 0.0,
                "total_qubits_used": 0
            }

            _initialization_complete = True
            logger.info("🎉 Application startup complete!")
            logger.info("=" * 50)

    yield

    # Cleanup
    logger.info("🛑 Shutting down Quantum Market Simulator...")
    if hasattr(app.state, 'market_simulator'):
        await app.state.market_simulator.cleanup()
    logger.info("👋 Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Quantum Market Simulator",
    description="Quantum computing for financial market prediction",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
            "message": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "quantum_backend": "ready",
        "version": "2.0.0",
        "components": {
            "sentiment_analyzer": hasattr(app.state, 'sentiment_analyzer'),
            "market_simulator": hasattr(app.state, 'market_simulator'),
            "news_processor": hasattr(app.state, 'news_processor'),
            "data_fetcher": hasattr(app.state, 'data_fetcher')
        }
    }


# Performance metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get application performance metrics"""
    if hasattr(app.state, 'performance_metrics'):
        return app.state.performance_metrics
    return {"error": "Metrics not available"}


# Include routers
app.include_router(routes.router, prefix="/api/v1")

# WebSocket fallback
try:
    from .api.websocket import websocket_endpoint

    @app.websocket("/ws")
    async def websocket_handler(websocket):
        """WebSocket endpoint for real-time updates"""
        await websocket_endpoint(websocket)
except ImportError as e:
    logger.warning(f"WebSocket not available: {e}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with application info"""
    return {
        "application": "Quantum Market Simulator", 
        "version": "2.0.0",
        "features": [
            "Quantum-enhanced sentiment analysis",
            "Advanced market simulation",
            "Real-time monitoring",
            "Portfolio analysis"
        ],
        "api_docs": "/docs",
        "health_check": "/health"
    }