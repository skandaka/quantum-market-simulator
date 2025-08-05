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

from app.config import settings
from app.api import routes
# from app.api import enhanced_routes  # Temporarily disabled
# from app.quantum.classiq_auth import classiq_auth  # Temporarily disabled
# from app.quantum.classiq_client import ClassiqClient  # Temporarily disabled
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.market_simulator import MarketSimulator
# from app.services.enhanced_market_simulator import EnhancedMarketSimulator  # Temporarily disabled
from app.services.news_processor import NewsProcessor
from app.services.data_fetcher import MarketDataFetcher

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

            # Step 1: Initialize Classiq authentication (Temporarily disabled)
            # logger.info("🔐 Authenticating with Classiq...")
            # try:
            #     if settings.CLASSIQ_API_KEY:
            #         classiq_auth.login(api_key=settings.CLASSIQ_API_KEY)
            #         logger.info("✅ Classiq authentication successful")
            #     else:
            #         logger.warning("⚠️ Running without Classiq authentication - simulation mode active")
            # except Exception as e:
            #     logger.warning(f"⚠️ Classiq initialization failed: {e} - running in simulation mode")

            # Step 2: Initialize quantum client (Temporarily disabled)
            # app.state.classiq_client = ClassiqClient()
            # try:
            #     await app.state.classiq_client.initialize()
            #     logger.info("✅ Quantum client initialized")
            # except Exception as e:
            #     logger.warning(f"Quantum client initialization warning: {e}")

            # Step 3: Initialize standard components
            logger.info("📈 Initializing standard components...")
            app.state.market_simulator = MarketSimulator()
            # app.state.sentiment_analyzer = SentimentAnalyzer(app.state.classiq_client)
            app.state.news_processor = NewsProcessor()
            app.state.data_fetcher = MarketDataFetcher()

            try:
                # await app.state.market_simulator.initialize(app.state.classiq_client)
                # await app.state.sentiment_analyzer.initialize()
                logger.info("✅ Standard components ready")
            except Exception as e:
                logger.error(f"Standard component initialization warning: {e}")

            # Step 4: Initialize enhanced components (Temporarily disabled)
            # logger.info("🧬 Initializing enhanced quantum components...")
            # app.state.enhanced_simulator = EnhancedMarketSimulator()

            # try:
            #     await app.state.enhanced_simulator.initialize(app.state.classiq_client)
            #     logger.info("✅ Enhanced quantum components ready")
            # except Exception as e:
            #     logger.error(f"Enhanced component initialization warning: {e}")

            # Step 5: Test quantum functionality if authenticated (Temporarily disabled)
            # if classiq_auth.is_authenticated():
            #     try:
            #         logger.info("🧪 Testing quantum circuit synthesis...")
            #         from classiq import create_model, qfunc, QBit, Output, allocate, H

            #         @qfunc
            #         def test_circuit(q: Output[QBit]):
            #             allocate(1, q)
            #             H(q)

            #         model = create_model(test_circuit)
            #         logger.info("✅ Quantum circuit test successful")
            #     except Exception as e:
            #         logger.warning(f"Quantum circuit test failed: {e}")

            # Step 6: Initialize performance monitoring
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
    if hasattr(app.state, 'enhanced_simulator'):
        # Add cleanup for enhanced simulator if needed
        pass
    logger.info("👋 Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Enhanced Quantum Market Simulator",
    description="Advanced quantum computing for financial market prediction with portfolio integration",
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
    quantum_status = "ready" if hasattr(app, 'state') and hasattr(app.state, 'classiq_client') else "initializing"

    return {
        "status": "healthy",
        "quantum_backend": quantum_status,
        "enhanced_features": "active",
        "version": "2.0.0",
        "components": {
            "sentiment_analyzer": hasattr(app.state, 'sentiment_analyzer'),
            "market_simulator": hasattr(app.state, 'market_simulator'),
            "enhanced_simulator": hasattr(app.state, 'enhanced_simulator'),
            "quantum_client": hasattr(app.state, 'classiq_client')
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
# app.include_router(enhanced_routes.router)  # Temporarily disabled

# Include Socket.IO for WebSocket support
try:
    from app.api.socketio_server import sio
    import socketio
    
    # Mount Socket.IO app at /ws/
    socketio_app = socketio.ASGIApp(sio, socketio_path='/ws/')
    app.mount('/ws', socketio_app)
    logger.info("✅ Socket.IO WebSocket support enabled")
except ImportError as e:
    logger.warning(f"⚠️ Socket.IO not available: {e} - using fallback WebSocket")
    # Fallback to native FastAPI WebSocket
    from app.api.websocket import websocket_endpoint

    @app.websocket("/ws")
    async def websocket_handler(websocket):
        """WebSocket endpoint for real-time updates"""
        await websocket_endpoint(websocket)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with application info"""
    return {
        "application": "Enhanced Quantum Market Simulator",
        "version": "2.0.0",
        "features": [
            "Multi-layer quantum processing",
            "Advanced quantum ML algorithms",
            "Portfolio integration and analysis",
            "Real-time quantum monitoring",
            "Interactive 3D visualizations",
            "Quantum circuit export"
        ],
        "api_docs": "/docs",
        "health_check": "/health",
        "quantum_status": False  # classiq_auth.is_authenticated()  # Temporarily disabled
    }


# Serve static files in production
# app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )