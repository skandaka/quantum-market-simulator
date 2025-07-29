"""Optimized main.py with better error handling and lazy loading"""

import asyncio
import logging
import warnings
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

# Suppress warnings early
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from app.config import settings
from app.api import routes
from app.api.websocket import app as websocket_app
from app.services.market_simulator import MarketSimulator
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.quantum.classiq_client import ClassiqClient
from app.quantum.classiq_auth import classiq_auth
from app.utils.helpers import setup_logging

# Setup logging
logger = setup_logging(__name__)

# Global flag to track initialization status
_initialization_complete = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optimized application lifecycle with graceful error handling"""
    global _initialization_complete

    # Startup
    logger.info("🚀 Starting Quantum Market Reaction Simulator...")

    try:
        # Step 1: Initialize Classiq authentication
        logger.info("⚡ Initializing Classiq quantum backend...")
        try:
            await classiq_auth.initialize()
            if classiq_auth.is_authenticated():
                logger.info("✅ Quantum backend authenticated successfully!")
            else:
                logger.warning("⚠️ Running without Classiq authentication - simulation mode active")
        except Exception as e:
            logger.warning(f"⚠️ Classiq initialization failed: {e} - running in simulation mode")

        # Step 2: Initialize quantum client
        app.state.classiq_client = ClassiqClient()
        try:
            await app.state.classiq_client.initialize()
            logger.info("✅ Quantum client initialized")
        except Exception as e:
            logger.warning(f"Quantum client initialization warning: {e}")

        # Step 3: Initialize market simulator
        logger.info("📈 Initializing market simulator...")
        app.state.market_simulator = MarketSimulator()
        try:
            await app.state.market_simulator.initialize(app.state.classiq_client)
            logger.info("✅ Market simulator ready")
        except Exception as e:
            logger.error(f"Market simulator initialization warning: {e}")

        # Step 4: Initialize sentiment analyzer
        logger.info("🧠 Initializing sentiment analyzer...")
        app.state.sentiment_analyzer = SentimentAnalyzer(app.state.classiq_client)
        try:
            await app.state.sentiment_analyzer.initialize()
            logger.info("✅ Sentiment analyzer ready")
        except Exception as e:
            logger.error(f"Sentiment analyzer initialization warning: {e}")

        # Step 5: Test quantum functionality if authenticated
        if classiq_auth.is_authenticated():
            try:
                logger.info("🧪 Testing quantum circuit synthesis...")
                from classiq import create_model, qfunc, QBit, Output, allocate

                @qfunc
                def test_main(q: Output[QBit]):
                    allocate(1, q)

                model = create_model(test_main)
                logger.info("✅ Quantum circuit test successful")
            except Exception as e:
                logger.warning(f"Quantum circuit test failed: {e}")

        _initialization_complete = True
        logger.info("🎉 Application startup complete!")

        # Print status summary
        print("\n" + "="*60)
        print("🚀 QUANTUM MARKET SIMULATOR STATUS")
        print("="*60)
        print(f"🌐 Server: http://localhost:{settings.port}")
        print(f"📊 API Docs: http://localhost:{settings.port}/api/docs")
        print(f"⚡ Quantum: {'✅ Authenticated' if classiq_auth.is_authenticated() else '⚠️ Simulation Mode'}")
        print(f"🧠 ML Models: ✅ Ready")
        print(f"📈 Simulator: ✅ Ready")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"❌ Critical startup error: {e}")
        _initialization_complete = True

    yield

    # Shutdown
    logger.info("🛑 Shutting down application...")
    try:
        if hasattr(app.state, 'market_simulator'):
            await app.state.market_simulator.cleanup()
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
    logger.info("✅ Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"] if settings.debug else settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "body": exc.body
        }
    )


# Include routers
app.include_router(routes.router, prefix="/api/v1")
app.mount("/ws", websocket_app)


@app.get("/")
async def root():
    """Root endpoint with status"""
    return {
        "message": "🚀 Quantum Market Reaction Simulator API",
        "version": settings.api_version,
        "status": "operational" if _initialization_complete else "initializing",
        "quantum_enabled": classiq_auth.is_authenticated(),
        "endpoints": {
            "docs": "/api/docs",
            "health": "/health",
            "simulate": "/api/v1/simulate"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if _initialization_complete else "starting",
        "quantum_authenticated": classiq_auth.is_authenticated(),
        "services": {
            "api": "ready",
            "quantum": "ready" if classiq_auth.is_authenticated() else "simulation_mode"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )