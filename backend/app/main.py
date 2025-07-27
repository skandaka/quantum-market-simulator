"""Main FastAPI application entry point with real quantum integration"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with real quantum initialization"""
    # Startup
    logger.info("Starting Quantum Market Reaction Simulator...")

    # Initialize Classiq authentication
    logger.info("Initializing Classiq quantum backend...")
    await classiq_auth.initialize()

    # Initialize quantum client
    app.state.classiq_client = ClassiqClient()
    await app.state.classiq_client.initialize()

    # Check quantum status
    if app.state.classiq_client.is_ready():
        logger.info("✅ Quantum backend connected successfully!")

        # Get backend status
        try:
            status = await app.state.classiq_client.get_backend_status()
            logger.info(f"Quantum backend status: {status}")
        except Exception as e:
            logger.warning(f"Could not get backend status: {e}")
    else:
        logger.warning("⚠️  Running without quantum backend - quantum features will be simulated")

    # Initialize services
    app.state.market_simulator = MarketSimulator()
    await app.state.market_simulator.initialize(app.state.classiq_client)

    # Initialize sentiment analyzer with quantum client
    app.state.sentiment_analyzer = SentimentAnalyzer(app.state.classiq_client)
    await app.state.sentiment_analyzer.initialize()

    # Warm up models
    logger.info("Warming up models...")

    # Test quantum circuit if available
    if app.state.classiq_client.is_ready():
        try:
            logger.info("Testing quantum circuit execution...")
            from classiq import create_model, QFunc, QBit, H, synthesize, execute

            @QFunc
            def test_circuit(q: QBit):
                H(q)

            model = create_model(test_circuit)
            quantum_program = synthesize(model)
            logger.info("✅ Quantum circuit synthesis successful")

        except Exception as e:
            logger.error(f"Quantum circuit test failed: {e}")

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await app.state.market_simulator.cleanup()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    description="""
    ## Quantum Market Reaction Simulator API
    
    This API provides quantum-enhanced financial market prediction capabilities:
    
    * **Real Quantum Computing**: Powered by Classiq's quantum platform
    * **Quantum NLP**: Quantum-enhanced sentiment analysis
    * **Quantum Monte Carlo**: Advanced market simulations
    * **Portfolio Optimization**: Quantum VQE-based optimization
    
    ### Authentication
    The API uses Classiq authentication. Set your CLASSIQ_API_KEY or use browser auth.
    """
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            "path": request.url.path
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


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(routes.router, prefix="/api/v1")
app.mount("/ws", websocket_app)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quantum Market Reaction Simulator API",
        "version": settings.api_version,
        "status": "operational",
        "quantum_enabled": app.state.classiq_client.is_ready() if hasattr(app.state, 'classiq_client') else False,
        "docs": "/api/docs"
    }


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint with quantum status"""

    quantum_status = "not_initialized"
    if hasattr(request.app.state, 'classiq_client'):
        if request.app.state.classiq_client.is_ready():
            quantum_status = "connected"
        else:
            quantum_status = "disconnected"

    return {
        "status": "healthy",
        "quantum_backend": settings.quantum_backend,
        "quantum_status": quantum_status,
        "services": {
            "market_simulator": "ready",
            "quantum_engine": quantum_status,
            "sentiment_analyzer": "ready",
            "data_pipeline": "ready"
        }
    }


@app.get("/api/v1/quantum/info")
async def quantum_info(request: Request):
    """Get detailed quantum backend information"""

    if not hasattr(request.app.state, 'classiq_client'):
        return {"error": "Quantum backend not initialized"}

    client = request.app.state.classiq_client

    if not client.is_ready():
        return {
            "status": "offline",
            "message": "Quantum backend not connected. Set CLASSIQ_API_KEY or run authentication."
        }

    try:
        # Get backend status
        status = await client.get_backend_status()

        # Get resource limits
        limits = classiq_auth.get_resource_limits()

        return {
            "status": "online",
            "backend_info": status,
            "resource_limits": limits,
            "configuration": {
                "max_qubits": classiq_auth.config.max_qubits,
                "optimization_level": classiq_auth.config.optimization_level,
                "use_hardware": classiq_auth.config.use_hardware,
                "shots": classiq_auth.config.shots
            }
        }

    except Exception as e:
        logger.error(f"Failed to get quantum info: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )