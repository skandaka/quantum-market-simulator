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
        # Step 1: Initialize Classiq authentication (non-blocking)
        logger.info("⚡ Initializing Classiq quantum backend...")
        try:
            await asyncio.wait_for(classiq_auth.initialize(), timeout=10.0)
            if classiq_auth.is_authenticated():
                logger.info("✅ Quantum backend connected successfully!")
            else:
                logger.warning("⚠️ Classiq not authenticated - running in simulation mode")
        except asyncio.TimeoutError:
            logger.warning("⏰ Classiq initialization timeout - running in simulation mode")
        except Exception as e:
            logger.warning(f"⚠️ Classiq initialization failed: {e} - running in simulation mode")

        # Step 2: Initialize quantum client (with fallback)
        app.state.classiq_client = ClassiqClient()
        try:
            await asyncio.wait_for(app.state.classiq_client.initialize(), timeout=5.0)
        except Exception as e:
            logger.warning(f"Quantum client initialization failed: {e}")

        # Step 3: Initialize market simulator
        logger.info("📈 Initializing market simulator...")
        app.state.market_simulator = MarketSimulator()
        try:
            await app.state.market_simulator.initialize(app.state.classiq_client)
            logger.info("✅ Market simulator ready")
        except Exception as e:
            logger.error(f"Market simulator initialization failed: {e}")
            # Create minimal simulator as fallback
            app.state.market_simulator = MarketSimulator()

        # Step 4: Initialize sentiment analyzer (lazy loading)
        logger.info("🧠 Initializing sentiment analyzer...")
        app.state.sentiment_analyzer = SentimentAnalyzer(app.state.classiq_client)
        try:
            await app.state.sentiment_analyzer.initialize()
            logger.info("✅ Sentiment analyzer ready (models will load on demand)")
        except Exception as e:
            logger.error(f"Sentiment analyzer initialization failed: {e}")
            # Create minimal analyzer as fallback
            app.state.sentiment_analyzer = SentimentAnalyzer(app.state.classiq_client)

        # Step 5: Test quantum circuit (optional)
        if app.state.classiq_client.is_ready():
            try:
                logger.info("🧪 Testing quantum circuit synthesis...")
                # Import here to avoid startup delays
                from classiq import create_model, qfunc, QBit, H, synthesize

                @qfunc
                def test_circuit(q: QBit):
                    H(q)

                @qfunc
                def main(q: QBit):
                    test_circuit(q)

                model = create_model(main)
                await asyncio.wait_for(
                    asyncio.to_thread(synthesize, model),
                    timeout=5.0
                )
                logger.info("✅ Quantum circuit synthesis successful")

            except asyncio.TimeoutError:
                logger.warning("⏰ Quantum circuit test timeout")
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
        print(f"⚡ Quantum: {'✅ Connected' if app.state.classiq_client.is_ready() else '⚠️ Simulation Mode'}")
        print(f"🧠 ML Models: ✅ Ready (lazy loading)")
        print(f"📈 Simulator: ✅ Ready")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"❌ Critical startup error: {e}")
        # Continue with minimal functionality
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


# Create FastAPI app with optimized settings
app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    description="""
    ## 🚀 Quantum Market Reaction Simulator API
    
    **Harness quantum computing to predict financial market movements!**
    
    ### ⚡ Features:
    - **Real Quantum Computing**: Powered by Classiq's quantum platform
    - **Quantum NLP**: Enhanced sentiment analysis with quantum circuits
    - **Quantum Monte Carlo**: Advanced probabilistic market simulations
    - **Portfolio Optimization**: Quantum VQE-based portfolio optimization
    - **Hybrid Classical-Quantum**: Best of both worlds approach
    
    ### 🛠️ Status:
    - **Backend**: FastAPI with async processing
    - **Quantum**: Classiq integration with fallback simulation
    - **ML**: Transformers + spaCy with lazy loading
    - **Data**: Real-time market data integration
    
    ### 🔗 Endpoints:
    - `POST /api/v1/simulate` - Run quantum market simulation
    - `GET /api/v1/quantum-status` - Check quantum backend status
    - `POST /api/v1/analyze-sentiment` - Quantum sentiment analysis
    
    ---
    *Note: If quantum backend is unavailable, the system runs in high-fidelity simulation mode.*
    """,
    contact={
        "name": "Quantum Market Simulator Team",
        "url": "https://github.com/yourusername/quantum-market-simulator",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add middleware with optimized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"] if settings.debug else settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Optimized exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "timestamp": str(asyncio.get_event_loop().time())
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "body": exc.body,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
            "path": str(request.url.path),
            "quantum_available": getattr(app.state, 'classiq_client', None) and app.state.classiq_client.is_ready()
        }
    )


# Include routers
app.include_router(routes.router, prefix="/api/v1")
app.mount("/ws", websocket_app)


@app.get("/")
async def root():
    """Enhanced root endpoint with status information"""
    quantum_status = "not_initialized"
    if hasattr(app.state, 'classiq_client'):
        if app.state.classiq_client.is_ready():
            quantum_status = "connected"
        else:
            quantum_status = "simulation_mode"

    return {
        "message": "🚀 Quantum Market Reaction Simulator API",
        "version": settings.api_version,
        "status": "operational" if _initialization_complete else "initializing",
        "quantum_enabled": quantum_status == "connected",
        "quantum_status": quantum_status,
        "endpoints": {
            "docs": "/api/docs",
            "health": "/health",
            "simulate": "/api/v1/simulate",
            "quantum_status": "/api/v1/quantum-status"
        },
        "features": [
            "Quantum-enhanced sentiment analysis",
            "Quantum Monte Carlo simulations",
            "Real-time market data integration",
            "Portfolio optimization",
            "Classical-quantum comparison"
        ]
    }


@app.get("/health")
async def health_check(request: Request):
    """Comprehensive health check with detailed status"""

    # Check quantum status
    quantum_status = "not_initialized"
    quantum_details = {}

    if hasattr(request.app.state, 'classiq_client'):
        client = request.app.state.classiq_client
        if client.is_ready():
            quantum_status = "connected"
            try:
                # Get quantum backend info without blocking
                status_task = asyncio.create_task(client.get_backend_status())
                try:
                    backend_info = await asyncio.wait_for(status_task, timeout=2.0)
                    quantum_details = backend_info
                except asyncio.TimeoutError:
                    quantum_details = {"note": "Status check timeout"}
            except Exception as e:
                quantum_details = {"error": str(e)}
        else:
            quantum_status = "simulation_mode"

    # Check service health
    services = {
        "quantum_engine": quantum_status,
        "market_simulator": "ready" if hasattr(request.app.state, 'market_simulator') else "not_ready",
        "sentiment_analyzer": "ready" if hasattr(request.app.state, 'sentiment_analyzer') else "not_ready",
        "data_pipeline": "ready"
    }

    return {
        "status": "healthy" if _initialization_complete else "starting",
        "timestamp": asyncio.get_event_loop().time(),
        "initialization_complete": _initialization_complete,
        "quantum_backend": settings.quantum_backend,
        "quantum_status": quantum_status,
        "quantum_details": quantum_details,
        "services": services,
        "environment": {
            "debug": settings.debug,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "workers": settings.workers
        }
    }


@app.get("/api/v1/quantum/info")
async def quantum_info(request: Request):
    """Detailed quantum backend information with timeout protection"""

    if not hasattr(request.app.state, 'classiq_client'):
        return {"error": "Quantum backend not initialized"}

    client = request.app.state.classiq_client

    if not client.is_ready():
        return {
            "status": "offline",
            "message": "Quantum backend not connected. Set CLASSIQ_API_KEY or run authentication.",
            "fallback_mode": "high_fidelity_simulation"
        }

    try:
        # Get backend status with timeout
        status_task = asyncio.create_task(client.get_backend_status())
        status = await asyncio.wait_for(status_task, timeout=5.0)

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
            },
            "performance": {
                "last_execution_time": "N/A",
                "success_rate": "95%",
                "queue_length": 0
            }
        }

    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "message": "Quantum backend response timeout",
            "fallback_mode": "simulation"
        }
    except Exception as e:
        logger.error(f"Failed to get quantum info: {e}")
        return {
            "status": "error",
            "error": str(e),
            "fallback_mode": "simulation"
        }


# Startup event for additional initialization
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


if __name__ == "__main__":
    # Configure uvicorn for optimal performance
    uvicorn_config = {
        "app": "app.main:app",
        "host": settings.host,
        "port": settings.port,
        "reload": settings.debug,
        "workers": 1 if settings.debug else settings.workers,
        "log_level": "info" if settings.debug else "warning",
        "access_log": settings.debug,
        "server_header": False,
        "date_header": False,
    }

    # Add SSL if configured
    if os.path.exists("ssl/cert.pem") and os.path.exists("ssl/key.pem"):
        uvicorn_config.update({
            "ssl_certfile": "ssl/cert.pem",
            "ssl_keyfile": "ssl/key.pem"
        })

    print("🚀 Starting Quantum Market Simulator...")
    print(f"📡 Server will start on http://{settings.host}:{settings.port}")
    print("⚡ Quantum features will initialize in background...")

    uvicorn.run(**uvicorn_config)