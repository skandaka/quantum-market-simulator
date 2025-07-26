"""Main FastAPI application entry point"""

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
from app.quantum.classiq_client import ClassiqClient
from app.utils.helpers import setup_logging

# Setup logging
logger = setup_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Quantum Market Reaction Simulator...")

    # Initialize services
    app.state.market_simulator = MarketSimulator()
    app.state.classiq_client = ClassiqClient(api_key=settings.classiq_api_key)

    # Warm up models
    logger.info("Warming up models...")
    await app.state.market_simulator.initialize()

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await app.state.market_simulator.cleanup()
    await app.state.classiq_client.close()


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
        "docs": "/api/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "quantum_backend": settings.quantum_backend,
        "services": {
            "market_simulator": "ready",
            "quantum_engine": "ready",
            "data_pipeline": "ready"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )