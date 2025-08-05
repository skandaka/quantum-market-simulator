# backend/main.py
"""
Main entry point for the Quantum Market Simulator backend
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Now we can import from app
from app.main import app
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Quantum Market Simulator Backend")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Port: {settings.port}")

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )