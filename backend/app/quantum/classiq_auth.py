"""
Classiq Authentication and Configuration Manager
Handles all Classiq-related authentication and configuration
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Check Classiq availability
try:
    import classiq
    from classiq import authenticate, set_credentials

    CLASSIQ_AVAILABLE = True
    logger.info(f"Classiq version {getattr(classiq, '__version__', 'unknown')} available")
except ImportError as e:
    logger.warning(f"Classiq not available: {e}")
    CLASSIQ_AVAILABLE = False


    # Mock functions for development
    def authenticate():
        raise RuntimeError("Classiq not installed")


    def set_credentials(**kwargs):
        raise RuntimeError("Classiq not installed")


@dataclass
class ClassiqConfig:
    """Configuration for Classiq quantum backend"""
    api_key: Optional[str] = None
    backend_provider: str = "IBM"
    max_qubits: int = 10
    optimization_level: int = 2
    shots: int = 1024
    use_hardware: bool = False
    timeout: int = 300  # 5 minutes


class ClassiqAuthManager:
    """Manages Classiq authentication and configuration"""

    def __init__(self):
        self.config = ClassiqConfig()
        self._authenticated = False
        self._initialized = False
        self._last_error: Optional[str] = None
        self._load_config()

    def _load_config(self):
        """Load configuration from environment"""
        # Load API key
        self.config.api_key = os.getenv("CLASSIQ_API_KEY")

        # Load other settings
        self.config.backend_provider = os.getenv("CLASSIQ_BACKEND_PROVIDER", "IBM")
        self.config.max_qubits = int(os.getenv("CLASSIQ_MAX_QUBITS", "10"))
        self.config.optimization_level = int(os.getenv("CLASSIQ_OPTIMIZATION_LEVEL", "2"))
        self.config.shots = int(os.getenv("CLASSIQ_SHOTS", "1024"))
        self.config.use_hardware = os.getenv("CLASSIQ_USE_HARDWARE", "false").lower() == "true"

        logger.info(f"Loaded Classiq config: provider={self.config.backend_provider}, "
                    f"qubits={self.config.max_qubits}, hardware={self.config.use_hardware}")

    async def initialize(self):
        """Initialize Classiq authentication"""
        if self._initialized:
            return

        if not CLASSIQ_AVAILABLE:
            logger.warning("Classiq not available - running in simulation mode")
            self._initialized = True
            return

        try:
            if self.config.api_key and self.config.api_key != "your_actual_api_key_here":
                # Use API key authentication
                logger.info("Authenticating with Classiq using API key...")
                set_credentials(api_key=self.config.api_key)
                self._authenticated = True
                logger.info("✅ Classiq authentication successful")
            else:
                logger.warning("No valid API key found - quantum features limited")
                self._authenticated = False

            self._initialized = True

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Classiq authentication failed: {e}")
            self._authenticated = False
            self._initialized = True

    def is_authenticated(self) -> bool:
        """Check if authenticated with Classiq"""
        return CLASSIQ_AVAILABLE and self._authenticated

    def update_config(self, **kwargs):
        """Update configuration dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated Classiq config: {key}={value}")

    def get_backend_preferences(self) -> Dict[str, Any]:
        """Get backend preferences for Classiq"""
        return {
            "backend_provider": self.config.backend_provider,
            "use_hardware": self.config.use_hardware,
            "max_qubits": self.config.max_qubits,
            "optimization_level": self.config.optimization_level
        }

    def get_execution_preferences(self) -> Dict[str, Any]:
        """Get execution preferences"""
        return {
            "num_shots": self.config.shots,
            "backend_preferences": self.get_backend_preferences()
        }

    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits based on configuration"""
        return {
            "max_qubits": self.config.max_qubits,
            "max_circuit_depth": 1000,
            "max_gates": 10000,
            "timeout_seconds": self.config.timeout
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "classiq_available": CLASSIQ_AVAILABLE,
            "authenticated": self._authenticated,
            "initialized": self._initialized,
            "last_error": self._last_error,
            "config": {
                "backend_provider": self.config.backend_provider,
                "max_qubits": self.config.max_qubits,
                "use_hardware": self.config.use_hardware,
                "optimization_level": self.config.optimization_level
            }
        }


# Global instance
classiq_auth = ClassiqAuthManager()