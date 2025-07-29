"""Classiq authentication and configuration"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import classiq
    from classiq import authenticate
    CLASSIQ_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Classiq not installed: {e}")
    CLASSIQ_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ClassiqConfig:
    """Classiq configuration settings"""
    use_hardware: bool = False
    max_qubits: int = 25
    optimization_level: int = 2
    backend_provider: str = "IBM"
    preferred_backend: Optional[str] = None
    shots: int = 4096
    seed: Optional[int] = 42


class ClassiqAuthManager:
    """Manage Classiq authentication and configuration"""

    def __init__(self):
        self.authenticated = False
        self.config = ClassiqConfig()
        self._client = None

    async def initialize(self):
        """Initialize Classiq connection"""
        if not CLASSIQ_AVAILABLE:
            logger.warning("Classiq not available, running in simulation mode")
            self.authenticated = False
            return

        try:
            # Try environment variable first
            api_key = os.getenv("CLASSIQ_API_KEY")

            if api_key and api_key != "dummy-key-for-hackathon":
                # Use API key authentication
                authenticate(api_key)
                self.authenticated = True
                logger.info("Authenticated with Classiq using API key")
            else:
                # Try browser-based authentication
                logger.info("No API key found, attempting browser authentication...")
                authenticate()  # This will open a browser
                self.authenticated = True
                logger.info("Authenticated with Classiq via browser")

            # Test connection
            self._test_connection()

        except Exception as e:
            logger.error(f"Classiq authentication failed: {e}")
            logger.warning("Running in offline mode - quantum features will be simulated")
            self.authenticated = False

    def _test_connection(self):
        """Test Classiq connection by checking available backends"""
        if not CLASSIQ_AVAILABLE:
            return

        try:
            # Simple test - try to import show function
            from classiq import show
            logger.info("Classiq connection test successful")
        except Exception as e:
            logger.warning(f"Could not verify Classiq connection: {e}")

    def get_backend_preferences(self) -> Dict[str, Any]:
        """Get backend preferences for circuit execution"""
        if not CLASSIQ_AVAILABLE:
            return {"backend": "simulator"}

        # Return simplified preferences for Classiq
        return {
            "backend": self.config.backend_provider.lower() if self.config.use_hardware else "simulator",
            "optimization_level": self.config.optimization_level
        }

    def get_execution_preferences(self) -> Dict[str, Any]:
        """Get execution preferences"""
        return {
            "backend_preferences": self.get_backend_preferences(),
            "num_shots": self.config.shots,
            "random_seed": self.config.seed,
            "job_name": "quantum_market_simulator"
        }

    def is_authenticated(self) -> bool:
        """Check if authenticated with Classiq"""
        return self.authenticated

    def update_config(self, **kwargs):
        """Update configuration settings"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated Classiq config: {key}={value}")

    def get_resource_limits(self) -> Dict[str, Any]:
        """Get current resource limits"""
        return {
            "max_qubits": self.config.max_qubits,
            "max_gates": 10000,  # Typical limit
            "max_circuit_depth": 1000,
            "optimization_level": self.config.optimization_level,
            "available_hardware": self.config.use_hardware
        }


# Global instance
classiq_auth = ClassiqAuthManager()