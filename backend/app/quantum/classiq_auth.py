# Path: /Users/skandaa/Desktop/quantum-market-simulator/backend/app/services/quantum/classiq_auth.py

"""
Classiq Authentication and Client Management
Handles Classiq API authentication with improved error handling and compatibility
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

# Handle pydantic imports with compatibility
try:
    from pydantic import BaseModel, Field
    # Try different import paths for StringConstraints
    try:
        from pydantic import StringConstraints
    except ImportError:
        try:
            from pydantic.types import StringConstraints
        except ImportError:
            try:
                from pydantic.v1 import StringConstraints
            except ImportError:
                # Fallback: create a simple constraint type
                StringConstraints = str
                logging.warning("StringConstraints not available, using str fallback")
except ImportError as e:
    logging.error(f"Failed to import pydantic: {e}")
    raise ImportError("Pydantic is required but not properly installed")

# Classiq imports with error handling
try:
    import classiq
    from classiq import authenticate, set_credentials
    CLASSIQ_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Classiq not available: {e}")
    CLASSIQ_AVAILABLE = False
    # Create mock functions for development
    def authenticate(*args, **kwargs):
        raise RuntimeError("Classiq not installed")
    
    def set_credentials(*args, **kwargs):
        raise RuntimeError("Classiq not installed")

logger = logging.getLogger(__name__)

class ClassiqConfig(BaseModel):
    """Configuration for Classiq client with flexible string constraints"""
    
    # Use conditional field definition based on StringConstraints availability
    if StringConstraints != str:
        api_key: Optional[str] = Field(
            default=None,
            description="Classiq API key for authentication"
        )
    else:
        api_key: Optional[str] = Field(
            default=None,
            description="Classiq API key for authentication"
        )
    
    api_base_url: str = Field(
        default="https://platform.classiq.io",
        description="Base URL for Classiq API"
    )
    
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )

class ClassiqClient:
    """
    Enhanced Classiq client with improved error handling and authentication management
    """
    
    def __init__(self, config: Optional[ClassiqConfig] = None):
        """
        Initialize Classiq client
        
        Args:
            config: Optional configuration. If None, loads from environment
        """
        self.config = config or self._load_config()
        self._authenticated = False
        self._client = None
        self._last_error: Optional[str] = None
        
        # Initialize client if Classiq is available
        if CLASSIQ_AVAILABLE:
            self._initialize_client()
        else:
            logger.warning("Classiq not available - running in simulation mode")
    
    def _load_config(self) -> ClassiqConfig:
        """Load configuration from environment variables and .env file"""
        
        # Try to load .env file
        env_path = Path(".env")
        env_vars = {}
        
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip().strip('"\'')
            except Exception as e:
                logger.warning(f"Could not read .env file: {e}")
        
        # Get API key from environment or .env file
        api_key = os.getenv('CLASSIQ_API_KEY') or env_vars.get('CLASSIQ_API_KEY')
        
        return ClassiqConfig(
            api_key=api_key,
            api_base_url=os.getenv('CLASSIQ_API_BASE_URL', 'https://platform.classiq.io'),
            timeout=int(os.getenv('CLASSIQ_TIMEOUT', '30')),
            max_retries=int(os.getenv('CLASSIQ_MAX_RETRIES', '3'))
        )
    
    def _initialize_client(self):
        """Initialize the Classiq client with authentication"""
        try:
            if self.config.api_key:
                # Use API key authentication
                set_credentials(api_key=self.config.api_key)
                self._authenticated = True
                logger.info("✅ Classiq client initialized with API key")
            else:
                logger.warning("⚠️ No API key provided - some features may be limited")
                self._authenticated = False
                
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"❌ Failed to initialize Classiq client: {e}")
            self._authenticated = False
    
    def authenticate_browser(self) -> bool:
        """
        Authenticate using browser-based flow
        
        Returns:
            bool: True if authentication successful
        """
        if not CLASSIQ_AVAILABLE:
            logger.error("Classiq not available for authentication")
            return False
            
        try:
            authenticate()
            self._authenticated = True
            logger.info("✅ Browser authentication successful")
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"❌ Browser authentication failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if client is properly connected and authenticated
        
        Returns:
            bool: True if connected and authenticated
        """
        if not CLASSIQ_AVAILABLE:
            return False
            
        return self._authenticated
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status information
        
        Returns:
            Dict with status information
        """
        return {
            "classiq_available": CLASSIQ_AVAILABLE,
            "authenticated": self._authenticated,
            "api_key_configured": bool(self.config.api_key),
            "last_error": self._last_error,
            "config": {
                "api_base_url": self.config.api_base_url,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries
            }
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Classiq services
        
        Returns:
            Dict with test results
        """
        results = {
            "success": False,
            "tests": {},
            "errors": []
        }
        
        if not CLASSIQ_AVAILABLE:
            results["errors"].append("Classiq library not available")
            return results
        
        # Test 1: Basic import
        try:
            import classiq
            results["tests"]["import"] = True
        except Exception as e:
            results["tests"]["import"] = False
            results["errors"].append(f"Import failed: {e}")
        
        # Test 2: Authentication status
        results["tests"]["authenticated"] = self._authenticated
        if not self._authenticated and not self.config.api_key:
            results["errors"].append("No API key configured")
        
        # Test 3: Basic quantum function (if authenticated)
        if self._authenticated:
            try:
                from classiq import qfunc, QArray, QBit, Output, allocate, hadamard
                
                @qfunc
                def test_func(q: Output[QArray[QBit]]):
                    allocate(1, q)
                    hadamard(q[0])
                
                results["tests"]["quantum_function"] = True
            except Exception as e:
                results["tests"]["quantum_function"] = False
                results["errors"].append(f"Quantum function test failed: {e}")
        
        results["success"] = all(results["tests"].values()) and len(results["errors"]) == 0
        return results
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get information about the Classiq client and environment
        
        Returns:
            Dict with client information
        """
        info = {
            "classiq_available": CLASSIQ_AVAILABLE,
            "client_initialized": self._client is not None,
            "authenticated": self._authenticated
        }
        
        if CLASSIQ_AVAILABLE:
            try:
                import classiq
                info["classiq_version"] = getattr(classiq, '__version__', 'unknown')
            except:
                info["classiq_version"] = 'unknown'
        
        return info

# Global client instance
_global_client: Optional[ClassiqClient] = None

def get_quantum_client() -> ClassiqClient:
    """
    Get or create global ClassiqClient instance
    
    Returns:
        ClassiqClient: Global client instance
    """
    global _global_client
    if _global_client is None:
        _global_client = ClassiqClient()
    return _global_client

def reset_quantum_client():
    """Reset the global client instance"""
    global _global_client
    _global_client = None

# Convenience functions
def is_quantum_available() -> bool:
    """Check if quantum computing is available"""
    return get_quantum_client().is_connected()

def get_quantum_status() -> Dict[str, Any]:
    """Get quantum backend status"""
    return get_quantum_client().get_status()

# For backward compatibility
ClassiqAuthenticator = ClassiqClient