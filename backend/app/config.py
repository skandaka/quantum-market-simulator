"""Configuration settings for the application"""

import os
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings as PydanticBaseSettings
    from pydantic_settings import SettingsConfigDict
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings as PydanticBaseSettings
    SettingsConfigDict = None

class Settings(PydanticBaseSettings):
    """Application settings"""

    # App settings
    app_name: str = "Quantum Market Reaction Simulator"
    api_version: str = "v1"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # CORS
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Database (if needed)
    database_url: Optional[str] = os.getenv("DATABASE_URL")

    # Redis (for caching)
    redis_url: Optional[str] = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Quantum settings
    quantum_backend: str = os.getenv("QUANTUM_BACKEND", "simulator")
    max_qubits: int = 10
    enable_quantum: bool = os.getenv("ENABLE_QUANTUM", "true").lower() == "true"  # Enable by default

    # API Keys
    alpha_vantage_api_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # News sources
    news_sources: dict = {
        "twitter": os.getenv("TWITTER_ENABLED", "false").lower() == "true",
        "reddit": os.getenv("REDDIT_ENABLED", "false").lower() == "true"
    }

    # Model settings
    sentiment_model: str = "ProsusAI/finbert"

    # Simulation settings
    default_scenarios: int = 1000
    confidence_intervals: List[float] = [0.68, 0.95]  # 1σ and 2σ

    class Config:
        if SettingsConfigDict:
            model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='allow')
        else:
            env_file = ".env"
            env_file_encoding = 'utf-8'
            extra = 'allow'


# Create settings instance
settings = Settings()