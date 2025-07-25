"""Utility functions and helpers"""

import asyncio
import functools
import hashlib
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from textblob import TextBlob
import spacy

# Cache for expensive operations
_cache = {}
_cache_timestamps = {}


def setup_logging(name: str) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def cache_result(ttl: int = 300):
    """Decorator to cache function results"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create cache key
            key = _make_cache_key(func.__name__, args, kwargs)

            # Check cache
            if key in _cache:
                timestamp = _cache_timestamps.get(key, 0)
                if time.time() - timestamp < ttl:
                    return _cache[key]

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            _cache[key] = result
            _cache_timestamps[key] = time.time()

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create cache key
            key = _make_cache_key(func.__name__, args, kwargs)

            # Check cache
            if key in _cache:
                timestamp = _cache_timestamps.get(key, 0)
                if time.time() - timestamp < ttl:
                    return _cache[key]

            # Call function
            result = func(*args, **kwargs)

            # Store in cache
            _cache[key] = result
            _cache_timestamps[key] = time.time()

            return result

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create cache key from function arguments"""
    key_data = {
        'func': func_name,
        'args': args,
        'kwargs': kwargs
    }

    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


class RateLimiter:
    """Rate limiter for API calls"""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        async with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_call = time.time()


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\!\?\-\@\#\$\%]', '', text)

    # Normalize case for consistency
    # Keep original case as it might be important for sentiment

    return text.strip()


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text"""

    # Simple entity extraction without spaCy
    # In production, use spaCy or other NER tools

    entities = {
        "organizations": [],
        "persons": [],
        "locations": [],
        "money": [],
        "percentages": [],
        "dates": []
    }

    # Extract money amounts
    money_pattern = r'\$[\d,]+\.?\d*[BMK]?|\d+\.?\d*\s*(billion|million|thousand|dollars?|usd)'
    for match in re.finditer(money_pattern, text, re.IGNORECASE):
        entities["money"].append(match.group())

    # Extract percentages
    percent_pattern = r'\d+\.?\d*\s*%'
    for match in re.finditer(percent_pattern, text):
        entities["percentages"].append(match.group())

    # Extract potential organization names (capitalized sequences)
    org_pattern = r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b'
    potential_orgs = re.findall(org_pattern, text)

    # Filter common words
    common_words = {"The", "This", "That", "These", "Those", "January", "February",
                    "March", "April", "May", "June", "July", "August", "September",
                    "October", "November", "December"}

    entities["organizations"] = [
                                    org for org in potential_orgs
                                    if org not in common_words and len(org) > 2
                                ][:5]  # Limit to top 5

    return entities


def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """Calculate various text statistics"""

    sentences = re.split(r'[.!?]+', text)
    words = text.split()

    stats = {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "lexical_diversity": len(set(words)) / max(len(words), 1),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1)
    }

    return stats


def detect_language(text: str) -> str:
    """Detect language of text"""
    try:
        blob = TextBlob(text)
        return blob.detect_language()
    except:
        return "en"  # Default to English


def calculate_correlation(series1: List[float], series2: List[float]) -> float:
    """Calculate correlation between two series"""

    if len(series1) != len(series2) or len(series1) < 2:
        return 0.0

    return float(np.corrcoef(series1, series2)[0, 1])


def exponential_moving_average(
        values: List[float],
        alpha: float = 0.1
) -> List[float]:
    """Calculate exponential moving average"""

    if not values:
        return []

    ema = [values[0]]

    for i in range(1, len(values)):
        ema_value = alpha * values[i] + (1 - alpha) * ema[-1]
        ema.append(ema_value)

    return ema


def detect_anomalies(
        values: List[float],
        threshold: float = 3.0
) -> List[int]:
    """Detect anomalies using z-score method"""

    if len(values) < 3:
        return []

    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        return []

    z_scores = [(v - mean) / std for v in values]
    anomalies = [i for i, z in enumerate(z_scores) if abs(z) > threshold]

    return anomalies


def format_number(value: float, decimals: int = 2) -> str:
    """Format number for display"""

    if abs(value) >= 1e9:
        return f"{value / 1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.02
) -> float:
    """Calculate Sharpe ratio"""

    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)

    if std_excess == 0:
        return 0.0

    return float(mean_excess / std_excess * np.sqrt(252))  # Annualized


def calculate_max_drawdown(prices: List[float]) -> float:
    """Calculate maximum drawdown"""

    if not prices or len(prices) < 2:
        return 0.0

    peak = prices[0]
    max_dd = 0.0

    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        if drawdown > max_dd:
            max_dd = drawdown

    return float(max_dd)


def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""

    # Basic validation
    if not symbol or len(symbol) > 10:
        return False

    # Check format
    if symbol.upper() != symbol:
        return False

    # Allow alphanumeric and some special chars
    pattern = r'^[A-Z0-9\-\.]+$'
    return bool(re.match(pattern, symbol))


def parse_time_period(period: str) -> timedelta:
    """Parse time period string to timedelta"""

    period_map = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "5y": timedelta(days=1825),
        "10y": timedelta(days=3650),
        "ytd": timedelta(days=(datetime.now() - datetime(datetime.now().year, 1, 1)).days),
        "max": timedelta(days=36500)  # ~100 years
    }

    return period_map.get(period, timedelta(days=30))


def generate_mock_sentiment_data() -> Dict[str, Any]:
    """Generate mock sentiment data for testing"""

    sentiments = ["very_negative", "negative", "neutral", "positive", "very_positive"]

    return {
        "sentiment": np.random.choice(sentiments),
        "confidence": np.random.uniform(0.6, 0.95),
        "quantum_vector": np.random.dirichlet(np.ones(5)).tolist(),
        "entities": ["AAPL", "Tim Cook", "iPhone"],
        "keywords": ["earnings", "beat", "revenue"]
    }