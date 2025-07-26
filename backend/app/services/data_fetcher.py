"""Market data fetching service"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies

from app.config import settings
from app.utils.helpers import cache_result, RateLimiter

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetch market data from multiple sources"""

    def __init__(self):
        self.session = None

        # Initialize API clients
        if settings.alpha_vantage_api_key:
            self.av_ts = TimeSeries(key=settings.alpha_vantage_api_key, output_format='pandas')
            self.av_crypto = CryptoCurrencies(key=settings.alpha_vantage_api_key, output_format='pandas')
        else:
            self.av_ts = None
            self.av_crypto = None

        # Rate limiters
        self.av_limiter = RateLimiter(calls_per_minute=5)  # Alpha Vantage free tier
        self.yf_limiter = RateLimiter(calls_per_minute=60)

    async def fetch_assets(
            self,
            assets: List[str],
            asset_type: str
    ) -> Dict[str, Any]:
        """Fetch data for multiple assets"""

        tasks = []
        for asset in assets:
            task = self.fetch_single_asset(asset, asset_type)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict
        asset_data = {}
        for asset, result in zip(assets, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {asset}: {result}")
                # Provide default data
                asset_data[asset] = self._get_default_asset_data(asset)
            else:
                asset_data[asset] = result

        return asset_data

    @cache_result(ttl=300)  # Cache for 5 minutes
    async def fetch_single_asset(
            self,
            symbol: str,
            asset_type: str,
            period: str = "1d"
    ) -> Dict[str, Any]:
        """Fetch data for a single asset"""

        try:
            if asset_type in ["stock", "index"]:
                return await self._fetch_stock_data(symbol, period)
            elif asset_type == "crypto":
                return await self._fetch_crypto_data(symbol, period)
            elif asset_type == "forex":
                return await self._fetch_forex_data(symbol, period)
            else:
                raise ValueError(f"Unknown asset type: {asset_type}")

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return self._get_default_asset_data(symbol)

    async def _fetch_stock_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Fetch stock data using yfinance"""

        await self.yf_limiter.acquire()

        try:
            # Get ticker object
            ticker = yf.Ticker(symbol)

            # Fetch recent data
            hist = ticker.history(period=period)

            if hist.empty:
                raise ValueError(f"No data found for {symbol}")

            # Get current info
            info = ticker.info

            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price

            # Calculate volatility (annualized)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2

            # Calculate trend (simple moving average)
            if len(hist) >= 20:
                sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                trend = (current_price - sma_20) / sma_20
            else:
                trend = 0.0

            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "previous_close": float(prev_close),
                "open": float(hist['Open'].iloc[-1]),
                "high": float(hist['High'].iloc[-1]),
                "low": float(hist['Low'].iloc[-1]),
                "volume": int(hist['Volume'].iloc[-1]),
                "volatility": float(volatility),
                "trend": float(trend),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "dividend_yield": info.get('dividendYield', 0),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")

            # Try Alpha Vantage as fallback
            if self.av_ts:
                return await self._fetch_stock_alpha_vantage(symbol)
            else:
                raise

    async def _fetch_stock_alpha_vantage(self, symbol: str) -> Dict[str, Any]:
        """Fetch stock data using Alpha Vantage"""

        await self.av_limiter.acquire()

        try:
            # Get daily data
            data, meta_data = await asyncio.to_thread(
                self.av_ts.get_daily, symbol=symbol
            )

            if data.empty:
                raise ValueError(f"No data from Alpha Vantage for {symbol}")

            # Get latest data
            latest = data.iloc[0]
            prev = data.iloc[1] if len(data) > 1 else latest

            # Calculate volatility
            returns = data['4. close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2

            return {
                "symbol": symbol,
                "current_price": float(latest['4. close']),
                "previous_close": float(prev['4. close']),
                "open": float(latest['1. open']),
                "high": float(latest['2. high']),
                "low": float(latest['3. low']),
                "volume": int(latest['5. volume']),
                "volatility": float(volatility),
                "trend": 0.0,  # Would need more data
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            raise

    async def _fetch_crypto_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Fetch cryptocurrency data"""

        # Convert symbol format if needed (BTC-USD -> BTC)
        crypto_symbol = symbol.split('-')[0] if '-' in symbol else symbol

        try:
            # Try yfinance first (supports crypto)
            if '-USD' in symbol or '-USDT' in symbol:
                return await self._fetch_stock_data(symbol, period)

            # Use Alpha Vantage for crypto
            if self.av_crypto:
                await self.av_limiter.acquire()

                data, meta_data = await asyncio.to_thread(
                    self.av_crypto.get_digital_currency_daily,
                    symbol=crypto_symbol,
                    market='USD'
                )

                if data.empty:
                    raise ValueError(f"No crypto data for {symbol}")

                latest = data.iloc[0]
                prev = data.iloc[1] if len(data) > 1 else latest

                # Calculate metrics
                returns = data['4a. close (USD)'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(365) if len(returns) > 0 else 0.4

                return {
                    "symbol": symbol,
                    "current_price": float(latest['4a. close (USD)']),
                    "previous_close": float(prev['4a. close (USD)']),
                    "open": float(latest['1a. open (USD)']),
                    "high": float(latest['2a. high (USD)']),
                    "low": float(latest['3a. low (USD)']),
                    "volume": float(latest['5. volume']),
                    "volatility": float(volatility),
                    "market_cap": float(latest.get('6. market cap (USD)', 0)),
                    "timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Crypto data error for {symbol}: {e}")
            return self._get_default_asset_data(symbol)

    async def _fetch_forex_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Fetch forex data"""

        # For forex, use Alpha Vantage or mock data
        # Symbol format: EURUSD, GBPUSD, etc.

        if len(symbol) != 6:
            raise ValueError(f"Invalid forex symbol: {symbol}")

        from_currency = symbol[:3]
        to_currency = symbol[3:]

        # Mock data for hackathon
        # In production, use real forex API
        base_rate = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 155.20,
            "AUDUSD": 0.6520
        }.get(symbol, 1.0)

        # Add some randomness
        current_price = base_rate * (1 + np.random.normal(0, 0.001))

        return {
            "symbol": symbol,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "current_price": float(current_price),
            "previous_close": float(base_rate),
            "bid": float(current_price - 0.0001),
            "ask": float(current_price + 0.0001),
            "volatility": 0.08,  # Typical forex volatility
            "timestamp": datetime.utcnow().isoformat()
        }

    def _get_default_asset_data(self, symbol: str) -> Dict[str, Any]:
        """Get default data when fetching fails"""

        # Use reasonable defaults
        base_prices = {
            "AAPL": 195.0,
            "GOOGL": 155.0,
            "MSFT": 420.0,
            "TSLA": 200.0,
            "BTC-USD": 65000.0,
            "ETH-USD": 3500.0
        }

        base_price = base_prices.get(symbol, 100.0)

        return {
            "symbol": symbol,
            "current_price": base_price,
            "previous_close": base_price * 0.99,
            "open": base_price * 0.995,
            "high": base_price * 1.01,
            "low": base_price * 0.99,
            "volume": 1000000,
            "volatility": 0.25,
            "trend": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "is_mock_data": True
        }

    async def get_historical_data(
            self,
            symbol: str,
            start_date: datetime,
            end_date: datetime
    ) -> pd.DataFrame:
        """Get historical data for backtesting"""

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if hist.empty:
                # Generate mock historical data
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                prices = self._generate_mock_prices(len(dates))

                hist = pd.DataFrame({
                    'Close': prices,
                    'Open': prices * 0.99,
                    'High': prices * 1.01,
                    'Low': prices * 0.98,
                    'Volume': np.random.randint(1000000, 10000000, len(dates))
                }, index=dates)

            return hist

        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            # Return mock data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            prices = self._generate_mock_prices(len(dates))

            return pd.DataFrame({
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)

    def _generate_mock_prices(self, n_days: int) -> np.ndarray:
        """Generate mock price series"""

        # Geometric Brownian Motion
        initial_price = 100
        drift = 0.0002  # Daily drift
        volatility = 0.02  # Daily volatility

        returns = np.random.normal(drift, volatility, n_days)
        price_series = initial_price * np.exp(np.cumsum(returns))

        return price_series