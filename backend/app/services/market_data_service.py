"""
Market Data Service
Handles fetching and processing of financial market data using yfinance.
"""

import yfinance as yf
import pandas as pd
import logging
from typing import Optional, List
from retry import retry

logger = logging.getLogger(__name__)

class MarketDataService:
    """
    A service to fetch market data for given tickers.
    """

    @retry(tries=3, delay=2, backoff=2, logger=logger)
    def get_price_history(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetches historical price data for a single ticker with retry logic.

        Args:
            ticker (str): The stock ticker symbol.
            period (str): The period for which to fetch data (e.g., '1d', '5d', '1mo', '1y').

        Returns:
            Optional[pd.DataFrame]: A DataFrame with historical data, or None if fetching fails.
        """
        try:
            logger.info(f"Attempting to fetch price history for {ticker} over period {period}...")
            stock = yf.Ticker(ticker)
            history = stock.history(period=period)

            if history.empty:
                logger.warning(f"No price data found for ticker '{ticker}' for the period '{period}'. The symbol may be delisted.")
                return None

            logger.info(f"Successfully fetched price history for {ticker}.")
            return history
        except Exception as e:
            logger.error(f"An error occurred while fetching data for {ticker} using yfinance: {e}", exc_info=True)
            # The @retry decorator will handle re-running this method.
            # If all retries fail, it will re-raise the last exception.
            raise e

    def get_market_caps(self, tickers: List[str]) -> dict:
        """
        Fetches market capitalization for a list of tickers.

        Args:
            tickers (List[str]): A list of stock ticker symbols.

        Returns:
            dict: A dictionary mapping tickers to their market caps.
        """
        caps = {}
        logger.info(f"Fetching market caps for tickers: {tickers}")
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # .info can be slow, so we fetch it once per ticker
                info = stock.info
                market_cap = info.get('marketCap')

                if market_cap:
                    caps[ticker] = market_cap
                else:
                    logger.warning(f"Could not retrieve market cap for {ticker}. Defaulting to 0.")
                    caps[ticker] = 0
            except Exception as e:
                logger.error(f"Error fetching market cap info for {ticker}: {e}", exc_info=True)
                caps[ticker] = 0 # Default to 0 on error to avoid breaking the simulation
        logger.info("Market cap fetching complete.")
        return caps