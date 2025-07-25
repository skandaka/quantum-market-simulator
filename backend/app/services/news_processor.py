"""News processing and parsing service"""

import re
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
import feedparser
import tweepy
from newspaper import Article

from app.models.schemas import NewsInput, NewsSourceType
from app.config import settings
from app.utils.helpers import clean_text, extract_entities

logger = logging.getLogger(__name__)


class NewsProcessor:
    """Process and parse news from various sources"""

    def __init__(self):
        self.session = None
        self.twitter_client = self._init_twitter() if settings.news_sources.get("twitter") else None

    def _init_twitter(self) -> Optional[tweepy.Client]:
        """Initialize Twitter client if credentials available"""
        # Would need Twitter API credentials
        return None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def process_batch(self, news_inputs: List[NewsInput]) -> List[Dict[str, Any]]:
        """Process multiple news inputs in parallel"""
        tasks = [self.process_single(news_input) for news_input in news_inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed processing
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process news item {i}: {result}")
            else:
                processed.append(result)

        return processed

    async def process_single(self, news_input: NewsInput) -> Dict[str, Any]:
        """Process a single news input"""
        # Clean and normalize text
        cleaned_text = clean_text(news_input.content)

        # Extract additional info based on source type
        enriched_data = await self._enrich_by_source(news_input)

        # Extract entities and keywords
        entities = extract_entities(cleaned_text)

        # Detect financial keywords
        financial_keywords = self._extract_financial_keywords(cleaned_text)

        # Analyze text structure
        text_metrics = self._analyze_text_structure(cleaned_text)

        return {
            "original": news_input.dict(),
            "cleaned_text": cleaned_text,
            "entities": entities,
            "financial_keywords": financial_keywords,
            "text_metrics": text_metrics,
            "enriched_data": enriched_data,
            "processed_at": datetime.utcnow()
        }

    async def _enrich_by_source(self, news_input: NewsInput) -> Dict[str, Any]:
        """Enrich news data based on source type"""
        enriched = {}

        if news_input.source_type == NewsSourceType.TWEET:
            # Extract hashtags, mentions, etc.
            enriched["hashtags"] = re.findall(r'#\w+', news_input.content)
            enriched["mentions"] = re.findall(r'@\w+', news_input.content)
            enriched["urls"] = re.findall(r'http[s]?://\S+', news_input.content)

        elif news_input.source_type == NewsSourceType.ARTICLE and news_input.source_url:
            # Fetch full article if URL provided
            try:
                article = await self._fetch_article(str(news_input.source_url))
                enriched.update(article)
            except Exception as e:
                logger.error(f"Failed to fetch article: {e}")

        elif news_input.source_type == NewsSourceType.SEC_FILING:
            # Parse SEC filing specifics
            enriched["filing_type"] = self._detect_filing_type(news_input.content)
            enriched["financial_figures"] = self._extract_financial_figures(news_input.content)

        return enriched

    async def _fetch_article(self, url: str) -> Dict[str, Any]:
        """Fetch and parse full article"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(url, timeout=10) as response:
                html = await response.text()

            # Parse with newspaper3k
            article = Article(url)
            article.set_html(html)
            article.parse()

            return {
                "title": article.title,
                "authors": article.authors,
                "publish_date": article.publish_date,
                "top_image": article.top_image,
                "summary": article.summary,
                "keywords": article.keywords
            }
        except Exception as e:
            logger.error(f"Article fetch failed for {url}: {e}")
            return {}

    def _extract_financial_keywords(self, text: str) -> List[str]:
        """Extract financial/market-related keywords"""
        financial_terms = {
            # Positive terms
            "bullish": ["bullish", "surge", "rally", "soar", "jump", "gain", "rise",
                        "breakthrough", "record high", "beat expectations", "upgrade"],
            # Negative terms
            "bearish": ["bearish", "plunge", "crash", "fall", "drop", "decline",
                        "miss expectations", "downgrade", "recession", "bear market"],
            # Neutral/important terms
            "neutral": ["earnings", "revenue", "guidance", "forecast", "Fed", "inflation",
                        "GDP", "unemployment", "merger", "acquisition", "IPO", "dividend"],
            # Action terms
            "actions": ["buy", "sell", "hold", "accumulate", "reduce", "exit", "enter"]
        }

        found_keywords = {}
        text_lower = text.lower()

        for category, terms in financial_terms.items():
            found_keywords[category] = [
                term for term in terms if term.lower() in text_lower
            ]

        return found_keywords

    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure and complexity"""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()

        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "exclamation_count": text.count('!'),
            "question_count": text.count('?'),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "complexity_score": self._calculate_complexity(text)
        }

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score"""
        # Simple complexity based on sentence length and word variety
        words = text.lower().split()
        unique_words = set(words)

        if not words:
            return 0.0

        variety_score = len(unique_words) / len(words)
        avg_word_length = sum(len(w) for w in words) / len(words)

        # Normalize to 0-1
        complexity = (variety_score * 0.5 + min(avg_word_length / 10, 1) * 0.5)
        return round(complexity, 3)

    def _detect_filing_type(self, text: str) -> Optional[str]:
        """Detect SEC filing type from text"""
        filing_patterns = {
            "10-K": r"10-K|annual report",
            "10-Q": r"10-Q|quarterly report",
            "8-K": r"8-K|current report",
            "DEF 14A": r"DEF 14A|proxy statement",
            "S-1": r"S-1|registration statement"
        }

        text_upper = text.upper()
        for filing_type, pattern in filing_patterns.items():
            if re.search(pattern, text_upper):
                return filing_type

        return None

    def _extract_financial_figures(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial figures from text"""
        # Pattern for currency amounts
        currency_pattern = r'\$?\d+\.?\d*\s*(billion|million|thousand|B|M|K)?'

        figures = []
        for match in re.finditer(currency_pattern, text, re.IGNORECASE):
            value_str = match.group()

            # Parse the value
            try:
                # Remove $ and convert multipliers
                clean_value = value_str.replace('$', '').strip()
                multipliers = {
                    'billion': 1e9, 'B': 1e9,
                    'million': 1e6, 'M': 1e6,
                    'thousand': 1e3, 'K': 1e3
                }

                for suffix, mult in multipliers.items():
                    if suffix in clean_value:
                        number = float(clean_value.replace(suffix, '').strip())
                        value = number * mult
                        break
                else:
                    value = float(clean_value)

                figures.append({
                    "raw_text": value_str,
                    "value": value,
                    "position": match.start()
                })
            except:
                continue

        return figures