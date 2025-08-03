"""
Pydantic Models for Sentiment Analysis
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class NewsArticle(BaseModel):
    """Represents a news article to be analyzed."""
    content: str = Field(..., min_length=20, description="The full text content of the news article.")
    source: Optional[str] = Field(None, description="The source of the news article (e.g., URL, publication).")

class DetectedEntity(BaseModel):
    """Represents a named entity detected in the text."""
    text: str = Field(..., description="The text of the detected entity.")
    type: str = Field(..., description="The type of the entity (e.g., ORG, PERSON, GPE).")
    start: int = Field(..., description="The starting character index of the entity.")
    end: int = Field(..., description="The ending character index of the entity.")

class SentimentAnalysis(BaseModel):
    """Represents the result of a sentiment analysis."""
    overall_sentiment: str = Field(..., description="The overall sentiment of the text (e.g., positive, negative, neutral).")
    sentiment_score: float = Field(..., description="The polarity score, from -1 (very negative) to 1 (very positive).")
    subjectivity_score: float = Field(..., description="The subjectivity score, from 0 (objective) to 1 (subjective).")
    entities_detected: List[DetectedEntity] = Field([], description="A list of named entities detected in the text.")