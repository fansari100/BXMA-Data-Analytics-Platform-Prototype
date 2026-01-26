"""
Semantic Alpha Pipeline
=======================

Processes text streams (news, filings, social media) into dense vector
embeddings using financial-domain LLMs (FinBERT, FinGPT, Fin-E5).

Computes "Semantic Divergence" - the distance between news sentiment
vectors and price action vectors - to identify potential mispricings.

Key Capabilities:
- Real-time text vectorization with financial domain models
- Semantic time-travel queries (KDB-X integration)
- Divergence-based alpha signals
- Entity extraction and linking

References:
- Yang et al. (2020): FinBERT - Financial Sentiment Analysis
- Wu et al. (2023): FinGPT - Open-source Financial LLM
- Araci (2019): FinBERT - Pre-trained NLP Model for Financial Text

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Literal
from enum import Enum, auto
import re
import hashlib


class TextSource(Enum):
    """Source types for text data."""
    NEWS = auto()
    SEC_FILING = auto()
    EARNINGS_CALL = auto()
    SOCIAL_TWITTER = auto()
    SOCIAL_REDDIT = auto()
    ANALYST_REPORT = auto()
    CENTRAL_BANK = auto()
    PRESS_RELEASE = auto()


@dataclass
class TextDocument:
    """A text document for processing."""
    
    id: str = ""
    source: TextSource = TextSource.NEWS
    
    # Content
    title: str = ""
    content: str = ""
    url: str = ""
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    author: str = ""
    entities: list[str] = field(default_factory=list)  # Extracted entities
    
    # Labels (if available)
    sentiment_label: str | None = None  # "positive", "negative", "neutral"


@dataclass
class TextEmbedding:
    """Embedding result for a text document."""
    
    document_id: str
    embedding: NDArray[np.float64]  # Dense vector
    
    # Sentiment scores
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_confidence: float = 0.0
    
    # Entity embeddings
    entity_embeddings: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    
    # Topic distribution
    topics: dict[str, float] = field(default_factory=dict)
    
    # Metadata
    model_name: str = ""
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SemanticDivergence:
    """
    Divergence between semantic signal and price action.
    
    High divergence may indicate mispricing or regime change.
    """
    
    entity: str  # Company or asset
    
    # Sentiment vs Price
    semantic_sentiment: float  # -1 to 1
    price_trend: float  # -1 to 1 (normalized returns)
    divergence_score: float  # Absolute difference
    
    # Confidence
    n_documents: int = 0
    sentiment_variance: float = 0.0
    
    # Signal
    signal_type: str = ""  # "bullish_divergence", "bearish_divergence", "aligned"
    signal_strength: float = 0.0  # 0 to 1
    
    # Time window
    window_start: datetime = field(default_factory=datetime.now)
    window_end: datetime = field(default_factory=datetime.now)


class FinancialEmbedder:
    """
    Embeds financial text into dense vectors using domain-specific models.
    
    Supports:
    - FinBERT: Sentiment-focused embeddings
    - Fin-E5: General financial embeddings
    - Custom models
    """
    
    def __init__(
        self,
        model_name: str = "finbert",
        embedding_dim: int = 768,
        use_gpu: bool = False,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        
        # In production, load actual models
        # self.model = AutoModel.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for embedding."""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if needed
        if len(text) > 4096:
            text = text[:4096]
        
        return text
    
    def embed(self, document: TextDocument) -> TextEmbedding:
        """
        Embed a single document.
        
        In production, this calls the actual model.
        """
        import time
        start_time = time.time()
        
        # Preprocess
        text = f"{document.title} {document.content}"
        text = self._preprocess(text)
        
        # Generate embedding (placeholder - real implementation uses model)
        # In production:
        # inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        # outputs = self.model(**inputs)
        # embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        
        # Simulated embedding based on text hash for consistency
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Simulated sentiment (would come from model)
        sentiment_keywords = {
            "positive": ["growth", "profit", "gain", "up", "rise", "beat", "strong"],
            "negative": ["loss", "decline", "fall", "down", "miss", "weak", "concern"],
        }
        
        text_lower = text.lower()
        pos_count = sum(1 for w in sentiment_keywords["positive"] if w in text_lower)
        neg_count = sum(1 for w in sentiment_keywords["negative"] if w in text_lower)
        
        if pos_count + neg_count > 0:
            sentiment = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            sentiment = 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        return TextEmbedding(
            document_id=document.id,
            embedding=embedding,
            sentiment_score=sentiment,
            sentiment_confidence=0.8,
            model_name=self.model_name,
            processing_time_ms=processing_time,
        )
    
    def embed_batch(self, documents: list[TextDocument]) -> list[TextEmbedding]:
        """Embed multiple documents."""
        return [self.embed(doc) for doc in documents]


class EntityExtractor:
    """
    Extracts and links financial entities from text.
    
    Entities include:
    - Companies (with ticker mapping)
    - People (executives, analysts)
    - Products/Services
    - Financial instruments
    """
    
    def __init__(self):
        # Ticker mapping (in production, load from database)
        self.ticker_map = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "meta": "META",
            "facebook": "META",
            "jpmorgan": "JPM",
            "goldman": "GS",
            "blackstone": "BX",
        }
    
    def extract(self, text: str) -> list[dict]:
        """
        Extract entities from text.
        
        In production, uses NER models (spaCy, HuggingFace).
        """
        entities = []
        text_lower = text.lower()
        
        # Simple keyword matching (placeholder for NER)
        for name, ticker in self.ticker_map.items():
            if name in text_lower:
                entities.append({
                    "text": name,
                    "type": "ORG",
                    "ticker": ticker,
                    "confidence": 0.9,
                })
        
        # Extract dollar amounts
        dollar_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:billion|million|B|M))?'
        for match in re.finditer(dollar_pattern, text, re.IGNORECASE):
            entities.append({
                "text": match.group(),
                "type": "MONEY",
                "confidence": 0.95,
            })
        
        # Extract percentages
        pct_pattern = r'[\d.]+\s*%'
        for match in re.finditer(pct_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "PERCENT",
                "confidence": 0.95,
            })
        
        return entities


class SemanticTimeSeries:
    """
    Time-series of semantic embeddings for an entity.
    
    Enables "semantic time travel" queries - finding historical periods
    with similar semantic conditions.
    """
    
    def __init__(self, entity: str):
        self.entity = entity
        self.embeddings: list[tuple[datetime, NDArray[np.float64]]] = []
        self.sentiments: list[tuple[datetime, float]] = []
    
    def add(self, timestamp: datetime, embedding: NDArray[np.float64], sentiment: float):
        """Add an embedding to the time series."""
        self.embeddings.append((timestamp, embedding))
        self.sentiments.append((timestamp, sentiment))
    
    def get_embedding_at(self, timestamp: datetime) -> NDArray[np.float64] | None:
        """Get embedding closest to a timestamp."""
        if not self.embeddings:
            return None
        
        closest = min(self.embeddings, key=lambda x: abs((x[0] - timestamp).total_seconds()))
        return closest[1]
    
    def find_similar_periods(
        self,
        query_embedding: NDArray[np.float64],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> list[tuple[datetime, float]]:
        """
        Find historical periods with similar semantic embeddings.
        
        Returns list of (timestamp, similarity_score).
        """
        similarities = []
        
        for timestamp, emb in self.embeddings:
            # Cosine similarity
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8
            )
            if sim >= min_similarity:
                similarities.append((timestamp, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]
    
    def compute_divergence(
        self,
        price_returns: list[tuple[datetime, float]],
        window_days: int = 5,
    ) -> list[SemanticDivergence]:
        """
        Compute semantic divergence vs price action.
        
        Args:
            price_returns: List of (timestamp, return)
            window_days: Rolling window in days
            
        Returns:
            List of divergence measurements
        """
        divergences = []
        
        # Group sentiments by date
        sentiment_by_date: dict[str, list[float]] = {}
        for ts, sent in self.sentiments:
            date_key = ts.strftime("%Y-%m-%d")
            if date_key not in sentiment_by_date:
                sentiment_by_date[date_key] = []
            sentiment_by_date[date_key].append(sent)
        
        # Match with price returns
        for ts, ret in price_returns:
            date_key = ts.strftime("%Y-%m-%d")
            
            if date_key in sentiment_by_date:
                avg_sentiment = np.mean(sentiment_by_date[date_key])
                
                # Normalize return to -1, 1 scale
                norm_return = np.tanh(ret * 20)  # Scale and bound
                
                # Divergence
                div_score = abs(avg_sentiment - norm_return)
                
                # Determine signal type
                if avg_sentiment > 0.2 and norm_return < -0.2:
                    signal_type = "bullish_divergence"
                elif avg_sentiment < -0.2 and norm_return > 0.2:
                    signal_type = "bearish_divergence"
                else:
                    signal_type = "aligned"
                
                divergences.append(SemanticDivergence(
                    entity=self.entity,
                    semantic_sentiment=avg_sentiment,
                    price_trend=norm_return,
                    divergence_score=div_score,
                    n_documents=len(sentiment_by_date[date_key]),
                    sentiment_variance=np.var(sentiment_by_date[date_key]),
                    signal_type=signal_type,
                    signal_strength=div_score if signal_type != "aligned" else 0,
                    window_start=ts,
                    window_end=ts,
                ))
        
        return divergences


class SemanticAlphaPipeline:
    """
    Main pipeline for semantic alpha signal generation.
    
    Flow:
    1. Ingest text streams
    2. Extract entities
    3. Embed documents
    4. Compute divergences
    5. Generate signals
    """
    
    def __init__(
        self,
        embedder: FinancialEmbedder | None = None,
        extractor: EntityExtractor | None = None,
    ):
        self.embedder = embedder or FinancialEmbedder()
        self.extractor = extractor or EntityExtractor()
        
        # Entity time series
        self.entity_series: dict[str, SemanticTimeSeries] = {}
        
        # Document store
        self.documents: list[TextDocument] = []
        self.embeddings: list[TextEmbedding] = []
    
    def ingest(self, document: TextDocument):
        """Ingest a new document."""
        # Extract entities
        entities = self.extractor.extract(f"{document.title} {document.content}")
        document.entities = [e.get("ticker", e["text"]) for e in entities if e["type"] == "ORG"]
        
        # Embed
        embedding = self.embedder.embed(document)
        
        # Store
        self.documents.append(document)
        self.embeddings.append(embedding)
        
        # Update entity time series
        for entity in document.entities:
            if entity not in self.entity_series:
                self.entity_series[entity] = SemanticTimeSeries(entity)
            
            self.entity_series[entity].add(
                timestamp=document.timestamp,
                embedding=embedding.embedding,
                sentiment=embedding.sentiment_score,
            )
    
    def ingest_batch(self, documents: list[TextDocument]):
        """Ingest multiple documents."""
        for doc in documents:
            self.ingest(doc)
    
    def get_entity_sentiment(
        self,
        entity: str,
        lookback_hours: int = 24,
    ) -> dict:
        """Get aggregated sentiment for an entity."""
        if entity not in self.entity_series:
            return {"entity": entity, "sentiment": 0, "n_docs": 0}
        
        series = self.entity_series[entity]
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        
        recent = [s for ts, s in series.sentiments if ts >= cutoff]
        
        if not recent:
            return {"entity": entity, "sentiment": 0, "n_docs": 0}
        
        return {
            "entity": entity,
            "sentiment": np.mean(recent),
            "sentiment_std": np.std(recent),
            "n_docs": len(recent),
            "min_sentiment": min(recent),
            "max_sentiment": max(recent),
        }
    
    def compute_divergence_signals(
        self,
        price_data: dict[str, list[tuple[datetime, float]]],
    ) -> list[SemanticDivergence]:
        """
        Compute divergence signals for all entities with price data.
        
        Args:
            price_data: Dict of entity -> [(timestamp, return)]
            
        Returns:
            List of divergence signals
        """
        all_divergences = []
        
        for entity, returns in price_data.items():
            if entity in self.entity_series:
                divs = self.entity_series[entity].compute_divergence(returns)
                all_divergences.extend(divs)
        
        # Sort by signal strength
        all_divergences.sort(key=lambda x: -x.signal_strength)
        
        return all_divergences
    
    def semantic_time_travel(
        self,
        entity: str,
        reference_text: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find historical periods with similar semantic conditions.
        
        Args:
            entity: Entity to search
            reference_text: Text describing current conditions
            top_k: Number of similar periods to return
            
        Returns:
            List of similar historical periods with metadata
        """
        if entity not in self.entity_series:
            return []
        
        # Embed reference text
        ref_doc = TextDocument(
            id="query",
            title="Query",
            content=reference_text,
        )
        ref_embedding = self.embedder.embed(ref_doc)
        
        # Find similar periods
        similar = self.entity_series[entity].find_similar_periods(
            ref_embedding.embedding,
            top_k=top_k,
        )
        
        return [
            {
                "entity": entity,
                "date": ts.strftime("%Y-%m-%d"),
                "similarity": sim,
            }
            for ts, sim in similar
        ]
    
    def generate_alpha_report(self) -> dict:
        """Generate a summary report of semantic alpha signals."""
        # Aggregate entity sentiments
        entity_sentiments = {}
        for entity in self.entity_series:
            entity_sentiments[entity] = self.get_entity_sentiment(entity)
        
        # Sort by absolute sentiment
        sorted_entities = sorted(
            entity_sentiments.items(),
            key=lambda x: abs(x[1]["sentiment"]),
            reverse=True,
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(self.documents),
            "entities_tracked": len(self.entity_series),
            "top_bullish": [
                e for e, s in sorted_entities
                if s["sentiment"] > 0.3 and s["n_docs"] >= 3
            ][:5],
            "top_bearish": [
                e for e, s in sorted_entities
                if s["sentiment"] < -0.3 and s["n_docs"] >= 3
            ][:5],
            "entity_sentiments": dict(sorted_entities[:20]),
        }
