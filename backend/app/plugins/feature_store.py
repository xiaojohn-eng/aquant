"""
AQuant Feature Store

Centralized feature computation and serving to eliminate training-serving skew.
Implements the Quant 2.0 Feature Store pattern.

Usage:
    from app.plugins.feature_store import FeatureStore
    
    store = FeatureStore("sqlite:///data/feature_store.db")
    store.compute_and_store(date="2024-01-15", codes=["000001.SZ", "600000.SH"])
    features = store.get_features(date="2024-01-15", code="000001.SZ")
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import Column, Date, Float, String, create_engine, text
from sqlalchemy.orm import declarative_base, Session, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class FeatureRecord(Base):
    """Single feature record per stock per day."""
    __tablename__ = "features"
    
    id = Column(String(32), primary_key=True)  # "date|code"
    trade_date = Column(Date, index=True, nullable=False)
    code = Column(String(16), index=True, nullable=False)
    
    # Raw features (stored as JSON for flexibility)
    features_json = Column(String(4096), nullable=False)
    
    # Computed score
    composite_score = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(String(32), nullable=False)
    version = Column(String(8), default="1.0")  # feature schema version


class FeatureStore:
    """
    Centralized feature computation and storage.
    
    Key principles:
    1. Point-in-time: all features computed using only data available at that timestamp
    2. Immutable: historical records never modified (append-only)
    3. Versioned: feature schema versioning for reproducibility
    4. Lazy: compute on-demand with caching
    """
    
    SCHEMA_VERSION = "2.0"
    
    def __init__(self, database_url: str = "sqlite:///data/feature_store.db"):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._cache: Dict[str, Dict[str, float]] = {}  # in-memory LRU cache
        self._max_cache_size = 5000
        
    def _make_id(self, trade_date: date, code: str) -> str:
        return f"{trade_date.isoformat()}|{code}"
    
    def compute_features(self, 
                         trade_date: date,
                         code: str,
                         price_df: pd.DataFrame,
                         volume_df: pd.DataFrame,
                         sector_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Compute all features for a single stock at a single date.
        
        This is the ONLY place where feature computation logic lives.
        All strategies consume pre-computed features from this store.
        """
        # Ensure we only use data up to and including trade_date
        price_hist = price_df[price_df.index <= pd.Timestamp(trade_date)].copy()
        volume_hist = volume_df[volume_df.index <= pd.Timestamp(trade_date)].copy()
        
        if len(price_hist) < 20:
            return {}  # insufficient history
        
        close = price_hist["close"].values
        high = price_hist["high"].values
        low = price_hist["low"].values
        open_p = price_hist["open"].values
        vol = volume_hist["volume"].values
        amt = volume_hist["amount"].values if "amount" in volume_hist.columns else vol * close
        
        # --- Feature 1: Intraday Momentum (core alpha) ---
        if len(close) >= 2:
            # Use only opening 30-min data if available, otherwise first bar
            momentum = (close[-1] / close[0] - 1) * 100 if close[0] > 0 else 0.0
        else:
            momentum = 0.0
        
        # --- Feature 2: Volume Ratio ---
        vol_ma20 = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
        volume_ratio = (vol[-1] / vol_ma20) if vol_ma20 > 0 else 1.0
        
        # --- Feature 3: ATR Ratio (risk control) ---
        if len(close) >= 14:
            tr1 = high[-14:] - low[-14:]
            tr2 = np.abs(high[-14:] - np.roll(close, 1)[-14:])
            tr3 = np.abs(low[-14:] - np.roll(close, 1)[-14:])
            atr = np.mean(np.maximum.reduce([tr1, tr2, tr3]))
            atr_ratio = (atr / close[-1] * 100) if close[-1] > 0 else 0.0
        else:
            atr_ratio = 0.0
        
        # --- Feature 4: Liquidity Score ---
        avg_amt = np.mean(amt[-20:]) if len(amt) >= 20 else np.mean(amt)
        liquidity = (amt[-1] / avg_amt) if avg_amt > 0 else 1.0
        
        # --- Feature 5: Auction Quality (opening gap) ---
        auction_gap = ((open_p[-1] / close[-2] - 1) * 100) if len(close) >= 2 and close[-2] > 0 else 0.0
        
        # --- Feature 6: Market Cap bias (log-transformed) ---
        # Expected to be injected from external data
        market_cap_log = 0.0  # placeholder, filled by caller
        
        # --- Feature 7: Sector Momentum ---
        sector_momentum = 0.0  # placeholder, filled by caller
        
        features = {
            "momentum": round(float(momentum), 4),
            "volume_ratio": round(float(volume_ratio), 4),
            "atr_ratio": round(float(atr_ratio), 4),
            "liquidity": round(float(liquidity), 4),
            "auction_gap": round(float(auction_gap), 4),
            "market_cap_log": round(float(market_cap_log), 4),
            "sector_momentum": round(float(sector_momentum), 4),
            "vol_ma20": round(float(vol_ma20), 2),
            "avg_amt": round(float(avg_amt), 2),
            "schema_version": self.SCHEMA_VERSION,
        }
        
        return features
    
    def store(self,
              trade_date: date,
              code: str,
              features: Dict[str, float],
              composite_score: Optional[float] = None) -> None:
        """Store computed features (immutable append)."""
        record_id = self._make_id(trade_date, code)
        
        session = self.Session()
        try:
            # Check if exists (should not happen in append-only mode, but defensive)
            existing = session.get(FeatureRecord, record_id)
            if existing:
                logger.warning(f"Feature already exists for {record_id}, skipping")
                return
            
            record = FeatureRecord(
                id=record_id,
                trade_date=trade_date,
                code=code,
                features_json=json.dumps(features, ensure_ascii=False),
                composite_score=composite_score,
                created_at=datetime.now().isoformat(),
                version=self.SCHEMA_VERSION,
            )
            session.add(record)
            session.commit()
            
            # Update cache
            self._cache[record_id] = features
            if len(self._cache) > self._max_cache_size:
                self._cache.pop(next(iter(self._cache)))
                
        finally:
            session.close()
    
    def get_features(self, trade_date: date, code: str) -> Optional[Dict[str, float]]:
        """Retrieve features with cache-first strategy."""
        record_id = self._make_id(trade_date, code)
        
        # Cache hit
        if record_id in self._cache:
            return self._cache[record_id]
        
        # DB hit
        session = self.Session()
        try:
            record = session.query(FeatureRecord).get(record_id)
            if record:
                features = json.loads(record.features_json)
                self._cache[record_id] = features
                return features
            return None
        finally:
            session.close()
    
    def get_batch(self, trade_date: date, codes: List[str]) -> pd.DataFrame:
        """Batch retrieve features for multiple stocks as DataFrame."""
        records = []
        for code in codes:
            feats = self.get_features(trade_date, code)
            if feats:
                row = {"code": code, **feats}
                records.append(row)
        
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records).set_index("code")
    
    def compute_composite_score(self, features: Dict[str, float], weights: Dict[str, float]) -> float:
        """Compute weighted composite score from raw features."""
        score = 0.0
        weight_sum = 0.0
        
        for feat_name, weight in weights.items():
            if feat_name in features and features[feat_name] is not None:
                # Z-score normalization using pre-computed means/stds
                score += features[feat_name] * weight
                weight_sum += abs(weight)
        
        return round(score / weight_sum, 4) if weight_sum > 0 else 0.0
    
    def list_dates(self, start: date, end: date) -> List[date]:
        """List all dates with stored features in range."""
        session = self.Session()
        try:
            result = session.query(FeatureRecord.trade_date).distinct().filter(
                FeatureRecord.trade_date >= start,
                FeatureRecord.trade_date <= end
            ).order_by(FeatureRecord.trade_date).all()
            return [r[0] for r in result]
        finally:
            session.close()
    
    def delete_old(self, before: date) -> int:
        """Delete records before a date (for data retention policy)."""
        session = self.Session()
        try:
            result = session.query(FeatureRecord).filter(FeatureRecord.trade_date < before).delete()
            session.commit()
            return result
        finally:
            session.close()
