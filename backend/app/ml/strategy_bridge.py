"""
strategy_bridge.py
==================
Bridge between MLRanker and StrategyEngine.

Enables hybrid scoring: linear weights + ML ranker ensemble.
When ML model is trained and available, it contributes 30% to final score.
Fallback to pure linear scoring when ML is unavailable.

Usage::

    from app.ml.strategy_bridge import MLStrategyBridge
    bridge = MLStrategyBridge()
    scores = bridge.score(factors_df)  # hybrid linear + ML
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLStrategyBridge:
    """
    Hybrid scoring bridge: 70% linear + 30% ML (when available).

    The ML component captures non-linear factor interactions
    that linear weights miss (e.g., momentum × volatility regimes).
    """

    def __init__(self, ml_weight: float = 0.30) -> None:
        self.ml_weight = ml_weight
        self.linear_weight = 1.0 - ml_weight
        self._ranker = None
        self._feature_cols = ["momentum", "volume_ratio", "atr_ratio", "liquidity", "market_cap", "overnight_return"]

    @property
    def ranker(self):
        """Lazy-load MLRanker."""
        if self._ranker is None:
            try:
                from app.ml.ml_ranker import MLRanker
                self._ranker = MLRanker()
            except ImportError as e:
                logger.warning("MLRanker not available: %s", e)
        return self._ranker

    def is_available(self) -> bool:
        """Check if ML model is loaded and ready."""
        r = self.ranker
        return r is not None and r.model is not None

    def score(
        self,
        factors_df: pd.DataFrame,
        linear_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute hybrid scores.

        Parameters
        ----------
        factors_df: pd.DataFrame
            Factor values (z-scored).
        linear_scores: ndarray, optional
            Pre-computed linear composite scores.

        Returns
        -------
        ndarray — hybrid scores (higher = better).
        """
        n = len(factors_df)
        if linear_scores is None:
            linear_scores = np.zeros(n)

        if not self.is_available():
            return linear_scores

        # ML prediction
        try:
            features = factors_df[self._feature_cols].fillna(0).values
            ml_scores = self.ranker.predict(features)

            # Normalize ML scores to same scale as linear
            if ml_scores.std() > 1e-12:
                ml_scores = (ml_scores - ml_scores.mean()) / ml_scores.std()

            # Ensemble: weighted combination
            hybrid = self.linear_weight * linear_scores + self.ml_weight * ml_scores
            logger.debug("Hybrid scoring: %.0f%% linear + %.0f%% ML", self.linear_weight * 100, self.ml_weight * 100)
            return hybrid

        except Exception as exc:
            logger.warning("ML scoring failed, falling back to linear: %s", exc)
            return linear_scores

    def train(self, features: pd.DataFrame, returns: pd.Series) -> None:
        """
        Train the ML ranker on historical data.

        Parameters
        ----------
        features: pd.DataFrame — factor values
        returns: pd.Series — forward returns (target)
        """
        try:
            from app.ml.ml_ranker import MLRanker
            self._ranker = MLRanker()
            self._ranker.train(features, returns)
            logger.info("ML ranker trained successfully")
        except Exception as exc:
            logger.error("ML training failed: %s", exc)
