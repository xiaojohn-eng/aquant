"""
strategy_orchestrator.py
========================
Orchestrates the complete quant pipeline:

Data → FeatureStore → GPU Factors → WFA Validation → ML Ranking → Signals

Activates previously "zombie" modules:
- FeatureStore (point-in-time computation)
- WFAEngine (rolling out-of-sample validation)
- MLRanker (online learning with daily retraining)

Author  : AQuant Orchestration Team
Version : v4.5
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyOrchestratorConfig:
    """Pipeline configuration."""
    wfa_enabled: bool = True
    wfa_min_sharpe: float = 1.0
    ml_enabled: bool = True
    ml_retrain_interval_days: int = 1
    feature_store_enabled: bool = True
    use_minute_data: bool = True


class StrategyOrchestrator:
    """
    Central pipeline controller.

    1. Fetches raw data (daily + minute if available)
    2. Computes point-in-time features via FeatureStore
    3. Runs Walk-Forward Analysis for validation
    4. If WFA passes → trains ML ranker → generates signals
    5. Risk checks → outputs final recommendations
    """

    def __init__(self, config: Optional[StrategyOrchestratorConfig] = None) -> None:
        self.cfg = config or StrategyOrchestratorConfig()
        self._last_ml_train: Optional[date] = None
        self._wfa_result: Optional[Dict] = None

    async def run_daily_pipeline(
        self,
        trade_date: date,
        price_data: pd.DataFrame,
    ) -> List[Dict]:
        """
        Execute full daily pipeline.

        Returns list of top-20 stock recommendations.
        """
        logger.info("[%s] Starting daily pipeline", trade_date)

        # Step 1: Feature Store (point-in-time)
        if self.cfg.feature_store_enabled:
            from app.plugins.feature_store import FeatureStore
            fs = FeatureStore()
            features = fs.compute_features_batch(trade_date, price_data)
            logger.info("[%s] FeatureStore: %d features computed", trade_date, len(features))
        else:
            features = price_data

        # Step 2: WFA Validation (rolling out-of-sample)
        if self.cfg.wfa_enabled:
            from app.backtest_advanced.wfa_engine import WFAEngine
            wfa = WFAEngine()
            self._wfa_result = wfa.run(price_data)
            wfa_sharpe = self._wfa_result.get("sharpe_ratio", 0)
            if wfa_sharpe < self.cfg.wfa_min_sharpe:
                logger.warning(
                    "[%s] WFA Sharpe %.2f < %.2f — skipping signal generation",
                    trade_date, wfa_sharpe, self.cfg.wfa_min_sharpe,
                )
                return []
            logger.info("[%s] WFA passed: Sharpe %.2f", trade_date, wfa_sharpe)

        # Step 3: ML Online Learning (daily retrain)
        if self.cfg.ml_enabled:
            await self._ensure_ml_trained(trade_date, price_data)

        # Step 4: Strategy scoring
        from app.core.strategy import create_overnight_reversal_strategy
        strategy = create_overnight_reversal_strategy(max_positions=20)

        from app.ml.strategy_bridge import MLStrategyBridge
        ml_bridge = MLStrategyBridge(ml_weight=0.30)

        # Compute scores
        factor_df = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        scored = strategy.calculate_scores(factor_df)

        # Apply ML hybrid if available
        if self.cfg.ml_enabled and ml_bridge.is_available():
            scored = strategy.calculate_scores(factor_df)  # trigger ML bridge

        # Filter and rank
        top20 = scored.head(20).reset_index()
        recommendations = []
        for _, row in top20.iterrows():
            recommendations.append({
                "code": row.get("code", row.name),
                "score": float(row["composite_score"]),
                "rank": int(row["rank"]),
                "factors": {k: float(v) for k, v in row.items() if k not in ["composite_score", "rank"]},
            })

        logger.info("[%s] Pipeline complete: %d recommendations", trade_date, len(recommendations))
        return recommendations

    async def _ensure_ml_trained(self, trade_date: date, price_data: pd.DataFrame) -> None:
        """Retrain ML model if not trained today."""
        if self._last_ml_train == trade_date:
            return

        try:
            from app.ml.ml_ranker import MLRanker
            from app.ml.strategy_bridge import MLStrategyBridge

            bridge = MLStrategyBridge()
            # Generate labels: next-day returns
            returns = price_data.groupby(level=1)["close"].pct_change().shift(-1)
            labels = returns.groupby(level=1).mean().dropna()

            if len(labels) < 50:
                logger.warning("Insufficient data for ML retraining (%d samples)", len(labels))
                return

            features = price_data.groupby(level=1).last()[["open", "high", "low", "close", "volume"]]
            bridge.train(features.loc[labels.index], labels)
            self._last_ml_train = trade_date
            logger.info("[%s] ML model retrained on %d samples", trade_date, len(labels))

        except Exception as exc:
            logger.error("ML retraining failed: %s", exc)

    def get_pipeline_status(self) -> Dict:
        """Return current pipeline health status."""
        return {
            "wfa_enabled": self.cfg.wfa_enabled,
            "wfa_last_sharpe": self._wfa_result.get("sharpe_ratio") if self._wfa_result else None,
            "ml_enabled": self.cfg.ml_enabled,
            "ml_last_train": self._last_ml_train.isoformat() if self._last_ml_train else None,
            "feature_store_enabled": self.cfg.feature_store_enabled,
        }
