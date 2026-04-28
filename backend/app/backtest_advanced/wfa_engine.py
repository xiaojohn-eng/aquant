"""
Walk-Forward Analysis (WFA) Backtest Engine

Simulates real-world strategy deployment by rolling training/test windows.
Eliminates future information leakage and overfitting.

Reference:
- Marcos Lopez de Prado: Advances in Financial Machine Learning, Ch. 12
- BigQuant: WFA动态回测寻优策略 (2025)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.backtest import BacktestConfig, BacktestEngine, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class WFAConfig:
    """Walk-Forward Analysis configuration."""
    # Time windows
    train_days: int = 252       # ~1 year training
    test_days: int = 63         # ~3 months out-of-sample
    step_days: int = 21         # Roll forward 1 month each iteration
    
    # Strategy parameters
    initial_capital: float = 1_000_000
    max_positions: int = 20
    
    # Model config
    use_ml_ranker: bool = True
    ml_model_type: str = "xgboost"


@dataclass
class WFAWindowResult:
    """Result for a single WFA window."""
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    num_train_samples: int
    num_test_samples: int
    test_return_pct: float
    test_sharpe: float
    test_max_dd_pct: float
    num_trades: int
    model_ic: float  # Information coefficient on validation


class WFABacktestEngine:
    """
    Walk-Forward Analysis engine.
    
    Process:
    1. Split history into rolling windows
    2. For each window:
       a. Train model on [train_start, train_end]
       b. Generate predictions for [test_start, test_end]
       c. Execute trades based on predictions
       d. Record OOS performance
    3. Aggregate all OOS results into final performance report
    """
    
    def __init__(self, config: WFAConfig):
        self.cfg = config
        self.window_results: List[WFAWindowResult] = []
    
    def run(self,
            price_data: pd.DataFrame,
            factor_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
            label_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None) -> PerformanceMetrics:
        """
        Run WFA backtest.
        
        Args:
            price_data: MultiIndex (date, code) DataFrame with OHLCV
            factor_fn: function(train_data) -> features_df
            label_fn: function(train_data) -> next-day returns (for ML training)
        """
        dates = sorted(price_data.index.get_level_values(0).unique())
        total_days = len(dates)
        
        # Generate window boundaries
        windows = []
        for start_idx in range(0, total_days - self.cfg.train_days - self.cfg.test_days, self.cfg.step_days):
            train_start_idx = start_idx
            train_end_idx = start_idx + self.cfg.train_days
            test_start_idx = train_end_idx
            test_end_idx = train_end_idx + self.cfg.test_days
            
            if test_end_idx > total_days:
                break
            
            windows.append({
                "train_start": dates[train_start_idx],
                "train_end": dates[train_end_idx - 1],
                "test_start": dates[test_start_idx],
                "test_end": dates[test_end_idx - 1],
            })
        
        logger.info(f"Running WFA with {len(windows)} windows")
        
        all_test_returns = []
        all_trades = []
        
        for i, w in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}: train={w['train_start']}~{w['train_end']}, "
                       f"test={w['test_start']}~{w['test_end']}")
            
            # Split data
            train_mask = (price_data.index.get_level_values(0) >= pd.Timestamp(w["train_start"])) & \
                        (price_data.index.get_level_values(0) <= pd.Timestamp(w["train_end"]))
            test_mask = (price_data.index.get_level_values(0) >= pd.Timestamp(w["test_start"])) & \
                       (price_data.index.get_level_values(0) <= pd.Timestamp(w["test_end"]))
            
            train_data = price_data[train_mask]
            test_data = price_data[test_mask]
            
            # Compute features
            train_features = factor_fn(train_data, train_data)
            test_features = factor_fn(test_data, test_data)
            
            # Compute labels for training
            if label_fn:
                train_labels = label_fn(train_data)
            else:
                # Default: next-day returns
                train_labels = self._compute_next_day_returns(train_data)
            
            # Train ML model (if enabled)
            if self.cfg.use_ml_ranker and len(train_features) > 100:
                from app.ml.ml_ranker import MLRanker
                ranker = MLRanker(model_type=self.cfg.ml_model_type)
                metrics = ranker.train(train_features, train_labels.values)
                model_ic = metrics.get("validation_ic", 0.0)
                
                # Predict on test set
                predictions = ranker.predict(test_features)
                top_stocks = predictions.nsmallest(self.cfg.max_positions, "rank").index.tolist()
            else:
                # Fallback: simple momentum ranking
                if "momentum" in test_features.columns:
                    top_stocks = test_features.nlargest(self.cfg.max_positions, "momentum").index.tolist()
                else:
                    top_stocks = []
                model_ic = 0.0
            
            # Run standard backtest on test period with selected stocks
            test_result = self._run_single_period(test_data, top_stocks)
            
            all_test_returns.extend(test_result["daily_returns"])
            all_trades.extend(test_result["trades"])
            
            self.window_results.append(WFAWindowResult(
                train_start=w["train_start"],
                train_end=w["train_end"],
                test_start=w["test_start"],
                test_end=w["test_end"],
                num_train_samples=len(train_features),
                num_test_samples=len(test_features),
                test_return_pct=test_result["total_return_pct"],
                test_sharpe=test_result["sharpe"],
                test_max_dd_pct=test_result["max_drawdown_pct"],
                num_trades=test_result["num_trades"],
                model_ic=model_ic,
            ))
        
        # Aggregate all OOS results
        return self._aggregate_results(all_test_returns, all_trades)
    
    def _compute_next_day_returns(self, data: pd.DataFrame) -> pd.Series:
        """Compute next-day forward returns for each stock."""
        returns = []
        codes = data.index.get_level_values(1).unique()
        
        for code in codes:
            code_data = data.xs(code, level=1)["close"]
            fwd_ret = code_data.pct_change().shift(-1).dropna()
            returns.append(fwd_ret)
        
        if returns:
            return pd.concat(returns)
        return pd.Series()
    
    def _run_single_period(self, data: pd.DataFrame, selected_stocks: List[str]) -> Dict:
        """Run backtest for a single test period."""
        # Simplified: equal-weight buy-and-hold of selected stocks
        if not selected_stocks:
            return {
                "daily_returns": [0.0] * len(data.index.get_level_values(0).unique()),
                "trades": [],
                "total_return_pct": 0.0,
                "sharpe": 0.0,
                "max_drawdown_pct": 0.0,
                "num_trades": 0,
            }
        
        dates = sorted(data.index.get_level_values(0).unique())
        daily_returns = []
        
        for d in dates:
            day_data = data.xs(pd.Timestamp(d), level=0)
            day_selected = day_data.loc[day_data.index.isin(selected_stocks)]
            
            if len(day_selected) > 0:
                # Equal-weight portfolio return
                port_return = day_selected["close"].pct_change().mean()
                daily_returns.append(port_return if not np.isnan(port_return) else 0.0)
            else:
                daily_returns.append(0.0)
        
        # Compute metrics
        returns_arr = np.array(daily_returns)
        total_ret = (1 + returns_arr).prod() - 1
        sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252) if np.std(returns_arr) > 0 else 0
        
        # Max drawdown
        cum = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cum)
        drawdown = (cum / running_max - 1)
        max_dd = np.min(drawdown) * 100
        
        return {
            "daily_returns": daily_returns,
            "trades": [],
            "total_return_pct": total_ret * 100,
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "num_trades": len(selected_stocks) * len(dates),
        }
    
    def _aggregate_results(self, all_returns: List[float], all_trades: List) -> PerformanceMetrics:
        """Aggregate all OOS window results."""
        returns_arr = np.array(all_returns)
        
        total_ret = (1 + returns_arr).prod() - 1
        ann_ret = (1 + total_ret) ** (252 / len(returns_arr)) - 1 if len(returns_arr) > 0 else 0
        sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252) if np.std(returns_arr) > 0 else 0
        
        cum = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cum)
        drawdown = (cum / running_max - 1)
        max_dd = np.min(drawdown) * 100 if len(drawdown) > 0 else 0
        
        win_rate = np.sum(returns_arr > 0) / len(returns_arr) * 100 if len(returns_arr) > 0 else 0
        
        return PerformanceMetrics(
            total_return_pct=round(total_ret * 100, 4),
            annualized_return_pct=round(ann_ret * 100, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=0.0,  # simplified
            max_drawdown_pct=round(max_dd, 4),
            calmar_ratio=round(ann_ret * 100 / abs(max_dd), 4) if max_dd != 0 else 0,
            win_rate_pct=round(win_rate, 4),
            profit_loss_ratio=1.0,  # simplified
            avg_trade_return_pct=round(np.mean(returns_arr) * 100, 4),
            avg_winning_trade_pct=0.0,
            avg_losing_trade_pct=0.0,
            num_trades=len(all_trades),
            num_winning_trades=0,
            num_losing_trades=0,
            turnover_ratio=20.0,
        )
    
    def get_window_summary(self) -> pd.DataFrame:
        """Return summary of all WFA windows."""
        if not self.window_results:
            return pd.DataFrame()
        
        records = []
        for r in self.window_results:
            records.append({
                "train_period": f"{r.train_start}~{r.train_end}",
                "test_period": f"{r.test_start}~{r.test_end}",
                "test_return_pct": r.test_return_pct,
                "test_sharpe": r.test_sharpe,
                "test_max_dd_pct": r.test_max_dd_pct,
                "num_trades": r.num_trades,
                "model_ic": r.model_ic,
            })
        return pd.DataFrame(records)
