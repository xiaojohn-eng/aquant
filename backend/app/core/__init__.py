"""
core/__init__.py
================
AQuant backend core package.

Exposes the five engine modules for strategy research, backtesting,
and live trading signal generation.
"""
from .data_fetcher import (
    get_stock_universe,
    get_minute_data,
    get_daily_data,
    get_bid_ask_data,
    batch_get_minute_data,
    batch_get_daily_data,
    get_trading_calendar,
    clear_cache,
)

from .gpu_compute import (
    compute_momentum_factor,
    compute_volume_ratio,
    compute_atr_ratio,
    compute_liquidity_score,
    compute_sector_momentum,
    batch_compute_all_factors,
    compute_composite_score,
    gpu_health_check,
    GPU_BACKEND,
)

from .strategy import (
    StrategyEngine,
    CostModel,
    Signal,
    Position,
    create_momentum_strategy,
    create_conservative_strategy,
)

from .recommender import (
    StockRecommender,
    StockRecommendation,
    is_trading_time,
    next_trading_day,
    previous_trading_day,
    get_trading_hours,
    trading_seconds_remaining,
    is_entry_window,
    format_recommendations,
)

from .backtest import (
    BacktestEngine,
    BacktestConfig,
    TradeRecord,
    PerformanceMetrics,
)

__all__ = [
    # data_fetcher
    "get_stock_universe",
    "get_minute_data",
    "get_daily_data",
    "get_bid_ask_data",
    "batch_get_minute_data",
    "batch_get_daily_data",
    "get_trading_calendar",
    "clear_cache",
    # gpu_compute
    "compute_momentum_factor",
    "compute_volume_ratio",
    "compute_atr_ratio",
    "compute_liquidity_score",
    "compute_sector_momentum",
    "batch_compute_all_factors",
    "compute_composite_score",
    "gpu_health_check",
    "GPU_BACKEND",
    # strategy
    "StrategyEngine",
    "CostModel",
    "Signal",
    "Position",
    "create_momentum_strategy",
    "create_conservative_strategy",
    # recommender
    "StockRecommender",
    "StockRecommendation",
    "is_trading_time",
    "next_trading_day",
    "previous_trading_day",
    "get_trading_hours",
    "trading_seconds_remaining",
    "is_entry_window",
    "format_recommendations",
    # backtest
    "BacktestEngine",
    "BacktestConfig",
    "TradeRecord",
    "PerformanceMetrics",
]
