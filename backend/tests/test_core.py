"""
Test suite for AQuant core engine.

Run with: pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# GPU Compute tests
# ---------------------------------------------------------------------------

def test_compute_momentum_factor():
    from app.core.gpu_compute import compute_momentum_factor
    np.random.seed(42)
    # prices shape: (n_stocks, n_time)
    prices = np.cumsum(np.random.randn(50, 100) * 0.02 + 0.001, axis=1) + 10.0
    volumes = np.random.randint(1e6, 1e8, size=(50, 100))
    result = compute_momentum_factor(prices, volumes)
    assert result.shape == (50,)
    assert not np.isnan(result).all()


def test_compute_composite_score():
    from app.core.gpu_compute import compute_composite_score
    np.random.seed(42)
    n = 100
    factors = {
        "momentum": np.random.randn(n),
        "volume_ratio": np.random.randn(n),
        "atr_ratio": np.random.randn(n),
        "liquidity": np.random.randn(n),
        "market_cap": np.random.randn(n),
    }
    weights = {"momentum": 0.25, "volume_ratio": 0.20, "atr_ratio": -0.10, "liquidity": 0.15, "market_cap": 0.10}
    scores = compute_composite_score(factors, weights)
    assert scores.shape == (n,)
    assert not np.isnan(scores).all()


def test_batch_compute_all_factors():
    from app.core.gpu_compute import batch_compute_all_factors
    np.random.seed(42)
    n = 20
    n_stocks = 5
    data = {
        "prices": np.random.rand(n_stocks, n) * 10 + 10,
        "highs": np.random.rand(n_stocks, n) * 10 + 11,
        "lows": np.random.rand(n_stocks, n) * 10 + 9,
        "closes": np.random.rand(n_stocks, n) * 10 + 10,
        "volumes": np.random.randint(1e6, 1e8, size=(n_stocks, n)),
        "amounts": np.random.rand(n_stocks, n) * 1e9,
        "volume_ma20": np.random.randint(1e6, 1e8, size=(n_stocks, n)),
    }
    result = batch_compute_all_factors(data)
    assert "momentum" in result
    assert "volume_ratio" in result
    assert "atr_ratio" in result
    assert "liquidity" in result


# ---------------------------------------------------------------------------
# Strategy engine tests
# ---------------------------------------------------------------------------

def test_strategy_calculate_scores():
    from app.core.strategy import StrategyEngine
    engine = StrategyEngine()
    np.random.seed(42)
    df = pd.DataFrame({
        "momentum": np.random.randn(30),
        "volume_ratio": np.random.randn(30),
        "atr_ratio": np.random.rand(30),
        "liquidity": np.random.rand(30),
        "market_cap": np.random.lognormal(10, 1, 30),
    }, index=[f"{i:06d}.SH" for i in range(30)])
    scored = engine.calculate_scores(df)
    assert "composite_score" in scored.columns
    assert "rank" in scored.columns
    assert len(scored) == 30
    assert scored["composite_score"].notna().sum() > 0


def test_filter_stocks_dataframe():
    from app.core.strategy import StrategyEngine
    engine = StrategyEngine()
    df = pd.DataFrame({
        "code": ["000001.SZ", "300001.SZ", "835001.BJ", "688001.SH", "000002.SZ"],
        "name": ["平安银行", "ST特力", "北交所股", "中芯国际", "万科A"],
        "composite_score": [80, 70, 60, 90, 75],
    }).set_index("code")
    # filter_stocks in strategy.py filters by halt/limit-up/quality gates,
    # NOT by ST/BJ (that's done in data_fetcher._filter_st_and_bj).
    filtered = engine.filter_stocks(df)
    codes = filtered.index.tolist()
    assert "688001.SH" in codes
    assert "000001.SZ" in codes
    assert "000002.SZ" in codes
    # All 5 should pass strategy-level filters (no halt/limit flags provided)
    assert len(codes) == 5


# ---------------------------------------------------------------------------
# Recommender tests
# ---------------------------------------------------------------------------

def test_trading_time_utils():
    from app.core.recommender import is_trading_time, next_trading_day, is_entry_window
    assert is_trading_time(datetime(2024, 6, 3, 10, 0)) is True
    assert is_trading_time(datetime(2024, 6, 3, 15, 30)) is False
    # is_entry_window has ±5 minute tolerance around 10:00
    assert is_entry_window(datetime(2024, 6, 3, 9, 57)) is True
    assert is_entry_window(datetime(2024, 6, 3, 10, 0)) is True
    assert is_entry_window(datetime(2024, 6, 3, 10, 3)) is True
    assert is_entry_window(datetime(2024, 6, 3, 9, 45)) is False  # before tolerance window
    assert is_entry_window(datetime(2024, 6, 3, 10, 30)) is False  # after tolerance window

    d = date(2024, 6, 3)  # Monday
    nxt = next_trading_day(d)
    assert nxt == date(2024, 6, 4)


def test_generate_reasons():
    from app.core.recommender import StockRecommender
    rec = StockRecommender()
    factors = {
        "momentum": 0.05,
        "volume_ratio": 2.5,
        "atr_ratio": 0.02,
        "liquidity": 0.8,
        "composite_score": 85.0,
    }
    reasons = rec.generate_reasons("000001.SZ", factors)
    assert len(reasons) >= 3
    assert any("早盘" in r or "动能" in r for r in reasons)


# ---------------------------------------------------------------------------
# Backtest engine tests
# ---------------------------------------------------------------------------

def test_backtest_with_synthetic_data():
    from app.core.backtest import BacktestConfig, BacktestEngine

    np.random.seed(42)
    n_days, n_stocks = 30, 20
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    codes = [f"{i:06d}.SH" for i in range(n_stocks)]

    records = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            open_p = 10.0 + np.random.randn() * 0.5
            close_p = open_p + np.random.randn() * 0.3
            high_p = max(open_p, close_p) + abs(np.random.randn()) * 0.2
            low_p = min(open_p, close_p) - abs(np.random.randn()) * 0.2
            records.append({
                "trade_date": pd.Timestamp(d),
                "code": c,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": int(1e6 * (1 + np.random.rand())),
                "amount": 1e7 * (1 + np.random.rand()),
            })

    df = pd.DataFrame(records).set_index(["trade_date", "code"]).sort_index()

    cfg = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-30",
        initial_capital=1_000_000,
        max_positions=5,
    )
    engine = BacktestEngine(cfg)
    metrics = engine.run(price_data=df)

    assert metrics.num_trades > 0
    assert metrics.total_return_pct is not None
    assert metrics.sharpe_ratio is not None
    assert metrics.max_drawdown_pct <= 0
    assert len(engine.daily_nav) > 0


def test_empty_data_backtest():
    from app.core.backtest import BacktestConfig, BacktestEngine
    cfg = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-05",
        initial_capital=1_000_000,
    )
    engine = BacktestEngine(cfg)
    metrics = engine.run(price_data=pd.DataFrame())
    assert metrics.num_trades == 0


# ---------------------------------------------------------------------------
# Data fetcher tests
# ---------------------------------------------------------------------------

def test_stock_universe_filtering():
    """Test that ST and BJ stocks are correctly filtered."""
    from app.core.data_fetcher import _filter_st_and_bj
    raw = [
        {"code": "000001.SZ", "name": "平安银行", "is_st": False},
        {"code": "300001.SZ", "name": "ST特力", "is_st": True},
        {"code": "835001.BJ", "name": "北交所股", "is_st": False},
        {"code": "688001.SH", "name": "中芯国际", "is_st": False},
    ]
    filtered = _filter_st_and_bj(raw)
    codes = [s["code"] for s in filtered]
    assert "300001.SZ" not in codes
    assert "835001.BJ" not in codes
    assert "000001.SZ" in codes
    assert "688001.SH" in codes


# ---------------------------------------------------------------------------
# API route tests (using TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import create_app
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Aquant" in response.json()["message"]


def test_monitor_gpu_endpoint(client):
    response = client.get("/api/monitor/gpu")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
