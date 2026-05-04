"""
Test suite for AQuant core engine (v4.3 — T+1 Adaptive Strategy).

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
    prices = np.cumsum(np.random.randn(50, 100) * 0.02 + 0.001, axis=1) + 10.0
    volumes = np.random.randint(1e6, 1e8, size=(50, 100))
    result = compute_momentum_factor(prices, volumes)
    assert result.shape == (50,)
    assert not np.isnan(result).all()


def test_compute_overnight_return():
    from app.core.gpu_compute import compute_overnight_return
    opens = np.array([10.0, 10.5, 9.8])
    pre_closes = np.array([10.2, 10.2, 10.2])
    result = compute_overnight_return(opens, pre_closes)
    assert result.shape == (3,)
    # Negative overnight return = gap down
    assert result[0] < 0  # 10.0 < 10.2 → gap down
    assert result[1] > 0  # 10.5 > 10.2 → gap up
    assert result[2] < 0  # 9.8 < 10.2 → gap down


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
        "overnight_return": np.random.randn(n) * 0.02,
    }
    weights = {
        "momentum": 0.15,
        "volume_ratio": 0.10,
        "atr_ratio": 0.10,
        "liquidity": 0.15,
        "market_cap": 0.20,
        "overnight_return": 0.20,
    }
    direction = {
        "momentum": -1,
        "volume_ratio": -1,
        "atr_ratio": -1,
        "liquidity": 1,
        "market_cap": 1,
        "overnight_return": -1,
    }
    scores = compute_composite_score(factors, weights, direction)
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
        "opens": np.random.rand(n_stocks, n) * 10 + 10,
        "pre_closes": np.random.rand(n_stocks, n) * 10 + 10,
    }
    result = batch_compute_all_factors(data)
    assert "momentum" in result
    assert "volume_ratio" in result
    assert "atr_ratio" in result
    assert "liquidity" in result
    assert "overnight_return" in result
    assert result["overnight_return"].shape == (n_stocks,)


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
        "overnight_return": np.random.randn(30) * 0.02,
    }, index=[f"{i:06d}.SH" for i in range(30)])
    scored = engine.calculate_scores(df)
    assert "composite_score" in scored.columns
    assert "rank" in scored.columns
    assert len(scored) == 30
    assert scored["composite_score"].notna().sum() > 0


def test_strategy_negative_direction():
    """v4.3: High momentum should result in LOWER score."""
    from app.core.strategy import StrategyEngine
    engine = StrategyEngine()
    df = pd.DataFrame({
        "momentum": [3.0, -3.0],  # First has high momentum (bad)
        "volume_ratio": [0, 0],
        "atr_ratio": [0, 0],
        "liquidity": [0.5, 0.5],
        "market_cap": [0.5, 0.5],
        "overnight_return": [0, 0],
    }, index=["000001.SH", "000002.SH"])
    scored = engine.calculate_scores(df)
    # With negative direction for momentum, high momentum = lower score
    assert scored.loc["000001.SH", "composite_score"] < scored.loc["000002.SH", "composite_score"]


def test_filter_stocks_dataframe():
    from app.core.strategy import StrategyEngine
    engine = StrategyEngine()
    df = pd.DataFrame({
        "code": ["000001.SZ", "300001.SZ", "835001.BJ", "688001.SH", "000002.SZ"],
        "name": ["平安银行", "ST特力", "北交所股", "中芯国际", "万科A"],
        "composite_score": [80, 70, 60, 90, 75],
    }).set_index("code")
    filtered = engine.filter_stocks(df)
    codes = filtered.index.tolist()
    assert "688001.SH" in codes
    assert "000001.SZ" in codes
    assert "000002.SZ" in codes
    assert len(codes) == 5


def test_overnight_reversal_strategy_preset():
    from app.core.strategy import create_overnight_reversal_strategy
    engine = create_overnight_reversal_strategy(max_positions=15)
    assert engine.max_positions == 15
    assert engine.weights["overnight_return"] == 0.25
    assert engine.weights["momentum"] == 0.15
    assert engine.weights["market_cap"] == 0.20


# ---------------------------------------------------------------------------
# Recommender tests
# ---------------------------------------------------------------------------

def test_trading_time_utils():
    from app.core.recommender import is_trading_time, next_trading_day, is_entry_window
    assert is_trading_time(datetime(2024, 6, 3, 10, 0)) is True
    assert is_trading_time(datetime(2024, 6, 3, 15, 30)) is False
    assert is_entry_window(datetime(2024, 6, 3, 9, 57)) is True
    assert is_entry_window(datetime(2024, 6, 3, 10, 0)) is True
    assert is_entry_window(datetime(2024, 6, 3, 10, 3)) is True
    assert is_entry_window(datetime(2024, 6, 3, 9, 45)) is False
    assert is_entry_window(datetime(2024, 6, 3, 10, 30)) is False

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


def test_backtest_market_timing():
    """v4.3: Test that market timing reduces trades in down markets."""
    from app.core.backtest import BacktestConfig, BacktestEngine

    np.random.seed(42)
    n_days, n_stocks = 40, 10
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    codes = [f"{i:06d}.SH" for i in range(n_stocks)]

    # Create a declining market
    records = []
    for i, d in enumerate(dates):
        trend = -0.005 * i  # declining trend
        for j, c in enumerate(codes):
            base = 10.0 * (1 + trend)
            open_p = base + np.random.randn() * 0.2
            close_p = open_p * (1 + np.random.randn() * 0.01 - 0.002)
            high_p = max(open_p, close_p) + abs(np.random.randn()) * 0.1
            low_p = min(open_p, close_p) - abs(np.random.randn()) * 0.1
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

    # With timing enabled
    cfg_timed = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-02-09",
        initial_capital=1_000_000,
        max_positions=5,
        market_timing_enabled=True,
        market_timing_lookback=10,
    )
    engine_timed = BacktestEngine(cfg_timed)
    metrics_timed = engine_timed.run(price_data=df)

    # With timing disabled
    cfg_notimed = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-02-09",
        initial_capital=1_000_000,
        max_positions=5,
        market_timing_enabled=False,
    )
    engine_notimed = BacktestEngine(cfg_notimed)
    metrics_notimed = engine_notimed.run(price_data=df)

    # Market timing should reduce number of trades in declining market
    assert metrics_timed.num_trades <= metrics_notimed.num_trades


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
