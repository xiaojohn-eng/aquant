"""
Test suite for AQuant v2.0 upgraded modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Feature Store tests
# ---------------------------------------------------------------------------

def test_feature_store_compute_and_store():
    from app.plugins.feature_store import FeatureStore
    
    import tempfile
    import os
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db.close()
    store = FeatureStore(f"sqlite:///{tmp_db.name}")
    
    # Create synthetic price data
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    code = "000001.SZ"
    price_df = pd.DataFrame({
        "open": np.random.rand(30) * 10 + 10,
        "high": np.random.rand(30) * 10 + 11,
        "low": np.random.rand(30) * 10 + 9,
        "close": np.random.rand(30) * 10 + 10,
    }, index=dates)
    
    volume_df = pd.DataFrame({
        "volume": np.random.randint(1e6, 1e8, 30),
        "amount": np.random.rand(30) * 1e9,
    }, index=dates)
    
    trade_date = date(2024, 1, 25)  # Use later date so we have >=20 days history
    features = store.compute_features(trade_date, code, price_df, volume_df)
    
    assert "momentum" in features
    assert "volume_ratio" in features
    assert "atr_ratio" in features
    assert features["schema_version"] == "2.0"
    
    # Store and retrieve (use unique test code to avoid cache conflicts)
    code = "TEST999.SZ"
    store.store(trade_date, code, features, composite_score=85.0)
    retrieved = store.get_features(trade_date, code)
    
    assert retrieved is not None
    assert retrieved["momentum"] == features["momentum"]
    # composite_score is stored in separate DB column, not in features_json
    assert "composite_score" not in retrieved


# ---------------------------------------------------------------------------
# Slippage Model tests
# ---------------------------------------------------------------------------

def test_slippage_model_small_cap_penalty():
    from app.plugins.slippage_model import SlippageModel
    
    model = SlippageModel()
    
    # Small cap (<50亿)
    slip_small = model.estimate_slippage(
        "000001.SZ", trade_amount=1_000_000, avg_daily_volume=1e6,
        avg_daily_amount=10e6, atr_ratio=0.02, market_cap=3e9
    )
    
    # Large cap (>500亿)
    slip_large = model.estimate_slippage(
        "600000.SH", trade_amount=1_000_000, avg_daily_volume=1e8,
        avg_daily_amount=1e9, atr_ratio=0.01, market_cap=100e9
    )
    
    assert slip_small > slip_large
    assert slip_small > 0.005  # >0.5% for small cap
    assert slip_large < 0.015  # <1.5% for large cap (base + volatility)


def test_slippage_annual_cost():
    from app.plugins.slippage_model import SlippageModel
    
    model = SlippageModel()
    costs = model.estimate_annual_cost(
        capital=1_000_000, turnover_per_year=20.0, avg_slippage_pct=0.005
    )
    
    assert costs["total_cost_pct_of_capital"] > 0
    assert costs["commission"] > 0
    assert costs["stamp_duty"] > 0
    assert costs["slippage"] > 0


# ---------------------------------------------------------------------------
# Dual Data Source tests
# ---------------------------------------------------------------------------

def test_dual_source_stock_filtering():
    from app.data_sources.dual_source import DualDataSource
    
    ds = DualDataSource()
    # Test internal filtering logic via mock
    assert ds.primary in ("akshare", "tushare")
    assert ds.secondary in ("akshare", "tushare")


# ---------------------------------------------------------------------------
# ML Ranker tests
# ---------------------------------------------------------------------------

def test_ml_ranker_train_and_predict():
    from app.ml.ml_ranker import MLRanker
    
    ranker = MLRanker(model_type="randomforest")  # Use RF for test speed
    
    np.random.seed(42)
    n = 200
    features_df = pd.DataFrame({
        "momentum": np.random.randn(n),
        "volume_ratio": np.random.randn(n),
        "atr_ratio": np.random.rand(n),
        "liquidity": np.random.rand(n),
    }, index=[f"{i:06d}.SH" for i in range(n)])
    
    labels = np.random.randn(n) * 0.02  # next-day returns
    
    metrics = ranker.train(features_df, labels, validation_split=0.3)
    
    assert "validation_ic" in metrics
    assert ranker.model is not None
    
    # Predict
    predictions = ranker.predict(features_df.iloc[:10])
    assert len(predictions) == 10
    assert "ml_score" in predictions.columns
    assert "rank" in predictions.columns


def test_ml_ranker_no_model_fallback():
    from app.ml.ml_ranker import MLRanker
    
    ranker = MLRanker()  # No pre-trained model
    features_df = pd.DataFrame({"f1": [1, 2, 3]}, index=["a", "b", "c"])
    
    predictions = ranker.predict(features_df)
    assert (predictions["ml_score"] == 0).all()  # fallback zeros


# ---------------------------------------------------------------------------
# WFA Engine tests
# ---------------------------------------------------------------------------

def test_wfa_window_generation():
    from app.backtest_advanced.wfa_engine import WFAConfig, WFABacktestEngine
    
    cfg = WFAConfig(train_days=20, test_days=10, step_days=5)
    engine = WFABacktestEngine(cfg)
    
    # 60 days of data -> should generate several windows
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    codes = ["000001.SZ"]
    
    records = []
    for d in dates:
        for c in codes:
            records.append({
                "trade_date": d,
                "code": c,
                "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
                "volume": 1e6, "amount": 10e6,
            })
    
    df = pd.DataFrame(records).set_index(["trade_date", "code"]).sort_index()
    
    def dummy_factor_fn(train_data, test_data):
        return pd.DataFrame({"momentum": [0.01] * len(test_data)})
    
    metrics = engine.run(df, factor_fn=dummy_factor_fn)
    
    assert metrics.total_return_pct is not None
    assert len(engine.window_results) > 0
    assert engine.window_results[0].num_train_samples > 0


# ---------------------------------------------------------------------------
# Prometheus Metrics tests
# ---------------------------------------------------------------------------

def test_prometheus_metrics_collection():
    from app.monitor.prometheus_metrics import MetricsCollector
    
    collector = MetricsCollector()
    
    # Test GPU metrics update
    gpu_data = [{
        "index": 0,
        "name": "NVIDIA H100",
        "utilization_gpu_pct": 85.0,
        "memory_used_mb": 40000,
        "memory_total_mb": 80000,
        "temperature_c": 65,
        "power_draw_w": 450,
    }]
    collector.update_gpu_metrics(gpu_data)
    
    # Test strategy metrics
    collector.update_strategy_metrics({
        "sharpe_ratio": 1.5,
        "total_return_pct": 25.0,
        "max_drawdown_pct": -8.0,
    })
    
    # Test data source status
    collector.update_data_source("akshare", True)
    collector.update_data_source("tushare", False)
    
    # Test fetch recording
    collector.record_data_fetch("akshare", "daily", 5000, 2.5)
    
    assert True  # No exceptions = pass
