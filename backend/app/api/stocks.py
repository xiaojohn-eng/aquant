"""Stock API routes."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    FactorData,
    KlinePoint,
    StockInfo,
    StockRecommendation,
    TradingDayCheck,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stocks", tags=["stocks"])

# In-memory caches (replaced by Redis in production)
_stock_universe_cache: List[StockInfo] = []
_recommendation_cache: dict[str, List[StockRecommendation]] = {}
_kline_cache: dict[str, List[KlinePoint]] = {}
_factor_cache: dict[str, FactorData] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_a_share_trading_day(d: date) -> bool:
    """Check if a date is an A-share trading day (exclude weekends + fixed holidays)."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # Simplified fixed holidays for 2024-2025
    fixed_holidays = {
        date(2024, 1, 1), date(2024, 2, 9), date(2024, 2, 12), date(2024, 2, 13),
        date(2024, 2, 14), date(2024, 2, 15), date(2024, 4, 4), date(2024, 4, 5),
        date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3), date(2024, 6, 10),
        date(2024, 9, 16), date(2024, 9, 17), date(2024, 10, 1), date(2024, 10, 2),
        date(2024, 10, 3), date(2024, 10, 4), date(2024, 10, 7),
        date(2025, 1, 1), date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30),
        date(2025, 1, 31), date(2025, 2, 3), date(2025, 2, 4), date(2025, 4, 4),
        date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 3), date(2025, 5, 31),
        date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3), date(2025, 10, 6),
        date(2025, 10, 7), date(2025, 10, 8),
    }
    return d not in fixed_holidays


async def _get_core_recommender():
    """Lazy import to avoid circular dependency at module load time."""
    try:
        from app.core.recommender import StockRecommender
        return StockRecommender()
    except ImportError:
        return None


async def _get_core_data_fetcher():
    """Lazy import of DataFetcher."""
    try:
        from app.core.data_fetcher import DataFetcher
        return DataFetcher()
    except ImportError:
        return None


async def _get_core_strategy():
    """Lazy import of StrategyEngine."""
    try:
        from app.core.strategy import StrategyEngine
        return StrategyEngine()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/universe", response_model=List[StockInfo])
async def get_stock_universe(
    refresh: bool = Query(False, description="Force refresh from core engine"),
) -> List[StockInfo]:
    """Get the full A-share stock universe (ST and 北交所 filtered)."""
    global _stock_universe_cache

    if _stock_universe_cache and not refresh:
        return _stock_universe_cache

    fetcher = await _get_core_data_fetcher()
    if fetcher and hasattr(fetcher, "get_stock_universe"):
        try:
            raw = fetcher.get_stock_universe()
            _stock_universe_cache = [
                StockInfo(
                    code=item.get("code"),
                    name=item.get("name"),
                    industry=item.get("industry"),
                    market_cap=item.get("market_cap"),
                    pe_ttm=item.get("pe_ttm"),
                    pb=item.get("pb"),
                    listing_date=item.get("listing_date"),
                    is_st=item.get("is_st", False),
                    exchange=item.get("exchange", ""),
                )
                for item in raw
            ]
            return _stock_universe_cache
        except Exception as exc:
            logger.warning("Core fetcher failed: %s", exc)

    # Fallback demo data
    _stock_universe_cache = [
        StockInfo(code="000001.SZ", name="平安银行", industry="银行", market_cap=2500.0, exchange="SZ"),
        StockInfo(code="000002.SZ", name="万科A", industry="房地产", market_cap=1800.0, exchange="SZ"),
        StockInfo(code="000858.SZ", name="五粮液", industry="白酒", market_cap=5200.0, exchange="SZ"),
        StockInfo(code="002594.SZ", name="比亚迪", industry="汽车", market_cap=6800.0, exchange="SZ"),
        StockInfo(code="300750.SZ", name="宁德时代", industry="新能源", market_cap=8200.0, exchange="SZ"),
        StockInfo(code="600519.SH", name="贵州茅台", industry="白酒", market_cap=21000.0, exchange="SH"),
        StockInfo(code="600036.SH", name="招商银行", industry="银行", market_cap=8500.0, exchange="SH"),
        StockInfo(code="601318.SH", name="中国平安", industry="保险", market_cap=7200.0, exchange="SH"),
        StockInfo(code="601888.SH", name="中国中免", industry="旅游零售", market_cap=2800.0, exchange="SH"),
        StockInfo(code="603259.SH", name="药明康德", industry="医药", market_cap=3200.0, exchange="SH"),
    ]
    return _stock_universe_cache


@router.get("/recommendations", response_model=List[StockRecommendation])
async def get_today_recommendations(
    top_n: int = Query(20, ge=1, le=100, description="Number of top picks to return"),
    refresh: bool = Query(False, description="Force recompute via core engine"),
) -> List[StockRecommendation]:
    """Get today's top stock recommendations (computed at ~10:00)."""
    global _recommendation_cache

    today = str(date.today())
    if today in _recommendation_cache and not refresh:
        return _recommendation_cache[today][:top_n]

    recommender = await _get_core_recommender()
    if recommender and hasattr(recommender, "get_recommendations"):
        try:
            raw = recommender.get_recommendations(top_n=top_n)
            recs = [
                StockRecommendation(
                    code=item.get("code"),
                    name=item.get("name"),
                    score=item.get("score", 0.0),
                    rank=item.get("rank", idx + 1),
                    reasons=item.get("reasons", []),
                    expected_return=item.get("expected_return"),
                    risk_level=item.get("risk_level"),
                    model_confidence=item.get("model_confidence"),
                    sector_alignment=item.get("sector_alignment"),
                )
                for idx, item in enumerate(raw)
            ]
            _recommendation_cache[today] = recs
            return recs[:top_n]
        except Exception as exc:
            logger.warning("Core recommender failed: %s", exc)

    # Fallback: generate from universe with fake scores
    universe = await get_stock_universe(refresh=False)
    import random
    random.seed(today)
    scored = sorted(
        [
            {
                **u.model_dump(),
                "score": round(random.uniform(55, 98), 2),
                "expected_return": round(random.uniform(1.5, 8.5), 2),
                "risk_level": random.choice(["low", "medium", "high"]),
                "reasons": ["动量因子占优", "量价配合良好", "资金持续流入"],
            }
            for u in universe
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    recs = [
        StockRecommendation(
            code=s["code"],
            name=s["name"],
            score=s["score"],
            rank=idx + 1,
            reasons=s["reasons"],
            expected_return=s["expected_return"],
            risk_level=s["risk_level"],
        )
        for idx, s in enumerate(scored[:top_n])
    ]
    _recommendation_cache[today] = recs
    return recs


@router.get("/recommendations/{target_date}", response_model=List[StockRecommendation])
async def get_historical_recommendations(
    target_date: date,
    top_n: int = Query(20, ge=1, le=100),
) -> List[StockRecommendation]:
    """Retrieve recommendations for a historical trading date."""
    global _recommendation_cache

    if not _is_a_share_trading_day(target_date):
        raise HTTPException(status_code=400, detail=f"{target_date} is not an A-share trading day")

    key = str(target_date)
    if key in _recommendation_cache:
        return _recommendation_cache[key][:top_n]

    # If the core backtest engine exists, pull from there
    try:
        from app.core.backtest import BacktestEngine
        bt = BacktestEngine()
        if hasattr(bt, "get_recommendations_for_date"):
            raw = bt.get_recommendations_for_date(str(target_date), top_n=top_n)
            recs = [
                StockRecommendation(
                    code=item.get("code"),
                    name=item.get("name"),
                    score=item.get("score", 0.0),
                    rank=item.get("rank", idx + 1),
                    reasons=item.get("reasons", []),
                    expected_return=item.get("expected_return"),
                )
                for idx, item in enumerate(raw)
            ]
            _recommendation_cache[key] = recs
            return recs[:top_n]
    except Exception as exc:
        logger.warning("Backtest engine unavailable for historical recs: %s", exc)

    raise HTTPException(status_code=404, detail=f"No recommendations cached for {target_date}")


@router.get("/factors/{code}", response_model=FactorData)
async def get_stock_factors(
    code: str,
    target_date: Optional[date] = Query(None, description="Optional historical date"),
) -> FactorData:
    """Get detailed factor breakdown for a single stock."""
    global _factor_cache

    cache_key = f"{code}_{target_date or date.today()}"
    if cache_key in _factor_cache:
        return _factor_cache[cache_key]

    strategy = await _get_core_strategy()
    if strategy and hasattr(strategy, "get_factors"):
        try:
            raw = strategy.get_factors(code, date=str(target_date) if target_date else None)
            fd = FactorData(
                code=code,
                trade_date=target_date or date.today(),
                momentum_5d=raw.get("momentum_5d"),
                momentum_20d=raw.get("momentum_20d"),
                momentum_60d=raw.get("momentum_60d"),
                volume_ratio=raw.get("volume_ratio"),
                turnover_rate=raw.get("turnover_rate"),
                money_flow=raw.get("money_flow"),
                volatility_20d=raw.get("volatility_20d"),
                atr_14=raw.get("atr_14"),
                roe_ttm=raw.get("roe_ttm"),
                profit_growth=raw.get("profit_growth"),
                revenue_growth=raw.get("revenue_growth"),
                composite_score=raw.get("composite_score"),
                rank=raw.get("rank"),
            )
            _factor_cache[cache_key] = fd
            return fd
        except Exception as exc:
            logger.warning("Core strategy factor fetch failed: %s", exc)

    # Fallback synthetic factors
    import random, math
    random.seed(code)
    fd = FactorData(
        code=code,
        trade_date=target_date or date.today(),
        momentum_5d=round(random.uniform(-5, 8), 2),
        momentum_20d=round(random.uniform(-10, 15), 2),
        momentum_60d=round(random.uniform(-20, 30), 2),
        volume_ratio=round(random.uniform(0.5, 3.5), 2),
        turnover_rate=round(random.uniform(1, 12), 2),
        money_flow=round(random.uniform(-5000, 15000), 2),
        volatility_20d=round(random.uniform(15, 45), 2),
        atr_14=round(random.uniform(1, 8), 2),
        roe_ttm=round(random.uniform(5, 25), 2),
        profit_growth=round(random.uniform(-15, 40), 2),
        revenue_growth=round(random.uniform(-5, 35), 2),
        composite_score=round(random.uniform(40, 95), 2),
        rank=random.randint(1, 5000),
    )
    _factor_cache[cache_key] = fd
    return fd


@router.get("/history/{code}", response_model=List[KlinePoint])
async def get_stock_history(
    code: str,
    start_date: Optional[date] = Query(None, description="Start date (inclusive)"),
    end_date: Optional[date] = Query(None, description="End date (inclusive)"),
    period: str = Query("daily", pattern="^(daily|weekly)$", description="K-line frequency"),
) -> List[KlinePoint]:
    """Get historical OHLCV K-line data for a stock."""
    global _kline_cache

    cache_key = f"{code}_{period}_{start_date}_{end_date}"
    if cache_key in _kline_cache:
        return _kline_cache[cache_key]

    fetcher = await _get_core_data_fetcher()
    if fetcher and hasattr(fetcher, "get_kline"):
        try:
            raw = fetcher.get_kline(
                code=code,
                start=str(start_date) if start_date else None,
                end=str(end_date) if end_date else None,
                period=period,
            )
            kline = [
                KlinePoint(
                    trade_date=item.get("date"),
                    open=item.get("open"),
                    high=item.get("high"),
                    low=item.get("low"),
                    close=item.get("close"),
                    volume=item.get("volume"),
                    turnover=item.get("turnover"),
                )
                for item in raw
            ]
            _kline_cache[cache_key] = kline
            return kline
        except Exception as exc:
            logger.warning("Core fetcher kline failed: %s", exc)

    # Fallback synthetic data
    import random
    random.seed(code)
    end = end_date or date.today()
    start = start_date or (end - timedelta(days=180))
    result: List[KlinePoint] = []
    price = round(random.uniform(10, 100), 2)
    d = start
    while d <= end:
        if _is_a_share_trading_day(d):
            change = random.uniform(-0.03, 0.03)
            o = round(price * (1 + random.uniform(-0.01, 0.01)), 2)
            c = round(price * (1 + change), 2)
            h = round(max(o, c) * (1 + random.uniform(0, 0.02)), 2)
            l = round(min(o, c) * (1 - random.uniform(0, 0.02)), 2)
            v = int(random.uniform(1_000_000, 50_000_000))
            result.append(KlinePoint(trade_date=d, open=o, high=h, low=l, close=c, volume=v))
            price = c
        d += timedelta(days=1)

    _kline_cache[cache_key] = result
    return result


@router.get("/trading-day/{check_date}", response_model=TradingDayCheck)
async def check_trading_day(check_date: date) -> TradingDayCheck:
    """Check whether a specific date is an A-share trading day."""
    ok = _is_a_share_trading_day(check_date)
    reason = None
    if check_date.weekday() >= 5:
        reason = "weekend"
    elif not ok:
        reason = "public holiday"
    return TradingDayCheck(trade_date=check_date, is_trading_day=ok, reason=reason)
