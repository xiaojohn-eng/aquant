"""
data_fetcher.py
===============
A-Share quantitative trading system — Data Acquisition & Cache Layer.

Responsibilities
----------------
* Retrieve **real-time / historical** A-share data via AkShare.
* Filter out ST / *ST / delisting-risk stocks and Beijing Stock Exchange (BSE).
* Provide **minute-level** (1/5/15/30/60 min), **daily** and **call-auction**
  data interfaces.
* Implement a **local cache** (SQLite + Parquet) to avoid repeated network
  requests and to accelerate GPU pipelines.

Architecture
------------
┌──────────────┐      ┌────────────────┐      ┌─────────────────┐
│   AkShare    │─────▶│  Local Cache   │─────▶│  GPU Compute    │
│   (HTTP)     │      │ (SQLite+Parquet)│      │ (CuPy/Numba)    │
└──────────────┘      └────────────────┘      └─────────────────┘

Key design choices
------------------
1. **ST / BSE filtering**  →  Regex on symbol name + code prefix (8/4/BJ).
2. **Cache strategy**      →  SQLite for meta info, Parquet for bulky OHLCV.
3. **Retry & back-off**    →  3 retries with exponential back-off for network.
4. **Type safety**         →  Full type annotations + runtime validation.

Author   : AQuant Core Team
Platform : NVIDIA DGX Spark (Grace Hopper)
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# AkShare is optional at import time so the module can be imported in
# environments without it (e.g. pure-GPU test nodes).
# ---------------------------------------------------------------------------
try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None  # type: ignore[assignment]

logger = logging.getLogger("aquant.data_fetcher")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = Path(os.getenv("AQUANT_CACHE_DIR", "/mnt/agents/output/aquant/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_DB = CACHE_DIR / "meta.db"
PARQUET_DIR = CACHE_DIR / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# A-share market prefixes we **exclude** (Beijing / 退市)
EXCLUDED_PREFIXES: Tuple[str, ...] = ("8", "4", "9", "BJ")
ST_REGEX = re.compile(r"(?:^|[^a-zA-Z])(ST|\*ST|退)(?:[^a-zA-Z]|$)")

# Minute frequencies supported by AkShare
MINUTE_FREQS: Tuple[str, ...] = ("1", "5", "15", "30", "60")

# Network retry configuration
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class DataFetcherError(Exception):
    """Base exception for data-fetcher failures."""

class NetworkError(DataFetcherError):
    """Raised when AkShare / network request fails after retries."""

class CacheError(DataFetcherError):
    """Raised when local cache read/write fails."""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StockMeta:
    """Lightweight metadata for a single A-share stock."""
    code: str
    name: str
    exchange: str  # "sh" | "sz" | "bj"
    industry: str
    total_mv: float  # 亿元
    list_date: Optional[str]

@dataclass(frozen=True)
class BidAskSnapshot:
    """Call-auction (09:15-09:25) bid/ask snapshot."""
    code: str
    timestamp: datetime
    bid_price_1: float
    ask_price_1: float
    bid_volume_1: int
    ask_volume_1: int
    pre_close: float
    auction_volume: int  # 集合竞价成交量
    auction_amount: float  # 集合竞价成交金额

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
def _init_meta_db() -> sqlite3.Connection:
    """Ensure SQLite meta DB exists with the required schema."""
    conn = sqlite3.connect(str(SQLITE_DB), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_meta (
            code        TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            exchange    TEXT NOT NULL,
            industry    TEXT,
            total_mv    REAL,
            list_date   TEXT,
            updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_registry (
            data_type   TEXT NOT NULL,
            stock_code  TEXT,
            freq        TEXT,
            start_date  TEXT,
            end_date    TEXT,
            parquet_path TEXT NOT NULL,
            updated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (data_type, stock_code, freq, start_date, end_date)
        )
        """
    )
    conn.commit()
    return conn

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _is_valid_stock(code: str, name: str) -> bool:
    """
    Exclude ST/*ST/退市 and Beijing Stock Exchange.

    Parameters
    ----------
    code: str
        6-digit stock code (e.g. "000001").
    name: str
        Stock name (e.g. "平安银行").

    Returns
    -------
    bool
        ``True`` if the stock is eligible for trading.
    """
    # Exclude by prefix
    if any(code.startswith(p) for p in EXCLUDED_PREFIXES):
        return False
    if name.upper().startswith("BJ"):
        return False
    # Exclude ST / *ST / 退
    if ST_REGEX.search(name):
        return False
    return True


def get_stock_universe(
    force_refresh: bool = False,
    min_listing_days: int = 60,
) -> pd.DataFrame:
    """
    Retrieve the **full A-share tradable universe** (excluding ST/BSE).

    The data source is ``ak.stock_zh_a_spot_em`` which returns real-time
    snapshot of ~5 300 stocks.  Results are cached in SQLite for 1 day.

    Parameters
    ----------
    force_refresh: bool, default False
        Ignore cache and re-fetch from AkShare.
    min_listing_days: int, default 60
        Filter out stocks listed fewer than *N* days ago.

    Returns
    -------
    pd.DataFrame
        Columns: ``code, name, exchange, industry, total_mv, list_date``
    """
    cache_key = f"stock_universe_{date.today().isoformat()}"
    cache_file = PARQUET_DIR / f"{cache_key}.parquet"

    if not force_refresh and cache_file.exists():
        logger.info("Loading stock universe from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    if ak is None:
        raise NetworkError("AkShare is not installed; cannot fetch universe.")

    logger.info("Fetching A-share universe from AkShare …")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = ak.stock_zh_a_spot_em()
            break
        except Exception as exc:
            logger.warning("Spot fetch attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise NetworkError(f"Failed to fetch spot data after {MAX_RETRIES} retries") from exc
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))

    # Normalise column names (AkShare sometimes varies)
    col_map = {
        "代码": "code",
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "change_pct",
        "总市值": "total_mv",
        "所属行业": "industry",
        "上市日期": "list_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Derive exchange from code prefix
    def _exchange(c: str) -> str:
        if c.startswith(("600", "601", "603", "605", "688", "689")):
            return "sh"
        if c.startswith(("000", "001", "002", "003", "300", "301")):
            return "sz"
        return "other"

    df["exchange"] = df["code"].apply(_exchange)
    df["industry"] = df.get("industry", "未知")
    df["total_mv"] = pd.to_numeric(df.get("total_mv", 0), errors="coerce").fillna(0)

    # Filter
    mask = df.apply(lambda r: _is_valid_stock(str(r["code"]), str(r["name"])), axis=1)
    df = df[mask].copy()

    # Min listing days
    if "list_date" in df.columns:
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        cutoff = (datetime.now() - timedelta(days=min_listing_days)).strftime("%Y-%m-%d")
        df = df[df["list_date"] <= cutoff]

    df = df[["code", "name", "exchange", "industry", "total_mv", "list_date"]].reset_index(drop=True)

    # Persist
    df.to_parquet(cache_file, index=False)
    # Also persist to SQLite meta table
    conn = _init_meta_db()
    try:
        records = df[["code", "name", "exchange", "industry", "total_mv", "list_date"]].values.tolist()
        conn.executemany(
            "INSERT OR REPLACE INTO stock_meta (code, name, exchange, industry, total_mv, list_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            records,
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("Stock universe loaded: %d symbols", len(df))
    return df


def get_minute_data(
    stock_code: str,
    period: str = "1",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = "qfq",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Retrieve minute-level OHLCV for a single stock via AkShare.

    Parameters
    ----------
    stock_code: str
        6-digit A-share code (e.g. "000001").
    period: {``"1"``, ``"5"``, ``"15"``, ``"30"``, ``"60"``}, default ``"1"``
        K-line period in minutes.
    start_date: str, optional
        ``"YYYYMMDD"`` or ``"YYYY-MM-DD"``. Defaults to 5 trading days ago.
    end_date: str, optional
        Defaults to today.
    adjust: {``""``, ``"qfq"``, ``"hfq"``}, default ``"qfq"``
        Price adjustment flag (forward / backward / none).
    use_cache: bool, default True
        Read from / write to local Parquet cache.

    Returns
    -------
    pd.DataFrame
        Columns: ``datetime, open, high, low, close, volume, amount``
    """
    if period not in MINUTE_FREQS:
        raise ValueError(f"period must be one of {MINUTE_FREQS}, got {period}")

    start_date = start_date or (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    end_date = end_date or datetime.now().strftime("%Y%m%d")
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")

    cache_path = PARQUET_DIR / f"min_{stock_code}_{period}_{start_date}_{end_date}_{adjust}.parquet"
    if use_cache and cache_path.exists():
        logger.debug("Minute cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    if ak is None:
        raise NetworkError("AkShare not available.")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # ak.stock_zh_a_hist_min_em supports multi-day minute data
            df = ak.stock_zh_a_hist_min_em(
                symbol=stock_code,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            break
        except Exception as exc:
            logger.warning("Minute fetch %s attempt %d failed: %s", stock_code, attempt, exc)
            if attempt == MAX_RETRIES:
                raise NetworkError(f"Failed minute data for {stock_code}") from exc
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))

    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])

    # Normalise columns
    rename_map = {
        "时间": "datetime",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    if use_cache:
        df.to_parquet(cache_path, index=False)

    return df


def get_daily_data(
    stock_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = "qfq",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Retrieve daily OHLCV for a single stock.

    Parameters
    ----------
    stock_code: str
        6-digit code.
    start_date, end_date: str, optional
        ``"YYYYMMDD"``.
    adjust: str, default ``"qfq"``
    use_cache: bool, default True

    Returns
    -------
    pd.DataFrame
        Columns: ``date, open, high, low, close, volume, amount, amplitude, pct_chg, turnover``
    """
    start_date = start_date or (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    end_date = end_date or datetime.now().strftime("%Y%m%d")
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")

    cache_path = PARQUET_DIR / f"daily_{stock_code}_{start_date}_{end_date}_{adjust}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    if ak is None:
        raise NetworkError("AkShare not available.")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            break
        except Exception as exc:
            logger.warning("Daily fetch %s attempt %d failed: %s", stock_code, attempt, exc)
            if attempt == MAX_RETRIES:
                raise NetworkError(f"Failed daily data for {stock_code}") from exc
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))

    if df is None or df.empty:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "turnover"]
        )

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "换手率": "turnover",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "turnover"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    if use_cache:
        df.to_parquet(cache_path, index=False)

    return df


def get_bid_ask_data(
    stock_codes: Optional[List[str]] = None,
    max_stocks: int = 100,
) -> pd.DataFrame:
    """
    Retrieve **call-auction / L1 bid-ask** snapshot for a batch of stocks.

    Uses ``ak.stock_bid_ask_em`` which returns the top-5 bid/ask queue for
    all listed stocks.  We filter down to the requested subset.

    Parameters
    ----------
    stock_codes: list of str, optional
        If ``None``, returns data for the **entire tradable universe** (slow).
    max_stocks: int, default 100
        Hard cap to prevent excessive network payload.

    Returns
    -------
    pd.DataFrame
        Bid/ask snapshot with derived metrics (bid_ask_ratio, auction_gap).
    """
    if ak is None:
        raise NetworkError("AkShare not available.")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = ak.stock_bid_ask_em()
            break
        except Exception as exc:
            logger.warning("Bid-ask fetch attempt %d failed: %s", attempt, exc)
            if attempt == MAX_RETRIES:
                raise NetworkError("Failed bid-ask data") from exc
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))

    # Normalise
    col_map = {
        "代码": "code",
        "名称": "name",
        "买一价": "bid_price_1",
        "卖一价": "ask_price_1",
        "买一量": "bid_volume_1",
        "卖一量": "ask_volume_1",
        "最新价": "last_price",
        "昨收": "pre_close",
        "开盘价": "open",
        "涨跌额": "change",
        "涨跌幅": "change_pct",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Filter
    if stock_codes is not None:
        df = df[df["code"].isin(stock_codes)].copy()
    else:
        df = df[df.apply(lambda r: _is_valid_stock(str(r["code"]), str(r.get("name", ""))), axis=1)].copy()

    if len(df) > max_stocks:
        logger.warning("Bid-ask result truncated from %d to %d", len(df), max_stocks)
        df = df.head(max_stocks)

    # Derived metrics
    numeric_cols = ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1", "last_price", "pre_close", "open"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["bid_ask_ratio"] = (df["bid_volume_1"] / (df["ask_volume_1"] + 1e-6)).round(4)
    df["auction_gap_pct"] = ((df["open"] - df["pre_close"]) / (df["pre_close"] + 1e-6) * 100).round(3)
    df["timestamp"] = datetime.now()

    return df.reset_index(drop=True)


def batch_get_minute_data(
    stock_codes: List[str],
    period: str = "1",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = "qfq",
    max_workers: int = 4,
) -> Dict[str, pd.DataFrame]:
    """
    Batch-fetch minute data for many stocks (sequential with caching).

    Parameters
    ----------
    stock_codes: list of str
    period: str
    start_date, end_date: str, optional
    adjust: str
    max_workers: int
        Placeholder for future parallel implementation (ProcessPoolExecutor).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``code → DataFrame``.
    """
    results: Dict[str, pd.DataFrame] = {}
    for code in stock_codes:
        try:
            df = get_minute_data(code, period, start_date, end_date, adjust)
            if not df.empty:
                results[code] = df
        except Exception as exc:
            logger.error("Skipping %s in batch minute fetch: %s", code, exc)
    logger.info("Batch minute fetch complete: %d/%d success", len(results), len(stock_codes))
    return results


def batch_get_daily_data(
    stock_codes: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = "qfq",
) -> Dict[str, pd.DataFrame]:
    """
    Batch-fetch daily data for many stocks.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    results: Dict[str, pd.DataFrame] = {}
    for code in stock_codes:
        try:
            df = get_daily_data(code, start_date, end_date, adjust)
            if not df.empty:
                results[code] = df
        except Exception as exc:
            logger.error("Skipping %s in batch daily fetch: %s", code, exc)
    logger.info("Batch daily fetch complete: %d/%d success", len(results), len(stock_codes))
    return results


def get_trading_calendar(
    exchange: str = "sh",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retrieve trading calendar (交易日历) for exchange.

    Parameters
    ----------
    exchange: str, default ``"sh"``
    start_year, end_year: int, optional

    Returns
    -------
    pd.DataFrame
        Columns: ``trade_date, is_open``
    """
    if ak is None:
        raise NetworkError("AkShare not available.")

    start_year = start_year or datetime.now().year - 1
    end_year = end_year or datetime.now().year

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = ak.tool_trade_date_hist_sina()
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise NetworkError("Failed trading calendar") from exc
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))

    df.columns = ["trade_date", "is_open"]
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    mask = (df["trade_date"].dt.year >= start_year) & (df["trade_date"].dt.year <= end_year)
    return df[mask].sort_values("trade_date").reset_index(drop=True)


def clear_cache(older_than_days: Optional[int] = None) -> None:
    """
    Remove stale cache files.

    Parameters
    ----------
    older_than_days: int, optional
        If given, only delete files older than *N* days.
    """
    cutoff = None
    if older_than_days is not None:
        cutoff = datetime.now() - timedelta(days=older_than_days)

    removed = 0
    for f in PARQUET_DIR.glob("*.parquet"):
        if cutoff and datetime.fromtimestamp(f.stat().st_mtime) > cutoff:
            continue
        f.unlink()
        removed += 1
    logger.info("Cache cleared: %d parquet files removed", removed)


# ---------------------------------------------------------------------------
# Convenience: pre-load universe into memory for GPU pipelines
# ---------------------------------------------------------------------------
_UNIVERSE_CACHE: Optional[pd.DataFrame] = None

def get_cached_universe() -> pd.DataFrame:
    """Return the in-memory cached universe (or load it)."""
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is None or _UNIVERSE_CACHE.empty:
        _UNIVERSE_CACHE = get_stock_universe()
    return _UNIVERSE_CACHE
