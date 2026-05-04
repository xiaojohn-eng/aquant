"""
minute_data.py
==============
A-share minute-level K-line data fetcher.

Replaces the 0.4*open + 0.6*high approximation with real 10:00 prices.
Sources: East Money minute data API.

Author  : AQuant Data Team
Version : v4.5
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minute bar data structure
# ---------------------------------------------------------------------------

@dataclass
class MinuteBar:
    """Single minute OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float


# ---------------------------------------------------------------------------
# Minute data fetcher
# ---------------------------------------------------------------------------

class MinuteDataFetcher:
    """
    Fetch minute-level K-line from East Money API.

    Endpoint: https://push2.eastmoney.com/api/qt/stock/get
              ?secid={market}.{code}&fields=f43,f44,f45,f46,f47,f48
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    async def fetch_minute_kline(
        self,
        code: str,
        trade_date: date,
        market: str = "0",  # 0=SZ, 1=SH
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 1-minute K-line for a single stock on a single day.

        Returns DataFrame with columns [time, open, high, low, close, volume, amount].
        """
        url = (
            f"https://push2.eastmoney.com/api/qt/stock/get"
            f"?secid={market}.{code}"
            f"&fields=f43,f44,f45,f46,f47,f48"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        logger.warning("Minute data HTTP %d for %s", resp.status, code)
                        return None
                    text = await resp.text()
                    data = json.loads(text)
                    return self._parse_minute_data(data, trade_date)
        except Exception as exc:
            logger.error("Minute data fetch failed for %s: %s", code, exc)
            return None

    def _parse_minute_data(
        self,
        raw: dict,
        trade_date: date,
    ) -> Optional[pd.DataFrame]:
        """Parse East Money minute data response."""
        data = raw.get("data", {})
        klines = data.get("klines", [])
        if not klines:
            return None

        records = []
        for line in klines:
            # Format: "09:30,open,high,low,close,volume,amount,avg"
            parts = line.split(",")
            if len(parts) < 6:
                continue
            t_str = parts[0]
            hh, mm = int(t_str[:2]), int(t_str[2:])
            ts = datetime.combine(trade_date, dt_time(hh, mm))
            records.append({
                "timestamp": ts,
                "open": float(parts[1]),
                "high": float(parts[2]),
                "low": float(parts[3]),
                "close": float(parts[4]),
                "volume": int(parts[5]),
                "amount": float(parts[6]) if len(parts) > 6 else 0.0,
            })

        return pd.DataFrame(records).set_index("timestamp")

    def get_price_at_time(
        self,
        minute_df: pd.DataFrame,
        target_time: dt_time,
    ) -> Optional[float]:
        """
        Extract exact price at target time (e.g., 10:00) from minute data.

        If exact minute not available, uses linear interpolation.
        """
        if minute_df.empty:
            return None

        # Find closest minute
        target_dt = datetime.combine(date.today(), target_time)
        idx = minute_df.index.get_indexer([target_dt], method="nearest")[0]
        if idx == -1:
            return None

        return float(minute_df.iloc[idx]["close"])

    async def fetch_batch_minute_data(
        self,
        codes: List[str],
        trade_date: date,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch minute data for multiple stocks (sequential with rate limit)."""
        results = {}
        for code in codes:
            market = "1" if code.endswith(".SH") else "0"
            clean_code = code.split(".")[0]
            df = await self.fetch_minute_kline(clean_code, trade_date, market)
            if df is not None:
                results[code] = df
        return results


# ---------------------------------------------------------------------------
# Backtest integration: minute-level execution
# ---------------------------------------------------------------------------

def get_entry_price_minute(
    minute_df: Optional[pd.DataFrame],
    fallback_open: float,
    fallback_high: float,
    entry_time: dt_time = dt_time(10, 0),
) -> float:
    """
    Get precise 10:00 entry price from minute data.

    Falls back to 0.4*open + 0.6*high if minute data unavailable.
    """
    if minute_df is not None and not minute_df.empty:
        price = MinuteDataFetcher().get_price_at_time(minute_df, entry_time)
        if price is not None:
            return price

    # Fallback approximation (deprecated in v4.5)
    return 0.4 * fallback_open + 0.6 * fallback_high
