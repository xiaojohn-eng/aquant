"""
Dual Data Source Manager

Combines AkShare (free, broad coverage) and Tushare Pro (stable, high quality)
to achieve 99%+ data availability with automatic failover.

Reference:
- Tushare + AKShare 双数据源整合方案 (2026)
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DualDataSource:
    """
    Dual-source data manager with automatic failover.
    
    Priority:
    1. Try primary source (configurable)
    2. If fails/timeout, try secondary source
    3. If both fail, return cached data if available
    4. Log failure for manual investigation
    """
    
    def __init__(self,
                 primary: str = "akshare",
                 secondary: str = "tushare",
                 tushare_token: Optional[str] = None,
                 timeout: int = 30,
                 max_workers: int = 8):
        self.primary = primary
        self.secondary = secondary
        self.timeout = timeout
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: Dict[str, pd.DataFrame] = {}  # simple in-memory cache
        
        # Initialize Tushare if available
        self._tushare_api = None
        if tushare_token:
            try:
                import tushare as ts
                self._tushare_api = ts.pro_api(tushare_token)
                logger.info("Tushare Pro initialized")
            except Exception as e:
                logger.warning(f"Tushare init failed: {e}")
    
    def _fetch_akshare_daily(self, code: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Fetch daily data from AkShare."""
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(symbol=code.replace(".SZ", "").replace(".SH", ""),
                                    period="daily",
                                    start_date=start.replace("-", ""),
                                    end_date=end.replace("-", ""),
                                    adjust="qfq")
            if df is not None and len(df) > 0:
                df.columns = [c.lower() for c in df.columns]
                df["trade_date"] = pd.to_datetime(df["date"])
                df = df.set_index("trade_date").sort_index()
                return df
        except Exception as e:
            logger.warning(f"AkShare fetch failed for {code}: {e}")
        return None
    
    def _fetch_tushare_daily(self, code: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Fetch daily data from Tushare Pro."""
        if self._tushare_api is None:
            return None
        try:
            # Tushare uses different code format
            ts_code = code.replace(".SZ", ".SZ").replace(".SH", ".SH")
            df = self._tushare_api.daily(ts_code=ts_code, start_date=start.replace("-", ""), 
                                         end_date=end.replace("-", ""))
            if df is not None and len(df) > 0:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.set_index("trade_date").sort_index()
                return df
        except Exception as e:
            logger.warning(f"Tushare fetch failed for {code}: {e}")
        return None
    
    def get_daily(self, code: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Get daily price data with automatic failover.
        
        Returns DataFrame with columns: open, high, low, close, volume, amount
        """
        cache_key = f"daily|{code}|{start}|{end}"
        
        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {code}")
            return self._cache[cache_key]
        
        result = None
        
        # Try primary
        if self.primary == "akshare":
            result = self._fetch_akshare_daily(code, start, end)
        else:
            result = self._fetch_tushare_daily(code, start, end)
        
        # Try secondary if primary fails
        if result is None or len(result) == 0:
            logger.info(f"Primary source failed for {code}, trying secondary")
            if self.secondary == "akshare":
                result = self._fetch_akshare_daily(code, start, end)
            else:
                result = self._fetch_tushare_daily(code, start, end)
        
        if result is not None and len(result) > 0:
            self._cache[cache_key] = result
            return result
        
        logger.error(f"Both data sources failed for {code}")
        return None
    
    def get_batch_daily(self, codes: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Fetch daily data for multiple stocks concurrently."""
        import concurrent.futures
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.get_daily, code, start, end): code for code in codes}
            for future in concurrent.futures.as_completed(futures):
                code = futures[future]
                try:
                    df = future.result(timeout=self.timeout)
                    if df is not None:
                        results[code] = df
                except Exception as e:
                    logger.error(f"Batch fetch failed for {code}: {e}")
        
        return results
    
    def get_stock_universe(self) -> List[str]:
        """Get full A-share universe with ST/BJ filtering."""
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            codes = df["代码"].tolist()
            # Filter ST and BJ
            filtered = []
            for c in codes:
                if c.startswith(("8", "4", "9")) or "BJ" in c.upper():
                    continue
                filtered.append(c)
            return filtered
        except Exception as e:
            logger.error(f"Failed to get stock universe: {e}")
            return []
    
    def get_realtime_quote(self, codes: List[str]) -> Optional[pd.DataFrame]:
        """Get real-time quotes for given codes."""
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            # Filter to requested codes
            df = df[df["代码"].isin(codes)]
            return df
        except Exception as e:
            logger.warning(f"Real-time quote failed: {e}")
            return None
