"""
gpu_pool.py
===========
Grace Hopper unified memory pool + multi-stream concurrency.

Maximises DGX Spark 624GB unified memory + NVLink-C2C 900GB/s.

Author  : AQuant GPU Team
Version : v4.5
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy CuPy import
cp = None
try:
    import cupy as cp
except Exception:
    pass


class GPUMemoryPool:
    """
    Pre-allocate GPU memory pools for factor computation.

    Avoids runtime allocation overhead and ensures all data stays
    in unified memory (no CPU↔GPU copies).
    """

    def __init__(self, max_stocks: int = 5500, max_days: int = 252) -> None:
        self.max_stocks = max_stocks
        self.max_days = max_days
        self._pools: Dict[str, "cp.ndarray"] = {}
        self._init = False

    def warm_up(self) -> None:
        """Pre-allocate memory pools for all factor arrays."""
        if cp is None:
            logger.warning("CuPy not available — memory pool disabled")
            return

        if self._init:
            return

        n_s, n_d = self.max_stocks, self.max_days
        logger.info("Warming up GPU memory pool: %d stocks × %d days", n_s, n_d)

        # Price/volume arrays (the largest)
        self._pools["prices"] = cp.empty((n_s, n_d), dtype=cp.float64)
        self._pools["volumes"] = cp.empty((n_s, n_d), dtype=cp.float64)
        self._pools["highs"] = cp.empty((n_s, n_d), dtype=cp.float64)
        self._pools["lows"] = cp.empty((n_s, n_d), dtype=cp.float64)
        self._pools["closes"] = cp.empty((n_s, n_d), dtype=cp.float64)
        self._pools["amounts"] = cp.empty((n_s, n_d), dtype=cp.float64)
        self._pools["opens"] = cp.empty((n_s, n_d), dtype=cp.float64)

        # Intermediate buffers
        self._pools["returns"] = cp.empty((n_s, n_d - 1), dtype=cp.float64)
        self._pools["atr_buffer"] = cp.empty((n_s, 14), dtype=cp.float64)

        # Force memory allocation
        cp.cuda.Device().synchronize()
        used = cp.get_default_memory_pool().used_bytes()
        logger.info("GPU memory pool allocated: %.1f MB", used / 1024 / 1024)
        self._init = True

    def get(self, name: str, shape: Optional[Tuple[int, ...]] = None) -> "cp.ndarray":
        """Get a pre-allocated buffer (sliced to requested shape)."""
        if name not in self._pools:
            raise KeyError(f"Pool '{name}' not pre-allocated")
        buf = self._pools[name]
        if shape is not None:
            return buf[: shape[0], : shape[1] if len(shape) > 1 else None]
        return buf


class GPUStreamPool:
    """
    Multi-stream concurrency for independent factor computations.

    Streams:
    - stream 0: momentum + overnight_return
    - stream 1: volume_ratio + liquidity
    - stream 2: atr_ratio + sector_momentum
    """

    def __init__(self, num_streams: int = 3) -> None:
        self.num_streams = num_streams
        self._streams: list = []
        if cp is not None:
            for _ in range(num_streams):
                self._streams.append(cp.cuda.Stream())
        else:
            self._streams = [None] * num_streams

    def get(self, stream_id: int):
        return self._streams[stream_id % self.num_streams]

    def synchronize_all(self) -> None:
        """Wait for all streams to complete."""
        for s in self._streams:
            if s is not None:
                s.synchronize()


# ---------------------------------------------------------------------------
# Multi-stream factor computation
# ---------------------------------------------------------------------------

def compute_factors_multistream(
    d_prices: "cp.ndarray",
    d_volumes: "cp.ndarray",
    d_highs: "cp.ndarray",
    d_lows: "cp.ndarray",
    d_closes: "cp.ndarray",
    d_amounts: "cp.ndarray",
    d_volume_ma20: "cp.ndarray",
    d_opens: Optional["cp.ndarray"] = None,
    d_pre_closes: Optional["cp.ndarray"] = None,
    stock_sectors: Optional[np.ndarray] = None,
    sector_returns: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute all 6 factors using 3 concurrent CUDA streams.

    Returns dict of CPU numpy arrays (single .get() at end).
    """
    if cp is None:
        raise RuntimeError("CuPy required for multi-stream computation")

    pool = GPUStreamPool(num_streams=3)
    mem = GPUMemoryPool()
    mem.warm_up()

    n_stocks = d_prices.shape[0]
    window = 30
    lookback = 14

    # Stream 0: momentum + overnight_return
    s0 = pool.get(0)
    with s0:
        start = max(0, d_prices.shape[1] - window)
        d_price_window = d_prices[:, start:]
        d_vol_window = d_volumes[:, start:]
        d_ret = cp.log(d_price_window[:, 1:] / (d_price_window[:, :-1] + 1e-12))
        d_avg_vol = d_vol_window[:, 1:].mean(axis=1, keepdims=True)
        d_vol_norm = d_vol_window[:, 1:] / (d_avg_vol + 1e-12)
        d_w_ret = d_ret * d_vol_norm
        d_momentum = d_w_ret.mean(axis=1)

        if d_opens is not None and d_pre_closes is not None:
            d_overnight = cp.log(d_opens[:, -1] / (d_pre_closes[:, -1] + 1e-12))
        else:
            d_overnight = cp.zeros(n_stocks, dtype=cp.float64)

    # Stream 1: volume_ratio + liquidity
    s1 = pool.get(1)
    with s1:
        d_vol_today = d_volumes[:, -1]
        d_vol_ma = d_volume_ma20[:, -1] if d_volume_ma20.ndim == 2 else d_volume_ma20
        d_vol_ratio = d_vol_today / (d_vol_ma + 1e-12)

        d_log_amt = cp.log1p(d_amounts)
        d_liq = d_log_amt.mean(axis=1)
        liq_min = d_liq.min()
        liq_max = d_liq.max()
        d_liq_norm = (d_liq - liq_min) / (liq_max - liq_min + 1e-12)

    # Stream 2: atr_ratio + sector_momentum
    s2 = pool.get(2)
    with s2:
        d_atr = cp.zeros(n_stocks, dtype=cp.float64)
        if d_highs.shape[1] >= lookback:
            d_h_win = d_highs[:, -lookback:]
            d_l_win = d_lows[:, -lookback:]
            d_c_win = d_closes[:, -lookback:]
            d_tr = cp.maximum(d_h_win - d_l_win, cp.abs(d_h_win - d_c_win))
            d_atr = d_tr.mean(axis=1)
        d_atr_ratio = d_atr / (d_closes[:, -1] + 1e-12)

        if stock_sectors is not None and sector_returns is not None:
            d_sectors = cp.asarray(stock_sectors)
            d_sret = cp.asarray(sector_returns)
            d_sector = d_sret[d_sectors]
        else:
            d_sector = cp.zeros(n_stocks, dtype=cp.float64)

    # Synchronize all streams before returning
    pool.synchronize_all()

    # Single CPU transfer at the end
    return {
        "momentum": d_momentum.get(),
        "volume_ratio": d_vol_ratio.get(),
        "atr_ratio": d_atr_ratio.get(),
        "liquidity": d_liq_norm.get(),
        "sector_momentum": d_sector.get(),
        "overnight_return": d_overnight.get(),
    }
