"""
gpu_compute.py
==============
A-Share quantitative trading system — GPU-Accelerated Factor Engine.

Responsibilities
----------------
* Compute **technical / statistical factors** for the entire A-share universe
  (~5 000 stocks) simultaneously on the GPU.
* Provide a **transparent fallback chain**: CuPy → Numba CUDA → NumPy.
* Manage GPU memory (pool allocator, explicit free, peak monitoring).

Supported hardware
------------------
* NVIDIA DGX Spark (Grace Hopper) — 96 GB HBM3
* Any CUDA-capable GPU with compute capability ≥ 7.0

Factor catalogue
----------------
1. ``momentum``       – 09:30-10:00 intra-day return (weighted by volume).
2. ``volume_ratio``   – Current volume / 20-day MA volume.
3. ``atr_ratio``      – Average True Range as % of close.
4. ``liquidity``      – Dollar-volume score (log-scaled, rank-based).
5. ``sector_momentum``– Cross-sectional sector return z-score.
6. ``composite``      – Weighted linear combination of all factors.

Implementation notes
--------------------
* CuPy uses the **RawKernel** interface for custom CUDA C++ when the
  built-in ``cupy.ElementwiseKernel`` or ``reduction`` is insufficient.
* Numba CUDA kernels are compiled lazily; the first call incurs JIT overhead.
* All inputs are expected as ``float64`` 2-D arrays (stocks × time) to maximise
  numerical stability.

Author   : AQuant Core Team
Platform : NVIDIA DGX Spark (Grace Hopper)
"""
from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np

# ---------------------------------------------------------------------------
# GPU backend detection & import
# ---------------------------------------------------------------------------
logger = logging.getLogger("aquant.gpu_compute")

GPU_BACKEND: str = "numpy"  # will be updated after probing

# CuPy probe
try:
    import cupy as cp
    import cupy.cuda.memory_hooks as cp_hooks

    cp.cuda.Device().use(0)
    _test_arr = cp.zeros(10)
    del _test_arr
    GPU_BACKEND = "cupy"
    logger.info("CuPy backend initialised (device %d)", cp.cuda.Device().id)
except Exception as _cpexc:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    logger.debug("CuPy not available: %s", _cpexc)

# Numba CUDA probe (only if CuPy failed)
if GPU_BACKEND == "numpy":
    try:
        from numba import cuda as numba_cuda
        from numba.cuda.cudadrv.error import CudaSupportError

        if numba_cuda.is_available():
            GPU_BACKEND = "numba_cuda"
            logger.info("Numba CUDA backend initialised")
        else:
            raise CudaSupportError("No CUDA device detected by Numba")
    except Exception as _numbaexc:  # pragma: no cover
        numba_cuda = None  # type: ignore[assignment]
        logger.debug("Numba CUDA not available: %s", _numbaexc)

logger.info("GPU compute backend selected: %s", GPU_BACKEND)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, "cp.ndarray"]
FactorDict = Dict[str, ArrayLike]

# ---------------------------------------------------------------------------
# GPU memory monitor
# ---------------------------------------------------------------------------
class GPUMemoryMonitor:
    """
    Lightweight peak-memory tracker for CuPy.

    Numba CUDA does not expose a unified memory pool, therefore peak tracking
    is best-effort via nvidia-ml-py or nvml wrappers.  This class abstracts
    the difference.
    """

    def __init__(self) -> None:
        self._peak_bytes = 0
        if GPU_BACKEND == "cupy" and cp is not None:
            self._hook = cp_hooks.DebugPrintHook()
        else:
            self._hook = None

    @contextmanager
    def track(self):
        """Context manager: logs peak GPU memory delta."""
        if GPU_BACKEND == "cupy" and cp is not None:
            mempool = cp.get_default_memory_pool()
            used_before = mempool.used_bytes()
            yield
            used_after = mempool.used_bytes()
            delta = used_after - used_before
            if delta > self._peak_bytes:
                self._peak_bytes = delta
            logger.debug("GPU memory delta: %.2f MB (peak %.2f MB)", delta / 1e6, self._peak_bytes / 1e6)
        else:
            yield

    @staticmethod
    def free_all() -> None:
        """Explicitly release all GPU memory back to the OS / pool."""
        if GPU_BACKEND == "cupy" and cp is not None:
            mempool = cp.get_default_memory_pool()
            pinned = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned.free_all_blocks()
            logger.debug("CuPy memory pools freed")
        elif GPU_BACKEND == "numba_cuda" and numba_cuda is not None:
            # Numba does not expose a global free API; rely on GC.
            import gc

            gc.collect()
            logger.debug("Numba CUDA GC triggered")


# Global monitor instance
_gpu_monitor = GPUMemoryMonitor()

# ---------------------------------------------------------------------------
# Helper: dispatch arrays to GPU if possible
# ---------------------------------------------------------------------------

def to_gpu(arr: np.ndarray) -> ArrayLike:
    """Move a NumPy array to the active GPU backend."""
    if GPU_BACKEND == "cupy" and cp is not None:
        return cp.asarray(arr)
    return arr


def to_cpu(arr: ArrayLike) -> np.ndarray:
    """Bring an array back to host memory."""
    if GPU_BACKEND == "cupy" and cp is not None and hasattr(arr, "get"):
        return arr.get()  # type: ignore[union-attr]
    return cast(np.ndarray, arr)


def zeros(shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> ArrayLike:
    """Allocate zero-filled array on the active compute device."""
    if GPU_BACKEND == "cupy" and cp is not None:
        return cp.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def ones(shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> ArrayLike:
    """Allocate ones-filled array on the active compute device."""
    if GPU_BACKEND == "cupy" and cp is not None:
        return cp.ones(shape, dtype=dtype)
    return np.ones(shape, dtype=dtype)


def empty(shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> ArrayLike:
    """Allocate uninitialized array on the active compute device."""
    if GPU_BACKEND == "cupy" and cp is not None:
        return cp.empty(shape, dtype=dtype)
    return np.empty(shape, dtype=dtype)

# ---------------------------------------------------------------------------
# CuPy RawKernel definitions (compiled on first use)
# ---------------------------------------------------------------------------
if GPU_BACKEND == "cupy" and cp is not None:
    # Kernel: momentum = weighted mean return over look-back window
    _MOMENTUM_KERNEL_CODE = r"""
    extern "C" __global__
    void momentum_kernel(const double* prices, const double* volumes,
                         double* out, int n_stocks, int n_time,
                         int window) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = idx; i < n_stocks; i += stride) {
            double weighted_sum = 0.0;
            double vol_sum = 0.0;
            int base = i * n_time;
            for (int t = n_time - window; t < n_time - 1; ++t) {
                double ret = (prices[base + t + 1] - prices[base + t])
                             / (prices[base + t] + 1e-12);
                double vol = volumes[base + t] + 1e-12;
                weighted_sum += ret * vol;
                vol_sum += vol;
            }
            out[i] = weighted_sum / (vol_sum + 1e-12);
        }
    }
    """
    _momentum_kernel = cp.RawKernel(_MOMENTUM_KERNEL_CODE, "momentum_kernel")

    # Kernel: ATR ratio = ATR / close
    _ATR_KERNEL_CODE = r"""
    extern "C" __global__
    void atr_kernel(const double* high, const double* low,
                    const double* close, double* out,
                    int n_stocks, int n_time, int window) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = idx; i < n_stocks; i += stride) {
            double atr = 0.0;
            int base = i * n_time;
            for (int t = n_time - window; t < n_time; ++t) {
                double tr1 = high[base + t] - low[base + t];
                double tr2 = fabs(high[base + t] - close[base + t - 1]);
                double tr3 = fabs(low[base + t] - close[base + t - 1]);
                double tr = tr1;
                if (tr2 > tr) tr = tr2;
                if (tr3 > tr) tr = tr3;
                atr += tr;
            }
            atr /= window;
            double last_close = close[base + n_time - 1];
            out[i] = atr / (last_close + 1e-12);
        }
    }
    """
    _atr_kernel = cp.RawKernel(_ATR_KERNEL_CODE, "atr_kernel")

# ---------------------------------------------------------------------------
# Numba CUDA kernels (compiled lazily)
# ---------------------------------------------------------------------------
if GPU_BACKEND == "numba_cuda" and numba_cuda is not None:

    @numba_cuda.jit  # type: ignore[misc]
    def _momentum_kernel_numba(prices, volumes, out, n_stocks, n_time, window):
        """Numba CUDA kernel for volume-weighted momentum."""
        i = numba_cuda.grid(1)
        stride = numba_cuda.gridsize(1)
        for stock in range(i, n_stocks, stride):
            weighted_sum = 0.0
            vol_sum = 0.0
            base = stock * n_time
            for t in range(n_time - window, n_time - 1):
                p_t = prices[base + t]
                p_next = prices[base + t + 1]
                ret = (p_next - p_t) / (p_t + 1e-12)
                vol = volumes[base + t] + 1e-12
                weighted_sum += ret * vol
                vol_sum += vol
            out[stock] = weighted_sum / (vol_sum + 1e-12)

    @numba_cuda.jit  # type: ignore[misc]
    def _atr_kernel_numba(high, low, close, out, n_stocks, n_time, window):
        """Numba CUDA kernel for ATR ratio."""
        i = numba_cuda.grid(1)
        stride = numba_cuda.gridsize(1)
        for stock in range(i, n_stocks, stride):
            atr = 0.0
            base = stock * n_time
            for t in range(n_time - window, n_time):
                tr1 = high[base + t] - low[base + t]
                tr2 = abs(high[base + t] - close[base + t - 1])
                tr3 = abs(low[base + t] - close[base + t - 1])
                tr = tr1
                if tr2 > tr:
                    tr = tr2
                if tr3 > tr:
                    tr = tr3
                atr += tr
            atr /= window
            last_close = close[base + n_time - 1]
            out[stock] = atr / (last_close + 1e-12)


# ---------------------------------------------------------------------------
# Factor computation API
# ---------------------------------------------------------------------------

def compute_momentum_factor(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """
    Volume-weighted intra-day momentum factor.

    For the A-share 10:00-entry strategy this is the return from 09:30 to
    10:00 (window ≈ 30 one-minute bars).  Higher values indicate stronger
    opening momentum.

    Parameters
    ----------
    prices: ndarray, shape (n_stocks, n_time)
    volumes: ndarray, shape (n_stocks, n_time)
    window: int, default 30
        Number of trailing bars to use.

    Returns
    -------
    ndarray, shape (n_stocks,)
        Momentum score per stock.
    """
    n_stocks, n_time = prices.shape
    if n_time < window + 1:
        window = max(1, n_time - 1)

    with _gpu_monitor.track():
        if GPU_BACKEND == "cupy" and cp is not None:
            d_prices = cp.asarray(prices)
            d_volumes = cp.asarray(volumes)
            d_out = cp.empty(n_stocks, dtype=cp.float64)
            threads = 256
            blocks = (n_stocks + threads - 1) // threads
            _momentum_kernel(
                (blocks,), (threads,),
                (d_prices, d_volumes, d_out, n_stocks, n_time, window),
            )
            cp.cuda.Device().synchronize()
            return d_out.get()

        elif GPU_BACKEND == "numba_cuda" and numba_cuda is not None:
            d_prices = numba_cuda.to_device(prices)
            d_volumes = numba_cuda.to_device(volumes)
            d_out = numba_cuda.device_array(n_stocks, dtype=np.float64)
            threads = 256
            blocks = (n_stocks + threads - 1) // threads
            _momentum_kernel_numba[blocks, threads](
                d_prices, d_volumes, d_out, n_stocks, n_time, window
            )
            numba_cuda.synchronize()
            return d_out.copy_to_host()

        # NumPy fallback
        out = np.empty(n_stocks, dtype=np.float64)
        for i in range(n_stocks):
            slice_p = prices[i, -window - 1 :]
            slice_v = volumes[i, -window - 1 :]
            rets = np.diff(slice_p) / (slice_p[:-1] + 1e-12)
            weighted = np.sum(rets * (slice_v[:-1] + 1e-12)) / (np.sum(slice_v[:-1]) + 1e-12)
            out[i] = weighted
        return out


def compute_volume_ratio(
    volume: np.ndarray,
    volume_ma20: np.ndarray,
) -> np.ndarray:
    """
    Volume ratio = current volume / 20-day moving-average volume.

    Parameters
    ----------
    volume: ndarray, shape (n_stocks,) or (n_stocks, n_time)
        Current (or latest) volume.
    volume_ma20: ndarray, same shape
        20-period moving average of volume.

    Returns
    -------
    ndarray
    """
    with _gpu_monitor.track():
        if GPU_BACKEND == "cupy" and cp is not None:
            d_vol = cp.asarray(volume)
            d_ma = cp.asarray(volume_ma20)
            ratio = d_vol / (d_ma + 1e-12)
            return ratio.get()
        # NumPy/Numba path
        return volume / (volume_ma20 + 1e-12)


def compute_atr_ratio(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    ATR (Average True Range) as a percentage of the last close.

    Lower values = tighter volatility = preferred for short-term holding.

    Parameters
    ----------
    high, low, close: ndarray, shape (n_stocks, n_time)
    window: int, default 14

    Returns
    -------
    ndarray, shape (n_stocks,)
    """
    n_stocks, n_time = high.shape
    if n_time < window + 1:
        window = max(1, n_time - 1)

    with _gpu_monitor.track():
        if GPU_BACKEND == "cupy" and cp is not None:
            d_high = cp.asarray(high)
            d_low = cp.asarray(low)
            d_close = cp.asarray(close)
            d_out = cp.empty(n_stocks, dtype=cp.float64)
            threads = 256
            blocks = (n_stocks + threads - 1) // threads
            _atr_kernel(
                (blocks,), (threads,),
                (d_high, d_low, d_close, d_out, n_stocks, n_time, window),
            )
            cp.cuda.Device().synchronize()
            return d_out.get()

        elif GPU_BACKEND == "numba_cuda" and numba_cuda is not None:
            d_high = numba_cuda.to_device(high)
            d_low = numba_cuda.to_device(low)
            d_close = numba_cuda.to_device(close)
            d_out = numba_cuda.device_array(n_stocks, dtype=np.float64)
            threads = 256
            blocks = (n_stocks + threads - 1) // threads
            _atr_kernel_numba[blocks, threads](
                d_high, d_low, d_close, d_out, n_stocks, n_time, window
            )
            numba_cuda.synchronize()
            return d_out.copy_to_host()

        # NumPy fallback
        out = np.empty(n_stocks, dtype=np.float64)
        for i in range(n_stocks):
            tr1 = high[i, -window:] - low[i, -window:]
            tr2 = np.abs(high[i, -window:] - close[i, -window - 1 : -1])
            tr3 = np.abs(low[i, -window:] - close[i, -window - 1 : -1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr = np.mean(tr)
            out[i] = atr / (close[i, -1] + 1e-12)
        return out


def compute_liquidity_score(
    amount: np.ndarray,
) -> np.ndarray:
    """
    Liquidity score based on log-scaled turnover amount.

    We first compute the raw amount, then apply log1p and min-max normalise
    to [0, 1] so that the score is comparable across market regimes.

    Parameters
    ----------
    amount: ndarray, shape (n_stocks,) or (n_stocks, n_time)
        Turnover amount in 万元 (or any consistent currency unit).

    Returns
    -------
    ndarray
        Normalised liquidity score ∈ [0, 1].
    """
    with _gpu_monitor.track():
        if GPU_BACKEND == "cupy" and cp is not None:
            d_amt = cp.asarray(amount)
            if d_amt.ndim == 2:
                d_amt = d_amt[:, -1]
            log_amt = cp.log1p(d_amt)
            amin = log_amt.min()
            amax = log_amt.max()
            score = (log_amt - amin) / (amax - amin + 1e-12)
            return score.get()

        if amount.ndim == 2:
            amount = amount[:, -1]
        log_amt = np.log1p(amount)
        amin, amax = log_amt.min(), log_amt.max()
        return (log_amt - amin) / (amax - amin + 1e-12)


def compute_sector_momentum(
    stock_sectors: np.ndarray,
    sector_returns: np.ndarray,
) -> np.ndarray:
    """
    Cross-sectional sector momentum factor.

    Each stock receives the z-score of its sector's return within the
    universe of sectors.

    Parameters
    ----------
    stock_sectors: ndarray, shape (n_stocks,)
        Integer sector label per stock.
    sector_returns: ndarray, shape (n_sectors,)
        Period return per sector.

    Returns
    -------
    ndarray, shape (n_stocks,)
        Sector momentum score (z-score centred).
    """
    with _gpu_monitor.track():
        if GPU_BACKEND == "cupy" and cp is not None:
            d_ret = cp.asarray(sector_returns)
            mu = d_ret.mean()
            sigma = d_ret.std()
            z = (d_ret - mu) / (sigma + 1e-12)
            d_sectors = cp.asarray(stock_sectors)
            return z[d_sectors].get()

        mu = sector_returns.mean()
        sigma = sector_returns.std()
        z = (sector_returns - mu) / (sigma + 1e-12)
        return z[stock_sectors]


def _numpy_batch_factors(stock_data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Pure-NumPy batch factor pipeline.

    Expected keys in *stock_data_dict*:
    ``prices, volumes, highs, lows, closes, amounts, volume_ma20,
    stock_sectors, sector_returns``.
    All 2-D arrays must share shape ``(n_stocks, n_time)``.
    """
    prices = stock_data_dict["prices"]
    volumes = stock_data_dict["volumes"]
    highs = stock_data_dict["highs"]
    lows = stock_data_dict["lows"]
    closes = stock_data_dict["closes"]
    amounts = stock_data_dict["amounts"]
    volume_ma20 = stock_data_dict["volume_ma20"]
    stock_sectors = stock_data_dict.get("stock_sectors", np.zeros(prices.shape[0], dtype=int))
    sector_returns = stock_data_dict.get("sector_returns", np.zeros(1, dtype=np.float64))

    n_stocks = prices.shape[0]
    logger.info("NumPy batch factor pipeline: %d stocks", n_stocks)

    t0 = time.perf_counter()
    momentum = compute_momentum_factor(prices, volumes, window=30)
    vol_ratio = compute_volume_ratio(volumes[:, -1], volume_ma20[:, -1])
    atr = compute_atr_ratio(highs, lows, closes, window=14)
    liquidity = compute_liquidity_score(amounts)
    sector = compute_sector_momentum(stock_sectors, sector_returns)
    logger.info("NumPy factor compute time: %.3f s", time.perf_counter() - t0)

    return {
        "momentum": momentum,
        "volume_ratio": vol_ratio,
        "atr_ratio": atr,
        "liquidity": liquidity,
        "sector_momentum": sector,
    }


def batch_compute_all_factors(
    stock_data_dict: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    **Batch compute all technical factors** for the entire universe.

    This is the primary entry point used by ``StrategyEngine``.  It
    automatically selects the fastest available backend and returns a
    dictionary of 1-D arrays indexed by stock.

    Parameters
    ----------
    stock_data_dict: dict
        Must contain the following **2-D** arrays (stocks × time):
        ``prices, volumes, highs, lows, closes, amounts``.
        Must also contain **1-D** arrays:
        ``volume_ma20`` (stocks × 1 or stocks × time),
        ``stock_sectors`` (stocks,),
        ``sector_returns`` (n_sectors,).

    Returns
    -------
    dict[str, ndarray]
        Keys: ``momentum, volume_ratio, atr_ratio, liquidity, sector_momentum``.
    """
    required = {"prices", "volumes", "highs", "lows", "closes", "amounts", "volume_ma20"}
    missing = required - set(stock_data_dict.keys())
    if missing:
        raise ValueError(f"batch_compute_all_factors missing keys: {missing}")

    # If CuPy is available, try to keep everything on GPU and return host arrays
    if GPU_BACKEND == "cupy" and cp is not None:
        logger.info("CuPy batch factor pipeline")
        with _gpu_monitor.track():
            prices = cp.asarray(stock_data_dict["prices"])
            volumes = cp.asarray(stock_data_dict["volumes"])
            highs = cp.asarray(stock_data_dict["highs"])
            lows = cp.asarray(stock_data_dict["lows"])
            closes = cp.asarray(stock_data_dict["closes"])
            amounts = cp.asarray(stock_data_dict["amounts"])
            volume_ma20 = cp.asarray(stock_data_dict["volume_ma20"])
            stock_sectors = cp.asarray(stock_data_dict.get("stock_sectors", np.zeros(prices.shape[0], dtype=int)))
            sector_returns = cp.asarray(stock_data_dict.get("sector_returns", np.zeros(1, dtype=np.float64)))

            t0 = time.perf_counter()
            momentum = compute_momentum_factor(cp.asnumpy(prices), cp.asnumpy(volumes), window=30)
            vol_ratio = compute_volume_ratio(
                cp.asnumpy(volumes[:, -1]),
                cp.asnumpy(volume_ma20[:, -1] if volume_ma20.ndim == 2 else volume_ma20),
            )
            atr = compute_atr_ratio(cp.asnumpy(highs), cp.asnumpy(lows), cp.asnumpy(closes), window=14)
            liquidity = compute_liquidity_score(cp.asnumpy(amounts))
            sector = compute_sector_momentum(cp.asnumpy(stock_sectors), cp.asnumpy(sector_returns))
            logger.info("CuPy batch factor time: %.3f s", time.perf_counter() - t0)

            return {
                "momentum": momentum,
                "volume_ratio": vol_ratio,
                "atr_ratio": atr,
                "liquidity": liquidity,
                "sector_momentum": sector,
            }

    # Numba CUDA or NumPy path
    return _numpy_batch_factors(stock_data_dict)


# ---------------------------------------------------------------------------
# Composite score helper
# ---------------------------------------------------------------------------

def compute_composite_score(
    factors: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    direction: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Linear-weighted composite score with z-score normalisation.

    Parameters
    ----------
    factors: dict[str, ndarray]
        Each value is a 1-D array of shape (n_stocks,).
    weights: dict[str, float], optional
        If ``None`` uses equal weighting across all supplied factors.
    direction: dict[str, int], optional
        ``+1`` = higher is better, ``-1`` = lower is better.
        Default: momentum/volume/liquidity/sector = +1, atr = -1.

    Returns
    -------
    ndarray, shape (n_stocks,)
        Higher = more attractive for long entry.
    """
    keys = list(factors.keys())
    n_stocks = factors[keys[0]].shape[0]

    if weights is None:
        w = {k: 1.0 / len(keys) for k in keys}
    else:
        w = {k: weights.get(k, 0.0) for k in keys}
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}

    if direction is None:
        direction = {k: 1 for k in keys}
        direction["atr_ratio"] = -1

    with _gpu_monitor.track():
        if GPU_BACKEND == "cupy" and cp is not None:
            score = cp.zeros(n_stocks, dtype=cp.float64)
            for k in keys:
                arr = cp.asarray(factors[k])
                mu = arr.mean()
                sigma = arr.std()
                z = (arr - mu) / (sigma + 1e-12)
                score += w[k] * z * direction.get(k, 1)
            return score.get()

        score = np.zeros(n_stocks, dtype=np.float64)
        for k in keys:
            arr = factors[k]
            mu = arr.mean()
            sigma = arr.std()
            z = (arr - mu) / (sigma + 1e-12)
            score += w[k] * z * direction.get(k, 1)
        return score


# ---------------------------------------------------------------------------
# Utility: warm-up / health check
# ---------------------------------------------------------------------------

def gpu_health_check() -> Dict[str, Any]:
    """
    Run a lightweight matmul to verify GPU is alive and report stats.

    Returns
    -------
    dict
        Keys: ``backend, device_name, memory_total, memory_free,
        compute_capability, warmup_time_ms``.
    """
    info: Dict[str, Any] = {"backend": GPU_BACKEND}

    if GPU_BACKEND == "cupy" and cp is not None:
        dev = cp.cuda.Device()
        info["device_name"] = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
        info["memory_total"] = dev.mem_info[1]
        info["memory_free"] = dev.mem_info[0]
        info["compute_capability"] = f"{dev.compute_capability}"

        t0 = time.perf_counter()
        a = cp.random.rand(1024, 1024)
        b = cp.random.rand(1024, 1024)
        c = a @ b
        cp.cuda.Device().synchronize()
        info["warmup_time_ms"] = (time.perf_counter() - t0) * 1000
        del a, b, c
        GPUMemoryMonitor.free_all()

    elif GPU_BACKEND == "numba_cuda" and numba_cuda is not None:
        dev = numba_cuda.get_current_device()
        info["device_name"] = dev.name
        info["compute_capability"] = f"{dev.compute_capability}"

        t0 = time.perf_counter()
        a = numba_cuda.to_device(np.random.rand(1024, 1024))
        b = numba_cuda.to_device(np.random.rand(1024, 1024))
        c = numba_cuda.device_array((1024, 1024))
        # No direct matmul in Numba; skip detailed timing
        numba_cuda.synchronize()
        info["warmup_time_ms"] = (time.perf_counter() - t0) * 1000
    else:
        info["device_name"] = "CPU (NumPy)"
        info["warmup_time_ms"] = 0.0

    return info
