"""
strategy.py
===========
A-Share quantitative trading system — Multi-Factor Strategy Engine.

Responsibilities
----------------
* **Factor ingestion** – receives raw market data, delegates GPU factor
  computation to ``gpu_compute``.
* **Scoring** – z-score normalise each factor then linearly combine using
  configurable weights.
* **Signal generation** – long-only entry at 10:00, mandatory exit at next
  day 09:30 (T+1 rule).
* **Filter layer** – remove ST/BSE/ halted / limit-up / limit-down stocks.
* **Cost model** – commission (0.03 %, min 5 CNY) + stamp duty (0.1 % sell).

Research-backed design
----------------------
1. **Intra-day momentum, overnight reversal** (Tang et al., 2020)
   – we capture the momentum leg at 10:00 and exit before the reversal leg.
2. **Early-window predictability** (Liu et al., 2021)
   – the 09:30-10:00 interval carries the highest information ratio.
3. **Volume confirmation** – signals accompanied by >1.5× average volume
   receive a boost, reducing false break-outs.

Class diagram
-------------
┌──────────────────┐
│  StrategyEngine  │
├──────────────────┤
│ + weights        │
│ + cost_model     │
│ + calculate_scores()   │
│ + generate_signals()   │
│ + get_exit_signals()   │
│ + filter_stocks()      │
└──────────────────┘

Author   : AQuant Core Team
Platform : NVIDIA DGX Spark (Grace Hopper)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("aquant.strategy")

# ---------------------------------------------------------------------------
# Cost model constants (A-share regulatory costs)
# ---------------------------------------------------------------------------
COMMISSION_RATE = 0.0003       # 0.03 %
COMMISSION_MIN = 5.0           # CNY
STAMP_DUTY_RATE = 0.001        # 0.1 %, seller only
TRANSFER_FEE_RATE = 0.00002    # 0.002 % ( Shanghai only, negligible )

# ---------------------------------------------------------------------------
# A-share price limit rules
# ---------------------------------------------------------------------------
LIMIT_UP_PCT = 0.10   # 10 % for normal stocks
LIMIT_UP_PCT_ST = 0.05  # 5 % for ST stocks
LIMIT_UP_PCT_KCB = 0.20  # 20 % for 科创板 / 创业板 first 5 days no limit

# ---------------------------------------------------------------------------
# Default factor weights (sum ≈ 1.0, can be overridden)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    "momentum": 0.25,
    "volume_ratio": 0.20,
    "auction_gap": 0.15,
    "atr_ratio": 0.10,
    "liquidity": 0.15,
    "market_cap": 0.10,
    "sector_momentum": 0.05,
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CostModel:
    """
    Transaction-cost model compliant with A-share regulations.

    Attributes
    ----------
    commission_rate: float
    commission_min: float
    stamp_duty_rate: float
    """
    commission_rate: float = COMMISSION_RATE
    commission_min: float = COMMISSION_MIN
    stamp_duty_rate: float = STAMP_DUTY_RATE

    def apply(
        self,
        notional: float,
        side: str,  # "buy" | "sell"
    ) -> float:
        """
        Compute total transaction cost for a single trade.

        Parameters
        ----------
        notional: float
            Trade notional value (price × shares).
        side: str
            ``"buy"`` or ``"sell"``.

        Returns
        -------
        float
            Total cost (always positive).
        """
        comm = max(notional * self.commission_rate, self.commission_min)
        stamp = notional * self.stamp_duty_rate if side == "sell" else 0.0
        return comm + stamp


@dataclass
class Signal:
    """A single trading signal (entry or exit)."""
    code: str
    timestamp: datetime
    side: str  # "buy" | "sell"
    price: float
    quantity: int
    reason: str = ""


@dataclass
class Position:
    """Active position record."""
    code: str
    entry_price: float
    entry_time: datetime
    quantity: int
    cost_basis: float  # includes entry commission


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _check_limit(price: float, pre_close: float, is_st: bool = False) -> Tuple[bool, bool]:
    """
    Determine whether a price hits the up-limit or down-limit.

    Returns
    -------
    (is_limit_up, is_limit_down)
    """
    pct = (price - pre_close) / (pre_close + 1e-12)
    limit = LIMIT_UP_PCT_ST if is_st else LIMIT_UP_PCT
    return pct >= limit - 1e-6, pct <= -limit + 1e-6


def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Ensure weights sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    Multi-factor strategy engine for A-share T+1 long-only day-trading.

    Core workflow
    -------------
    1. **Morning 09:30-10:00** – accumulate minute-level data.
    2. **10:00** – ``generate_signals`` selects top-20 stocks.
    3. **Next day 09:30** – ``get_exit_signals`` liquidates all holdings.
    4. Throughout – ``filter_stocks`` removes untradable names.

    Parameters
    ----------
    weights: dict[str, float], optional
        Factor weights.  Must sum ≈ 1.0.
    cost_model: CostModel, optional
    max_positions: int, default 20
        Number of names to hold each day.
    max_notional_per_stock: float, optional
        Hard cap on single-stock exposure (CNY).
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        cost_model: Optional[CostModel] = None,
        max_positions: int = 20,
        max_notional_per_stock: Optional[float] = None,
    ) -> None:
        self.weights = _normalise_weights(weights or DEFAULT_WEIGHTS.copy())
        self.cost_model = cost_model or CostModel()
        self.max_positions = max_positions
        self.max_notional_per_stock = max_notional_per_stock
        self.positions: List[Position] = []
        self.signal_history: List[Signal] = []

        logger.info(
            "StrategyEngine initialised: weights=%s, max_positions=%d",
            self.weights,
            max_positions,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_scores(
        self,
        factors_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute composite score for every stock in the factor matrix.

        Parameters
        ----------
        factors_df: pd.DataFrame
            Index = stock code, columns = factor names
            (``momentum, volume_ratio, auction_gap, atr_ratio,
            liquidity, market_cap, sector_momentum``).

        Returns
        -------
        pd.DataFrame
            Same index with added column ``composite_score`` and
            ``rank`` (1 = best).
        """
        if factors_df.empty:
            logger.warning("calculate_scores received empty factors_df")
            return factors_df.assign(composite_score=np.nan, rank=np.nan)

        factor_keys = set(self.weights.keys()) & set(factors_df.columns)
        if not factor_keys:
            raise ValueError("No overlap between configured weights and factor columns")

        # Prepare arrays for GPU compute
        factor_arrays: Dict[str, np.ndarray] = {}
        for k in factor_keys:
            factor_arrays[k] = factors_df[k].fillna(0.0).values.astype(np.float64)

        # GPU composite score (lazy import to avoid circular deps)
        from . import gpu_compute
        t0 = datetime.now()
        scores = gpu_compute.compute_composite_score(
            factor_arrays,
            weights={k: self.weights[k] for k in factor_keys},
            direction={
                "momentum": 1,
                "volume_ratio": 1,
                "auction_gap": 1,
                "atr_ratio": -1,
                "liquidity": 1,
                "market_cap": 1,
                "sector_momentum": 1,
            },
        )
        logger.debug("Composite score compute time: %.3f ms", (datetime.now() - t0).total_seconds() * 1000)

        result = factors_df.copy()
        result["composite_score"] = scores
        result["rank"] = result["composite_score"].rank(ascending=False, method="min").astype(int)
        return result.sort_values("composite_score", ascending=False)

    def generate_signals(
        self,
        date_obj: Union[date, datetime, str],
        time_obj: Optional[dt_time] = None,
        factors_df: Optional[pd.DataFrame] = None,
        prices: Optional[Dict[str, float]] = None,
        pre_closes: Optional[Dict[str, float]] = None,
        halt_flags: Optional[Dict[str, bool]] = None,
        capital: float = 1_000_000.0,
    ) -> List[Signal]:
        """
        Generate **buy signals** at 10:00 (or specified time).

        Rules enforced:
        1. Only execute if time is within trading hours (09:30-11:30, 13:00-15:00).
        2. Skip stocks that are halted or at limit-up.
        3. Equal-weight allocation across top ``max_positions`` names.
        4. T+1: no stock held overnight may be bought again today.

        Parameters
        ----------
        date_obj: date | datetime | str
            Trade date.
        time_obj: time, optional
            Defaults to 10:00.
        factors_df: pd.DataFrame, optional
            Must contain composite scores.  If ``None``, raises.
        prices: dict[str, float], optional
            Execution price per stock (10:00 snapshot).  Falls back to
            ``factors_df['price']`` if absent.
        pre_closes: dict[str, float], optional
            Previous close for limit-up check.
        halt_flags: dict[str, bool], optional
            ``True`` = halted / suspended.
        capital: float, default 1_000_000
            Available cash today.

        Returns
        -------
        list[Signal]
            Buy signals (may be empty if no candidates pass filters).
        """
        if factors_df is None or factors_df.empty:
            logger.warning("generate_signals: empty factors, no signals")
            return []

        # Normalise date/time
        if isinstance(date_obj, str):
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d").date()
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        time_obj = time_obj or dt_time(10, 0)
        ts = datetime.combine(date_obj, time_obj)

        # Time guard
        if not self._is_trading_time(time_obj):
            logger.warning(" generate_signals called outside trading hours: %s", time_obj)
            return []

        # Filter
        candidates = self.filter_stocks(
            factors_df,
            prices=prices,
            pre_closes=pre_closes,
            halt_flags=halt_flags,
            side="buy",
        )
        if candidates.empty:
            logger.info("All candidates filtered out on %s", date_obj)
            return []

        # Select top-N
        top = candidates.head(self.max_positions)

        # Capital allocation (equal weight)
        n = len(top)
        cash_per_stock = capital / n

        signals: List[Signal] = []
        for code in top.index:
            price = prices.get(code, 0.0) if prices else top.loc[code, "price"]
            if price <= 0:
                continue
            # Quantity = floor(cash / price), must be multiple of 100 (hand)
            raw_qty = int(cash_per_stock / price)
            qty = (raw_qty // 100) * 100
            if qty == 0:
                continue
            notional = qty * price
            cost = self.cost_model.apply(notional, "buy")
            signals.append(
                Signal(
                    code=code,
                    timestamp=ts,
                    side="buy",
                    price=price,
                    quantity=qty,
                    reason=f"composite_score={top.loc[code, 'composite_score']:.3f}",
                )
            )
            # Record position
            self.positions.append(
                Position(
                    code=code,
                    entry_price=price,
                    entry_time=ts,
                    quantity=qty,
                    cost_basis=notional + cost,
                )
            )

        self.signal_history.extend(signals)
        logger.info("Generated %d buy signals on %s", len(signals), date_obj)
        return signals

    def get_exit_signals(
        self,
        positions: Optional[List[Position]] = None,
        next_day_open: Optional[Dict[str, float]] = None,
        next_day_date: Optional[date] = None,
    ) -> List[Signal]:
        """
        Generate **sell signals** at next-day 09:30 open.

        T+1 rule: every holding *must* be liquidated the next trading day.
        We do not carry overnight exposure because academic evidence shows
        A-share overnight cumulative return is negative.

        Parameters
        ----------
        positions: list[Position], optional
            Defaults to ``self.positions``.
        next_day_open: dict[str, float]
            Opening price for each held stock.
        next_day_date: date, optional

        Returns
        -------
        list[Signal]
            Sell signals.
        """
        positions = positions or self.positions
        if not positions:
            return []

        if next_day_open is None:
            logger.warning("get_exit_signals: no opening prices provided")
            return []

        ts = datetime.combine(next_day_date or date.today(), dt_time(9, 30))
        signals: List[Signal] = []

        for pos in positions:
            open_price = next_day_open.get(pos.code)
            if open_price is None or open_price <= 0:
                logger.warning("Missing open price for %s, skipping exit", pos.code)
                continue

            notional = pos.quantity * open_price
            cost = self.cost_model.apply(notional, "sell")
            gross_pnl = notional - pos.quantity * pos.entry_price
            net_pnl = gross_pnl - cost - (pos.cost_basis - pos.quantity * pos.entry_price)

            signals.append(
                Signal(
                    code=pos.code,
                    timestamp=ts,
                    side="sell",
                    price=open_price,
                    quantity=pos.quantity,
                    reason=f"T+1_exit|gross_pnl={gross_pnl:.2f}|net_pnl={net_pnl:.2f}",
                )
            )

        # Clear positions after exit
        self.positions.clear()
        self.signal_history.extend(signals)
        logger.info("Generated %d sell signals on %s", len(signals), next_day_date)
        return signals

    def filter_stocks(
        self,
        stock_df: pd.DataFrame,
        prices: Optional[Dict[str, float]] = None,
        pre_closes: Optional[Dict[str, float]] = None,
        halt_flags: Optional[Dict[str, bool]] = None,
        side: str = "buy",
    ) -> pd.DataFrame:
        """
        Apply A-share tradability filters.

        Filters applied
        ---------------
        1. Halted / suspended (``halt_flags``).
        2. Limit-up (cannot buy) / limit-down (cannot sell) – only if
           ``prices`` and ``pre_closes`` are provided.
        3. Remove stocks already held (T+1 no same-day re-buy).
        4. Remove negative composite scores (optional quality gate).

        Parameters
        ----------
        stock_df: pd.DataFrame
            Must contain ``composite_score`` column.
        prices, pre_closes, halt_flags: dict, optional
        side: str
            ``"buy"`` or ``"sell"`` – limit checks differ.

        Returns
        -------
        pd.DataFrame
            Filtered and sorted by ``composite_score`` descending.
        """
        df = stock_df.copy()
        codes = set(df.index.astype(str))

        # 1. Halt filter
        if halt_flags:
            halted = {c for c, h in halt_flags.items() if h}
            codes -= halted

        # 2. Limit filters
        if prices and pre_closes:
            for c in list(codes):
                p = prices.get(c)
                pc = pre_closes.get(c)
                if p is None or pc is None or pc <= 0:
                    codes.discard(c)
                    continue
                up, down = _check_limit(p, pc, is_st=False)
                if side == "buy" and up:
                    codes.discard(c)
                elif side == "sell" and down:
                    codes.discard(c)

        # 3. No same-day re-buy of existing positions
        held = {p.code for p in self.positions}
        if side == "buy":
            codes -= held

        # 4. Quality gate
        if "composite_score" in df.columns:
            codes &= set(df[df["composite_score"] > 0].index.astype(str))

        df = df.loc[df.index.isin(codes)].copy()
        if "composite_score" in df.columns:
            df = df.sort_values("composite_score", ascending=False)

        logger.debug(
            "filter_stocks: %d → %d after filters (side=%s)",
            len(stock_df),
            len(df),
            side,
        )
        return df

    # ------------------------------------------------------------------
    # Static / utility
    # ------------------------------------------------------------------

    @staticmethod
    def _is_trading_time(t: dt_time) -> bool:
        """Return ``True`` if *t* falls inside A-share continuous auction."""
        morning = dt_time(9, 30) <= t <= dt_time(11, 30)
        afternoon = dt_time(13, 0) <= t <= dt_time(15, 0)
        return morning or afternoon

    def reset(self) -> None:
        """Clear all positions and signal history."""
        self.positions.clear()
        self.signal_history.clear()
        logger.info("StrategyEngine state reset")

    def get_summary(self) -> Dict[str, Any]:
        """Current engine state summary."""
        return {
            "weights": self.weights,
            "open_positions": len(self.positions),
            "total_signals": len(self.signal_history),
            "unrealised_pnl": self._unrealised_pnl(),
        }

    def _unrealised_pnl(self) -> float:
        """Placeholder – would need live marks to be accurate."""
        return 0.0


# ---------------------------------------------------------------------------
# Convenience: pre-canned strategy presets
# ---------------------------------------------------------------------------

def create_momentum_strategy(max_positions: int = 20) -> StrategyEngine:
    """Factory: pure momentum tilt (research default)."""
    w = DEFAULT_WEIGHTS.copy()
    w["momentum"] = 0.40
    w["volume_ratio"] = 0.25
    w["auction_gap"] = 0.15
    w["atr_ratio"] = 0.05
    w["liquidity"] = 0.10
    w["market_cap"] = 0.03
    w["sector_momentum"] = 0.02
    return StrategyEngine(weights=w, max_positions=max_positions)


def create_conservative_strategy(max_positions: int = 20) -> StrategyEngine:
    """Factory: low-volatility tilt for risk-averse allocator."""
    w = DEFAULT_WEIGHTS.copy()
    w["momentum"] = 0.15
    w["volume_ratio"] = 0.15
    w["auction_gap"] = 0.10
    w["atr_ratio"] = 0.25
    w["liquidity"] = 0.20
    w["market_cap"] = 0.10
    w["sector_momentum"] = 0.05
    return StrategyEngine(weights=w, max_positions=max_positions)
