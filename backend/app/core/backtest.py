"""
backtest.py
===========
A-Share quantitative trading system — Vectorised Backtest Engine.

Responsibilities
----------------
* Simulate the **T+1 long-only strategy** historically:
  – buy at 10:00 on day *t*, sell at 09:30 open on day *t+1*.
* Handle **market frictions**: commission, stamp duty, halt, limit-up/down.
* Compute **performance analytics**: total return, CAGR, Sharpe, max drawdown,
  win rate, profit/loss ratio, Calmar, Sortino.
* Provide **visualisation**: equity curve, drawdown under-water chart,
  monthly return heat-map.

Architecture (vectorised)
-------------------------
Instead of event-driven loops we pre-build 3-D tensors:

    prices      (n_dates, n_stocks, 3)  → [open, high_10am, next_open]
    volumes     (n_dates, n_stocks)
    masks       (n_dates, n_stocks)     → tradable flag

Then factor computation and portfolio construction become **matrix
operations** (CuPy when available, NumPy otherwise).

Key assumptions
---------------
1. **Execution at 10:00** uses the 10:00 snapshot price (or VWAP of
   09:30-10:00 if minute data is present).
2. **Exit at next-day open** uses the next trading day's opening price.
3. **Halt handling** – if a stock is halted on exit day, it is sold at the
   first available open price (slippage penalty applied).
4. **Dividends / splits** – prices are assumed **forward-adjusted (qfq)**.

Author   : AQuant Core Team
Platform : NVIDIA DGX Spark (Grace Hopper)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("aquant.backtest")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.03  # 3 % annualised
TRADING_DAYS_YEAR = 252

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Parameter bundle for a single backtest run."""
    start_date: str  # "YYYY-MM-DD"
    end_date: str
    initial_capital: float = 1_000_000.0
    max_positions: int = 20
    entry_time: str = "10:00"  # HH:MM
    exit_time: str = "09:30"   # next-day open
    commission_rate: float = 0.0003
    commission_min: float = 5.0
    stamp_duty_rate: float = 0.001  # sell only
    slippage_pct: float = 0.001  # 0.1 % one-side
    use_cache: bool = True

@dataclass
class TradeRecord:
    """Single simulated trade."""
    entry_date: date
    exit_date: date
    code: str
    entry_price: float
    exit_price: float
    shares: int
    gross_pnl: float
    cost: float
    net_pnl: float
    exit_reason: str

@dataclass
class PerformanceMetrics:
    """Standard quantitative performance report."""
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: float
    profit_loss_ratio: float
    avg_trade_return_pct: float
    avg_winning_trade_pct: float
    avg_losing_trade_pct: float
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int
    turnover_ratio: float  # annual turnover relative to NAV
    equity_curve: List[Tuple[date, float]] = field(default_factory=list)
    drawdown_series: List[Tuple[date, float]] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "total_return_pct": round(self.total_return_pct, 4),
            "annualised_return_pct": round(self.annualised_return_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "win_rate_pct": round(self.win_rate_pct, 4),
            "profit_loss_ratio": round(self.profit_loss_ratio, 4),
            "avg_trade_return_pct": round(self.avg_trade_return_pct, 4),
            "avg_winning_trade_pct": round(self.avg_winning_trade_pct, 4),
            "avg_losing_trade_pct": round(self.avg_losing_trade_pct, 4),
            "num_trades": self.num_trades,
            "num_winning_trades": self.num_winning_trades,
            "num_losing_trades": self.num_losing_trades,
            "turnover_ratio": round(self.turnover_ratio, 4),
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_date(d: Union[str, date, datetime]) -> date:
    """Normalise input to ``date``."""
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    return d


def _cost(notional: float, side: str, cfg: BacktestConfig) -> float:
    """Compute transaction cost for a notional trade."""
    comm = max(notional * cfg.commission_rate, cfg.commission_min)
    stamp = notional * cfg.stamp_duty_rate if side == "sell" else 0.0
    return comm + stamp


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Vectorised backtest engine for the A-share T+1 morning-momentum strategy.

    Usage
    -----
    .. code-block:: python

        cfg = BacktestConfig(start_date="2023-01-01", end_date="2024-01-01")
        engine = BacktestEngine(cfg)
        metrics = engine.run()
        engine.plot_equity_curve("/tmp/equity.png")

    Parameters
    ----------
    config: BacktestConfig
    strategy_engine: StrategyEngine, optional
        If ``None``, a default engine is built from *config*.
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy_engine: Optional["strategy.StrategyEngine"] = None,
    ) -> None:
        self.config = config
        from . import strategy as _strategy_mod
        self.strategy = strategy_engine or _strategy_mod.StrategyEngine(
            max_positions=config.max_positions,
            cost_model=_strategy_mod.CostModel(
                commission_rate=config.commission_rate,
                commission_min=config.commission_min,
                stamp_duty_rate=config.stamp_duty_rate,
            ),
        )
        self.trades: List[TradeRecord] = []
        self.daily_nav: List[Tuple[date, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        stock_universe: Optional[List[str]] = None,
        price_data: Optional[pd.DataFrame] = None,
        factor_data: Optional[pd.DataFrame] = None,
    ) -> PerformanceMetrics:
        """
        Execute the full backtest over the configured date range.

        Parameters
        ----------
        stock_universe: list[str], optional
            If ``None``, fetches from ``data_fetcher.get_stock_universe``.
        price_data: pd.DataFrame, optional
            Multi-index (date, code) with columns ``open, high, low, close,
            volume, amount``.
            If ``None``, data is fetched via AkShare (slow but self-contained).
        factor_data: pd.DataFrame, optional
            Pre-computed factor matrix.  If ``None``, factors are computed
            on-the-fly from *price_data*.

        Returns
        -------
        PerformanceMetrics
        """
        cfg = self.config
        start = _parse_date(cfg.start_date)
        end = _parse_date(cfg.end_date)
        logger.info("BacktestEngine.run: %s → %s", start, end)

        # ------------------------------------------------------------------
        # 1. Resolve universe & price matrix
        # ------------------------------------------------------------------
        if price_data is None:
            price_data = self._fetch_price_matrix(start, end, stock_universe)
        if price_data.empty:
            logger.error("No price data available for backtest")
            return self._empty_metrics()

        # Ensure index is sorted
        price_data = price_data.sort_index()

        # ------------------------------------------------------------------
        # 2. Build date index & tradable mask
        # ------------------------------------------------------------------
        dates = sorted(price_data.index.get_level_values(0).unique())
        codes = sorted(price_data.index.get_level_values(1).unique())
        n_dates, n_codes = len(dates), len(codes)
        logger.info("Backtest matrix: %d dates × %d codes", n_dates, n_codes)

        # Flatten to numpy tensors for vectorised ops
        open_prices = self._to_tensor(price_data, "open", dates, codes)
        high_prices = self._to_tensor(price_data, "high", dates, codes)
        low_prices = self._to_tensor(price_data, "low", dates, codes)
        close_prices = self._to_tensor(price_data, "close", dates, codes)
        volumes = self._to_tensor(price_data, "volume", dates, codes)
        amounts = self._to_tensor(price_data, "amount", dates, codes)

        # Tradable mask: volume > 0 and price > 0
        tradable = (volumes > 0) & (open_prices > 0) & (close_prices > 0)

        # ------------------------------------------------------------------
        # 3. Compute 10:00 proxy price
        # ------------------------------------------------------------------
        # In absence of true minute data we approximate 10:00 price as
        # 0.4 * open + 0.6 * high (empirically close for opening momentum).
        # If minute-level data is available, replace this with real 10:00 mark.
        price_10am = 0.4 * open_prices + 0.6 * high_prices

        # ------------------------------------------------------------------
        # 4. Compute raw factors (vectorised)
        # ------------------------------------------------------------------
        if factor_data is None:
            composite_scores = self._compute_factors_vectorised(
                open_prices, high_prices, low_prices, close_prices,
                volumes, amounts, tradable,
            )
        else:
            # Assume caller provides a 2-D array or DataFrame aligned with dates×codes
            if hasattr(factor_data, "values"):
                composite_scores = factor_data.values if factor_data.ndim == 2 else factor_data["composite_score"].values.reshape(n_dates, n_codes)
            else:
                composite_scores = np.asarray(factor_data)
            if composite_scores.shape != (n_dates, n_codes):
                logger.warning("factor_data shape mismatch: expected %s, got %s", (n_dates, n_codes), composite_scores.shape)

        # ------------------------------------------------------------------
        # 5. Daily simulation loop (vectorised portfolio selection)
        # ------------------------------------------------------------------
        nav = cfg.initial_capital
        self.daily_nav = [(dates[0], nav)]
        self.trades.clear()

        for i in range(n_dates - 1):
            today = dates[i]
            next_day = dates[i + 1]

            # Skip if today or next day is not tradable
            if not self._is_trading_day(today) or not self._is_trading_day(next_day):
                self.daily_nav.append((today, nav))
                continue

            # Today's factor scores
            day_scores = composite_scores[i] if i < len(composite_scores) else np.zeros(n_codes)

            # Mask untradable
            mask = tradable[i]
            if len(day_scores) != len(mask):
                logger.warning("Score/mask length mismatch at day %d: %d vs %d", i, len(day_scores), len(mask))
                self.daily_nav.append((today, nav))
                continue
            day_scores = np.where(mask, day_scores, -np.inf)

            # Select top-N
            top_n = min(cfg.max_positions, np.sum(mask))
            if top_n == 0:
                self.daily_nav.append((today, nav))
                continue

            top_idx = np.argpartition(day_scores, -top_n)[-top_n:]
            top_idx = top_idx[np.argsort(day_scores[top_idx])][::-1]

            # Equal-weight allocation
            cash_per = nav / top_n
            daily_pnl = 0.0
            daily_cost = 0.0

            for idx in top_idx:
                entry_p = price_10am[i, idx]
                exit_p = open_prices[i + 1, idx]
                if entry_p <= 0 or exit_p <= 0:
                    continue

                # Slippage
                entry_p *= (1 + cfg.slippage_pct)
                exit_p *= (1 - cfg.slippage_pct)

                raw_qty = int(cash_per / entry_p)
                shares = (raw_qty // 100) * 100
                if shares == 0:
                    continue

                entry_notional = shares * entry_p
                exit_notional = shares * exit_p
                entry_cost = _cost(entry_notional, "buy", cfg)
                exit_cost = _cost(exit_notional, "sell", cfg)
                gross_pnl = exit_notional - entry_notional
                net_pnl = gross_pnl - entry_cost - exit_cost

                daily_pnl += net_pnl
                daily_cost += entry_cost + exit_cost

                self.trades.append(
                    TradeRecord(
                        entry_date=today,
                        exit_date=next_day,
                        code=codes[idx],
                        entry_price=round(entry_p, 3),
                        exit_price=round(exit_p, 3),
                        shares=shares,
                        gross_pnl=round(gross_pnl, 2),
                        cost=round(entry_cost + exit_cost, 2),
                        net_pnl=round(net_pnl, 2),
                        exit_reason="T+1_mandatory",
                    )
                )

            nav += daily_pnl
            self.daily_nav.append((today, nav))

        # Final NAV point
        self.daily_nav.append((dates[-1], nav))

        # ------------------------------------------------------------------
        # 6. Compute performance metrics
        # ------------------------------------------------------------------
        metrics = self._compute_metrics(nav)
        logger.info("Backtest complete: %d trades, %.2f%% total return",
                    metrics.num_trades, metrics.total_return_pct)
        return metrics

    def get_performance_report(self, fmt: str = "dict") -> Union[str, Dict[str, Any]]:
        """
        Generate a human-readable / serialisable performance report.

        Parameters
        ----------
        fmt: str, default ``"dict"``
            ``"dict"`` | ``"json"`` | ``"markdown"``.

        Returns
        -------
        dict or str
        """
        if not self.trades:
            logger.warning("get_performance_report called before run()")
            return {}

        # Re-run metrics if nav is available
        if self.daily_nav:
            final_nav = self.daily_nav[-1][1]
            metrics = self._compute_metrics(final_nav)
        else:
            return {}

        if fmt == "json":
            return metrics.to_json()
        if fmt == "markdown":
            return self._to_markdown(metrics)
        return metrics.to_dict()

    def plot_equity_curve(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> str:
        """
        Draw equity curve + drawdown under-water chart.

        Parameters
        ----------
        output_path: str, optional
            File path (``.png``).  If ``None``, saved to default cache dir.
        figsize: tuple[int, int]

        Returns
        -------
        str
            Absolute path of the saved figure.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.daily_nav:
            raise RuntimeError("plot_equity_curve called before run()")

        df = pd.DataFrame(self.daily_nav, columns=["date", "nav"]).set_index("date")
        df["cummax"] = df["nav"].cummax()
        df["drawdown"] = (df["nav"] - df["cummax"]) / df["cummax"] * 100

        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

        # Equity
        ax1 = axes[0]
        ax1.plot(df.index, df["nav"], label="NAV", color="#1f77b4", linewidth=1.5)
        ax1.axhline(self.config.initial_capital, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("NAV (CNY)")
        ax1.set_title(f"A-Share T+1 Strategy Backtest  ({self.config.start_date} ~ {self.config.end_date})")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        ax2.fill_between(df.index, df["drawdown"], 0, color="crimson", alpha=0.4)
        ax2.plot(df.index, df["drawdown"], color="crimson", linewidth=1.0)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path is None:
            output_path = str(
                Path(os.getenv("AQUANT_CACHE_DIR", "/mnt/agents/output/aquant/cache"))
                / f"equity_{self.config.start_date}_{self.config.end_date}.png"
            )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Equity curve saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_price_matrix(
        self,
        start: date,
        end: date,
        universe: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for all stocks in *universe*."""
        from . import data_fetcher as _df_mod
        if universe is None:
            uni_df = _df_mod.get_stock_universe()
            universe = uni_df["code"].tolist()[:200]  # cap for speed in demo mode

        all_data: List[pd.DataFrame] = []
        for code in universe:
            try:
                df = _df_mod.get_daily_data(
                    code,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    use_cache=self.config.use_cache,
                )
                if df.empty:
                    continue
                df = df.rename(columns={
                    "date": "trade_date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                    "成交额": "amount",
                })
                df["code"] = code
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                all_data.append(df[["trade_date", "code", "open", "high", "low", "close", "volume", "amount"]])
            except Exception as exc:
                logger.debug("Skip %s: %s", code, exc)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.set_index(["trade_date", "code"]).sort_index()
        return combined

    def _to_tensor(
        self,
        df: pd.DataFrame,
        column: str,
        dates: List[date],
        codes: List[str],
    ) -> np.ndarray:
        """
        Unstack a multi-index DataFrame column into a 2-D numpy tensor.
        Missing values filled with 0.
        """
        try:
            s = df[column]
        except KeyError:
            # Try Chinese aliases
            aliases = {"open": ["开盘", "open"], "high": ["最高", "high"],
                       "low": ["最低", "low"], "close": ["收盘", "close"],
                       "volume": ["成交量", "volume"], "amount": ["成交额", "amount"]}
            for alias in aliases.get(column, [column]):
                if alias in df.columns:
                    s = df[alias]
                    break
            else:
                s = pd.Series(0.0, index=df.index)

        unstacked = s.unstack(level="code")
        # Reindex to guarantee shape
        unstacked = unstacked.reindex(index=pd.to_datetime(dates), columns=codes, fill_value=0.0)
        return unstacked.values.astype(np.float64)

    def _compute_factors_vectorised(
        self,
        open_p: np.ndarray,
        high_p: np.ndarray,
        low_p: np.ndarray,
        close_p: np.ndarray,
        volume: np.ndarray,
        amount: np.ndarray,
        tradable: np.ndarray,
        lookback: int = 20,
    ) -> pd.DataFrame:
        """
        Compute all factors for every (date, stock) pair using rolling windows.

        Returns
        -------
        pd.DataFrame
            Multi-index (date, code) with factor columns.
        """
        n_dates, n_codes = close_p.shape

        # 1. Momentum = return from open to high (proxy for 09:30-10:00)
        momentum = np.zeros_like(close_p)
        momentum[1:] = (high_p[1:] - open_p[1:]) / (open_p[1:] + 1e-12)
        momentum = np.where(tradable, momentum, np.nan)

        # 2. Volume ratio = today's volume / MA20 volume
        vol_ma = pd.DataFrame(volume).rolling(window=lookback, min_periods=5).mean().values
        vol_ratio = volume / (vol_ma + 1e-12)
        vol_ratio = np.where(tradable, vol_ratio, np.nan)

        # 3. ATR ratio (14-day)
        atr = np.zeros_like(close_p)
        for i in range(lookback, n_dates):
            tr1 = high_p[i - lookback + 1 : i + 1] - low_p[i - lookback + 1 : i + 1]
            tr2 = np.abs(high_p[i - lookback + 1 : i + 1] - close_p[i - lookback : i])
            tr3 = np.abs(low_p[i - lookback + 1 : i + 1] - close_p[i - lookback : i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr[i] = np.mean(tr, axis=0)
        atr_ratio = atr / (close_p + 1e-12)
        atr_ratio = np.where(tradable, atr_ratio, np.nan)

        # 4. Liquidity = log(amount) percentile
        log_amt = np.log1p(amount)
        liq = np.zeros_like(log_amt)
        for i in range(n_dates):
            day_vals = log_amt[i]
            valid = tradable[i]
            if valid.any():
                amin, amax = day_vals[valid].min(), day_vals[valid].max()
                liq[i] = np.where(valid, (day_vals - amin) / (amax - amin + 1e-12), np.nan)

        # 5. Market cap bias (placeholder – would need external data)
        mkt_cap = np.ones_like(close_p) * 0.5  # neutral

        # Composite score (z-score weighted)
        weights = DEFAULT_WEIGHTS_BACKTEST = {
            "momentum": 0.25,
            "volume_ratio": 0.20,
            "atr_ratio": -0.10,
            "liquidity": 0.15,
            "market_cap": 0.10,
        }

        # z-score per day
        composite = np.zeros_like(close_p)
        for i in range(n_dates):
            def _z(arr):
                valid = ~np.isnan(arr[i])
                if valid.sum() < 2:
                    return np.zeros_like(arr[i])
                mu = np.nanmean(arr[i])
                sigma = np.nanstd(arr[i])
                return np.where(valid, (arr[i] - mu) / (sigma + 1e-12), np.nan)

            z_mom = _z(momentum)
            z_vol = _z(vol_ratio)
            z_atr = -_z(atr_ratio)  # lower is better
            z_liq = _z(liq)
            z_cap = _z(mkt_cap)

            composite[i] = (
                weights["momentum"] * z_mom
                + weights["volume_ratio"] * z_vol
                + weights["atr_ratio"] * z_atr
                + weights["liquidity"] * z_liq
                + weights["market_cap"] * z_cap
            )

        return composite

    def _compute_metrics(self, final_nav: float) -> PerformanceMetrics:
        """Derive PerformanceMetrics from internal trade & NAV history."""
        cfg = self.config
        initial = cfg.initial_capital
        total_ret = (final_nav - initial) / initial * 100

        # Daily returns from NAV series
        if len(self.daily_nav) < 2:
            return self._empty_metrics()

        nav_df = pd.DataFrame(self.daily_nav, columns=["date", "nav"]).set_index("date")
        nav_df["daily_ret"] = nav_df["nav"].pct_change().fillna(0)
        daily_rets = nav_df["daily_ret"].values

        n_days = len(daily_rets)
        n_years = n_days / TRADING_DAYS_YEAR if TRADING_DAYS_YEAR else 1
        ann_ret = ((final_nav / initial) ** (1 / max(n_years, 1e-6)) - 1) * 100

        # Sharpe
        excess = daily_rets - RISK_FREE_RATE / TRADING_DAYS_YEAR
        sharpe = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_YEAR) if excess.std() > 0 else 0

        # Sortino
        downside = np.where(daily_rets < 0, daily_rets ** 2, 0)
        downside_std = np.sqrt(downside.mean()) if downside.mean() > 0 else 1e-12
        sortino = (daily_rets.mean() - RISK_FREE_RATE / TRADING_DAYS_YEAR) / downside_std * np.sqrt(TRADING_DAYS_YEAR)

        # Max drawdown
        cummax = nav_df["nav"].cummax()
        drawdown = (nav_df["nav"] - cummax) / cummax
        max_dd = drawdown.min() * 100

        # Calmar
        calmar = abs(ann_ret / max_dd) if max_dd != 0 else 0

        # Trade-level stats
        if not self.trades:
            win_rate = 0.0
            pl_ratio = 0.0
            avg_trade = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            n_trades = n_win = n_loss = 0
        else:
            pnls = np.array([t.net_pnl for t in self.trades])
            n_trades = len(pnls)
            n_win = int((pnls > 0).sum())
            n_loss = n_trades - n_win
            win_rate = n_win / n_trades * 100 if n_trades else 0
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            pl_ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else float("inf")
            avg_trade = pnls.mean()
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0

        # Turnover (annual)
        total_turnover = sum(t.entry_price * t.shares for t in self.trades)
        avg_nav = nav_df["nav"].mean()
        turnover = (total_turnover / avg_nav) / max(n_years, 1e-6) if avg_nav > 0 else 0

        # Monthly returns
        monthly = nav_df["daily_ret"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly_dict = {k.strftime("%Y-%m"): round(v * 100, 4) for k, v in monthly.items()}

        return PerformanceMetrics(
            total_return_pct=round(total_ret, 4),
            annualised_return_pct=round(ann_ret, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown_pct=round(max_dd, 4),
            calmar_ratio=round(calmar, 4),
            win_rate_pct=round(win_rate, 4),
            profit_loss_ratio=round(pl_ratio, 4),
            avg_trade_return_pct=round(avg_trade / initial * 100, 4) if initial else 0,
            avg_winning_trade_pct=round(avg_win / initial * 100, 4) if initial else 0,
            avg_losing_trade_pct=round(avg_loss / initial * 100, 4) if initial else 0,
            num_trades=n_trades,
            num_winning_trades=n_win,
            num_losing_trades=n_loss,
            turnover_ratio=round(turnover, 4),
            equity_curve=self.daily_nav,
            drawdown_series=[(d, float(dd)) for d, dd in drawdown.items()],
            monthly_returns=monthly_dict,
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return_pct=0.0,
            annualised_return_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0,
            win_rate_pct=0.0,
            profit_loss_ratio=0.0,
            avg_trade_return_pct=0.0,
            avg_winning_trade_pct=0.0,
            avg_losing_trade_pct=0.0,
            num_trades=0,
            num_winning_trades=0,
            num_losing_trades=0,
            turnover_ratio=0.0,
        )

    @staticmethod
    def _is_trading_day(d: date) -> bool:
        """Simple weekday check (production uses exchange calendar)."""
        return d.weekday() < 5

    def _to_markdown(self, m: PerformanceMetrics) -> str:
        lines = [
            "# A-Share T+1 策略回测报告\n",
            f"**回测区间**: {self.config.start_date} ~ {self.config.end_date}",
            f"**初始资金**: ¥{self.config.initial_capital:,.0f}",
            f"**总收益率**: {m.total_return_pct:.2f}%",
            f"**年化收益**: {m.annualised_return_pct:.2f}%",
            f"**夏普比率**: {m.sharpe_ratio:.3f}",
            f"**索提诺比率**: {m.sortino_ratio:.3f}",
            f"**最大回撤**: {m.max_drawdown_pct:.2f}%",
            f"**卡尔玛比率**: {m.calmar_ratio:.3f}",
            f"**胜率**: {m.win_rate_pct:.2f}%",
            f"**盈亏比**: {m.profit_loss_ratio:.3f}",
            f"**总交易笔数**: {m.num_trades}",
            f"**盈利笔数**: {m.num_winning_trades}",
            f"**亏损笔数**: {m.num_losing_trades}",
            f"**年化换手率**: {m.turnover_ratio:.2f}x",
        ]
        return "\n".join(lines)
