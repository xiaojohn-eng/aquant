"""
risk_engine.py
==============
Risk management engine for AQuant T+1 strategy.

Implements:
- Daily loss limit (default -3.0%)
- Consecutive loss day circuit breaker (default 5 days)
- Per-stock position limit (default 10.0% of capital)
- Abnormal price movement filter (>7% reject)
- Volatility regime detection (pause in high-vol)

Author  : AQuant Risk Team
Version : v4.4 (integrated with backtest + live trading)
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """Risk management parameters."""

    # Daily P&L controls
    daily_loss_limit_pct: float = -3.0      # Stop trading if daily P&L < -3%
    daily_gain_limit_pct: float = 5.0       # Take profit if daily P&L > +5%

    # Circuit breaker
    consecutive_loss_days: int = 5          # Pause after N consecutive loss days
    consecutive_loss_pct: float = -5.0      # Or cumulative loss over N days

    # Position limits
    max_position_pct: float = 10.0          # No single stock > 10% of capital
    max_total_exposure_pct: float = 95.0    # Total long exposure < 95%
    min_cash_reserve_pct: float = 5.0       # Always keep 5% cash

    # Price movement filters
    max_entry_gap_pct: float = 7.0          # Reject if stock gapped > 7%
    max_intraday_move_pct: float = 10.0     # Reject if already moved > 10%

    # Volatility regime
    high_volatility_threshold: float = 0.025  # ATR/Close > 2.5% = high vol
    pause_on_high_vol: bool = True

    # Cooldown after circuit breaker (days)
    circuit_breaker_cooldown: int = 3


# ---------------------------------------------------------------------------
# Risk State
# ---------------------------------------------------------------------------

@dataclass
class RiskState:
    """Mutable risk tracking state."""

    consecutive_loss_days: int = 0
    cumulative_loss_pct: float = 0.0
    circuit_breaker_active: bool = False
    circuit_breaker_until: Optional[date] = None
    daily_pnl_history: List[Tuple[date, float]] = field(default_factory=list)
    last_trade_date: Optional[date] = None


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Central risk controller for the T+1 strategy.

    Usage::

        risk = RiskManager(RiskConfig(daily_loss_limit_pct=-3.0))

        # Before entry
        if not risk.check_entry_allowed(nav, daily_pnl, today):
            return []  # skip trading

        # After exit — update P&L
        risk.update_pnl(today, daily_pnl)

        # Check if circuit breaker triggered
        if risk.is_circuit_breaker_active(today):
            logger.warning("Circuit breaker active — all trading paused")
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        self.state = RiskState()
        logger.info(
            "RiskManager initialized: daily_loss=%.1f%%, circuit_after=%d days, "
            "max_position=%.1f%%",
            self.config.daily_loss_limit_pct,
            self.config.consecutive_loss_days,
            self.config.max_position_pct,
        )

    # ------------------------------------------------------------------
    # Entry checks
    # ------------------------------------------------------------------

    def check_entry_allowed(
        self,
        nav: float,
        daily_pnl_pct: float,
        trade_date: date,
        stock_code: Optional[str] = None,
        stock_gap_pct: Optional[float] = None,
        stock_intraday_pct: Optional[float] = None,
        atr_ratio: Optional[float] = None,
    ) -> bool:
        """
        Comprehensive pre-trade risk check.

        Returns ``True`` only if ALL conditions pass.
        """
        # 1. Circuit breaker
        if self.is_circuit_breaker_active(trade_date):
            logger.warning("[%s] ENTRY REJECTED: circuit breaker active", trade_date)
            return False

        # 2. Daily loss limit
        if daily_pnl_pct <= self.config.daily_loss_limit_pct:
            logger.warning(
                "[%s] ENTRY REJECTED: daily PnL %.2f%% <= limit %.1f%%",
                trade_date, daily_pnl_pct, self.config.daily_loss_limit_pct,
            )
            return False

        # 3. High volatility regime
        if self.config.pause_on_high_vol and atr_ratio is not None:
            if atr_ratio > self.config.high_volatility_threshold:
                logger.warning(
                    "[%s] ENTRY REJECTED: high volatility (ATR ratio %.2f%%)",
                    trade_date, atr_ratio * 100,
                )
                return False

        # 4. Per-stock price movement filters
        if stock_gap_pct is not None and abs(stock_gap_pct) > self.config.max_entry_gap_pct:
            logger.warning(
                "[%s %s] ENTRY REJECTED: gap %.2f%% > limit %.1f%%",
                trade_date, stock_code, stock_gap_pct, self.config.max_entry_gap_pct,
            )
            return False

        if stock_intraday_pct is not None and abs(stock_intraday_pct) > self.config.max_intraday_move_pct:
            logger.warning(
                "[%s %s] ENTRY REJECTED: intraday move %.2f%% > limit %.1f%%",
                trade_date, stock_code, stock_intraday_pct, self.config.max_intraday_move_pct,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def check_position_size(
        self,
        proposed_notional: float,
        current_position_notional: float,
        nav: float,
    ) -> Tuple[bool, float]:
        """
        Check if proposed position size is within limits.

        Returns (allowed, adjusted_notional).
        """
        max_single = nav * self.config.max_position_pct / 100.0
        max_total = nav * self.config.max_total_exposure_pct / 100.0

        if current_position_notional + proposed_notional > max_total:
            adjusted = max(0.0, max_total - current_position_notional)
            logger.warning(
                "Position capped: %.0f → %.0f (total exposure limit)",
                proposed_notional, adjusted,
            )
            return adjusted > 0, adjusted

        if proposed_notional > max_single:
            logger.warning(
                "Position capped: %.0f → %.0f (single stock limit %.1f%%)",
                proposed_notional, max_single, self.config.max_position_pct,
            )
            return True, max_single

        return True, proposed_notional

    # ------------------------------------------------------------------
    # P&L tracking
    # ------------------------------------------------------------------

    def update_pnl(self, trade_date: date, daily_pnl_pct: float) -> None:
        """Record daily P&L and update circuit breaker state."""
        self.state.daily_pnl_history.append((trade_date, daily_pnl_pct))
        self.state.last_trade_date = trade_date

        if daily_pnl_pct < 0:
            self.state.consecutive_loss_days += 1
            self.state.cumulative_loss_pct += daily_pnl_pct
        else:
            # Reset consecutive counter on profit
            self.state.consecutive_loss_days = 0
            self.state.cumulative_loss_pct = 0.0

        # Check circuit breaker triggers
        cb_triggered = False
        if self.state.consecutive_loss_days >= self.config.consecutive_loss_days:
            logger.error(
                "CIRCUIT BREAKER: %d consecutive loss days (limit %d)",
                self.state.consecutive_loss_days, self.config.consecutive_loss_days,
            )
            cb_triggered = True

        if self.state.cumulative_loss_pct <= self.config.consecutive_loss_pct:
            logger.error(
                "CIRCUIT BREAKER: cumulative loss %.2f%% (limit %.1f%%)",
                self.state.cumulative_loss_pct, self.config.consecutive_loss_pct,
            )
            cb_triggered = True

        if cb_triggered:
            self.state.circuit_breaker_active = True
            self.state.circuit_breaker_until = trade_date + timedelta(
                days=self.config.circuit_breaker_cooldown,
            )

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def is_circuit_breaker_active(self, trade_date: date) -> bool:
        """Check if trading is currently paused."""
        if not self.state.circuit_breaker_active:
            return False

        if self.state.circuit_breaker_until is None:
            return False

        if trade_date >= self.state.circuit_breaker_until:
            # Auto-reset after cooldown
            logger.info("Circuit breaker auto-reset on %s", trade_date)
            self.state.circuit_breaker_active = False
            self.state.circuit_breaker_until = None
            self.state.consecutive_loss_days = 0
            self.state.cumulative_loss_pct = 0.0
            return False

        return True

    def manual_reset(self) -> None:
        """Manually reset circuit breaker (admin override)."""
        self.state.circuit_breaker_active = False
        self.state.circuit_breaker_until = None
        self.state.consecutive_loss_days = 0
        self.state.cumulative_loss_pct = 0.0
        logger.info("Circuit breaker manually reset")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Return current risk status for monitoring."""
        return {
            "circuit_breaker_active": self.state.circuit_breaker_active,
            "circuit_breaker_until": self.state.circuit_breaker_until.isoformat() if self.state.circuit_breaker_until else None,
            "consecutive_loss_days": self.state.consecutive_loss_days,
            "cumulative_loss_pct": round(self.state.cumulative_loss_pct, 4),
            "daily_pnl_count": len(self.state.daily_pnl_history),
            "config": {
                "daily_loss_limit_pct": self.config.daily_loss_limit_pct,
                "consecutive_loss_days": self.config.consecutive_loss_days,
                "max_position_pct": self.config.max_position_pct,
            },
        }
