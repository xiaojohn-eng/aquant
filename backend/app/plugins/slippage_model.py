"""
Slippage Model for A-Share Small-Cap Stocks

Models realistic transaction costs including:
- Commission (0.03%)
- Stamp duty (0.1% on sell)
- Slippage (liquidity-dependent, 0.1% - 2.0%)
- Market impact (position size / daily volume ratio)

References:
- 明汯投教: 滑点与冲击成本 (2023)
- JoinQuant: 小市值策略交易成本分析 (2025)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeCost:
    """Complete trade cost breakdown for a single trade."""
    commission: float = 0.0
    stamp_duty: float = 0.0
    slippage: float = 0.0
    impact_cost: float = 0.0
    total_cost: float = 0.0
    total_cost_pct: float = 0.0


class SlippageModel:
    """
    Liquidity-aware slippage and impact cost model.
    
    Key insight from research:
    - Large-cap (沪深300): slippage ~0.1%, impact minimal
    - Mid-cap (中证500): slippage ~0.3-0.5%
    - Small-cap (<50亿): slippage ~0.5-2.0%, impact significant
    """
    
    def __init__(self,
                 commission_rate: float = 0.0003,      # 0.03%
                 stamp_duty_rate: float = 0.001,       # 0.1% (sell only)
                 min_commission: float = 5.0,          # RMB
                 slippage_base: float = 0.001,         # 0.1% base
                 slippage_volatility_factor: float = 0.5,
                 impact_coefficient: float = 1.0):
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_commission = min_commission
        self.slippage_base = slippage_base
        self.slippage_volatility_factor = slippage_volatility_factor
        self.impact_coefficient = impact_coefficient
    
    def estimate_slippage(self,
                          stock_code: str,
                          trade_amount: float,
                          avg_daily_volume: float,
                          avg_daily_amount: float,
                          atr_ratio: float,
                          market_cap: Optional[float] = None) -> float:
        """
        Estimate slippage percentage for a trade.
        
        Args:
            trade_amount: intended trade amount in RMB
            avg_daily_volume: 20-day average daily volume (shares)
            avg_daily_amount: 20-day average daily turnover (RMB)
            atr_ratio: ATR / price (volatility proxy)
            market_cap: market capitalization in RMB
        
        Returns:
            Slippage as percentage (e.g., 0.005 = 0.5%)
        """
        # Base slippage: higher for small-caps
        base = self.slippage_base
        
        # Size effect: trade size relative to daily volume
        if avg_daily_amount > 0:
            size_ratio = trade_amount / avg_daily_amount
            # Exponential penalty for large relative size
            size_penalty = np.exp(size_ratio * 5) - 1  # e.g., 10% of ADV → ~0.65% extra
        else:
            size_penalty = 0.0
        
        # Volatility effect: higher ATR → higher slippage
        vol_penalty = atr_ratio * self.slippage_volatility_factor
        
        # Market cap effect: small-cap penalty
        cap_penalty = 0.0
        if market_cap and market_cap > 0:
            if market_cap < 5e9:      # < 50亿
                cap_penalty = 0.008   # +0.8%
            elif market_cap < 2e10:   # < 200亿
                cap_penalty = 0.003   # +0.3%
            elif market_cap < 5e10:   # < 500亿
                cap_penalty = 0.001   # +0.1%
        
        total_slippage = base + size_penalty + vol_penalty + cap_penalty
        return min(total_slippage, 0.05)  # Cap at 5%
    
    def calculate_cost(self,
                       stock_code: str,
                       price: float,
                       quantity: int,
                       side: str,  # "buy" or "sell"
                       avg_daily_volume: float = 0,
                       avg_daily_amount: float = 0,
                       atr_ratio: float = 0.01,
                       market_cap: Optional[float] = None) -> TradeCost:
        """
        Calculate total transaction cost for a trade.
        
        Args:
            side: "buy" or "sell"
        """
        trade_amount = price * quantity
        
        # Commission (both sides)
        commission = max(trade_amount * self.commission_rate, self.min_commission)
        
        # Stamp duty (sell only)
        stamp_duty = trade_amount * self.stamp_duty_rate if side == "sell" else 0.0
        
        # Slippage (both sides)
        slippage_pct = self.estimate_slippage(
            stock_code, trade_amount, avg_daily_volume, avg_daily_amount,
            atr_ratio, market_cap
        )
        slippage = trade_amount * slippage_pct
        
        # Market impact (additional cost from moving the market)
        if avg_daily_amount > 0:
            participation = trade_amount / avg_daily_amount
            impact = trade_amount * (participation ** 0.5) * 0.01 * self.impact_coefficient
        else:
            impact = 0.0
        
        total = commission + stamp_duty + slippage + impact
        total_pct = (total / trade_amount * 100) if trade_amount > 0 else 0.0
        
        return TradeCost(
            commission=round(commission, 2),
            stamp_duty=round(stamp_duty, 2),
            slippage=round(slippage, 2),
            impact_cost=round(impact, 2),
            total_cost=round(total, 2),
            total_cost_pct=round(total_pct, 4),
        )
    
    def estimate_annual_cost(self,
                             capital: float,
                             turnover_per_year: float = 20.0,  # 20x for daily strategy
                             avg_slippage_pct: float = 0.005) -> Dict[str, float]:
        """
        Estimate annual transaction cost impact.
        
        Args:
            turnover_per_year: annual turnover ratio (e.g., 20x for daily 20-stock strategy)
            avg_slippage_pct: average slippage estimate
        """
        annual_turnover_amount = capital * turnover_per_year
        
        commission = annual_turnover_amount * self.commission_rate * 2  # buy + sell
        stamp_duty = annual_turnover_amount * self.stamp_duty_rate      # sell side only
        slippage = annual_turnover_amount * avg_slippage_pct * 2       # both sides
        
        total = commission + stamp_duty + slippage
        total_pct_of_capital = (total / capital * 100) if capital > 0 else 0.0
        
        return {
            "annual_turnover": annual_turnover_amount,
            "commission": round(commission, 2),
            "stamp_duty": round(stamp_duty, 2),
            "slippage": round(slippage, 2),
            "total_cost": round(total, 2),
            "total_cost_pct_of_capital": round(total_pct_of_capital, 2),
        }
