"""
recommender.py
==============
A-Share quantitative trading system — Stock Recommendation Engine.

Responsibilities
----------------
* Produce a **daily top-N recommendation list** (default 20) with
  human-readable, structured justifications.
* Generate **per-stock reasoning templates** grounded in the six factor
  families (momentum, volume, auction, volatility, liquidity, composite).
* Provide **A-share market-hour utilities** (calendar, trading-time checks,
  next-trading-day lookup).

Design philosophy
-----------------
Every recommendation must be **explainable** — the user sees not only
*which* stock but *why* it was selected.  Chinese-language templates are
used because the end-user base is Mandarin-speaking retail & institutional
investors.

Author   : AQuant Core Team
Platform : NVIDIA DGX Spark (Grace Hopper)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("aquant.recommender")

# ---------------------------------------------------------------------------
# Structured reason templates (Chinese)
# ---------------------------------------------------------------------------
REASON_TEMPLATE: Dict[str, str] = {
    "momentum": "早盘动能：开盘后涨幅{value:.2f}%，位列市场前{percentile:.1f}%",
    "volume": "量能配合：成交量为20日均量的{ratio:.2f}倍，{signal}",
    "auction": "竞价质量：开盘价高于竞价加权价{gap:.2f}%",
    "volatility": "风险指标：ATR波动率{atr:.2f}%，{assessment}",
    "liquidity": "流动性：30分钟成交额{amount:.0f}万元，{rank}",
    "composite": "综合评分：{score:.2f}分（击败{beat_pct:.1f}%的股票）",
}

# ---------------------------------------------------------------------------
# A-share trading calendar (simplified; production uses exchange API)
# ---------------------------------------------------------------------------
_TRADING_HOURS: Tuple[Tuple[dt_time, dt_time], ...] = (
    (dt_time(9, 15), dt_time(9, 25)),   # 集合竞价
    (dt_time(9, 30), dt_time(11, 30)),  # 上午连续竞价
    (dt_time(13, 0), dt_time(15, 0)),   # 下午连续竞价
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class StockRecommendation:
    """
    Single recommendation with structured human-readable reasons.

    Attributes
    ----------
    code: str
    name: str
    rank: int
        1 = best candidate.
    composite_score: float
    entry_price: float
        Suggested execution price (10:00 mark).
    reasons: list[str]
        Chinese sentences explaining the pick.
    risk_label: str
        One of ``"低风险"``, ``"中风险"``, ``"高风险"``.
    expected_hold: str
        Always ``"T+1（次日开盘卖出）"`` for this strategy.
    """
    code: str
    name: str
    rank: int
    composite_score: float
    entry_price: float
    reasons: List[str]
    risk_label: str
    expected_hold: str = "T+1（次日开盘卖出）"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class StockRecommender:
    """
    End-of-morning-scan recommendation engine.

    Workflow
    --------
    1. Fetch / refresh factor matrix (delegated to ``gpu_compute``).
    2. Run ``StrategyEngine.calculate_scores``.
    3. Filter out untradable names.
    4. For each top-N stock call ``generate_reasons``.
    5. Package into ``StockRecommendation`` objects.

    Parameters
    ----------
    strategy_engine: StrategyEngine
    top_n: int, default 20
    """

    def __init__(
        self,
        strategy_engine: Optional["strategy.StrategyEngine"] = None,
        top_n: int = 20,
    ) -> None:
        from . import strategy as _strategy_mod
        self.engine = strategy_engine or _strategy_mod.StrategyEngine()
        self.top_n = top_n
        self._calendar_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def recommend(
        self,
        date_obj: Union[date, str],
        factors_df: Optional[pd.DataFrame] = None,
        prices: Optional[Dict[str, float]] = None,
        pre_closes: Optional[Dict[str, float]] = None,
        stock_names: Optional[Dict[str, str]] = None,
        halt_flags: Optional[Dict[str, bool]] = None,
    ) -> List[StockRecommendation]:
        """
        Generate the daily top-N recommendation list.

        Parameters
        ----------
        date_obj: date or str
        factors_df: pd.DataFrame, optional
            Index = stock code, columns = raw factors + ``composite_score``.
            If ``None``, the caller must have pre-computed factors.
        prices: dict[str, float], optional
            10:00 execution price per stock.
        pre_closes: dict[str, float], optional
        stock_names: dict[str, str], optional
            Mapping code → human-readable name.
        halt_flags: dict[str, bool], optional

        Returns
        -------
        list[StockRecommendation]
            Sorted by ``rank`` ascending (best first).
        """
        if isinstance(date_obj, str):
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d").date()

        if factors_df is None or factors_df.empty:
            logger.warning("recommend: no factor data, returning empty list")
            return []

        # Ensure composite_score exists
        if "composite_score" not in factors_df.columns:
            factors_df = self.engine.calculate_scores(factors_df)

        # Filter
        filtered = self.engine.filter_stocks(
            factors_df,
            prices=prices,
            pre_closes=pre_closes,
            halt_flags=halt_flags,
            side="buy",
        )

        top = filtered.head(self.top_n)
        recommendations: List[StockRecommendation] = []

        for rank, code in enumerate(top.index, start=1):
            row = top.loc[code]
            name = (stock_names or {}).get(str(code), str(code))
            price = prices.get(str(code), 0.0) if prices else 0.0
            reasons = self.generate_reasons(str(code), row.to_dict())
            risk = self._risk_label(row)
            recommendations.append(
                StockRecommendation(
                    code=str(code),
                    name=name,
                    rank=rank,
                    composite_score=float(row.get("composite_score", 0.0)),
                    entry_price=price,
                    reasons=reasons,
                    risk_label=risk,
                )
            )

        logger.info("Generated %d recommendations for %s", len(recommendations), date_obj)
        return recommendations

    def generate_reasons(
        self,
        stock_code: str,
        factors: Dict[str, Any],
    ) -> List[str]:
        """
        Build a list of Chinese justification sentences for *one* stock.

        Parameters
        ----------
        stock_code: str
        factors: dict
            Must contain keys that match the factor families:
            ``momentum, volume_ratio, auction_gap, atr_ratio,
            liquidity, composite_score``.

        Returns
        -------
        list[str]
            Ordered reasons (most important first).
        """
        reasons: List[str] = []

        # 1. Momentum
        mom = factors.get("momentum")
        if mom is not None:
            pct = float(mom) * 100
            # Approximate percentile (will be refined by caller if full cross-section known)
            reasons.append(
                REASON_TEMPLATE["momentum"].format(value=pct, percentile=0.0)
            )

        # 2. Volume
        vr = factors.get("volume_ratio")
        if vr is not None:
            ratio = float(vr)
            sig = "放量显著，资金介入积极" if ratio > 1.5 else "量能温和" if ratio > 1.0 else "量能萎缩"
            reasons.append(REASON_TEMPLATE["volume"].format(ratio=ratio, signal=sig))

        # 3. Auction gap
        gap = factors.get("auction_gap")
        if gap is not None:
            gap_pct = float(gap) * 100
            reasons.append(REASON_TEMPLATE["auction"].format(gap=gap_pct))

        # 4. Volatility
        atr = factors.get("atr_ratio")
        if atr is not None:
            atr_pct = float(atr) * 100
            assess = "波动适中" if 0.5 <= atr_pct <= 3.0 else "波动偏大" if atr_pct > 3.0 else "波动极低"
            reasons.append(REASON_TEMPLATE["volatility"].format(atr=atr_pct, assessment=assess))

        # 5. Liquidity
        liq = factors.get("liquidity")
        amt = factors.get("amount_30min")
        if amt is not None:
            amt_wan = float(amt) / 10_000  # convert to 万元
            rk = "成交活跃" if amt_wan > 5000 else "成交一般" if amt_wan > 1000 else "成交清淡"
            reasons.append(REASON_TEMPLATE["liquidity"].format(amount=amt_wan, rank=rk))
        elif liq is not None:
            reasons.append(f"流动性评分：{float(liq):.3f}（归一化）")

        # 6. Composite (always last, summary)
        score = factors.get("composite_score")
        if score is not None:
            sc = float(score)
            # beat_pct placeholder; caller may override
            reasons.append(REASON_TEMPLATE["composite"].format(score=sc, beat_pct=0.0))

        return reasons

    @staticmethod
    def enrich_reasons_with_percentiles(
        reasons: List[str],
        factors_df: pd.DataFrame,
        stock_code: str,
    ) -> List[str]:
        """
        Replace placeholder percentiles in reason strings with real values
        computed from the full cross-section.

        Parameters
        ----------
        reasons: list[str]
        factors_df: pd.DataFrame
            Full universe factor matrix (needed for percentile calculation).
        stock_code: str

        Returns
        -------
        list[str]
            Updated reasons with real percentiles.
        """
        if stock_code not in factors_df.index:
            return reasons

        enriched: List[str] = []
        for r in reasons:
            if "早盘动能" in r and "percentile" in r:
                mom = factors_df.loc[stock_code, "momentum"]
                pct = (factors_df["momentum"] <= mom).mean() * 100
                r = re.sub(r"位列市场前[\d.]+%", f"位列市场前{pct:.1f}%", r)
            elif "综合评分" in r and "beat_pct" in r:
                score = factors_df.loc[stock_code, "composite_score"]
                beat = (factors_df["composite_score"] < score).mean() * 100
                r = re.sub(r"击败[\d.]+%的股票", f"击败{beat:.1f}%的股票", r)
            enriched.append(r)
        return enriched

    # ------------------------------------------------------------------
    # Risk label
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_label(row: pd.Series) -> str:
        """Classify risk based on ATR and momentum extremes."""
        atr = row.get("atr_ratio", 0.0)
        mom = row.get("momentum", 0.0)
        if atr > 0.05 or abs(mom) > 0.07:
            return "高风险"
        if atr > 0.03 or abs(mom) > 0.04:
            return "中风险"
        return "低风险"


# ---------------------------------------------------------------------------
# A-share market-hour utilities
# ---------------------------------------------------------------------------

def is_trading_time(
    dt: Optional[datetime] = None,
    include_auction: bool = True,
) -> bool:
    """
    Check whether *dt* falls inside A-share trading hours.

    Parameters
    ----------
    dt: datetime, optional
        Defaults to ``datetime.now()``.
    include_auction: bool, default True
        Count 09:15-09:25 call auction as trading time.

    Returns
    -------
    bool
    """
    dt = dt or datetime.now()
    t = dt.time()
    weekday = dt.weekday()
    if weekday >= 5:  # Saturday = 5, Sunday = 6
        return False

    periods = list(_TRADING_HOURS)
    if not include_auction:
        periods = periods[1:]

    for start, end in periods:
        if start <= t <= end:
            return True
    return False


def get_trading_hours() -> List[Tuple[dt_time, dt_time]]:
    """Return A-share trading hour segments."""
    return list(_TRADING_HOURS)


def next_trading_day(
    from_date: Optional[Union[date, str]] = None,
    calendar_df: Optional[pd.DataFrame] = None,
) -> date:
    """
    Return the next trading day after *from_date*.

    Parameters
    ----------
    from_date: date or str, optional
    calendar_df: pd.DataFrame, optional
        Trading calendar from ``data_fetcher.get_trading_calendar()``.
        If ``None``, uses a simple weekday heuristic (no holiday handling).

    Returns
    -------
    date
    """
    if from_date is None:
        from_date = date.today()
    elif isinstance(from_date, str):
        from_date = datetime.strptime(from_date, "%Y-%m-%d").date()

    if calendar_df is not None and not calendar_df.empty:
        cals = pd.to_datetime(calendar_df["trade_date"]).dt.date
        future = cals[cals > from_date]
        if not future.empty:
            return future.iloc[0]

    # Simple heuristic: skip weekends
    nxt = from_date + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def previous_trading_day(
    from_date: Optional[Union[date, str]] = None,
    calendar_df: Optional[pd.DataFrame] = None,
) -> date:
    """Return the previous trading day before *from_date*."""
    if from_date is None:
        from_date = date.today()
    elif isinstance(from_date, str):
        from_date = datetime.strptime(from_date, "%Y-%m-%d").date()

    if calendar_df is not None and not calendar_df.empty:
        cals = pd.to_datetime(calendar_df["trade_date"]).dt.date
        past = cals[cals < from_date]
        if not past.empty:
            return past.iloc[-1]

    prev = from_date - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def trading_seconds_remaining(dt: Optional[datetime] = None) -> int:
    """
    Seconds until the current trading session ends.

    Returns 0 if already closed.
    """
    dt = dt or datetime.now()
    t = dt.time()
    weekday = dt.weekday()
    if weekday >= 5:
        return 0

    # Morning session
    if dt_time(9, 30) <= t <= dt_time(11, 30):
        close = datetime.combine(dt.date(), dt_time(11, 30))
        return max(0, int((close - dt).total_seconds()))
    # Afternoon session
    if dt_time(13, 0) <= t <= dt_time(15, 0):
        close = datetime.combine(dt.date(), dt_time(15, 0))
        return max(0, int((close - dt).total_seconds()))
    return 0


def is_entry_window(
    dt: Optional[datetime] = None,
    entry_time: dt_time = dt_time(10, 0),
    tolerance_minutes: int = 5,
) -> bool:
    """
    Check whether *dt* is within the strategy's entry tolerance window.

    Parameters
    ----------
    dt: datetime, optional
    entry_time: time, default 10:00
    tolerance_minutes: int, default 5
        ± tolerance around entry_time.

    Returns
    -------
    bool
    """
    dt = dt or datetime.now()
    lower = datetime.combine(dt.date(), entry_time) - timedelta(minutes=tolerance_minutes)
    upper = datetime.combine(dt.date(), entry_time) + timedelta(minutes=tolerance_minutes)
    return lower <= dt <= upper


# ---------------------------------------------------------------------------
# Convenience: format recommendation list for frontend
# ---------------------------------------------------------------------------

def format_recommendations(
    recommendations: List[StockRecommendation],
    fmt: str = "json",
) -> Union[str, List[Dict[str, Any]]]:
    """
    Serialise recommendations for API / frontend consumption.

    Parameters
    ----------
    recommendations: list[StockRecommendation]
    fmt: str, default ``"json"``
        ``"json"`` | ``"dict"`` | ``"markdown"``.

    Returns
    -------
    str or list[dict]
    """
    if fmt == "dict":
        return [r.to_dict() for r in recommendations]
    if fmt == "json":
        return json.dumps(
            [r.to_dict() for r in recommendations],
            ensure_ascii=False,
            indent=2,
        )
    if fmt == "markdown":
        lines = ["# 每日推荐 TOP-{n}\n".format(n=len(recommendations))]
        for rec in recommendations:
            lines.append(f"## {rec.rank}. {rec.name} ({rec.code})")
            lines.append(f"- **综合评分**: {rec.composite_score:.3f}")
            lines.append(f"- **建议买入价**: ¥{rec.entry_price:.2f}")
            lines.append(f"- **风险等级**: {rec.risk_label}")
            lines.append(f"- **持有周期**: {rec.expected_hold}")
            lines.append("- **推荐理由**:")
            for rsn in rec.reasons:
                lines.append(f"  - {rsn}")
            lines.append("")
        return "\n".join(lines)
    raise ValueError(f"Unknown format: {fmt}")
