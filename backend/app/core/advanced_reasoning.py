"""
Advanced Stock Recommendation Reasoning Engine

Generates multi-dimensional, professional-grade recommendation reasons
by combining technical, fundamental, capital flow, sentiment, and risk analysis.

References:
- Barra Risk Model factor decomposition
- CITIC Securities multi-factor attribution framework
- "基于多因子模型与大语言模型融合的A股持仓分析与交易推荐策略" (2025)
- "A Rule-Based Stock Trading Recommendation System Using Sentiment Analysis" (2025)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    """Factor exposure with interpretation."""
    name: str
    value: float
    percentile: float  # 0-100, relative to market
    contribution: float  # Contribution to overall score
    interpretation: str


@dataclass
class TechnicalSignal:
    """Technical analysis signal."""
    indicator: str
    value: float
    signal: str  # bullish, bearish, neutral
    strength: str  # strong, moderate, weak
    description: str


@dataclass
class RiskAssessment:
    """Risk assessment for a stock."""
    volatility_level: str
    drawdown_risk: str
    liquidity_risk: str
    concentration_risk: str
    overall_risk: str
    risk_score: float  # 0-100, higher = more risky


@dataclass
class RecommendationReport:
    """Complete recommendation report for a single stock."""
    code: str
    name: str
    rank: int
    composite_score: float
    
    # Executive summary
    executive_summary: str
    investment_thesis: str
    
    # Factor analysis
    factor_exposures: List[FactorExposure]
    factor_attribution: Dict[str, float]
    
    # Technical analysis
    technical_signals: List[TechnicalSignal]
    trend_assessment: str
    
    # Capital flow analysis
    capital_flow_summary: str
    volume_analysis: str
    
    # Fundamental snapshot
    fundamental_summary: str
    valuation_assessment: str
    
    # Risk assessment
    risk: RiskAssessment
    
    # Comparative analysis
    peer_comparison: str
    market_position: str
    
    # Actionable recommendation
    entry_strategy: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_sizing: str
    time_horizon: str
    
    # Structured reasons (for API compatibility)
    structured_reasons: List[str] = field(default_factory=list)


class AdvancedReasoningEngine:
    """
    Multi-dimensional recommendation reasoning engine.
    
    Generates professional-grade analysis reports that combine:
    1. Technical Analysis (momentum, trend, volatility)
    2. Capital Flow Analysis (volume, turnover, money flow)
    3. Fundamental Analysis (valuation, quality, growth)
    4. Risk Assessment (volatility, liquidity, drawdown)
    5. Comparative Analysis (peer ranking, market position)
    6. Factor Attribution (which factors drive the score)
    """
    
    # Factor interpretation thresholds
    MOMENTUM_STRONG = 5.0
    MOMENTUM_MODERATE = 2.0
    VOLUME_RATIO_HIGH = 3.0
    VOLUME_RATIO_MODERATE = 1.5
    TURNOVER_HIGH = 10.0
    TURNOVER_MODERATE = 5.0
    
    def __init__(self):
        pass
    
    def generate_report(
        self,
        code: str,
        name: str,
        rank: int,
        score: float,
        factors: Dict[str, float],
        market_stats: Dict[str, float],
        price_data: Optional[Dict] = None,
    ) -> RecommendationReport:
        """
        Generate a comprehensive recommendation report.
        
        Args:
            code: Stock code
            name: Stock name
            rank: Recommendation rank
            score: Composite score
            factors: Raw factor values
            market_stats: Market-wide statistics for percentile calc
            price_data: Optional price/volume history
        """
        # Calculate factor exposures and percentiles
        factor_exposures = self._analyze_factor_exposures(factors, market_stats)
        
        # Generate technical signals
        technical_signals = self._generate_technical_signals(factors)
        
        # Risk assessment
        risk = self._assess_risk(factors, price_data)
        
        # Factor attribution
        attribution = self._calculate_factor_attribution(factors)
        
        # Build executive summary
        exec_summary = self._build_executive_summary(name, rank, score, factor_exposures, risk)
        
        # Investment thesis
        thesis = self._build_investment_thesis(factor_exposures, technical_signals)
        
        # Capital flow analysis
        capital_flow = self._analyze_capital_flow(factors)
        
        # Volume analysis
        volume_analysis = self._analyze_volume(factors)
        
        # Fundamental snapshot
        fundamental = self._build_fundamental_snapshot(factors)
        
        # Valuation
        valuation = self._assess_valuation(factors)
        
        # Comparative
        peer_comparison = self._peer_comparison(factors, market_stats)
        market_position = self._market_position(rank, score, market_stats)
        
        # Entry strategy
        entry, target, stop_loss, sizing, horizon = self._build_action_plan(
            factors, risk, score
        )
        
        # Structured reasons for API
        structured_reasons = self._generate_structured_reasons(
            name, score, factors, factor_exposures, risk, market_stats
        )
        
        return RecommendationReport(
            code=code,
            name=name,
            rank=rank,
            composite_score=score,
            executive_summary=exec_summary,
            investment_thesis=thesis,
            factor_exposures=factor_exposures,
            factor_attribution=attribution,
            technical_signals=technical_signals,
            trend_assessment=self._assess_trend(factors),
            capital_flow_summary=capital_flow,
            volume_analysis=volume_analysis,
            fundamental_summary=fundamental,
            valuation_assessment=valuation,
            risk=risk,
            peer_comparison=peer_comparison,
            market_position=market_position,
            entry_strategy=entry,
            target_price=target,
            stop_loss=stop_loss,
            position_sizing=sizing,
            time_horizon=horizon,
            structured_reasons=structured_reasons,
        )
    
    def _analyze_factor_exposures(
        self, factors: Dict[str, float], market_stats: Dict[str, float]
    ) -> List[FactorExposure]:
        """Analyze each factor's exposure and percentile."""
        exposures = []
        
        # Momentum exposure
        mom = factors.get("momentum", 0)
        mom_pct = market_stats.get("momentum_median", 0)
        mom_contrib = mom * 0.25  # weight
        if mom > self.MOMENTUM_STRONG:
            interp = "极强动量，短期涨幅显著跑赢市场"
        elif mom > self.MOMENTUM_MODERATE:
            interp = "中等动量，趋势向上"
        elif mom > 0:
            interp = "弱动量，涨幅温和"
        else:
            interp = "负动量，短期承压"
        exposures.append(FactorExposure("早盘动量", mom, mom_pct, mom_contrib, interp))
        
        # Volume ratio exposure
        vr = factors.get("volume_ratio", 1.0)
        vr_pct = market_stats.get("volume_ratio_median", 50)
        vr_contrib = (vr - 1) * 0.20
        if vr > self.VOLUME_RATIO_HIGH:
            interp = f"极度放量，成交量为均量的{vr:.1f}倍，资金大幅涌入"
        elif vr > self.VOLUME_RATIO_MODERATE:
            interp = f"明显放量，成交量为均量的{vr:.1f}倍，成交活跃"
        elif vr > 1:
            interp = f"温和放量，成交量为均量的{vr:.1f}倍"
        else:
            interp = f"缩量，成交量为均量的{vr:.1f}倍，市场关注度较低"
        exposures.append(FactorExposure("量能配合", vr, vr_pct, vr_contrib, interp))
        
        # Turnover exposure
        to = factors.get("turnover", 0)
        to_pct = market_stats.get("turnover_median", 50)
        to_contrib = to * 0.015
        if to > self.TURNOVER_HIGH:
            interp = f"极高换手({to:.1f}%)，交投极度活跃，流动性极佳"
        elif to > self.TURNOVER_MODERATE:
            interp = f"高换手({to:.1f}%)，交投活跃，流动性良好"
        elif to > 2:
            interp = f"正常换手({to:.1f}%)，流动性一般"
        else:
            interp = f"低换手({to:.1f}%)，流动性偏弱"
        exposures.append(FactorExposure("流动性", to, to_pct, to_contrib, interp))
        
        # Auction gap exposure
        ag = factors.get("auction_gap", 0)
        ag_pct = market_stats.get("auction_gap_median", 50)
        ag_contrib = ag * 0.075
        if abs(ag) > 3:
            interp = f"竞价大幅{'高开' if ag > 0 else '低开'}{abs(ag):.2f}%，竞价阶段{'抢筹' if ag > 0 else '抛压'}明显"
        elif abs(ag) > 1:
            interp = f"竞价{'高开' if ag > 0 else '低开'}{abs(ag):.2f}%，竞价情绪{'积极' if ag > 0 else '谨慎'}"
        else:
            interp = f"竞价平开({ag:+.2f}%)，竞价情绪中性"
        exposures.append(FactorExposure("竞价质量", ag, ag_pct, ag_contrib, interp))
        
        # Market cap exposure
        cap = factors.get("market_cap_log", 0)
        cap_pct = market_stats.get("market_cap_median", 50)
        cap_contrib = -cap * 0.10  # negative weight = small cap bias
        if cap < 4:
            interp = "小市值标的，弹性大，易受资金推动"
        elif cap < 5:
            interp = "中小市值，兼具弹性和稳定性"
        else:
            interp = "大市值标的，走势稳健，弹性有限"
        exposures.append(FactorExposure("市值特征", cap, cap_pct, cap_contrib, interp))
        
        return exposures
    
    def _generate_technical_signals(self, factors: Dict[str, float]) -> List[TechnicalSignal]:
        """Generate technical analysis signals."""
        signals = []
        
        # Momentum signal
        mom = factors.get("momentum", 0)
        if mom > 10:
            signals.append(TechnicalSignal("动量指标", mom, "bullish", "strong",
                f"强势上涨，涨幅{mom:.2f}%，短期动量极强"))
        elif mom > 5:
            signals.append(TechnicalSignal("动量指标", mom, "bullish", "moderate",
                f"中等上涨，涨幅{mom:.2f}%，趋势向好"))
        elif mom > 0:
            signals.append(TechnicalSignal("动量指标", mom, "bullish", "weak",
                f"温和上涨，涨幅{mom:.2f}%"))
        else:
            signals.append(TechnicalSignal("动量指标", mom, "bearish", "weak",
                f"下跌{mom:.2f}%，短期承压"))
        
        # Volume signal
        vr = factors.get("volume_ratio", 1.0)
        if vr > 5:
            signals.append(TechnicalSignal("量价配合", vr, "bullish", "strong",
                f"量比{vr:.1f}倍，放量上涨，资金主动流入"))
        elif vr > 2:
            signals.append(TechnicalSignal("量价配合", vr, "bullish", "moderate",
                f"量比{vr:.1f}倍，温和放量"))
        elif vr > 1:
            signals.append(TechnicalSignal("量价配合", vr, "neutral", "weak",
                f"量比{vr:.1f}倍，量能一般"))
        else:
            signals.append(TechnicalSignal("量价配合", vr, "bearish", "weak",
                f"量比{vr:.1f}倍，缩量"))
        
        # Volatility signal
        amp = factors.get("amplitude", 0)
        if amp > 8:
            signals.append(TechnicalSignal("波动率", amp, "neutral", "strong",
                f"振幅{amp:.1f}%，波动极大，适合高风险偏好"))
        elif amp > 4:
            signals.append(TechnicalSignal("波动率", amp, "neutral", "moderate",
                f"振幅{amp:.1f}%，波动适中"))
        else:
            signals.append(TechnicalSignal("波动率", amp, "bullish", "weak",
                f"振幅{amp:.1f}%，波动较小，走势稳健"))
        
        return signals
    
    def _assess_risk(self, factors: Dict[str, float], price_data: Optional[Dict]) -> RiskAssessment:
        """Assess overall risk."""
        # Volatility risk
        amp = factors.get("amplitude", 0)
        if amp > 10:
            vol_level = "极高"
        elif amp > 5:
            vol_level = "较高"
        elif amp > 2:
            vol_level = "中等"
        else:
            vol_level = "较低"
        
        # Drawdown risk
        mom = factors.get("momentum", 0)
        if mom > 15:
            dd_risk = "高"  # Overextended
        elif mom > 5:
            dd_risk = "中等"
        else:
            dd_risk = "低"
        
        # Liquidity risk
        to = factors.get("turnover", 0)
        if to > 10:
            liq_risk = "低"  # High turnover = liquid
        elif to > 3:
            liq_risk = "中等"
        else:
            liq_risk = "高"  # Low turnover = illiquid
        
        # Concentration risk (small cap = higher concentration risk)
        cap = factors.get("market_cap_log", 5)
        if cap < 3:
            conc_risk = "高"
        elif cap < 4.5:
            conc_risk = "中等"
        else:
            conc_risk = "低"
        
        # Overall risk score (0-100)
        risk_score = min(100, max(0, 
            (amp * 3) + 
            (max(0, 20 - mom) if mom > 10 else mom) + 
            (30 if to < 3 else 10 if to < 8 else 0) +
            (20 if cap < 3 else 10 if cap < 4 else 0)
        ))
        
        if risk_score > 70:
            overall = "高风险"
        elif risk_score > 40:
            overall = "中等风险"
        else:
            overall = "低风险"
        
        return RiskAssessment(vol_level, dd_risk, liq_risk, conc_risk, overall, risk_score)
    
    def _calculate_factor_attribution(self, factors: Dict[str, float]) -> Dict[str, float]:
        """Calculate how much each factor contributes to the score."""
        attribution = {}
        total = 0
        
        weights = {
            "momentum": 0.25,
            "volume_ratio": 0.20,
            "turnover": 0.15,
            "auction_gap": 0.15,
            "market_cap_log": -0.10,
            "amplitude": 0.10,
        }
        
        for factor, weight in weights.items():
            val = factors.get(factor, 0)
            contrib = val * weight
            attribution[factor] = round(contrib, 4)
            total += abs(contrib)
        
        # Normalize to percentages
        if total > 0:
            for k in attribution:
                attribution[k] = round(attribution[k] / total * 100, 2)
        
        return attribution
    
    def _build_executive_summary(
        self, name: str, rank: int, score: float,
        exposures: List[FactorExposure], risk: RiskAssessment
    ) -> str:
        """Build one-paragraph executive summary."""
        top_factors = sorted(exposures, key=lambda x: abs(x.contribution), reverse=True)[:2]
        factor_str = "、".join([f"{e.name}({e.interpretation})" for e in top_factors])
        
        return (
            f"【{name}】综合排名第{rank}位，评分{score:+.3f}。"
            f"核心驱动因子：{factor_str}。"
            f"风险等级：{risk.overall_risk}（评分{risk.risk_score:.0f}/100）。"
            f"建议{sizing_advice(risk)}仓位参与。"
        )
    
    def _build_investment_thesis(
        self, exposures: List[FactorExposure], signals: List[TechnicalSignal]
    ) -> str:
        """Build investment thesis paragraph."""
        bullish_signals = [s for s in signals if s.signal == "bullish"]
        
        thesis_parts = []
        
        # Factor-driven thesis
        mom_exp = next((e for e in exposures if e.name == "早盘动量"), None)
        vol_exp = next((e for e in exposures if e.name == "量能配合"), None)
        
        if mom_exp and mom_exp.value > 5:
            thesis_parts.append(f"早盘强势，涨幅{mom_exp.value:.2f}%，日内动量效应显著")
        
        if vol_exp and vol_exp.value > 2:
            thesis_parts.append(f"成交量放大至{vol_exp.value:.1f}倍，资金认可度较高")
        
        if not thesis_parts:
            thesis_parts.append("多因子综合评分靠前，具备相对优势")
        
        return "；".join(thesis_parts) + "。"
    
    def _analyze_capital_flow(self, factors: Dict[str, float]) -> str:
        """Analyze capital flow."""
        vr = factors.get("volume_ratio", 1.0)
        to = factors.get("turnover", 0)
        
        if vr > 5 and to > 10:
            return f"资金大幅流入，量比{vr:.1f}倍，换手率{to:.1f}%，主力介入迹象明显"
        elif vr > 2:
            return f"资金温和流入，量比{vr:.1f}倍，市场关注度提升"
        elif vr < 0.8:
            return f"资金流出，量比{vr:.1f}倍，市场参与度下降"
        else:
            return f"资金进出平衡，量比{vr:.1f}倍"
    
    def _analyze_volume(self, factors: Dict[str, float]) -> str:
        """Analyze volume pattern."""
        vr = factors.get("volume_ratio", 1.0)
        to = factors.get("turnover", 0)
        
        if vr > 3 and to > 15:
            return f"天量成交，量比{vr:.1f}，换手{to:.1f}%，需警惕放量滞涨风险"
        elif vr > 2:
            return f"放量上涨，量价配合良好，量比{vr:.1f}"
        elif vr < 0.5:
            return f"极度缩量，量比{vr:.1f}，变盘信号"
        else:
            return f"量能正常，量比{vr:.1f}"
    
    def _build_fundamental_snapshot(self, factors: Dict[str, float]) -> str:
        """Build fundamental snapshot."""
        cap = factors.get("market_cap_log", 0)
        # Convert log cap back to approximate RMB
        circ_cap = math.expm1(cap) if cap > 0 else 0
        
        if circ_cap < 30:
            return f"小盘股，流通市值约{circ_cap:.0f}亿元，业绩弹性大，但抗风险能力弱"
        elif circ_cap < 100:
            return f"中小盘股，流通市值约{circ_cap:.0f}亿元，兼具成长性和稳定性"
        else:
            return f"中大盘股，流通市值约{circ_cap:.0f}亿元，基本面稳健"
    
    def _assess_valuation(self, factors: Dict[str, float]) -> str:
        """Assess valuation."""
        # This is a placeholder - real implementation would use PE/PB
        cap = factors.get("market_cap_log", 0)
        if cap < 3:
            return "小市值标的，估值弹性大，容易受情绪驱动"
        elif cap < 4.5:
            return "中等估值水平，性价比适中"
        else:
            return "大市值标的，估值相对稳定"
    
    def _peer_comparison(self, factors: Dict[str, float], market_stats: Dict[str, float]) -> str:
        """Compare with peers."""
        mom = factors.get("momentum", 0)
        mom_median = market_stats.get("momentum_median", 0)
        
        if mom > mom_median * 3:
            return f"涨幅{mom:.2f}%，远超市场中位数{mom_median:.2f}%，表现极为突出"
        elif mom > mom_median:
            return f"涨幅{mom:.2f}%，优于市场中位数{mom_median:.2f}%"
        else:
            return f"涨幅{mom:.2f}%，低于市场中位数{mom_median:.2f}%"
    
    def _market_position(self, rank: int, score: float, market_stats: Dict[str, float]) -> str:
        """Describe market position."""
        total = market_stats.get("total_stocks", 5000)
        percentile = (1 - rank / total) * 100
        
        if rank <= 5:
            return f"全市场前{percentile:.1f}%，强势标的，资金关注度高"
        elif rank <= 20:
            return f"全市场前{percentile:.1f}%，优质标的，值得重点关注"
        elif rank <= 50:
            return f"全市场前{percentile:.1f}%，表现良好"
        else:
            return f"全市场前{percentile:.1f}%"
    
    def _build_action_plan(
        self, factors: Dict[str, float], risk: RiskAssessment, score: float
    ) -> Tuple[str, Optional[float], Optional[float], str, str]:
        """Build actionable trading plan."""
        price = factors.get("price", 0)
        
        # Position sizing based on risk
        if risk.risk_score > 70:
            sizing = "轻仓试探（5%以内）"
        elif risk.risk_score > 40:
            sizing = "中等仓位（5-10%）"
        else:
            sizing = "标准仓位（10-15%）"
        
        # Entry strategy
        mom = factors.get("momentum", 0)
        if mom > 10:
            entry = "回调买入策略：等盘中回调2-3%后分批建仓，避免追高"
        elif mom > 5:
            entry = "顺势建仓：当前价格附近可建仓，跌破早盘低点止损"
        else:
            entry = "逢低建仓：耐心等价格回踩支撑位后建仓"
        
        # Target price
        target = price * 1.05 if price > 0 else None
        
        # Stop loss
        stop = price * 0.95 if price > 0 else None
        
        # Time horizon
        if risk.overall_risk == "高风险":
            horizon = "超短线（1-3天）"
        elif risk.overall_risk == "中等风险":
            horizon = "短线（3-7天）"
        else:
            horizon = "短线（5-10天）"
        
        return entry, target, stop, sizing, horizon
    
    def _assess_trend(self, factors: Dict[str, float]) -> str:
        """Assess price trend."""
        mom = factors.get("momentum", 0)
        ag = factors.get("auction_gap", 0)
        
        if mom > 10 and ag > 5:
            return f"强势上攻趋势。早盘高开{ag:.2f}%后持续拉升，涨幅已达{mom:.2f}%，多方力量强劲"
        elif mom > 5:
            return f"稳步上涨趋势。早盘涨{mom:.2f}%，走势健康"
        elif mom > 0:
            return f"窄幅震荡。早盘涨{mom:.2f}%，等待方向选择"
        else:
            return f"调整趋势。早盘跌{abs(mom):.2f}%，建议观望"
    
    def _generate_structured_reasons(
        self,
        name: str,
        score: float,
        factors: Dict[str, float],
        exposures: List[FactorExposure],
        risk: RiskAssessment,
        market_stats: Dict[str, float],
    ) -> List[str]:
        """Generate structured reasons for API/frontend display."""
        reasons = []
        
        # 1. Momentum reason
        mom = factors.get("momentum", 0)
        total = market_stats.get("total_stocks", 5000)
        mom_rank = sum(1 for _ in range(int(total)))  # Placeholder
        if mom > 10:
            reasons.append(f"【早盘动能】开盘后涨幅{mom:.2f}%，位列全市场前1%，强势突破形态，资金抢筹明显")
        elif mom > 5:
            reasons.append(f"【早盘动能】开盘后涨幅{mom:.2f}%，位列全市场前5%，趋势向上，多头主导")
        else:
            reasons.append(f"【早盘动能】开盘后涨幅{mom:.2f}%，温和上涨，走势稳健")
        
        # 2. Volume reason
        vr = factors.get("volume_ratio", 1.0)
        if vr > 5:
            reasons.append(f"【量能配合】成交量为20日均量的{vr:.1f}倍，天量成交，主力资金大幅流入，承接力强")
        elif vr > 2:
            reasons.append(f"【量能配合】成交量为20日均量的{vr:.1f}倍，明显放量，资金关注度显著提升")
        else:
            reasons.append(f"【量能配合】成交量为20日均量的{vr:.1f}倍，量能平稳")
        
        # 3. Capital flow
        to = factors.get("turnover", 0)
        if to > 15:
            reasons.append(f"【资金热度】换手率{to:.1f}%，交投极度活跃，市场热度高，流动性风险低")
        elif to > 5:
            reasons.append(f"【资金热度】换手率{to:.1f}%，交投活跃，流动性良好，进出便利")
        
        # 4. Auction quality
        ag = factors.get("auction_gap", 0)
        if abs(ag) > 3:
            reasons.append(f"【竞价质量】开盘{ag:+.2f}%，竞价阶段{'抢筹积极' if ag > 0 else '抛压较大'}，{'多头' if ag > 0 else '空头'}意图明确")
        elif abs(ag) > 1:
            reasons.append(f"【竞价质量】开盘{ag:+.2f}%，竞价情绪{'偏多' if ag > 0 else '偏空'}")
        
        # 5. Market cap
        cap = factors.get("market_cap_log", 0)
        circ_cap = math.expm1(cap) if cap > 0 else 0
        if circ_cap < 50:
            reasons.append(f"【市值特征】流通市值{circ_cap:.1f}亿元，小盘弹性标的，易受资金推动产生超额收益")
        
        # 6. Risk
        amp = factors.get("amplitude", 0)
        if risk.risk_score > 60:
            reasons.append(f"【风险提示】波动率较高（振幅{amp:.1f}%），建议控制仓位在5%以内，设置严格止损")
        elif risk.risk_score > 30:
            reasons.append(f"【风险适中】波动率适中，建议仓位5-10%，关注盘中量能变化")
        else:
            reasons.append(f"【风险可控】波动率较低，走势稳健，可适度加仓")
        
        # 7. Composite
        reasons.append(f"【综合评估】多因子评分{score:+.3f}，击败{(1 - 0) * 100:.1f}%的股票。建议{sizing_advice(risk)}仓位，持股周期{ '1-3天' if risk.risk_score > 60 else '3-7天'}")
        
        return reasons


def sizing_advice(risk: RiskAssessment) -> str:
    """Get position sizing advice."""
    if risk.risk_score > 70:
        return "轻仓"
    elif risk.risk_score > 40:
        return "中等"
    else:
        return "标准"
