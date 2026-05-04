"""Pydantic models for request/response schemas."""
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TradeDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SystemState(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    COMPUTING = "computing"
    TRADING = "trading"
    SYNCING = "syncing"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Base schemas
# ---------------------------------------------------------------------------

class StockInfo(BaseModel):
    """Basic stock information."""

    code: str = Field(..., description="Stock code, e.g. 000001.SZ")
    name: str = Field(..., description="Stock name, e.g. 平安银行")
    industry: Optional[str] = Field(None, description="Industry classification")
    market_cap: Optional[float] = Field(None, description="Market capitalization in 亿 CNY")
    pe_ttm: Optional[float] = Field(None, description="Trailing P/E ratio")
    pb: Optional[float] = Field(None, description="Price-to-book ratio")
    listing_date: Optional[date] = Field(None, description="IPO date")
    is_st: bool = Field(False, description="Whether the stock is ST/*ST")
    exchange: str = Field(..., description="Exchange: SZ / SH / BJ")

    model_config = {"from_attributes": True}


class FactorData(BaseModel):
    """Factor data for a single stock."""

    code: str = Field(..., description="Stock code")
    trade_date: date = Field(..., description="Calculation date")

    # Momentum factors
    momentum_5d: Optional[float] = Field(None, description="5-day return momentum")
    momentum_20d: Optional[float] = Field(None, description="20-day return momentum")
    momentum_60d: Optional[float] = Field(None, description="60-day return momentum")

    # Volume factors
    volume_ratio: Optional[float] = Field(None, description="Volume / MA20 volume")
    turnover_rate: Optional[float] = Field(None, description="Turnover rate %")
    money_flow: Optional[float] = Field(None, description="Net money flow in 万 CNY")

    # Volatility factors
    volatility_20d: Optional[float] = Field(None, description="20-day realized volatility")
    atr_14: Optional[float] = Field(None, description="14-day Average True Range")

    # Quality / valuation
    roe_ttm: Optional[float] = Field(None, description="Return on equity TTM")
    profit_growth: Optional[float] = Field(None, description="YoY profit growth %")
    revenue_growth: Optional[float] = Field(None, description="YoY revenue growth %")

    # Composite score
    composite_score: Optional[float] = Field(None, description="Composite factor score (0-100)")
    rank: Optional[int] = Field(None, description="Rank within universe")

    model_config = {"from_attributes": True}


class StockRecommendation(BaseModel):
    """A recommended stock with rationale."""

    code: str = Field(..., description="Stock code")
    name: str = Field(..., description="Stock name")
    score: float = Field(..., ge=0, le=100, description="Composite score 0-100")
    rank: int = Field(..., ge=1, description="Rank in today's recommendations")

    reasons: List[str] = Field(default_factory=list, description="Human-readable rationale list")
    expected_return: Optional[float] = Field(None, description="Expected 1-day return %")
    risk_level: Optional[str] = Field(None, description="Risk level: low / medium / high")

    factors: Optional[FactorData] = Field(None, description="Detailed factor breakdown")

    # Internal metadata
    model_confidence: Optional[float] = Field(None, description="Model confidence 0-1")
    sector_alignment: Optional[str] = Field(None, description="Aligned sector theme")

    model_config = {"from_attributes": True}


class PortfolioPosition(BaseModel):
    """A single open position."""

    id: Optional[int] = Field(None, description="Position ID in DB")
    code: str = Field(..., description="Stock code")
    name: str = Field(..., description="Stock name")

    buy_price: float = Field(..., description="Execution buy price")
    buy_time: datetime = Field(..., description="Buy datetime")
    quantity: int = Field(..., gt=0, description="Number of shares")

    current_price: Optional[float] = Field(None, description="Latest market price")
    current_value: Optional[float] = Field(None, description="Current position value")

    pnl: Optional[float] = Field(None, description="Absolute P&L in CNY")
    pnl_pct: Optional[float] = Field(None, description="P&L percentage")

    # Metadata
    strategy: Optional[str] = Field("default", description="Strategy that opened position")
    status: str = Field("open", description="Position status: open / closed")

    @field_validator("pnl_pct", mode="before")
    @classmethod
    def compute_pnl_pct(cls, v, info):
        values = info.data
        if v is None and values.get("buy_price") and values.get("current_price"):
            return round(
                (values["current_price"] - values["buy_price"]) / values["buy_price"] * 100,
                4,
            )
        return v

    model_config = {"from_attributes": True}


class TradeSignal(BaseModel):
    """A trading signal or executed order."""

    id: Optional[int] = Field(None, description="Trade ID in DB")
    direction: TradeDirection = Field(..., description="Buy or sell")
    code: str = Field(..., description="Stock code")
    name: Optional[str] = Field(None, description="Stock name")

    price: Optional[float] = Field(None, description="Execution or signal price")
    quantity: Optional[int] = Field(None, description="Planned or executed quantity")
    amount: Optional[float] = Field(None, description="Trade amount in CNY")

    signal_time: datetime = Field(default_factory=datetime.now, description="Signal generation time")
    execute_time: Optional[datetime] = Field(None, description="Actual execution time")

    status: TradeStatus = Field(TradeStatus.PENDING, description="Order status")
    reason: Optional[str] = Field(None, description="Signal reason / memo")

    model_config = {"from_attributes": True}


class PerformanceMetrics(BaseModel):
    """Portfolio performance summary."""

    total_return_pct: float = Field(..., description="Total return since inception %")
    annualized_return_pct: Optional[float] = Field(None, description="Annualized return %")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio (risk-free=3%)")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    max_drawdown_pct: Optional[float] = Field(None, description="Maximum drawdown %")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")

    win_rate_pct: Optional[float] = Field(None, description="Winning trade percentage")
    profit_factor: Optional[float] = Field(None, description="Gross profit / gross loss")
    profit_loss_ratio: Optional[float] = Field(None, description="Profit/Loss ratio")
    avg_trade_return_pct: Optional[float] = Field(None, description="Average trade return %")
    avg_winning_trade_pct: Optional[float] = Field(None, description="Average winning trade %")
    avg_losing_trade_pct: Optional[float] = Field(None, description="Average losing trade %")

    total_trades: int = Field(0, description="Total number of closed trades")
    num_trades: int = Field(0, description="Total trades (alias)")
    num_winning_trades: int = Field(0, description="Winning trade count")
    num_losing_trades: int = Field(0, description="Losing trade count")
    open_positions: int = Field(0, description="Currently open positions")
    turnover_ratio: Optional[float] = Field(None, description="Annual turnover ratio")

    equity_curve: Optional[List[Dict[str, Any]]] = Field(None, description="Daily NAV series")
    drawdown_series: Optional[List[Dict[str, Any]]] = Field(None, description="Drawdown series")
    monthly_returns: Optional[Dict[str, float]] = Field(None, description="Monthly return map")

    last_updated: Optional[datetime] = Field(None, description="Metrics calculation time")

    model_config = {"from_attributes": True}


class GpuStatus(BaseModel):
    """GPU telemetry snapshot."""

    index: int = Field(..., ge=0, description="GPU device index")
    name: str = Field(..., description="GPU name, e.g. NVIDIA H100 80GB HBM3")

    utilization_gpu_pct: Optional[float] = Field(None, description="GPU utilization %")
    utilization_mem_pct: Optional[float] = Field(None, description="Memory controller utilization %")

    memory_total_mb: Optional[float] = Field(None, description="Total VRAM in MB")
    memory_used_mb: Optional[float] = Field(None, description="Used VRAM in MB")
    memory_free_mb: Optional[float] = Field(None, description="Free VRAM in MB")

    temperature_c: Optional[float] = Field(None, description="GPU temperature °C")
    power_draw_w: Optional[float] = Field(None, description="Current power draw in W")
    power_limit_w: Optional[float] = Field(None, description="Power limit in W")

    fan_speed_pct: Optional[float] = Field(None, description="Fan speed %")
    clock_sm_mhz: Optional[float] = Field(None, description="SM clock MHz")
    clock_mem_mhz: Optional[float] = Field(None, description="Memory clock MHz")

    processes: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Running compute processes")

    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"from_attributes": True}


class SystemStatus(BaseModel):
    """Overall system health and state."""

    state: SystemState = Field(SystemState.IDLE, description="Current system state")
    version: str = Field("1.0.0", description="Backend version")

    is_trading_day: bool = Field(True, description="Whether today is an A-share trading day")
    market_open: bool = Field(False, description="Whether market is currently open")

    last_update_time: Optional[datetime] = Field(None, description="Last data update")
    next_buy_time: Optional[datetime] = Field(None, description="Next scheduled buy (10:00)")
    next_sell_time: Optional[datetime] = Field(None, description="Next scheduled sell (09:30)")
    next_scan_time: Optional[datetime] = Field(None, description="Next scheduled scan (09:15)")
    next_sync_time: Optional[datetime] = Field(None, description="Next scheduled data sync (15:30)")

    db_connected: bool = Field(True, description="SQLite connection status")
    gpu_available: bool = Field(False, description="GPU compute available")
    gpu_count: int = Field(0, ge=0, description="Number of GPUs detected")

    active_positions: int = Field(0, description="Count of open positions")
    today_recommendations: int = Field(0, description="Count of today's top picks")

    uptime_seconds: Optional[float] = Field(None, description="Process uptime in seconds")
    message: Optional[str] = Field(None, description="Status message / latest log line")

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Request / response wrappers
# ---------------------------------------------------------------------------

class ApiResponse(BaseModel):
    """Generic API envelope."""

    success: bool = Field(True)
    message: Optional[str] = Field(None)
    data: Optional[dict] = Field(None)


class BuyRequest(BaseModel):
    """Manual buy request body."""

    code: str = Field(..., description="Stock code to buy")
    price: Optional[float] = Field(None, description="Optional limit price")
    quantity: Optional[int] = Field(None, description="Optional quantity (auto-calculated if omitted)")
    reason: Optional[str] = Field("manual", description="Reason tag")


class SellRequest(BaseModel):
    """Manual sell request body."""

    code: str = Field(..., description="Stock code to sell")
    price: Optional[float] = Field(None, description="Optional limit price")
    quantity: Optional[int] = Field(None, description="Optional partial quantity")
    reason: Optional[str] = Field("manual", description="Reason tag")


class KlinePoint(BaseModel):
    """Single OHLCV point."""

    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    turnover: Optional[float] = None

    model_config = {"from_attributes": True}


class LogEntry(BaseModel):
    """A system log line."""

    timestamp: datetime
    level: str = Field(..., description="DEBUG / INFO / WARNING / ERROR")
    source: str = Field(..., description="Module or job name")
    message: str

    model_config = {"from_attributes": True}


class ScheduleJobInfo(BaseModel):
    """APScheduler job metadata."""

    id: str
    name: str
    next_run_time: Optional[datetime]
    trigger: str
    last_run_time: Optional[datetime]
    last_run_status: Optional[str] = Field(None, description="success / failed / never")

    model_config = {"from_attributes": True}


class TradingDayCheck(BaseModel):
    """Trading day check result."""

    trade_date: date
    is_trading_day: bool
    reason: Optional[str] = Field(None, description="holiday name if not trading day")
