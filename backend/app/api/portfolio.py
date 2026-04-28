"""Portfolio & trading API routes."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import aiosqlite
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    BuyRequest,
    PerformanceMetrics,
    PortfolioPosition,
    SellRequest,
    TradeSignal,
    TradeStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------

DB_DIR = Path(__file__).resolve().parents[3] / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "portfolio.db"

# Global connection pool reference (set by lifespan)
_db_conn: Optional[aiosqlite.Connection] = None

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_POSITIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    buy_price REAL NOT NULL,
    buy_time TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    current_price REAL,
    strategy TEXT DEFAULT 'default',
    status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed')),
    sell_price REAL,
    sell_time TEXT,
    pnl REAL,
    pnl_pct REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

_TRADES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    direction TEXT NOT NULL CHECK(direction IN ('buy', 'sell')),
    code TEXT NOT NULL,
    name TEXT,
    price REAL,
    quantity INTEGER,
    amount REAL,
    signal_time TEXT NOT NULL,
    execute_time TEXT,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'executed', 'cancelled', 'failed')),
    reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


async def _ensure_tables() -> None:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(_POSITIONS_TABLE_SQL)
        await db.execute(_TRADES_TABLE_SQL)
        await db.commit()
        logger.info("Portfolio database initialized at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


async def _get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def _fetch_open_positions(db: aiosqlite.Connection) -> List[PortfolioPosition]:
    async with db.execute(
        "SELECT * FROM positions WHERE status = 'open' ORDER BY buy_time DESC"
    ) as cursor:
        rows = await cursor.fetchall()
    positions = []
    for row in rows:
        d = dict(row)
        # Refresh current price from latest market if possible
        d["current_price"] = d.get("current_price") or d["buy_price"]
        d["current_value"] = round(d["current_price"] * d["quantity"], 2)
        d["pnl"] = round(d["current_value"] - d["buy_price"] * d["quantity"], 2)
        d["pnl_pct"] = round(
            (d["current_price"] - d["buy_price"]) / d["buy_price"] * 100, 4
        ) if d["buy_price"] else 0.0
        positions.append(PortfolioPosition(**d))
    return positions


async def _fetch_trade_history(
    db: aiosqlite.Connection,
    limit: int = 200,
) -> List[TradeSignal]:
    async with db.execute(
        "SELECT * FROM trades ORDER BY signal_time DESC LIMIT ?",
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [TradeSignal(**dict(row)) for row in rows]


async def _compute_performance(db: aiosqlite.Connection) -> PerformanceMetrics:
    """Compute performance metrics from closed trades."""
    async with db.execute(
        "SELECT pnl, pnl_pct FROM positions WHERE status = 'closed' AND pnl IS NOT NULL"
    ) as cursor:
        rows = await cursor.fetchall()

    total_trades = len(rows)
    if total_trades == 0:
        return PerformanceMetrics(
            total_return_pct=0.0,
            annualized_return_pct=None,
            sharpe_ratio=None,
            max_drawdown_pct=None,
            win_rate_pct=None,
            profit_factor=None,
            total_trades=0,
            open_positions=0,
        )

    pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
    pnl_pcts = [r["pnl_pct"] for r in rows if r["pnl_pct"] is not None]

    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]

    total_return = sum(pnls)
    initial_capital = 1_000_000.0  # Assume 1M CNY base
    total_return_pct = round(total_return / initial_capital * 100, 4)

    win_rate = round(len(wins) / total_trades * 100, 2) if total_trades else 0.0
    avg_win = round(sum(wins) / len(wins), 4) if wins else 0.0
    avg_loss = round(sum(losses) / len(losses), 4) if losses else 0.0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 0 else None

    # Simple max drawdown from pnl_pcts series
    peak = 0.0
    max_dd = 0.0
    cum = 0.0
    for pct in pnl_pcts:
        cum += pct
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    # Annualized return (assume ~250 trading days per year)
    days = (datetime.now() - datetime(2024, 1, 1)).days or 1
    annualized = round(((1 + total_return_pct / 100) ** (250 / days) - 1) * 100, 4) if days > 0 else None

    # Sharpe (simplified, risk-free=3%)
    import statistics
    sharpe = None
    if len(pnl_pcts) > 1:
        try:
            avg_ret = statistics.mean(pnl_pcts)
            std_ret = statistics.stdev(pnl_pcts)
            if std_ret > 0:
                sharpe = round((avg_ret - 0.03) / std_ret, 4)
        except Exception:
            pass

    async with db.execute(
        "SELECT COUNT(*) AS cnt FROM positions WHERE status = 'open'"
    ) as cursor:
        row = await cursor.fetchone()
        open_pos = row["cnt"] if row else 0

    return PerformanceMetrics(
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized,
        sharpe_ratio=sharpe,
        max_drawdown_pct=round(max_dd, 4),
        calmar_ratio=round(annualized / max_dd, 4) if (annualized and max_dd > 0) else None,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        total_trades=total_trades,
        open_positions=open_pos,
        last_updated=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/positions", response_model=List[PortfolioPosition])
async def get_positions() -> List[PortfolioPosition]:
    """Get all currently open positions."""
    async for db in _get_db():
        return await _fetch_open_positions(db)
    return []


@router.get("/history", response_model=List[TradeSignal])
async def get_trade_history(
    limit: int = 200,
) -> List[TradeSignal]:
    """Get trade history (signals + executed orders)."""
    async for db in _get_db():
        return await _fetch_trade_history(db, limit=limit)
    return []


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance() -> PerformanceMetrics:
    """Get portfolio performance metrics."""
    async for db in _get_db():
        return await _compute_performance(db)
    return PerformanceMetrics(total_return_pct=0.0, total_trades=0, open_positions=0)


@router.post("/simulate-buy", response_model=TradeSignal)
async def simulate_buy(request: BuyRequest) -> TradeSignal:
    """Manually trigger a simulated buy order."""
    async for db in _get_db():
        # Resolve name and latest price
        name = request.code
        price = request.price
        if price is None:
            # Try to fetch from data layer
            try:
                from app.core.data_fetcher import DataFetcher
                df = DataFetcher()
                if hasattr(df, "get_latest_price"):
                    price = df.get_latest_price(request.code)
            except Exception as exc:
                logger.warning("Price fetch failed, using fallback: %s", exc)
                price = 100.0
        if price is None or price <= 0:
            price = 100.0

        qty = request.quantity or int(100_000 / price)
        amount = round(price * qty, 2)
        now = datetime.now()

        # Insert trade record
        await db.execute(
            """
            INSERT INTO trades (direction, code, name, price, quantity, amount, signal_time, execute_time, status, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("buy", request.code, name, price, qty, amount, now, now, "executed", request.reason),
        )
        trade_id = db.lastrowid

        # Insert open position
        await db.execute(
            """
            INSERT INTO positions (code, name, buy_price, buy_time, quantity, current_price, strategy, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (request.code, name, price, now, qty, price, request.reason or "manual", "open"),
        )
        pos_id = db.lastrowid
        await db.commit()

        logger.info("Simulated BUY %s @ %.2f x %d (trade_id=%s, pos_id=%s)",
                    request.code, price, qty, trade_id, pos_id)

        return TradeSignal(
            id=trade_id,
            direction="buy",
            code=request.code,
            name=name,
            price=price,
            quantity=qty,
            amount=amount,
            signal_time=now,
            execute_time=now,
            status=TradeStatus.EXECUTED,
            reason=request.reason,
        )

    raise HTTPException(status_code=500, detail="Database unavailable")


@router.post("/simulate-sell", response_model=TradeSignal)
async def simulate_sell(request: SellRequest) -> TradeSignal:
    """Manually trigger a simulated sell order (close matching open position)."""
    async for db in _get_db():
        # Find matching open position
        async with db.execute(
            "SELECT * FROM positions WHERE code = ? AND status = 'open' ORDER BY buy_time DESC LIMIT 1",
            (request.code,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"No open position found for {request.code}")

        pos = dict(row)
        buy_price = pos["buy_price"]
        open_qty = pos["quantity"]
        sell_qty = request.quantity or open_qty
        if sell_qty > open_qty:
            raise HTTPException(status_code=400, detail="Sell quantity exceeds open position")

        # Resolve sell price
        price = request.price
        if price is None:
            try:
                from app.core.data_fetcher import DataFetcher
                df = DataFetcher()
                if hasattr(df, "get_latest_price"):
                    price = df.get_latest_price(request.code)
            except Exception as exc:
                logger.warning("Sell price fetch failed: %s", exc)
                price = buy_price * 1.02  # Fallback: assume +2%
        if price is None or price <= 0:
            price = buy_price * 1.02

        amount = round(price * sell_qty, 2)
        pnl = round((price - buy_price) * sell_qty, 2)
        pnl_pct = round((price - buy_price) / buy_price * 100, 4) if buy_price else 0.0
        now = datetime.now()

        # Insert trade record
        await db.execute(
            """
            INSERT INTO trades (direction, code, name, price, quantity, amount, signal_time, execute_time, status, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("sell", request.code, pos["name"], price, sell_qty, amount, now, now, "executed", request.reason),
        )
        trade_id = db.lastrowid

        # Update / close position
        if sell_qty == open_qty:
            await db.execute(
                """
                UPDATE positions
                SET status = 'closed', sell_price = ?, sell_time = ?, current_price = ?, pnl = ?, pnl_pct = ?
                WHERE id = ?
                """,
                (price, now, price, pnl, pnl_pct, pos["id"]),
            )
        else:
            # Partial close: reduce quantity, keep position open
            await db.execute(
                "UPDATE positions SET quantity = quantity - ? WHERE id = ?",
                (sell_qty, pos["id"]),
            )

        await db.commit()

        logger.info("Simulated SELL %s @ %.2f x %d pnl=%.2f (trade_id=%s)",
                    request.code, price, sell_qty, pnl, trade_id)

        return TradeSignal(
            id=trade_id,
            direction="sell",
            code=request.code,
            name=pos["name"],
            price=price,
            quantity=sell_qty,
            amount=amount,
            signal_time=now,
            execute_time=now,
            status=TradeStatus.EXECUTED,
            reason=request.reason,
        )

    raise HTTPException(status_code=500, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Init hook
# ---------------------------------------------------------------------------

async def init_portfolio_db() -> None:
    await _ensure_tables()
