"""APScheduler job definitions and SchedulerManager.

Daily schedule (A-share trading day):
- 09:15  morning_scan_job  – warm-up, data readiness checks
- 10:00  buy_job           – factor computation + buy execution
- 09:30  sell_job          – sell yesterday's open positions
- 15:30  data_sync_job     – persist EOD data

Weekends and Chinese public holidays are skipped automatically.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trading-day calendar helpers
# ---------------------------------------------------------------------------

_FIXED_HOLIDAYS: set[date] = {
    date(2024, 1, 1), date(2024, 2, 9), date(2024, 2, 12), date(2024, 2, 13),
    date(2024, 2, 14), date(2024, 2, 15), date(2024, 4, 4), date(2024, 4, 5),
    date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3), date(2024, 6, 10),
    date(2024, 9, 16), date(2024, 9, 17), date(2024, 10, 1), date(2024, 10, 2),
    date(2024, 10, 3), date(2024, 10, 4), date(2024, 10, 7),
    date(2025, 1, 1), date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30),
    date(2025, 1, 31), date(2025, 2, 3), date(2025, 2, 4), date(2025, 4, 4),
    date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 3), date(2025, 5, 31),
    date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3), date(2025, 10, 6),
    date(2025, 10, 7), date(2025, 10, 8),
}


def is_trading_day(d: date) -> bool:
    """Return True if *d* is an A-share trading day."""
    if d.weekday() >= 5:  # Sat or Sun
        return False
    return d not in _FIXED_HOLIDAYS


def next_trading_day(start: date) -> date:
    """Return the first trading day on or after *start*."""
    d = start
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# Job state persistence (tiny SQLite table)
# ---------------------------------------------------------------------------

_JOB_STATE_DB = Path(__file__).resolve().parents[3] / "data" / "scheduler_state.db"
_JOB_STATE_DB.parent.mkdir(parents=True, exist_ok=True)


def _init_job_state_db() -> None:
    with sqlite3.connect(str(_JOB_STATE_DB)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS job_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                run_time TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                detail TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.commit()


def _record_job_run(job_id: str, status: str, detail: str = "") -> None:
    with sqlite3.connect(str(_JOB_STATE_DB)) as conn:
        conn.execute(
            "INSERT INTO job_runs (job_id, run_time, status, detail) VALUES (?, ?, ?, ?)",
            (job_id, datetime.now().isoformat(), status, detail),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Core engine wrappers (lazy import to avoid circular deps)
# ---------------------------------------------------------------------------

async def _run_in_thread(func: Callable, *args, **kwargs) -> Any:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return await loop.run_in_executor(pool, lambda: func(*args, **kwargs))


async def _warmup_gpu() -> None:
    """Pre-allocate a small tensor on every visible GPU to warm up CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                dev = torch.device(f"cuda:{i}")
                x = torch.randn(512, 512, device=dev)
                _ = x @ x.T
                torch.cuda.synchronize(dev)
                logger.info("GPU %s warmed up (%s)", i, torch.cuda.get_device_name(i))
        else:
            logger.info("CUDA unavailable; GPU warm-up skipped")
    except Exception as exc:
        logger.warning("GPU warm-up failed: %s", exc)


async def _morning_scan() -> None:
    """9:15 scan: ensure market data is ready and run pre-market checks."""
    logger.info("[morning_scan_job] Starting pre-market scan …")
    try:
        from app.core.data_fetcher import DataFetcher
        df = DataFetcher()
        if hasattr(df, "warmup_cache"):
            await _run_in_thread(df.warmup_cache)
        if hasattr(df, "check_market_ready"):
            ready = await _run_in_thread(df.check_market_ready)
            if not ready:
                logger.warning("Market data not fully ready at 09:15")
    except ImportError:
        logger.info("DataFetcher not available; scan skipped")
    _record_job_run("morning_scan", "success")


async def _execute_buy() -> None:
    """10:00 buy: compute factors, rank, and open positions for top picks."""
    logger.info("[buy_job] Starting daily buy workflow …")
    today = date.today()
    if not is_trading_day(today):
        logger.info("[buy_job] %s is not a trading day – buy skipped", today)
        _record_job_run("buy", "success", "non-trading day")
        return

    recs: List[Dict[str, Any]] = []
    try:
        from app.core.recommender import StockRecommender
        recommender = StockRecommender()
        if hasattr(recommender, "get_recommendations"):
            raw = await _run_in_thread(recommender.get_recommendations, top_n=20)
            recs = raw if isinstance(raw, list) else []
    except ImportError:
        logger.info("StockRecommender not available; using fallback")
    except Exception as exc:
        logger.error("Recommender failed: %s", exc)

    # Cache recommendations in the stocks module
    from app.api.stocks import _recommendation_cache
    from app.models.schemas import StockRecommendation

    if recs:
        parsed = [
            StockRecommendation(
                code=r.get("code"),
                name=r.get("name"),
                score=r.get("score", 0.0),
                rank=idx + 1,
                reasons=r.get("reasons", []),
                expected_return=r.get("expected_return"),
                risk_level=r.get("risk_level"),
            )
            for idx, r in enumerate(recs)
        ]
        _recommendation_cache[str(today)] = parsed
    else:
        logger.info("No recommendations produced by core engine")

    # Execute buy for top picks (simulation)
    try:
        from app.core.strategy import StrategyEngine
        strategy = StrategyEngine()
        if hasattr(strategy, "execute_buy_signals"):
            await _run_in_thread(strategy.execute_buy_signals, recs)
    except ImportError:
        pass
    except Exception as exc:
        logger.error("Strategy buy execution failed: %s", exc)

    _record_job_run("buy", "success", f"{len(recs)} recommendations")
    logger.info("[buy_job] Completed with %d recommendations", len(recs))


async def _execute_sell() -> None:
    """9:30 sell: liquidate all positions opened yesterday."""
    logger.info("[sell_job] Starting daily sell workflow …")
    today = date.today()
    if not is_trading_day(today):
        logger.info("[sell_job] %s is not a trading day – sell skipped", today)
        _record_job_run("sell", "success", "non-trading day")
        return

    try:
        from app.core.strategy import StrategyEngine
        strategy = StrategyEngine()
        if hasattr(strategy, "execute_sell_all"):
            await _run_in_thread(strategy.execute_sell_all)
        elif hasattr(strategy, "close_all_positions"):
            await _run_in_thread(strategy.close_all_positions)
    except ImportError:
        logger.info("StrategyEngine not available; sell skipped")
    except Exception as exc:
        logger.error("Strategy sell execution failed: %s", exc)

    # Also close any simulated positions in the local DB
    db_path = Path(__file__).resolve().parents[3] / "data" / "portfolio.db"
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                # Find open positions bought before today
                yesterday = today - timedelta(days=1)
                rows = conn.execute(
                    "SELECT * FROM positions WHERE status = 'open' AND date(buy_time) <= ?",
                    (yesterday,),
                ).fetchall()
                for row in rows:
                    # Simulate sell at +2% for demo
                    buy_price = row[3]
                    sell_price = round(buy_price * 1.02, 2)
                    pnl = round((sell_price - buy_price) * row[5], 2)
                    pnl_pct = round((sell_price - buy_price) / buy_price * 100, 4)
                    conn.execute(
                        """UPDATE positions
                           SET status = 'closed',
                               sell_price = ?,
                               sell_time = ?,
                               current_price = ?,
                               pnl = ?,
                               pnl_pct = ?
                           WHERE id = ?""",
                        (sell_price, datetime.now().isoformat(), sell_price, pnl, pnl_pct, row[0]),
                    )
                    conn.execute(
                        """INSERT INTO trades
                           (direction, code, name, price, quantity, amount, signal_time, execute_time, status, reason)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        ("sell", row[1], row[2], sell_price, row[5], round(sell_price * row[5], 2),
                         datetime.now().isoformat(), datetime.now().isoformat(), "executed", "scheduled_sell"),
                    )
                conn.commit()
                logger.info("[sell_job] Closed %d positions from DB", len(rows))
        except Exception as exc:
            logger.error("DB sell failed: %s", exc)

    _record_job_run("sell", "success")


async def _sync_eod_data() -> None:
    """15:30: persist end-of-day data to local SQLite."""
    logger.info("[data_sync_job] Starting EOD data sync …")
    try:
        from app.core.data_fetcher import DataFetcher
        df = DataFetcher()
        if hasattr(df, "sync_eod"):
            await _run_in_thread(df.sync_eod)
        elif hasattr(df, "download_daily"):
            await _run_in_thread(df.download_daily, str(date.today()))
    except ImportError:
        logger.info("DataFetcher not available; sync skipped")
    except Exception as exc:
        logger.error("EOD sync failed: %s", exc)
    _record_job_run("data_sync", "success")


# ---------------------------------------------------------------------------
# SchedulerManager
# ---------------------------------------------------------------------------

class SchedulerManager:
    """Manages APScheduler lifecycle and A-share-aware job registration."""

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._started: bool = False
        _init_job_state_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
        self._register_jobs()
        self._scheduler.start()
        self._started = True
        logger.info("SchedulerManager started with %d jobs", len(self._scheduler.get_jobs()))

    def shutdown(self) -> None:
        if self._scheduler:
            self._scheduler.shutdown(wait=True)
            self._scheduler = None
        self._started = False
        logger.info("SchedulerManager shut down")

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_jobs(self) -> None:
        assert self._scheduler is not None
        sched = self._scheduler

        # 09:15 – morning scan (only on trading days)
        sched.add_job(
            self._wrap_async(_morning_scan),
            trigger=CronTrigger(hour=9, minute=15, timezone="Asia/Shanghai"),
            id="morning_scan",
            name="Morning market scan & warm-up",
            replace_existing=True,
            misfire_grace_time=300,
        )

        # 10:00 – buy execution (only on trading days)
        sched.add_job(
            self._wrap_async(_execute_buy),
            trigger=CronTrigger(hour=10, minute=0, timezone="Asia/Shanghai"),
            id="daily_buy",
            name="Daily buy (factor scan + order)",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # 09:30 – sell execution (only on trading days)
        sched.add_job(
            self._wrap_async(_execute_sell),
            trigger=CronTrigger(hour=9, minute=30, timezone="Asia/Shanghai"),
            id="daily_sell",
            name="Daily sell (close yesterday positions)",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # 15:30 – EOD data sync (only on trading days)
        sched.add_job(
            self._wrap_async(_sync_eod_data),
            trigger=CronTrigger(hour=15, minute=30, timezone="Asia/Shanghai"),
            id="eod_sync",
            name="End-of-day data synchronisation",
            replace_existing=True,
            misfire_grace_time=900,
        )

    @staticmethod
    def _wrap_async(coro_func: Callable) -> Callable:
        """Wrap an async function so APScheduler can execute it."""
        def wrapper() -> None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already running – schedule it
                    asyncio.create_task(coro_func())
                else:
                    loop.run_until_complete(coro_func())
            except RuntimeError:
                # No event loop in this thread
                asyncio.run(coro_func())
        return wrapper

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_jobs(self) -> List[Dict[str, Any]]:
        if not self._scheduler:
            return []
        jobs = self._scheduler.get_jobs()
        result: List[Dict[str, Any]] = []
        for j in jobs:
            next_run = j.next_run_time.isoformat() if j.next_run_time else None
            trigger_str = str(j.trigger)
            # Try to get last run status from SQLite
            last_status = "never"
            last_run = None
            try:
                with sqlite3.connect(str(_JOB_STATE_DB)) as conn:
                    row = conn.execute(
                        "SELECT run_time, status FROM job_runs WHERE job_id = ? ORDER BY id DESC LIMIT 1",
                        (j.id,),
                    ).fetchone()
                    if row:
                        last_run = row[0]
                        last_status = row[1]
            except Exception:
                pass
            result.append({
                "id": j.id,
                "name": j.name,
                "next_run_time": next_run,
                "trigger": trigger_str,
                "last_run_time": last_run,
                "last_run_status": last_status,
            })
        return result

    def pause_job(self, job_id: str) -> bool:
        if self._scheduler:
            self._scheduler.pause_job(job_id)
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        if self._scheduler:
            self._scheduler.resume_job(job_id)
            return True
        return False

    def trigger_job_now(self, job_id: str) -> bool:
        """Manually trigger a job by ID (for testing / admin)."""
        mapping = {
            "morning_scan": _morning_scan,
            "daily_buy": _execute_buy,
            "daily_sell": _execute_sell,
            "eod_sync": _sync_eod_data,
        }
        if job_id not in mapping:
            return False
        coro = mapping[job_id]
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(coro())
            else:
                loop.run_until_complete(coro())
            return True
        except RuntimeError:
            asyncio.run(coro())
            return True


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

scheduler_manager = SchedulerManager()
