"""System monitoring API routes (GPU, logs, schedule, health)."""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.models.schemas import GpuStatus, LogEntry, ScheduleJobInfo, SystemStatus, SystemState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitor", tags=["monitor"])

# ---------------------------------------------------------------------------
# GPU utilities (nvidia-ml-py / pynvml)
# ---------------------------------------------------------------------------

_nvml_initialized: bool = False

def _init_nvml() -> bool:
    global _nvml_initialized
    if _nvml_initialized:
        return True
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_initialized = True
        return True
    except Exception as exc:
        logger.warning("NVML init failed: %s", exc)
        return False


def _shutdown_nvml() -> None:
    global _nvml_initialized
    if _nvml_initialized:
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass
        _nvml_initialized = False


def _get_gpu_status() -> List[GpuStatus]:
    """Fetch GPU telemetry using pynvml (nvidia-ml-py)."""
    if not _init_nvml():
        return []

    import pynvml
    devices: List[GpuStatus] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception as exc:
        logger.error("NVML device count failed: %s", exc)
        return []

    for i in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            mem_util = util.memory

            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_total_mb = mem_info.total / 1024 / 1024
            mem_used_mb = mem_info.used / 1024 / 1024
            mem_free_mb = mem_info.free / 1024 / 1024

            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            # Power
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except Exception:
                power_draw = None
                power_limit = None

            # Fan
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                fan_speed = None

            # Clocks
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                sm_clock = None
                mem_clock = None

            # Processes
            processes: List[Dict[str, Any]] = []
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    proc_name = None
                    try:
                        proc_info = pynvml.nvmlSystemGetProcessName(p.pid)
                        if isinstance(proc_info, bytes):
                            proc_name = proc_info.decode("utf-8", errors="ignore")
                        else:
                            proc_name = proc_info
                    except Exception:
                        proc_name = "unknown"
                    processes.append({
                        "pid": p.pid,
                        "name": proc_name,
                        "used_memory_mb": getattr(p, "usedGpuMemory", 0) / 1024 / 1024,
                    })
            except Exception:
                pass

            devices.append(
                GpuStatus(
                    index=i,
                    name=name,
                    utilization_gpu_pct=gpu_util,
                    utilization_mem_pct=mem_util,
                    memory_total_mb=round(mem_total_mb, 2),
                    memory_used_mb=round(mem_used_mb, 2),
                    memory_free_mb=round(mem_free_mb, 2),
                    temperature_c=temp,
                    power_draw_w=round(power_draw, 2) if power_draw is not None else None,
                    power_limit_w=round(power_limit, 2) if power_limit is not None else None,
                    fan_speed_pct=fan_speed,
                    clock_sm_mhz=sm_clock,
                    clock_mem_mhz=mem_clock,
                    processes=processes,
                    timestamp=datetime.now(),
                )
            )
        except Exception as exc:
            logger.warning("NVML read failed for GPU %d: %s", i, exc)
            devices.append(
                GpuStatus(
                    index=i,
                    name=f"GPU-{i}",
                    timestamp=datetime.now(),
                )
            )
    return devices


# ---------------------------------------------------------------------------
# System state helpers
# ---------------------------------------------------------------------------

_start_time: Optional[datetime] = None

def _is_a_share_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    fixed_holidays = {
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
    return d not in fixed_holidays


def _next_time_on_trading_day(hour: int, minute: int) -> Optional[datetime]:
    now = datetime.now()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while not _is_a_share_trading_day(candidate.date()):
        candidate += timedelta(days=1)
    return candidate


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/system", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """Overall system health, state, and next scheduled actions."""
    global _start_time

    now = datetime.now()
    is_trading = _is_a_share_trading_day(now.date())
    market_open = False
    if is_trading and (9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30)):
        market_open = True

    gpu_count = 0
    gpu_available = False
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpu_available = gpu_count > 0
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # Position count from DB
    active_positions = 0
    try:
        db_path = Path(__file__).resolve().parents[3] / "data" / "portfolio.db"
        if db_path.exists():
            with sqlite3.connect(str(db_path)) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
                active_positions = cur.fetchone()[0]
    except Exception as exc:
        logger.warning("Could not read position count: %s", exc)

    uptime = (now - _start_time).total_seconds() if _start_time else None

    # Count today's recommendations (from cache or compute)
    today_recs = 0
    try:
        from app.api.stocks import _recommendation_cache
        today_recs = len(_recommendation_cache.get(str(date.today()), []))
    except Exception:
        pass

    return SystemStatus(
        state=SystemState.IDLE,
        is_trading_day=is_trading,
        market_open=market_open,
        last_update_time=now,
        next_buy_time=_next_time_on_trading_day(10, 0),
        next_sell_time=_next_time_on_trading_day(9, 30),
        next_scan_time=_next_time_on_trading_day(9, 15),
        next_sync_time=_next_time_on_trading_day(15, 30),
        db_connected=True,
        gpu_available=gpu_available,
        gpu_count=gpu_count,
        active_positions=active_positions,
        today_recommendations=today_recs,
        uptime_seconds=uptime,
    )


@router.get("/gpu", response_model=List[GpuStatus])
async def get_gpu_monitor() -> List[GpuStatus]:
    """Get real-time GPU telemetry for all detected NVIDIA GPUs."""
    gpus = _get_gpu_status()
    if not gpus:
        # Return a mock entry when no GPU is present so the frontend doesn't crash
        return [
            GpuStatus(
                index=0,
                name="NVIDIA H100 80GB HBM3 (Mock – no driver)",
                utilization_gpu_pct=0.0,
                utilization_mem_pct=0.0,
                memory_total_mb=81920.0,
                memory_used_mb=1024.0,
                memory_free_mb=80896.0,
                temperature_c=42.0,
                power_draw_w=85.0,
                power_limit_w=700.0,
                fan_speed_pct=30.0,
                clock_sm_mhz=1200,
                clock_mem_mhz=1593,
                timestamp=datetime.now(),
            )
        ]
    return gpus


@router.get("/logs", response_model=List[LogEntry])
async def get_recent_logs(
    limit: int = 100,
    level: Optional[str] = None,
) -> List[LogEntry]:
    """Get recent system / scheduler operation logs."""
    # We store logs in a simple SQLite table for persistence
    log_db = Path(__file__).resolve().parents[3] / "data" / "logs.db"
    log_db.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sqlite3.connect(str(log_db)) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    level TEXT,
                    source TEXT,
                    message TEXT
                )"""
            )
            if level:
                rows = conn.execute(
                    "SELECT timestamp, level, source, message FROM logs WHERE level = ? ORDER BY id DESC LIMIT ?",
                    (level.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT timestamp, level, source, message FROM logs ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [LogEntry(timestamp=r[0], level=r[1], source=r[2], message=r[3]) for r in rows]
    except Exception as exc:
        logger.error("Log DB read failed: %s", exc)
        # Fallback: return empty
        return []


@router.get("/schedule", response_model=List[ScheduleJobInfo])
async def get_schedule_status() -> List[ScheduleJobInfo]:
    """List all registered APScheduler jobs with next run times."""
    try:
        from app.scheduler.jobs import scheduler_manager
        jobs = scheduler_manager.list_jobs()
        return [
            ScheduleJobInfo(
                id=j["id"],
                name=j["name"],
                next_run_time=j["next_run_time"],
                trigger=j["trigger"],
                last_run_time=j.get("last_run_time"),
                last_run_status=j.get("last_run_status"),
            )
            for j in jobs
        ]
    except Exception as exc:
        logger.error("Schedule query failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Logging bridge: capture Python logging into SQLite
# ---------------------------------------------------------------------------

class SQLiteLogHandler(logging.Handler):
    """Custom handler to persist logs to SQLite for the /api/monitor/logs endpoint."""

    def __init__(self, db_path: Path, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    level TEXT,
                    source TEXT,
                    message TEXT
                )"""
            )
            conn.commit()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "INSERT INTO logs (timestamp, level, source, message) VALUES (?, ?, ?, ?)",
                    (
                        datetime.fromtimestamp(record.created).isoformat(),
                        record.levelname,
                        record.name,
                        self.format(record),
                    ),
                )
                conn.commit()
        except Exception:
            self.handleError(record)


def setup_log_persistence() -> None:
    db_path = Path(__file__).resolve().parents[3] / "data" / "logs.db"
    handler = SQLiteLogHandler(db_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    # Also attach to specific modules
    for name in ("app.scheduler.jobs", "app.api.stocks", "app.api.portfolio", "app.main"):
        logging.getLogger(name).addHandler(handler)
