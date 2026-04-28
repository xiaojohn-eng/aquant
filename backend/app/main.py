"""FastAPI main application entry point.

Features:
- Lifespan context manager (init DB, scheduler, GPU warm-up)
- REST routers (stocks, portfolio, monitor)
- WebSocket endpoints (/ws/market, /ws/gpu)
- CORS for frontend dev server (port 5173)
- Static file mount for production build
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging setup (before anything else)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app.main")

# ---------------------------------------------------------------------------
# Import sub-modules
# ---------------------------------------------------------------------------

from app.models.schemas import (
    GpuStatus,
    PortfolioPosition,
    StockRecommendation,
    SystemState,
    SystemStatus,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_start_time: Optional[datetime] = None
_market_ws_clients: Set[WebSocket] = set()
_gpu_ws_clients: Set[WebSocket] = set()
_system_state: SystemState = SystemState.IDLE

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _start_time
    _start_time = datetime.now()
    logger.info("=== Aquant Backend Starting ===")

    # 1. Initialise portfolio database
    try:
        from app.api.portfolio import init_portfolio_db
        await init_portfolio_db()
        logger.info("Portfolio database ready")
    except Exception as exc:
        logger.error("Portfolio DB init failed: %s", exc)

    # 2. Initialise log persistence (SQLite handler)
    try:
        from app.api.monitor import setup_log_persistence
        setup_log_persistence()
        logger.info("Log persistence enabled")
    except Exception as exc:
        logger.warning("Log persistence setup failed: %s", exc)

    # 3. Start APScheduler
    try:
        from app.scheduler.jobs import scheduler_manager
        scheduler_manager.start()
        logger.info("Scheduler started")
    except Exception as exc:
        logger.error("Scheduler start failed: %s", exc)

    # 4. GPU warm-up
    try:
        from app.scheduler.jobs import _warmup_gpu
        await _warmup_gpu()
        logger.info("GPU warm-up completed")
    except Exception as exc:
        logger.warning("GPU warm-up failed: %s", exc)

    # 5. Initialise Feature Store
    try:
        from app.plugins.feature_store import FeatureStore
        feature_store = FeatureStore()
        logger.info("Feature Store ready (schema v%s)", FeatureStore.SCHEMA_VERSION)
    except Exception as exc:
        logger.warning("Feature Store init failed: %s", exc)

    # 6. Setup Prometheus metrics
    try:
        from app.monitor.prometheus_metrics import setup_prometheus
        setup_prometheus(app)
        logger.info("Prometheus metrics mounted at /metrics")
    except Exception as exc:
        logger.warning("Prometheus setup failed: %s", exc)

    # 7. Seed system state
    _update_system_state(SystemState.IDLE, "System ready v2.0")

    # 8. Start background GPU push loop
    asyncio.create_task(_gpu_push_loop())
    logger.info("Background GPU push loop started")

    logger.info("=== Startup complete – serving requests ===")
    yield

    # ---- Shutdown ----
    logger.info("=== Aquant Backend Shutting Down ===")
    try:
        from app.scheduler.jobs import scheduler_manager
        scheduler_manager.shutdown()
        logger.info("Scheduler shut down")
    except Exception as exc:
        logger.warning("Scheduler shutdown error: %s", exc)

    # Close remaining WebSockets gracefully
    for ws in list(_market_ws_clients):
        try:
            await ws.close(code=1001, reason="Server shutting down")
        except Exception:
            pass
    _market_ws_clients.clear()
    for ws in list(_gpu_ws_clients):
        try:
            await ws.close(code=1001, reason="Server shutting down")
        except Exception:
            pass
    _gpu_ws_clients.clear()

    logger.info("=== Shutdown complete ===")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Aquant Quantitative Trading API",
        description="HTTP and WebSocket backend for A-share quantitative strategy execution. "
                    "v2.0 with Feature Store, ML Ranker, Dual Data Sources, and Prometheus monitoring.",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS – allow frontend dev server (Vite default 5173) and production origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://localhost:3000",
            "https://aquant.local",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health / root
    @app.get("/health", tags=["health"])
    async def health_check() -> Dict[str, Any]:
        uptime = (datetime.now() - _start_time).total_seconds() if _start_time else 0
        return {"status": "ok", "version": "2.0.0", "uptime_sec": uptime, "features": ["feature_store", "ml_ranker", "dual_source", "prometheus", "wfa_backtest"]}

    @app.get("/", tags=["health"])
    async def root() -> Dict[str, str]:
        return {"message": "Aquant Backend API", "docs": "/docs", "version": "1.0.0"}

    # Include routers
    from app.api import monitor, portfolio, stocks
    app.include_router(stocks.router)
    app.include_router(portfolio.router)
    app.include_router(monitor.router)

    # Static files (production build)
    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
        logger.info("Mounted static files from %s", frontend_dist)
    else:
        logger.info("No frontend/dist found – static mount skipped")

    # Register WebSocket routes
    _register_websocket_routes(app)

    # Exception handler
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": str(exc), "detail": "Internal server error"},
        )

    return app


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------

def _register_websocket_routes(app: FastAPI) -> None:

    @app.websocket("/ws/market")
    async def ws_market(websocket: WebSocket) -> None:
        """Push real-time recommendations and portfolio updates to clients."""
        await websocket.accept()
        _market_ws_clients.add(websocket)
        logger.info("Market WebSocket client connected (%d total)", len(_market_ws_clients))
        try:
            # Send initial snapshot
            await _send_market_snapshot(websocket)
            # Keep connection alive and handle client pings / commands
            async for message in websocket.iter_text():
                if message == "ping":
                    await websocket.send_text("pong")
                elif message == "refresh":
                    await _send_market_snapshot(websocket)
                else:
                    # Could be a subscription command in future
                    await websocket.send_json({"type": "ack", "payload": message})
        except WebSocketDisconnect:
            logger.info("Market WebSocket client disconnected")
        except Exception as exc:
            logger.warning("Market WS error: %s", exc)
        finally:
            _market_ws_clients.discard(websocket)

    @app.websocket("/ws/gpu")
    async def ws_gpu(websocket: WebSocket) -> None:
        """Push real-time GPU telemetry to clients."""
        await websocket.accept()
        _gpu_ws_clients.add(websocket)
        logger.info("GPU WebSocket client connected (%d total)", len(_gpu_ws_clients))
        try:
            # Send initial snapshot
            await _send_gpu_snapshot(websocket)
            async for message in websocket.iter_text():
                if message == "ping":
                    await websocket.send_text("pong")
                elif message == "refresh":
                    await _send_gpu_snapshot(websocket)
                else:
                    await websocket.send_json({"type": "ack", "payload": message})
        except WebSocketDisconnect:
            logger.info("GPU WebSocket client disconnected")
        except Exception as exc:
            logger.warning("GPU WS error: %s", exc)
        finally:
            _gpu_ws_clients.discard(websocket)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

async def _send_market_snapshot(websocket: WebSocket) -> None:
    """Send current recommendations + positions to a single client."""
    payload: Dict[str, Any] = {"type": "market_snapshot", "timestamp": datetime.now().isoformat()}
    try:
        from app.api.stocks import get_today_recommendations
        recs = await get_today_recommendations(top_n=20, refresh=False)
        payload["recommendations"] = [r.model_dump() for r in recs]
    except Exception as exc:
        payload["recommendations"] = []
        payload["recommendations_error"] = str(exc)

    try:
        from app.api.portfolio import get_positions
        positions = await get_positions()
        payload["positions"] = [p.model_dump() for p in positions]
    except Exception as exc:
        payload["positions"] = []
        payload["positions_error"] = str(exc)

    await websocket.send_json(payload)


async def _send_gpu_snapshot(websocket: WebSocket) -> None:
    """Send current GPU status to a single client."""
    try:
        from app.api.monitor import _get_gpu_status
        gpus = _get_gpu_status()
        payload = {
            "type": "gpu_snapshot",
            "timestamp": datetime.now().isoformat(),
            "gpus": [g.model_dump() for g in gpus],
        }
    except Exception as exc:
        payload = {
            "type": "gpu_snapshot",
            "timestamp": datetime.now().isoformat(),
            "gpus": [],
            "error": str(exc),
        }
    await websocket.send_json(payload)


# ---------------------------------------------------------------------------
# Broadcast helpers (used by scheduler jobs / API layer)
# ---------------------------------------------------------------------------

async def broadcast_market_update(data: Dict[str, Any]) -> None:
    """Push a JSON message to all connected market WebSocket clients."""
    if not _market_ws_clients:
        return
    dead: Set[WebSocket] = set()
    message = {**data, "timestamp": datetime.now().isoformat()}
    for ws in _market_ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _market_ws_clients.discard(ws)


async def broadcast_gpu_update(data: Dict[str, Any]) -> None:
    """Push a JSON message to all connected GPU WebSocket clients."""
    if not _gpu_ws_clients:
        return
    dead: Set[WebSocket] = set()
    message = {**data, "timestamp": datetime.now().isoformat()}
    for ws in _gpu_ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _gpu_ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# System state helpers
# ---------------------------------------------------------------------------

def _update_system_state(state: SystemState, message: str = "") -> None:
    global _system_state
    _system_state = state
    logger.info("System state -> %s | %s", state.value, message)


# ---------------------------------------------------------------------------
# Background loop: periodic GPU push when clients are connected
# ---------------------------------------------------------------------------

async def _gpu_push_loop() -> None:
    """Periodically broadcast GPU status (~1 Hz) when at least one client is connected."""
    while True:
        await asyncio.sleep(5)
        if _gpu_ws_clients:
            try:
                from app.api.monitor import _get_gpu_status
                gpus = _get_gpu_status()
                await broadcast_gpu_update({"type": "gpu_tick", "gpus": [g.model_dump() for g in gpus]})
            except Exception as exc:
                logger.debug("GPU push loop error: %s", exc)


# ---------------------------------------------------------------------------
# Application instance (module-level for uvicorn)
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
