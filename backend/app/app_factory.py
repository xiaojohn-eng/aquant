"""
app_factory.py
==============
FastAPI application factory.

Creates and configures the AQuant FastAPI app with all routes,
middleware, event handlers, and lifespan management.
"""
from __future__ import annotations

import asyncio
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.state_machine import StateMachine, SystemState
from app.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

# Shared state (injected into app.state)
_state_machine = StateMachine()
_ws_manager = WebSocketManager()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan — startup and shutdown."""
    logger.info("=" * 60)
    logger.info("AQuant v4.4 — Starting up...")
    logger.info("=" * 60)

    # Startup
    await _state_machine.transition(SystemState.RUNNING, "startup complete")

    # Schedule background tasks
    app.state._gpu_task = asyncio.create_task(_gpu_push_loop())
    app.state._snapshot_task = asyncio.create_task(_snapshot_loop())

    yield

    # Shutdown
    await _state_machine.transition(SystemState.IDLE, "shutdown")
    app.state._gpu_task.cancel()
    app.state._snapshot_task.cancel()
    logger.info("AQuant — Shut down complete")


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _gpu_push_loop() -> None:
    """Periodically push GPU status to WebSocket clients."""
    while True:
        try:
            await asyncio.sleep(5)
            from app.api.monitor import get_gpu_status
            status = get_gpu_status()
            await _ws_manager.broadcast_gpu(json.dumps(status, default=str))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("GPU push error: %s", exc)


async def _snapshot_loop() -> None:
    """Periodically send market snapshot to WebSocket clients."""
    while True:
        try:
            await asyncio.sleep(30)
            snapshot = {
                "type": "snapshot",
                "time": datetime.now().isoformat(),
                "state": _state_machine.state_sync.value,
            }
            await _ws_manager.broadcast_market(json.dumps(snapshot))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("Snapshot error: %s", exc)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the AQuant FastAPI application."""
    app = FastAPI(
        title="AQuant — A-Share Quantitative Trading System",
        description="GPU-accelerated T+1 overnight-reversal strategy",
        version="4.4.0",
        lifespan=app_lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Exception handler
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        await _state_machine.transition(SystemState.ERROR, str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(exc).__name__},
        )

    # Inject shared state
    app.state.sm = _state_machine
    app.state.ws = _ws_manager

    # Register routers
    from app.api.stocks import router as stocks_router
    from app.api.portfolio import router as portfolio_router
    from app.api.monitor import router as monitor_router

    app.include_router(stocks_router, prefix="/api")
    app.include_router(portfolio_router, prefix="/api")
    app.include_router(monitor_router, prefix="/api")

    # Root
    @app.get("/")
    async def root():
        return {
            "message": "AQuant API v4.4 — A-Share Quantitative Trading System",
            "docs": "/docs",
            "version": "4.4.0",
        }

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "4.4.0", "timestamp": datetime.now().isoformat()}

    # WebSocket endpoints
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws/market")
    async def market_ws(websocket: WebSocket):
        await app.state.ws.connect_market(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(f"Echo: {data}")
        except WebSocketDisconnect:
            await app.state.ws.disconnect_market(websocket)

    @app.websocket("/ws/gpu")
    async def gpu_ws(websocket: WebSocket):
        await app.state.ws.connect_gpu(websocket)
        try:
            while True:
                await websocket.receive_text()  # keepalive
        except WebSocketDisconnect:
            await app.state.ws.disconnect_gpu(websocket)

    logger.info("FastAPI app created — 4.4.0")
    return app
