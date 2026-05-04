"""
websocket_manager.py
====================
Centralised WebSocket connection management.

- Tracks active market and GPU monitor connections
- Provides broadcast helpers for real-time data推送
- Handles connection lifecycle (connect/disconnect/cleanup)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages all WebSocket client connections."""

    def __init__(self) -> None:
        self._market_clients: Set[WebSocket] = set()
        self._gpu_clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect_market(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._market_clients.add(ws)
        logger.info("Market WS client connected (total: %d)", len(self._market_clients))

    async def disconnect_market(self, ws: WebSocket) -> None:
        async with self._lock:
            self._market_clients.discard(ws)
        logger.info("Market WS client disconnected (total: %d)", len(self._market_clients))

    async def connect_gpu(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._gpu_clients.add(ws)
        logger.info("GPU WS client connected (total: %d)", len(self._gpu_clients))

    async def disconnect_gpu(self, ws: WebSocket) -> None:
        async with self._lock:
            self._gpu_clients.discard(ws)
        logger.info("GPU WS client disconnected (total: %d)", len(self._gpu_clients))

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    async def broadcast_market(self, message: str) -> None:
        """Send message to all connected market WebSocket clients."""
        dead: Set[WebSocket] = set()
        for ws in list(self._market_clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._market_clients -= dead

    async def broadcast_gpu(self, message: str) -> None:
        """Send message to all connected GPU monitor WebSocket clients."""
        dead: Set[WebSocket] = set()
        for ws in list(self._gpu_clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._gpu_clients -= dead

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def market_client_count(self) -> int:
        return len(self._market_clients)

    @property
    def gpu_client_count(self) -> int:
        return len(self._gpu_clients)
