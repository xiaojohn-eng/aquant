"""
state_machine.py
================
System state management for AQuant.

Tracks: IDLE → RUNNING → PAUSED → ERROR → IDLE
Used by: scheduler, WebSocket broadcast, API endpoints
"""
from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SystemState(str, Enum):
    """Finite state machine states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class StateMachine:
    """Thread-safe system state machine."""

    def __init__(self) -> None:
        self._state = SystemState.IDLE
        self._lock = asyncio.Lock()
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def transition(self, new_state: SystemState, reason: str = "") -> bool:
        async with self._lock:
            old = self._state
            if old == new_state:
                return True
            # Validate transition
            valid = self._validate_transition(old, new_state)
            if valid:
                self._state = new_state
                if new_state == SystemState.ERROR:
                    self._last_error = reason
                logger.info("State: %s → %s (%s)", old.value, new_state.value, reason)
            else:
                logger.warning("Invalid transition: %s → %s", old.value, new_state.value)
            return valid

    def _validate_transition(self, old: SystemState, new: SystemState) -> bool:
        """Define valid state transitions."""
        valid_map = {
            SystemState.IDLE: {SystemState.RUNNING, SystemState.ERROR},
            SystemState.RUNNING: {SystemState.PAUSED, SystemState.ERROR, SystemState.IDLE},
            SystemState.PAUSED: {SystemState.RUNNING, SystemState.IDLE, SystemState.ERROR},
            SystemState.ERROR: {SystemState.IDLE},
        }
        return new in valid_map.get(old, set())

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    async def state(self) -> SystemState:
        async with self._lock:
            return self._state

    @property
    def state_sync(self) -> SystemState:
        return self._state

    @property
    async def last_error(self) -> Optional[str]:
        async with self._lock:
            return self._last_error

    def is_running(self) -> bool:
        return self._state == SystemState.RUNNING
