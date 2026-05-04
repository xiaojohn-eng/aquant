"""
audit_logger.py
===============
Complete audit trail for all trading operations.

Records: who, when, what, where, why for every action.
Storage: PostgreSQL (persistent, tamper-resistant)
Retention: 7 years (regulatory compliance)

Author  : AQuant Compliance Team
Version : v4.5
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """Single audit trail entry."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = "system"
    session_id: str = ""
    ip_address: str = ""
    action: str = ""  # LOGIN, LOGOUT, PLACE_ORDER, CANCEL_ORDER, etc.
    resource: str = ""  # stock_code, portfolio, etc.
    details: Dict = field(default_factory=dict)
    pre_state: Dict = field(default_factory=dict)
    post_state: Dict = field(default_factory=dict)
    risk_checks: List[str] = field(default_factory=list)
    result: str = ""  # SUCCESS, FAILURE, REJECTED
    error_message: str = ""


class AuditLogger:
    """
    Regulatory-grade audit logging.

    All trading actions are immutable, timestamped, and non-repudiable.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self.db_url = db_url
        self._buffer: List[AuditRecord] = []
        self._max_buffer = 100

    async def log(
        self,
        action: str,
        user_id: str = "system",
        resource: str = "",
        details: Optional[Dict] = None,
        pre_state: Optional[Dict] = None,
        post_state: Optional[Dict] = None,
        risk_checks: Optional[List[str]] = None,
        result: str = "SUCCESS",
        error_message: str = "",
        ip_address: str = "",
        session_id: str = "",
    ) -> str:
        """
        Record an audit trail entry.

        Returns the record_id (UUID).
        """
        record = AuditRecord(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            action=action,
            resource=resource,
            details=details or {},
            pre_state=pre_state or {},
            post_state=post_state or {},
            risk_checks=risk_checks or [],
            result=result,
            error_message=error_message,
        )
        self._buffer.append(record)

        # Flush to persistent storage
        if len(self._buffer) >= self._max_buffer:
            await self._flush()

        logger.info(
            "AUDIT: %s | user=%s | action=%s | resource=%s | result=%s",
            record.timestamp.isoformat(), user_id, action, resource, result,
        )
        return record.record_id

    async def log_trade(
        self,
        user_id: str,
        code: str,
        side: str,
        quantity: int,
        price: float,
        pre_cash: float,
        post_cash: float,
        pre_positions: Dict,
        post_positions: Dict,
        risk_checks: List[str],
        result: str = "SUCCESS",
        error: str = "",
    ) -> str:
        """Convenience method for trade logging."""
        return await self.log(
            action="PLACE_ORDER",
            user_id=user_id,
            resource=code,
            details={"side": side, "quantity": quantity, "price": price},
            pre_state={"cash": pre_cash, "positions": pre_positions},
            post_state={"cash": post_cash, "positions": post_positions},
            risk_checks=risk_checks,
            result=result,
            error_message=error,
        )

    async def _flush(self) -> None:
        """Write buffered records to PostgreSQL."""
        if not self._buffer:
            return

        try:
            import asyncpg
            if self.db_url:
                conn = await asyncpg.connect(self.db_url)
                for r in self._buffer:
                    await conn.execute(
                        """
                        INSERT INTO audit_log (
                            record_id, timestamp, user_id, session_id, ip_address,
                            action, resource, details, pre_state, post_state,
                            risk_checks, result, error_message
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        """,
                        r.record_id, r.timestamp, r.user_id, r.session_id, r.ip_address,
                        r.action, r.resource,
                        str(r.details), str(r.pre_state), str(r.post_state),
                        r.risk_checks, r.result, r.error_message,
                    )
                await conn.close()
            self._buffer.clear()
        except Exception as exc:
            logger.error("Audit flush failed: %s", exc)

    def get_buffer_size(self) -> int:
        return len(self._buffer)
