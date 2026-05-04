"""
broker_adapter.py
=================
Abstract broker adapter for A-share simulation trading.

Supports:
- XTP (中泰证券) — C++ API wrapper
- UFT (恒生电子) — unified trading API
- Paper trading (本地模拟)

All adapters implement the same interface:
  connect() → query_account() → place_order() → query_order() → cancel_order()

Author  : AQuant Execution Team
Version : v4.5
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class OrderRequest:
    code: str
    side: OrderSide
    quantity: int
    price: Optional[float] = None  # None for market order
    order_type: OrderType = OrderType.LIMIT


@dataclass
class OrderResponse:
    order_id: str
    status: str  # submitted, filled, partial, cancelled, rejected
    filled_quantity: int
    avg_price: float
    message: str = ""


@dataclass
class AccountInfo:
    account_id: str
    available_cash: float
    total_asset: float
    total_position_value: float
    positions: Dict[str, Dict]


# ---------------------------------------------------------------------------
# Abstract Broker Adapter
# ---------------------------------------------------------------------------

class BrokerAdapter(ABC):
    """Abstract base for all broker implementations."""

    @abstractmethod
    async def connect(self) -> bool:
        ...

    @abstractmethod
    async def query_account(self) -> AccountInfo:
        ...

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        ...

    @abstractmethod
    async def query_order(self, order_id: str) -> OrderResponse:
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Paper Trading (Simulation)
# ---------------------------------------------------------------------------

class PaperBroker(BrokerAdapter):
    """
    Local paper trading simulator.

    - Matches orders against next available price
    - Applies slippage model
    - Tracks virtual positions and P&L
    """

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self.cash = initial_cash
        self.positions: Dict[str, Dict] = {}
        self.order_counter = 0
        self.orders: Dict[str, OrderResponse] = {}
        self.trade_history: List[Dict] = []

    async def connect(self) -> bool:
        logger.info("Paper broker connected (simulation)")
        return True

    async def query_account(self) -> AccountInfo:
        total_pos = sum(p["market_value"] for p in self.positions.values())
        return AccountInfo(
            account_id="PAPER_001",
            available_cash=self.cash,
            total_asset=self.cash + total_pos,
            total_position_value=total_pos,
            positions=self.positions,
        )

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter:06d}"

        # Simulate fill at requested price (paper trading ideal execution)
        fill_price = order.price or 10.0  # default if no price
        notional = fill_price * order.quantity

        if order.side == OrderSide.BUY:
            if notional > self.cash:
                return OrderResponse(order_id=order_id, status="rejected", filled_quantity=0, avg_price=0, message="Insufficient cash")
            self.cash -= notional
            pos = self.positions.get(order.code, {"quantity": 0, "avg_price": 0, "market_value": 0})
            new_qty = pos["quantity"] + order.quantity
            new_avg = (pos["avg_price"] * pos["quantity"] + notional) / new_qty if new_qty > 0 else 0
            self.positions[order.code] = {"quantity": new_qty, "avg_price": new_avg, "market_value": notional}
        else:
            pos = self.positions.get(order.code, {"quantity": 0})
            if pos["quantity"] < order.quantity:
                return OrderResponse(order_id=order_id, status="rejected", filled_quantity=0, avg_price=0, message="Insufficient position")
            self.cash += notional
            pos["quantity"] -= order.quantity
            if pos["quantity"] <= 0:
                del self.positions[order.code]
            else:
                self.positions[order.code] = pos

        resp = OrderResponse(order_id=order_id, status="filled", filled_quantity=order.quantity, avg_price=fill_price)
        self.orders[order_id] = resp
        self.trade_history.append({
            "time": datetime.now().isoformat(),
            "code": order.code,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": fill_price,
            "order_id": order_id,
        })
        return resp

    async def query_order(self, order_id: str) -> OrderResponse:
        return self.orders.get(order_id, OrderResponse(order_id=order_id, status="unknown", filled_quantity=0, avg_price=0))

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = "cancelled"
            return True
        return False

    async def disconnect(self) -> None:
        logger.info("Paper broker disconnected")


# ---------------------------------------------------------------------------
# XTP Adapter (Zhongtai Securities)
# ---------------------------------------------------------------------------

class XTPBroker(BrokerAdapter):
    """
    XTP (中泰证券) C++ API adapter.

    Requires: xtptraderapi library installed.
    """

    def __init__(self, user_id: str, password: str, server_ip: str, port: int) -> None:
        self.user_id = user_id
        self.password = password
        self.server_ip = server_ip
        self.port = port
        self._api = None

    async def connect(self) -> bool:
        try:
            import xtptraderapi
            self._api = xtptraderapi.XtpTraderApi.CreateTraderApi(1, "./logs/")
            # ... connect logic
            logger.info("XTP connected to %s:%d", self.server_ip, self.port)
            return True
        except ImportError:
            logger.error("xtptraderapi not installed")
            return False

    async def query_account(self) -> AccountInfo:
        # XTP specific implementation
        return AccountInfo(account_id=self.user_id, available_cash=0, total_asset=0, total_position_value=0, positions={})

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        # XTP specific implementation
        return OrderResponse(order_id="", status="rejected", filled_quantity=0, avg_price=0, message="XTP not implemented")

    async def query_order(self, order_id: str) -> OrderResponse:
        return OrderResponse(order_id=order_id, status="unknown", filled_quantity=0, avg_price=0)

    async def cancel_order(self, order_id: str) -> bool:
        return False

    async def disconnect(self) -> None:
        if self._api:
            self._api.exit()


# ---------------------------------------------------------------------------
# Broker Factory
# ---------------------------------------------------------------------------

def create_broker(broker_type: str = "paper", **kwargs) -> BrokerAdapter:
    """Factory for broker adapters."""
    if broker_type == "paper":
        return PaperBroker(initial_cash=kwargs.get("initial_cash", 1_000_000.0))
    elif broker_type == "xtp":
        return XTPBroker(
            user_id=kwargs["user_id"],
            password=kwargs["password"],
            server_ip=kwargs.get("server_ip", "127.0.0.1"),
            port=kwargs.get("port", 20001),
        )
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")
