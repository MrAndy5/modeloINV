"""WebSocket connection management and real-time streaming for modeloINV dashboard.

Provides:
- ConnectionManager: Handles WebSocket subscriber lifecycles, heartbeats, and
  thread-safe broadcast of market ticks, order fills, equity updates, and system mode toggles.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import threading
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts real-time streaming events."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self._thread_lock = threading.RLock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts an incoming WebSocket connection and registers it."""
        await websocket.accept()
        with self._thread_lock:
            async with self._lock:
                self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Active connections: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Removes a disconnected WebSocket client with thread-safe synchronization."""
        with self._thread_lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                logger.info("WebSocket client removed. Active connections: %d", len(self.active_connections))

    async def send_json(self, message: Dict[str, Any], websocket: WebSocket) -> bool:
        """Sends a JSON payload to a specific client. Returns True if successful."""
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.debug("Failed to send message to websocket client: %s", e)
            self.disconnect(websocket)
            return False

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Broadcasts a JSON message to all active subscribers.

        Returns:
            The number of clients successfully notified.
        """
        with self._thread_lock:
            async with self._lock:
                subscribers = list(self.active_connections)

        if not subscribers:
            return 0

        success_count = 0
        dead_sockets: List[WebSocket] = []

        for ws in subscribers:
            try:
                await ws.send_json(message)
                success_count += 1
            except Exception as e:
                logger.debug("Broadcast failed for socket: %s", e)
                dead_sockets.append(ws)

        if dead_sockets:
            with self._thread_lock:
                async with self._lock:
                    for dead in dead_sockets:
                        if dead in self.active_connections:
                            self.active_connections.remove(dead)

        return success_count

    async def broadcast_tick(
        self,
        symbol: str,
        price: float,
        portfolio_equity_eur: float,
        timestamp: Optional[str] = None,
    ) -> int:
        """Pushes a real-time price tick and current portfolio equity."""
        if timestamp is None:
            timestamp = datetime.datetime.utcnow().isoformat()

        payload = {
            "event": "tick",
            "symbol": symbol,
            "price": round(float(price), 4),
            "portfolio_equity_eur": round(float(portfolio_equity_eur), 4),
            "timestamp": timestamp,
        }
        return await self.broadcast(payload)

    async def broadcast_trade(self, trade_data: Dict[str, Any]) -> int:
        """Pushes an executed trade fill notification."""
        payload = {
            "event": "trade_fill",
            "trade": trade_data,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        return await self.broadcast(payload)

    async def broadcast_equity(
        self,
        equity_eur: float,
        cash_eur: float,
        realized_pnl_eur: float,
        unrealized_pnl_eur: float,
    ) -> int:
        """Pushes an updated portfolio balance and P&L snapshot."""
        payload = {
            "event": "equity_update",
            "equity_eur": round(float(equity_eur), 4),
            "cash_eur": round(float(cash_eur), 4),
            "realized_pnl_eur": round(float(realized_pnl_eur), 4),
            "unrealized_pnl_eur": round(float(unrealized_pnl_eur), 4),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        return await self.broadcast(payload)

    async def broadcast_mode(self, active_mode: str, is_live: bool) -> int:
        """Pushes an execution mode transition event."""
        payload = {
            "event": "mode_change",
            "active_mode": active_mode,
            "is_live": is_live,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        return await self.broadcast(payload)
