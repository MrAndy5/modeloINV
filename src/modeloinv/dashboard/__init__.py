"""modeloINV Live Performance Dashboard Web UI & REST API Backend.

Provides:
- create_app: Application factory for the FastAPI server
- ConnectionManager: Real-time WebSocket event broadcaster
- REST API routes for portfolio, positions, trades, performance, benchmark, learning, readiness, mode toggle
"""

from modeloinv.dashboard.app import create_app
from modeloinv.dashboard.routes import router
from modeloinv.dashboard.websockets import ConnectionManager

__all__ = [
    "create_app",
    "router",
    "ConnectionManager",
]
