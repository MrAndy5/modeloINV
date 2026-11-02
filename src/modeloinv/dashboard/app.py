"""FastAPI Application Factory for modeloINV Performance Dashboard.

Provides:
- create_app: Assembles REST API routes, WebSocket streamer, static file mounting,
  and dependency injection of brokers, database managers, model registries, and live readiness evaluators.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from modeloinv.config import get_config
from modeloinv.dashboard.routes import router as api_router
from modeloinv.dashboard.websockets import ConnectionManager
from modeloinv.execution.paper import SimulatedPaperBroker
from modeloinv.learning.readiness import LiveReadinessEvaluator
from modeloinv.learning.registry import ModelRegistry
from modeloinv.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


def create_app(
    broker: Optional[Any] = None,
    learning_manager: Optional[Any] = None,
    config: Optional[Any] = None,
    db_manager: Optional[Any] = None,
    model_registry: Optional[Any] = None,
    readiness_evaluator: Optional[Any] = None,
) -> FastAPI:
    """Instantiates and configures the FastAPI application."""
    app = FastAPI(
        title="modeloINV Trading API",
        description="Institutional Algorithmic Trading Real-Time Dashboard and REST API",
        version="1.0.0",
    )

    # Configure permissive CORS for local and web UI access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize configuration fallback
    app_config = config
    if app_config is None:
        try:
            app_config = get_config()
        except Exception:
            app_config = None

    # Dependency Injection with fallbacks
    if broker is None:
        broker = SimulatedPaperBroker(initial_capital_eur=100.0)

    if db_manager is None:
        db_path = ":memory:" if os.environ.get("MODELOINV_ENV") == "test" else "data/modeloinv.db"
        try:
            db_manager = DatabaseManager(db_path=db_path)
        except Exception:
            db_manager = None

    if model_registry is None:
        try:
            model_registry = ModelRegistry()
        except Exception:
            model_registry = None

    if readiness_evaluator is None:
        readiness_evaluator = LiveReadinessEvaluator()

    connection_manager = ConnectionManager()

    # Store state on application instance
    app.state.broker = broker
    app.state.learning_manager = learning_manager
    app.state.config = app_config
    app.state.db_manager = db_manager
    app.state.model_registry = model_registry
    app.state.readiness_evaluator = readiness_evaluator
    app.state.connection_manager = connection_manager

    # Register API routers
    app.include_router(api_router)

    # Register WebSocket streaming route
    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        ws_mgr: ConnectionManager = app.state.connection_manager
        await ws_mgr.connect(websocket)
        try:
            # Push initial state tick on connection
            acc = app.state.broker.get_account()
            await websocket.send_json({
                "event": "tick",
                "symbol": "SPY",
                "price": 450.0,
                "portfolio_equity_eur": round(float(acc.equity_eur), 4),
                "timestamp": datetime.datetime.utcnow().isoformat(),
            })
            while True:
                # Maintain connection alive and listen for client ping/messages
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_mgr.disconnect(websocket)
        except Exception as e:
            logger.debug("WebSocket client disconnected or error: %s", e)
            ws_mgr.disconnect(websocket)

    # Mount static assets
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def serve_index():
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return JSONResponse({"status": "ok", "message": "modeloINV Dashboard"})

        @app.get("/index.html")
        async def serve_index_html():
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return JSONResponse({"status": "ok", "message": "modeloINV Dashboard"})

    return app
