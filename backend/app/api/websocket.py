"""WebSocket implementation for real-time updates"""

import json
import asyncio
import logging
from typing import Dict, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from starlette.applications import WebSocketRoute
from starlette.routing import Mount
from starlette.websockets import WebSocketState

from app.models.schemas import WSMessage, SimulationProgress

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of channels

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        """Remove connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.subscriptions[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def subscribe(self, client_id: str, channel: str):
        """Subscribe client to channel"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].add(channel)
            await self.send_personal_message(
                f"Subscribed to {channel}",
                client_id,
                msg_type="subscription"
            )

    async def unsubscribe(self, client_id: str, channel: str):
        """Unsubscribe client from channel"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(channel)
            await self.send_personal_message(
                f"Unsubscribed from {channel}",
                client_id,
                msg_type="subscription"
            )

    async def send_personal_message(
            self,
            message: str,
            client_id: str,
            msg_type: str = "info"
    ):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]

            ws_message = WSMessage(
                type=msg_type,
                data={"message": message},
                timestamp=datetime.utcnow()
            )

            try:
                await websocket.send_json(ws_message.dict())
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast_to_channel(
            self,
            channel: str,
            message: Dict,
            msg_type: str = "update"
    ):
        """Broadcast message to all clients subscribed to channel"""
        ws_message = WSMessage(
            type=msg_type,
            data=message,
            timestamp=datetime.utcnow()
        )

        # Find all clients subscribed to this channel
        for client_id, channels in self.subscriptions.items():
            if channel in channels:
                if client_id in self.active_connections:
                    websocket = self.active_connections[client_id]
                    try:
                        await websocket.send_json(ws_message.dict())
                    except Exception as e:
                        logger.error(f"Failed to broadcast to {client_id}: {e}")
                        self.disconnect(client_id)

    async def send_simulation_progress(
            self,
            request_id: str,
            stage: str,
            progress: float,
            message: str
    ):
        """Send simulation progress update"""
        progress_update = SimulationProgress(
            request_id=request_id,
            stage=stage,
            progress=progress,
            message=message
        )

        await self.broadcast_to_channel(
            f"simulation:{request_id}",
            progress_update.dict(),
            msg_type="progress"
        )


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint"""
    client_id = websocket.query_params.get("client_id", str(id(websocket)))

    await manager.connect(websocket, client_id)

    try:
        # Send welcome message
        await manager.send_personal_message(
            "Connected to Quantum Market Simulator",
            client_id,
            msg_type="welcome"
        )

        # Listen for messages
        while True:
            data = await websocket.receive_json()

            # Handle different message types
            if data.get("type") == "subscribe":
                channel = data.get("channel")
                if channel:
                    await manager.subscribe(client_id, channel)

            elif data.get("type") == "unsubscribe":
                channel = data.get("channel")
                if channel:
                    await manager.unsubscribe(client_id, channel)

            elif data.get("type") == "ping":
                # Respond to ping
                await manager.send_personal_message(
                    "pong",
                    client_id,
                    msg_type="pong"
                )

            else:
                # Echo unknown messages
                await manager.send_personal_message(
                    f"Unknown message type: {data.get('type')}",
                    client_id,
                    msg_type="error"
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


# Create WebSocket app
app = Mount("/", routes=[
    WebSocketRoute("/", websocket_endpoint)
])


# Helper functions for other services to send updates
async def send_market_update(asset: str, data: Dict):
    """Send market data update"""
    await manager.broadcast_to_channel(
        f"market:{asset}",
        data,
        msg_type="market_update"
    )


async def send_simulation_update(request_id: str, update: Dict):
    """Send simulation update"""
    await manager.broadcast_to_channel(
        f"simulation:{request_id}",
        update,
        msg_type="simulation_update"
    )