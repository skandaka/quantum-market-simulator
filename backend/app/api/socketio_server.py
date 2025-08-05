"""Socket.IO implementation for real-time updates"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import socketio
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Create Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False
)

# Connection management
connected_clients: Dict[str, Any] = {}


@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    logger.info(f"Client {sid} connected")
    connected_clients[sid] = {
        "connected_at": datetime.utcnow(),
        "subscriptions": set()
    }
    await sio.emit('welcome', {'message': 'Connected to Quantum Market Simulator'}, to=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client {sid} disconnected")
    if sid in connected_clients:
        del connected_clients[sid]


@sio.event
async def subscribe(sid, data):
    """Handle subscription to channels"""
    channel = data.get('channel')
    if channel and sid in connected_clients:
        connected_clients[sid]['subscriptions'].add(channel)
        await sio.emit('subscription', {
            'message': f'Subscribed to {channel}',
            'channel': channel
        }, to=sid)


@sio.event
async def unsubscribe(sid, data):
    """Handle unsubscription from channels"""
    channel = data.get('channel')
    if channel and sid in connected_clients:
        connected_clients[sid]['subscriptions'].discard(channel)
        await sio.emit('subscription', {
            'message': f'Unsubscribed from {channel}',
            'channel': channel
        }, to=sid)


@sio.event
async def ping(sid, data):
    """Handle ping-pong for connection health"""
    await sio.emit('pong', {'timestamp': datetime.utcnow().isoformat()}, to=sid)


# Helper functions for broadcasting
async def broadcast_to_channel(channel: str, message: Dict, event_type: str = 'update'):
    """Broadcast message to all clients subscribed to a channel"""
    if not connected_clients:
        return
    
    # Find all clients subscribed to this channel
    target_clients = [
        sid for sid, client_data in connected_clients.items()
        if channel in client_data['subscriptions']
    ]
    
    if target_clients:
        await sio.emit(event_type, {
            'channel': channel,
            'data': message,
            'timestamp': datetime.utcnow().isoformat()
        }, to=target_clients)


async def send_simulation_progress(request_id: str, stage: str, progress: float, message: str):
    """Send simulation progress update"""
    await broadcast_to_channel(
        f"simulation:{request_id}",
        {
            'request_id': request_id,
            'stage': stage,
            'progress': progress,
            'message': message
        },
        'progress'
    )


async def send_simulation_result(request_id: str, result: Dict):
    """Send simulation completion result"""
    await broadcast_to_channel(
        f"simulation:{request_id}",
        result,
        'result'
    )


async def send_error(request_id: str, error_message: str):
    """Send error message"""
    await broadcast_to_channel(
        f"simulation:{request_id}",
        {
            'request_id': request_id,
            'error': error_message
        },
        'error'
    )


def create_socketio_app():
    """Create FastAPI app with Socket.IO integration"""
    # Create a minimal Socket.IO app
    socketio_app = socketio.ASGIApp(sio, socketio_path='/ws/')
    return socketio_app


# Export the Socket.IO server for use in main app
__all__ = ['sio', 'create_socketio_app', 'send_simulation_progress', 'send_simulation_result', 'send_error']
