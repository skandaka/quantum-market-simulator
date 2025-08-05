"""Minimal FastAPI server for testing WebSocket and simulation endpoints"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn
import json
import asyncio
from typing import Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False
)

# Create FastAPI app
app = FastAPI(title="Quantum Market Simulator - Minimal")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Mount Socket.IO
socketio_app = socketio.ASGIApp(sio, socketio_path='/ws/')
app.mount('/ws', socketio_app)

# API Routes
@app.get("/")
async def root():
    return {
        "message": "Quantum Market Simulator - Minimal Version",
        "status": "running",
        "endpoints": {
            "simulate": "/api/v1/simulate",
            "health": "/health",
            "websocket": "/ws/"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "minimal-1.0"
    }

@app.post("/api/v1/simulate")
async def simulate(request: Dict[str, Any]):
    """Simple simulation endpoint that returns mock data"""
    logger.info(f"Received simulation request: {request}")
    
    # Generate mock response
    request_id = f"sim_{int(datetime.utcnow().timestamp())}"
    
    # Send progress updates via WebSocket
    await send_progress(request_id, "starting", 0.1, "Initializing simulation...")
    await asyncio.sleep(0.5)
    
    await send_progress(request_id, "processing", 0.5, "Processing market data...")
    await asyncio.sleep(0.5)
    
    await send_progress(request_id, "calculating", 0.8, "Running quantum calculations...")
    await asyncio.sleep(0.5)
    
    await send_progress(request_id, "completed", 1.0, "Simulation complete!")
    
    # Mock simulation result
    result = {
        "request_id": request_id,
        "status": "completed",
        "timestamp": datetime.utcnow().isoformat(),
        "execution_time_seconds": 2.0,
        "quantum_advantage_ratio": 1.15,
        "predictions": [
            {
                "asset": asset,
                "predicted_return": 0.025,  # 2.5% return
                "confidence_interval": [0.01, 0.04],
                "quantum_probability": 0.87,
                "classical_probability": 0.73,
                "sentiment_impact": 0.12,
                "volatility": 0.18
            }
            for asset in request.get('target_assets', ['AAPL', 'GOOGL', 'MSFT'])
        ],
        "quantum_metrics": {
            "circuit_depth": 12,
            "num_qubits": 8,
            "gate_count": 48,
            "coherence_time": 0.94,
            "fidelity": 0.98
        },
        "enhanced_features": {
            "used_quantum_nlp": True,
            "used_quantum_monte_carlo": True,
            "portfolio_optimization": False,
            "risk_analysis": True
        }
    }
    
    return result

async def send_progress(request_id: str, stage: str, progress: float, message: str):
    """Send progress update via WebSocket"""
    await sio.emit('progress', {
        'request_id': request_id,
        'stage': stage,
        'progress': progress,
        'message': message,
        'timestamp': datetime.utcnow().isoformat()
    })

if __name__ == "__main__":
    logger.info("Starting Quantum Market Simulator - Minimal Version")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
