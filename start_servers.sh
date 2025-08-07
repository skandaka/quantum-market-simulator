#!/bin/bash

# Quantum Market Simulator - Server Startup Script

echo "🚀 Starting Quantum Market Simulator..."
echo "=================================="

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $port is already in use"
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

# Check ports
echo "🔍 Checking ports..."
check_port 8001
check_port 5173

# Start backend
echo ""
echo "🔧 Starting Backend Server (Port 8001)..."
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Test backend health
echo "🏥 Testing backend health..."
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
fi

# Start frontend
echo ""
echo "🎨 Starting Frontend Server (Port 5173)..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "🎉 Servers started successfully!"
echo "=================================="
echo "📊 Frontend: http://localhost:5173"
echo "🔧 Backend:  http://localhost:8001"
echo "📚 API Docs: http://localhost:8001/docs"
echo ""
echo "📝 PIDs saved for cleanup:"
echo "   Backend:  $BACKEND_PID"
echo "   Frontend: $FRONTEND_PID"
echo ""
echo "🛑 To stop servers:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "✨ Happy coding! The Quantum Market Simulator is ready."

# Keep script running
wait
