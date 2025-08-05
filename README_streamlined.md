# 🚀 Quantum Market Simulator

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue)](https://typescriptlang.org)
[![Classiq](https://img.shields.io/badge/Powered%20by-Classiq-orange)](https://classiq.io)

A cutting-edge financial simulation platform that combines quantum computing with classical machine learning for advanced market prediction and portfolio optimization.

## 🚀 Features

- **Quantum-Enhanced Predictions**: Leverages quantum computing for superior market analysis
- **Hybrid ML Models**: Combines quantum and classical machine learning approaches
- **Real-time Sentiment Analysis**: Advanced NLP with quantum-enhanced sentiment vectors
- **Portfolio Optimization**: Quantum algorithms for optimal asset allocation
- **Interactive Visualizations**: 3D quantum state visualizations and market dashboards
- **WebSocket Support**: Real-time data streaming and updates

## 🏗️ Architecture

```
quantum-market-simulator/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/            # REST API endpoints & WebSocket
│   │   ├── quantum/        # Quantum computing modules
│   │   ├── services/       # Business logic & simulators
│   │   ├── models/         # Data models & schemas
│   │   └── utils/          # Helper functions
│   ├── examples/           # Usage examples
│   └── scripts/            # Setup & maintenance scripts
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API clients
│   │   ├── store/          # Redux state management
│   │   └── utils/          # Frontend utilities
│   └── public/             # Static assets
└── infrastructure/         # Deployment configs
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/skandaka/quantum-market-simulator.git
   cd quantum-market-simulator
   ```

2. **Automatic Setup** (Recommended):
   ```bash
   chmod +x setup.sh && ./setup.sh
   ```

3. **Manual Setup**:

   **Backend**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python scripts/setup_and_test.py
   ```

   **Frontend**:
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

**Option 1: Using Makefile**
```bash
make run  # Starts both backend and frontend
```

**Option 2: Manual Start**

Terminal 1 (Backend):
```bash
cd backend
source venv/bin/activate
python -m app.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

**Option 3: Docker**
```bash
docker-compose up
```

### Access Points

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## ⚛️ Quantum Features

### Quantum Computing Integration

- **Classiq Platform**: Professional quantum development
- **Quantum ML Algorithms**: Advanced quantum machine learning
- **Quantum Portfolio Optimization**: Optimal asset allocation
- **Quantum Random Sampling**: True quantum randomness

### Enabling Quantum Features

1. Install Classiq SDK:
   ```bash
   pip install classiq
   ```

2. Set up Classiq API key:
   ```bash
   cp backend/.env.example backend/.env
   # Add your Classiq API key to .env
   ```

3. Run quantum setup:
   ```bash
   cd backend
   python scripts/setup_and_test.py
   ```

## 📊 Usage Examples

### Basic Market Simulation

```python
from app.services.unified_market_simulator import UnifiedMarketSimulator
from app.models.schemas import SentimentAnalysis, SentimentType

# Initialize simulator
simulator = UnifiedMarketSimulator()
await simulator.initialize()

# Create sentiment data
sentiment = SentimentAnalysis(
    text="Apple reports strong earnings",
    sentiment=SentimentType.POSITIVE,
    confidence=0.85,
    entities_detected=[{"text": "Apple", "label": "ORG"}],
    quantum_sentiment_vector=[0.1, 0.2, 0.4, 0.25, 0.05]
)

# Run simulation
predictions = await simulator.simulate(
    sentiment_results=[sentiment],
    market_data={"AAPL": {"current_price": 150.0}},
    simulation_params={
        "target_assets": ["AAPL"],
        "method": "hybrid_qml",
        "time_horizon": 7
    }
)
```

### Quantum vs Classical Comparison

The platform automatically compares quantum and classical approaches:

- **Quantum Method**: Uses quantum circuits for parameter estimation
- **Hybrid Method**: Combines quantum parameter estimation with classical Monte Carlo
- **Classical Method**: Traditional financial modeling

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd backend
python scripts/setup_and_test.py
```

This will:
- Check all dependencies
- Test imports and basic functionality
- Verify quantum features (if available)
- Provide setup recommendations

## 🔧 Configuration

### Environment Variables

Create `backend/.env` with:

```env
# Core Settings
DEBUG=false
LOG_LEVEL=INFO

# Quantum Computing
CLASSIQ_API_KEY=your_classiq_api_key_here
QUANTUM_ENABLED=true

# API Keys (Optional)
ALPHA_VANTAGE_API_KEY=your_key
NEWS_API_KEY=your_key

# Database (if using external DB)
DATABASE_URL=postgresql://...
```

### Simulation Parameters

- **time_horizon**: Prediction timeframe (days)
- **num_scenarios**: Number of Monte Carlo scenarios
- **method**: 'quantum', 'hybrid_qml', or 'classical'
- **risk_tolerance**: Portfolio optimization parameter (0-1)

## 🚀 Deployment

### Production Deployment

1. **Using Docker**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Manual Deployment**:
   - Backend: Deploy with Gunicorn + Nginx
   - Frontend: Build and serve with Nginx
   - Infrastructure: Use provided Nginx config

### Performance Optimization

- Enable quantum acceleration for large-scale simulations
- Use caching for frequently accessed market data
- Implement connection pooling for databases
- Configure CDN for frontend assets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Classiq Platform](https://classiq.io/) - Quantum computing platform
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Backend framework
- [React Documentation](https://reactjs.org/) - Frontend framework

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the documentation at `/docs` endpoint
- Run the diagnostic tools in `scripts/`

---

**Note**: The application runs in classical mode if quantum dependencies are not available, ensuring full functionality regardless of quantum setup.
