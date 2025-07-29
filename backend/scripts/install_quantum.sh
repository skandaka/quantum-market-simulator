#!/bin/bash

# Path: /Users/skandaa/Desktop/quantum-market-simulator/backend/scripts/install_quantum.sh

echo "🚀 Quantum Market Simulator - Clean Installation"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found. Please run this from the backend directory."
    exit 1
fi

# Activate virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment already activated: $VIRTUAL_ENV"
else
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "📦 Creating new virtual environment..."
        python -m venv venv
        source venv/bin/activate
        echo "✅ New virtual environment created and activated"
    fi
fi

echo ""
echo "🧹 Step 1: Cleaning existing installation..."

# Uninstall potentially conflicting packages
pip uninstall -y pydantic pydantic-core pydantic-settings classiq thinc spacy numpy scipy

echo ""
echo "📦 Step 2: Upgrading pip..."
pip install --upgrade pip

echo ""
echo "🔧 Step 3: Installing core dependencies..."

# Install in specific order to avoid conflicts
pip install "numpy>=1.26.0,<2.0.0"
pip install "scipy>=1.11.0,<2.0.0"

echo ""
echo "🏗️  Step 4: Installing pydantic stack..."
pip install "pydantic>=2.8.0,<2.10.0"
pip install "pydantic-settings>=2.4.0,<3.0.0"

echo ""
echo "⚡ Step 5: Installing quantum computing packages..."
pip install classiq

echo ""
echo "🌐 Step 6: Installing web framework..."
pip install fastapi "uvicorn[standard]"

echo ""
echo "📊 Step 7: Installing data processing packages..."
pip install pandas matplotlib plotly httpx python-dotenv

echo ""
echo "🧪 Step 8: Installing development tools..."
pip install pytest pytest-asyncio black isort

echo ""
echo "✅ Step 9: Verifying installation..."

python -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import numpy as np
    print(f'✅ NumPy {np.__version__}')
except Exception as e:
    print(f'❌ NumPy: {e}')

try:
    import pydantic
    print(f'✅ Pydantic {pydantic.__version__}')
except Exception as e:
    print(f'❌ Pydantic: {e}')

try:
    import classiq
    print('✅ Classiq imported successfully')
except Exception as e:
    print(f'❌ Classiq: {e}')

try:
    import fastapi
    print('✅ FastAPI imported successfully')
except Exception as e:
    print(f'❌ FastAPI: {e}')
"

echo ""
echo "🔑 Step 10: Setting up environment..."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Quantum Computing Configuration
CLASSIQ_API_KEY=your_actual_api_key_here

# Application Configuration
APP_NAME=Quantum Market Simulator
APP_VERSION=1.0.0
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
EOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🧪 Step 11: Running quantum test..."
python scripts/test_quantum.py

echo ""
echo "================================================"
echo "🎉 Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. 🔑 Get your Classiq API key:"
echo "   - Visit: https://platform.classiq.io/"
echo "   - Sign up/Log in"
echo "   - Copy your API key"
echo ""
echo "2. ✏️  Update your .env file:"
echo "   - Open: nano .env"
echo "   - Replace 'your_actual_api_key_here' with your real API key"
echo ""
echo "3. 🚀 Start the application:"
echo "   python -m app.main"
echo ""
echo "4. 🧪 Test everything:"
echo "   python scripts/test_quantum.py"
echo ""
echo "Need help? Check the logs or run the test script for diagnostics."