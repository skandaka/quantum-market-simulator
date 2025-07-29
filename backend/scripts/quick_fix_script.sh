#!/bin/bash

# Quick Fix Script for Quantum Market Simulator
# Run this from the backend directory

echo "🚀 Quantum Market Simulator - Quick Fix"
echo "======================================"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run: python -m venv venv"
    exit 1
fi

# Step 1: Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Step 2: Fix pydantic conflicts
echo "🔧 Fixing pydantic conflicts..."
python -m pip uninstall -y pydantic pydantic-core pydantic-settings fastapi

# Install compatible versions
python -m pip install "pydantic>=2.4.0,<2.10.0"
python -m pip install "pydantic-settings>=2.4.0,<3.0.0"
python -m pip install "fastapi==0.104.1"

# Step 3: Install core dependencies
echo "📚 Installing core dependencies..."
python -m pip install "uvicorn[standard]==0.24.0"
python -m pip install "python-multipart==0.0.6"
python -m pip install "python-dotenv==1.0.0"

# Step 4: Install ML dependencies
echo "🧠 Installing ML dependencies..."
python -m pip install "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "transformers==4.35.2"
python -m pip install "scikit-learn==1.3.2"
python -m pip install "textblob==0.17.1"

# Step 5: Install other dependencies
echo "🔗 Installing other dependencies..."
python -m pip install "numpy==1.26.2"
python -m pip install "pandas==2.1.4"
python -m pip install "beautifulsoup4==4.12.2"
python -m pip install "lxml==4.9.3"
python -m pip install "yfinance==0.2.33"
python -m pip install "spacy==3.7.2"

# Step 6: Download spaCy model
echo "🔤 Downloading spaCy model..."
python -m spacy download en_core_web_sm || echo "⚠️ SpaCy model download failed - will work without it"

# Step 7: Create necessary directories
echo "📁 Creating directories..."
mkdir -p models data

# Step 8: Test basic imports
echo "🧪 Testing basic imports..."
python -c "
try:
    import fastapi
    import pydantic
    import uvicorn
    print('✅ FastAPI stack working')
except Exception as e:
    print(f'❌ FastAPI error: {e}')

try:
    import torch
    import transformers
    print('✅ ML stack working')
except Exception as e:
    print(f'⚠️ ML error: {e}')

try:
    import spacy
    print('✅ NLP stack working')
except Exception as e:
    print(f'⚠️ NLP error: {e}')
"

echo ""
echo "🎯 Fix complete!"
echo "To run the application:"
echo "  python -m app.main"
echo ""
echo "If you still get errors, the app will run in fallback mode."