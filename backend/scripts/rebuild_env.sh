#!/bin/bash

echo "🚀 Quantum Market Simulator - Environment Repair Script"
echo "=================================================="

# Move out of any active virtualenv
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating current virtual environment: $VIRTUAL_ENV"
    deactivate
fi

echo "📂 Current directory: $(pwd)"

# Create fresh environment directory
echo "🧹 Removing old virtual environment..."
rm -rf fresh_venv
rm -rf venv

# Determine which Python version to use
echo "🐍 Finding available Python installation..."
PYTHON_PATH=""

for cmd in "python3.9" "python3.10" "python3.11" "python3.8" "python3" "python"; do
    if command -v $cmd &> /dev/null; then
        PYTHON_PATH=$(which $cmd)
        PYTHON_VERSION=$($cmd --version)
        echo "✅ Found $PYTHON_VERSION at $PYTHON_PATH"
        break
    fi
done

if [ -z "$PYTHON_PATH" ]; then
    echo "❌ No Python installation found! Please install Python 3.8 or newer."
    exit 1
fi

# Create new virtual environment without using symlinks
echo "🏗️ Creating fresh virtual environment..."
$PYTHON_PATH -m venv fresh_venv --without-pip

# Activate the new environment
echo "🔌 Activating fresh virtual environment..."
source fresh_venv/bin/activate

# Bootstrap pip into the environment
echo "📦 Bootstrapping pip..."
curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py --force-reinstall
rm get-pip.py

# Verify pip installation
echo "🔍 Verifying pip installation..."
python -m pip --version

# Install core dependencies one by one
echo "📚 Installing core dependencies..."
python -m pip install wheel setuptools

# Fast AI Stack - install these first
echo "📊 Installing NumPy (foundation package)..."
python -m pip install numpy==1.24.4

echo "📈 Installing scientific packages..."
python -m pip install scipy==1.10.1
python -m pip install pandas==1.5.3

echo "🌐 Installing FastAPI and web dependencies..."
python -m pip install fastapi==0.104.1
python -m pip install uvicorn==0.24.0
python -m pip install pydantic==2.0.0
python -m pip install python-multipart==0.0.6
python -m pip install python-dotenv==1.0.0

echo "🤖 Installing ML dependencies..."
python -m pip install scikit-learn==1.3.2

# Only if needed and available
if python -m pip install torch --dry-run &> /dev/null; then
    echo "🔥 Installing PyTorch..."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install transformers==4.35.2
else
    echo "⚠️ Skipping PyTorch installation"
fi

echo "🔠 Installing spaCy..."
python -m pip install spacy==3.7.2
python -m spacy download en_core_web_sm --no-deps

echo "📡 Installing data fetching tools..."
python -m pip install httpx==0.24.1
python -m pip install beautifulsoup4==4.12.2
python -m pip install lxml==4.9.3
python -m pip install retry==0.9.2
python -m pip install yfinance==0.2.33

echo "✅ Core dependencies installed successfully!"

# Finalize
echo ""
echo "🎯 Environment setup complete!"
echo ""
echo "To use this environment:"
echo "  source fresh_venv/bin/activate"
echo ""
echo "To run the application:"
echo "  python -m app.main"
echo ""
echo "If this environment works, you can replace your old one with:"
echo "  rm -rf venv"
echo "  mv fresh_venv venv"
