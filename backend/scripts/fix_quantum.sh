#!/bin/bash

# Path: /Users/skandaa/Desktop/quantum-market-simulator/backend/scripts/fix_quantum.sh

echo "🔧 Fixing Quantum Backend Integration (v2)"
echo "========================================"

# Activate virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment already activated"
else
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    elif [ -f "../venv/bin/activate" ]; then
        source ../venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found. Please create one first."
        exit 1
    fi
fi

echo ""
echo "🔍 Step 1: Cleaning up conflicting packages..."

# Remove all problematic packages first
pip uninstall -y pydantic pydantic-core pydantic-settings classiq thinc spacy numpy

echo ""
echo "📦 Step 2: Installing compatible base packages..."

# Install numpy first (compatible version)
pip install "numpy>=1.26.0,<2.0.0"

# Install pydantic v2 with compatible versions
pip install "pydantic>=2.8.0,<2.10.0" "pydantic-core>=2.20.0,<2.24.0"

# Install pydantic-settings
pip install "pydantic-settings>=2.4.0,<3.0.0"

echo ""
echo "⚡ Step 3: Installing Classiq with compatible dependencies..."

# Install classiq
pip install classiq

echo ""
echo "🔧 Step 4: Fixing any remaining conflicts..."

# If thinc was reinstalled, update it to a compatible version
pip install --upgrade "thinc>=8.2.0" --no-deps

echo ""
echo "🧪 Step 5: Testing the installation..."

python -c "
try:
    import pydantic
    print(f'✅ Pydantic {pydantic.__version__} imported successfully')

    # Test StringConstraints import
    try:
        from pydantic import StringConstraints
        print('✅ StringConstraints imported successfully')
    except ImportError:
        try:
            from pydantic.types import StringConstraints
            print('✅ StringConstraints imported from pydantic.types')
        except ImportError:
            print('⚠️  StringConstraints not available in this pydantic version')

    import classiq
    print('✅ Classiq imported successfully')

    import numpy as np
    print(f'✅ Numpy {np.__version__} imported successfully')

except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo ""
echo "🎉 Installation test completed!"

echo ""
echo "🚀 Step 6: Testing quantum circuit creation (fixed syntax)..."

python -c "
import classiq
from classiq import *

# Test with corrected quantum function
@qfunc
def simple_circuit(q: QArray[QBit]):
    hadamard(q[0])
    cnot(q[0], q[1])

@qfunc
def main(q: Output[QArray[QBit]]):
    allocate(2, q)
    simple_circuit(q)

try:
    model = create_model(main)
    print('✅ Quantum circuit created successfully!')
except Exception as e:
    print(f'⚠️  Circuit creation issue (may need API key): {e}')
"

echo ""
echo "🔑 Step 7: Setting up Classiq authentication..."
echo ""
echo "You have two options for Classiq authentication:"
echo ""
echo "Option 1 - API Key (Recommended):"
echo "1. Go to https://platform.classiq.io/"
echo "2. Sign up/Log in to get your API key"
echo "3. Add it to your .env file: CLASSIQ_API_KEY=your_actual_api_key"
echo ""
echo "Option 2 - Browser Authentication:"
echo "Run: python -c \"import classiq; classiq.authenticate()\""

echo ""
echo "📋 Step 8: Final verification..."

python -c "
try:
    from app.services.quantum.classiq_auth import ClassiqClient
    print('✅ ClassiqClient imported successfully')

    client = ClassiqClient()
    print('✅ ClassiqClient created')

except Exception as e:
    print(f'⚠️  ClassiqClient issue: {e}')
"

echo ""
echo "======================================"
echo "🎯 QUANTUM BACKEND FIX COMPLETE!"
echo "======================================"
echo ""
echo "Next steps to enable REAL quantum computing:"
echo ""
echo "1. 🔑 Get your Classiq API key:"
echo "   - Visit: https://platform.classiq.io/"
echo "   - Sign up/Log in"
echo "   - Get your API key from the dashboard"
echo ""
echo "2. ✏️  Update your .env file:"
echo "   - Open backend/.env"
echo "   - Replace 'your_actual_api_key_here' with your real API key"
echo ""
echo "3. 🚀 Run the application:"
echo "   python -m app.main"
echo ""
echo "4. ✅ Verify it's working:"
echo "   - Check the startup logs for '✅ Quantum backend connected successfully!'"
echo "   - Look for 'Quantum: ✅ Connected' in the status display"
echo ""
echo "If you see 'Simulation Mode' in the logs, check your API key!"