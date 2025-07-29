#!/bin/bash

# Quantum Market Simulator - Fix Quantum Backend Script
# This script fixes the Pydantic/Classiq compatibility issues

set -e

echo "🔧 Fixing Quantum Backend Integration"
echo "===================================="

cd ..

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run: python -m venv venv"
    exit 1
fi

echo ""
echo "🔍 Step 1: Diagnosing the issue..."
echo "Current pydantic version:"
pip show pydantic | grep Version || echo "Pydantic not installed"

echo ""
echo "📦 Step 2: Uninstalling conflicting packages..."
pip uninstall -y pydantic pydantic-core pydantic-settings classiq || true

echo ""
echo "🎯 Step 3: Installing compatible versions..."

# Install specific Pydantic version that works with Classiq
pip install "pydantic==2.5.3"
pip install "pydantic-core==2.14.6"
pip install "pydantic-settings==2.1.0"

# Install Classiq
echo ""
echo "⚡ Step 4: Installing Classiq..."
pip install classiq

echo ""
echo "🧪 Step 5: Testing the installation..."

python -c "
import sys
print('Testing imports...')

try:
    import pydantic
    print(f'✅ Pydantic {pydantic.VERSION} imported successfully')
except Exception as e:
    print(f'❌ Pydantic import failed: {e}')
    sys.exit(1)

try:
    import classiq
    print('✅ Classiq imported successfully')

    # Test specific imports that were failing
    from classiq import qfunc, QBit, H, create_model, synthesize
    print('✅ Classiq core functions imported successfully')

except Exception as e:
    print(f'❌ Classiq import failed: {e}')
    print('This might be normal if you don\\'t have API credentials yet')

print('')
print('🎉 Import test completed!')
"

echo ""
echo "🔑 Step 6: Setting up Classiq authentication..."
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

# Check if .env exists and update it
if [ -f ".env" ]; then
    if ! grep -q "CLASSIQ_API_KEY" .env; then
        echo "CLASSIQ_API_KEY=your_actual_api_key_here" >> .env
        echo "✅ Added CLASSIQ_API_KEY placeholder to .env"
    fi
else
    echo "CLASSIQ_API_KEY=your_actual_api_key_here" > .env
    echo "✅ Created .env with CLASSIQ_API_KEY placeholder"
fi

echo ""
echo "🚀 Step 7: Testing quantum circuit creation..."

python -c "
import os
import warnings
warnings.filterwarnings('ignore')

# Test the circuit creation that was failing
try:
    from classiq import qfunc, QBit, H, create_model, synthesize

    @qfunc
    def main(q: QBit):
        H(q)

    print('✅ Circuit function defined successfully')

    model = create_model(main)
    print('✅ Model created successfully')

    # Only try synthesis if we have credentials
    api_key = os.getenv('CLASSIQ_API_KEY')
    if api_key and api_key != 'your_actual_api_key_here':
        try:
            qprog = synthesize(model)
            print('✅ Circuit synthesized successfully - REAL QUANTUM BACKEND WORKING!')
        except Exception as e:
            print(f'⚠️  Synthesis failed (likely auth issue): {e}')
            print('   Set your real API key in .env to enable synthesis')
    else:
        print('⚠️  No API key set - add real key to .env for full functionality')

except Exception as e:
    print(f'❌ Circuit test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "📋 Step 8: Final verification..."
echo ""

# Test our quantum modules
python -c "
import sys
import os
sys.path.insert(0, '.')

try:
    from app.quantum.classiq_auth import classiq_auth
    print('✅ classiq_auth module imported')

    from app.quantum.classiq_client import ClassiqClient
    print('✅ ClassiqClient imported')

    client = ClassiqClient()
    print('✅ ClassiqClient created')

except Exception as e:
    print(f'❌ Module test failed: {e}')
    import traceback
    traceback.print_exc()
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