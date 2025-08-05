#!/usr/bin/env python3
"""
Consolidated script for quantum setup and testing
"""

import sys
import os
import subprocess
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'numpy', 
        'transformers', 'torch', 'scikit-learn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    return missing


def install_dependencies():
    """Install missing dependencies"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 
            str(backend_dir / 'requirements.txt')
        ])
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def test_imports():
    """Test basic imports"""
    print("\nTesting imports...")
    
    try:
        from app.config import settings
        print("✅ Config imported")
        
        from app.services.sentiment_analyzer import SentimentAnalyzer
        print("✅ Sentiment analyzer imported")
        
        from app.services.unified_market_simulator import UnifiedMarketSimulator
        print("✅ Market simulator imported")
        
        from app.services.news_processor import NewsProcessor
        print("✅ News processor imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_quantum():
    """Test quantum functionality if available"""
    print("\nTesting quantum functionality...")
    
    try:
        import classiq
        print("✅ Classiq imported")
        
        # Test basic quantum circuit
        from classiq import qfunc, QBit, H, create_model, synthesize
        
        @qfunc
        def main(q: QBit):
            H(q)
        
        model = create_model(main)
        print("✅ Quantum model created")
        
        qprog = synthesize(model)
        print("✅ Quantum synthesis works")
        
        return True
        
    except ImportError:
        print("⚠️  Classiq not available - quantum features disabled")
        return False
    except Exception as e:
        print(f"❌ Quantum test failed: {e}")
        return False


def main():
    """Main setup and test function"""
    print("=" * 60)
    print("🚀 Quantum Market Simulator Setup & Test")
    print("=" * 60)
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\nMissing dependencies: {missing}")
        if input("Install missing dependencies? (y/n): ").lower() == 'y':
            if not install_dependencies():
                return 1
        else:
            print("Cannot proceed without dependencies")
            return 1
    
    # Test imports
    if not test_imports():
        print("❌ Basic imports failed")
        return 1
    
    # Test quantum (optional)
    quantum_works = test_quantum()
    
    print("\n" + "=" * 60)
    print("📊 Setup Summary:")
    print("=" * 60)
    print("✅ Core functionality: Ready")
    print(f"{'✅' if quantum_works else '⚠️ '} Quantum features: {'Available' if quantum_works else 'Disabled'}")
    
    if not quantum_works:
        print("\nTo enable quantum features:")
        print("1. Install Classiq: pip install classiq")
        print("2. Set up Classiq API key in .env file")
    
    print("\n✅ Setup complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
