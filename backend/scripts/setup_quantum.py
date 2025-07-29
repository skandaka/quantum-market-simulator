#!/usr/bin/env python3
"""Setup script for Classiq quantum backend"""

import os
import sys
import asyncio
import logging
import warnings
from pathlib import Path

# Suppress pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.quantum.classiq_auth import classiq_auth
from app.quantum.classiq_client import ClassiqClient

try:
    from classiq import authenticate
    CLASSIQ_AVAILABLE = True
except ImportError:
    CLASSIQ_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_quantum():
    """Interactive setup for Classiq quantum backend"""

    print("\n" + "=" * 60)
    print("🚀 Quantum Market Simulator - Classiq Setup")
    print("=" * 60 + "\n")

    if not CLASSIQ_AVAILABLE:
        print("❌ Classiq is not installed!")
        print("\nTo install Classiq, run:")
        print("  pip install classiq")
        print("\nOr update your requirements.txt to include 'classiq'")
        return False

    # Check for existing API key
    api_key = os.getenv("CLASSIQ_API_KEY")

    if api_key and api_key != "dummy-key-for-hackathon":
        print(f"✅ Found CLASSIQ_API_KEY in environment")
        use_existing = input("Use existing API key? (y/n): ").lower().strip()
        if use_existing != 'y':
            api_key = None

    if not api_key:
        print("\n📝 Classiq Authentication Options:")
        print("1. Enter API key manually")
        print("2. Authenticate via browser (recommended)")
        print("3. Skip quantum setup (run in classical mode)")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == "1":
            api_key = input("Enter your Classiq API key: ").strip()
            os.environ["CLASSIQ_API_KEY"] = api_key

        elif choice == "2":
            print("\n🌐 Opening browser for Classiq authentication...")
            print("Please log in to your Classiq account.")
            try:
                authenticate()
                print("✅ Authentication successful!")
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
                return False

        elif choice == "3":
            print("\n⚠️  Skipping quantum setup. The app will run in classical mode.")
            return True
        else:
            print("❌ Invalid option")
            return False

    # Initialize and test connection
    print("\n🔧 Testing Classiq connection...")

    try:
        await classiq_auth.initialize()

        if classiq_auth.is_authenticated():
            print("✅ Successfully connected to Classiq!")

            # Test quantum client
            print("\n🧪 Testing quantum circuit synthesis...")
            client = ClassiqClient()
            await client.initialize()

            if client.is_ready():
                print("✅ Quantum client operational!")

                # Ask about hardware usage
                use_hardware = input("\n🖥️  Use real quantum hardware? (y/n, default=n): ").lower().strip()
                if use_hardware == 'y':
                    classiq_auth.update_config(use_hardware=True)
                    print("⚡ Configured for quantum hardware execution")
                else:
                    print("💻 Configured for quantum simulation")

                # Save configuration
                save_config = input("\n💾 Save configuration to .env? (y/n): ").lower().strip()
                if save_config == 'y':
                    env_path = backend_dir / ".env"

                    # Read existing .env
                    existing_lines = []
                    if env_path.exists():
                        with open(env_path, 'r') as f:
                            existing_lines = [line for line in f.readlines()
                                              if not line.startswith("CLASSIQ_API_KEY")]

                    # Write updated .env
                    with open(env_path, 'w') as f:
                        f.write(f"CLASSIQ_API_KEY={api_key}\n")
                        f.writelines(existing_lines)

                    print(f"✅ Configuration saved to {env_path}")

                return True
            else:
                print("❌ Quantum client initialization failed")
                return False

        else:
            print("❌ Not authenticated with Classiq")
            return False

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


async def test_quantum_circuit():
    """Test a simple quantum circuit"""

    print("\n🔬 Running quantum circuit test...")

    if not CLASSIQ_AVAILABLE:
        print("❌ Classiq not available")
        return False

    try:
        from classiq import create_model, qfunc, QBit, H, synthesize, CX, Output

        @qfunc
        def bell_state_prep(q0: QBit, q1: QBit):
            H(q0)
            CX(q0, q1)

        @qfunc
        def main(q0: QBit, q1: QBit):
            bell_state_prep(q0, q1)

        model = create_model(main)
        quantum_program = synthesize(model)

        print("✅ Quantum circuit synthesized successfully!")

        # Show circuit if possible
        try:
            from classiq import show
            show(quantum_program)
        except:
            pass

        return True

    except Exception as e:
        print(f"❌ Circuit test failed: {e}")
        return False


async def main():
    """Main setup function"""

    # Run setup
    success = await setup_quantum()

    if success and classiq_auth.is_authenticated():
        # Run circuit test
        await test_quantum_circuit()

        print("\n" + "=" * 60)
        print("✅ Quantum setup complete!")
        print("\nYou can now run the application with:")
        print("  cd backend && python -m app.main")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("⚠️  Setup incomplete")
        print("\nThe application will run in classical mode.")
        print("To enable quantum features, run this setup again.")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())