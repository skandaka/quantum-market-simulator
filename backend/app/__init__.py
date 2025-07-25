"""Quantum Market Reaction Simulator Backend Package"""

__version__ = "1.0.0"
__author__ = "Quantum Market Sim Team"

# Make app a proper package
import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)