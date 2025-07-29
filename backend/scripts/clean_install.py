#!/usr/bin/env python3
"""Clean install script to fix all dependencies"""

import subprocess
import sys
import os


def run_command(cmd):
    """Run a command and print output"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("Quantum Market Simulator - Clean Dependency Install")
    print("=" * 60)

    # Step 1: Update pip
    print("\n1. Updating pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Step 2: Uninstall problematic packages
    print("\n2. Removing problematic packages...")
    packages_to_remove = ["newspaper3k", "newspaper"]
    for pkg in packages_to_remove:
        run_command(f"{sys.executable} -m pip uninstall -y {pkg}")

    # Step 3: Install core dependencies
    print("\n3. Installing core dependencies...")
    core_deps = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.0.0",
        "pydantic-settings==2.0.0",
        "python-multipart==0.0.6",
        "python-dotenv==1.0.0"
    ]

    for dep in core_deps:
        if not run_command(f"{sys.executable} -m pip install '{dep}'"):
            print(f"Failed to install {dep}")

    # Step 4: Install ML dependencies
    print("\n4. Installing ML dependencies...")
    ml_deps = [
        "numpy==1.26.2",
        "pandas==2.1.4",
        "scipy==1.11.4",
        "torch>=2.2.0",
        "transformers==4.35.2",
        "scikit-learn==1.3.2",
        "openai"
    ]

    for dep in ml_deps:
        if not run_command(f"{sys.executable} -m pip install '{dep}'"):
            print(f"Failed to install {dep}")

    # Step 5: Install other dependencies
    print("\n5. Installing other dependencies...")
    other_deps = [
        "beautifulsoup4==4.12.2",
        "lxml==4.9.3",
        "lxml_html_clean",
        "yfinance==0.2.33",
        "spacy==3.7.2",
        "textblob==0.17.1"
    ]

    for dep in other_deps:
        if not run_command(f"{sys.executable} -m pip install '{dep}'"):
            print(f"Failed to install {dep}")

    # Step 6: Download spacy model
    print("\n6. Downloading spacy model...")
    run_command(f"{sys.executable} -m spacy download en_core_web_sm")

    # Step 7: Install remaining requirements
    print("\n7. Installing remaining requirements...")
    if os.path.exists("requirements.txt"):
        run_command(f"{sys.executable} -m pip install -r requirements.txt")

    print("\n" + "=" * 60)
    print("Installation complete!")
    print("\nNow you should:")
    print("1. Make sure news_processor.py is using the updated version without newspaper imports")
    print("2. Run: python -m app.main")


if __name__ == "__main__":
    main()