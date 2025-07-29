#!/bin/bash

echo "Installing/Updating dependencies..."
echo "=================================="

# Update pip first
pip install --upgrade pip

# Install dependencies one by one to handle conflicts
echo "Installing core dependencies..."
pip install fastapi==0.104.1
pip install "uvicorn[standard]==0.24.0"
pip install pydantic==2.0.0
pip install pydantic-settings==2.0.0
pip install python-multipart==0.0.6
pip install python-dotenv==1.0.0

echo "Installing ML dependencies..."
pip install "torch>=2.2.0"
pip install transformers==4.35.2
pip install scikit-learn==1.3.2
pip install openai

echo "Installing other dependencies..."
pip install beautifulsoup4==4.12.2
pip install lxml==4.9.3
pip install lxml_html_clean

echo "Dependencies installed!"