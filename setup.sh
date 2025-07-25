#!/bin/bash

# Quantum Market Simulator - Quick Setup
# Usage: chmod +x setup.sh && ./setup.sh

set -e

echo "🚀 Setting up Quantum Market Simulator..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Python version
check_python() {
    echo -e "${YELLOW}Checking Python version...${NC}"
    
    # Try to find Python 3.10+ executable
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "❌ Python 3.10+ is required"
        exit 1
    fi
    
    # Verify version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Found Python $PYTHON_VERSION${NC}"
}

# Create project structure
create_structure() {
    echo -e "${YELLOW}Creating project structure...${NC}"
    mkdir -p frontend/src/{components,pages,services,hooks,utils,store/slices}
    mkdir -p backend/app/{api,services,quantum,ml,models,utils}
    mkdir -p data notebooks
    echo -e "${GREEN}✓ Structure created${NC}"
}

# Setup Python environment
setup_python() {
    echo -e "${YELLOW}Setting up Python environment...${NC}"
    
    cd backend
    
    # Create virtual environment with discovered Python
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    # Download spaCy model
    python -m spacy download en_core_web_sm || echo "Warning: Could not download spaCy model"
    
    cd ..
    echo -e "${GREEN}✓ Python environment ready${NC}"
}

# Setup Frontend
setup_frontend() {
    echo -e "${YELLOW}Setting up React frontend...${NC}"
    
    cd frontend
    
    # Install dependencies
    npm install
    
    cd ..
    echo -e "${GREEN}✓ Frontend ready${NC}"
}

# Create initial files
create_initial_files() {
    echo -e "${YELLOW}Creating initial files...${NC}"
    
    # Create .env file if it doesn't exist
    if [ ! -f backend/.env ]; then
        cp backend/.env.example backend/.env 2>/dev/null || echo -e "${YELLOW}Warning: .env.example not found${NC}"
    fi
    
    echo -e "${GREEN}✓ Initial files created${NC}"
}

# Main setup flow
main() {
    check_python
    create_structure
    setup_python
    setup_frontend
    create_initial_files
    
    echo -e "\n${GREEN}✅ Setup complete!${NC}"
    echo -e "\nNext steps:"
    echo -e "1. Edit backend/.env and add your API keys"
    echo -e "2. Run 'make run' to start both frontend and backend"
    echo -e "3. Open http://localhost:5173 in your browser"
}

# Run main
main
