.PHONY: help setup install-backend install-frontend run-backend run-frontend run test build clean docker-up docker-down directories

# Colors
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

help:
	@echo "$(GREEN)Quantum Market Reaction Simulator$(NC)"
	@echo "Available commands:"
	@echo "  $(YELLOW)make setup$(NC)       - Complete project setup"
	@echo "  $(YELLOW)make run$(NC)         - Run both frontend and backend"
	@echo "  $(YELLOW)make test$(NC)        - Run all tests"
	@echo "  $(YELLOW)make build$(NC)       - Build production version"
	@echo "  $(YELLOW)make docker-up$(NC)   - Start with Docker"
	@echo "  $(YELLOW)make clean$(NC)       - Clean project files"

directories:
	@echo "$(YELLOW)Creating directory structure...$(NC)"
	@mkdir -p backend/app/{api,services,quantum,ml,models,utils}
	@mkdir -p frontend/src/{components,pages,services,hooks,utils,store/slices}
	@mkdir -p infrastructure/nginx
	@mkdir -p data notebooks

setup: directories install-backend install-frontend
	@echo "$(GREEN)✓ Setup complete!$(NC)"

install-backend:
	@echo "$(YELLOW)Installing backend dependencies...$(NC)"
	cd backend && python3 -m venv venv
	cd backend && . venv/bin/activate && pip install --upgrade pip
	cd backend && . venv/bin/activate && pip install -r requirements.txt
	cd backend && . venv/bin/activate && python -m spacy download en_core_web_sm || echo "SpaCy model download skipped"
	@if [ ! -f backend/.env ]; then \
		cp backend/.env.example backend/.env 2>/dev/null || echo "$(YELLOW)Please create backend/.env file$(NC)"; \
	fi

install-frontend:
	@echo "$(YELLOW)Installing frontend dependencies...$(NC)"
	cd frontend && npm install

run-backend:
	@echo "$(YELLOW)Starting backend server...$(NC)"
	cd backend && . venv/bin/activate && uvicorn app.main:app --reload --port 8000

run-frontend:
	@echo "$(YELLOW)Starting frontend development server...$(NC)"
	cd frontend && npm run dev

run:
	@echo "$(GREEN)Starting Quantum Market Simulator...$(NC)"
	@make -j2 run-backend run-frontend

test-backend:
	cd backend && . venv/bin/activate && pytest -v

test-frontend:
	cd frontend && npm test

test: test-backend test-frontend

build-backend:
	cd backend && docker build -t quantum-simulator-backend .

build-frontend:
	cd frontend && npm run build

build: build-backend build-frontend

docker-up:
	docker-compose up -d
	@echo "$(GREEN)✓ Application running at http://localhost$(NC)"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf backend/venv
	rm -rf frontend/node_modules
	rm -rf frontend/dist
	rm -rf .pytest_cache
	@echo "$(GREEN)✓ Project cleaned$(NC)"

format:
	cd backend && . venv/bin/activate && black app/ || echo "Black not installed"
	cd frontend && npm run format || echo "Prettier not configured"

lint:
	cd backend && . venv/bin/activate && flake8 app/ || echo "Flake8 not installed"
	cd frontend && npm run lint || echo "ESLint not configured"

# Development helpers
dev-reset-db:
	cd backend && . venv/bin/activate && python scripts/setup_db.py

dev-seed-data:
	cd backend && . venv/bin/activate && python scripts/populate_test_data.py

dev-quantum-test:
	cd backend && . venv/bin/activate && python scripts/test_quantum_connection.py

# Quick start commands
quick-start: setup
	@echo "$(GREEN)Setup complete! Now:$(NC)"
	@echo "1. Edit backend/.env with your API keys"
	@echo "2. Run 'make run' to start the application"
	@echo "3. Open http://localhost:5173"

check-deps:
	@echo "$(YELLOW)Checking dependencies...$(NC)"
	@command -v python3 >/dev/null 2>&1 || { echo "$(RED)Python 3 is required but not installed.$(NC)" >&2; exit 1; }
	@command -v node >/dev/null 2>&1 || { echo "$(RED)Node.js is required but not installed.$(NC)" >&2; exit 1; }
	@command -v npm >/dev/null 2>&1 || { echo "$(RED)npm is required but not installed.$(NC)" >&2; exit 1; }
	@echo "$(GREEN)✓ All dependencies found$(NC)"