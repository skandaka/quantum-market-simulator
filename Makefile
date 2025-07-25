.PHONY: help setup install-backend install-frontend run-backend run-frontend run test build clean docker-up docker-down

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

setup: install-backend install-frontend
	@echo "$(GREEN)✓ Setup complete!$(NC)"

install-backend:
	@echo "$(YELLOW)Installing backend dependencies...$(NC)"
	cd backend && python -m venv venv
	cd backend && . venv/bin/activate && pip install -r requirements.txt
	cd backend && . venv/bin/activate && python -m spacy download en_core_web_sm

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
	cd backend && . venv/bin/activate && pytest

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
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf backend/venv
	rm -rf frontend/node_modules
	rm -rf frontend/dist
	rm -rf .pytest_cache
	@echo "$(GREEN)✓ Project cleaned$(NC)"

format:
	cd backend && . venv/bin/activate && black app/
	cd frontend && npm run format

lint:
	cd backend && . venv/bin/activate && flake8 app/
	cd frontend && npm run lint

# Development helpers
dev-reset-db:
	cd backend && . venv/bin/activate && python scripts/setup_db.py

dev-seed-data:
	cd backend && . venv/bin/activate && python scripts/populate_test_data.py

dev-quantum-test:
	cd backend && . venv/bin/activate && python scripts/test_quantum_connection.py