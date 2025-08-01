# Multi-Disease Prediction Makefile

.PHONY: help install test train run clean docker-build docker-run

# Default target
help:
	@echo "Multi-Disease Prediction System"
	@echo "================================"
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  train        - Train models"
	@echo "  run          - Start Streamlit app"
	@echo "  clean        - Clean generated files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v

# Train models
train:
	@echo "Training models..."
	python run_pipeline.py

# Start Streamlit app
run:
	@echo "Starting Streamlit app..."
	streamlit run app/app.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf app/__pycache__/
	rm -rf .pytest_cache/
	rm -rf logs/*.log
	rm -rf models/*.joblib
	rm -rf models/*.json
	rm -rf visualizations/*.png
	rm -rf data/processed/*.csv

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -f docker/Dockerfile -t multi-disease-prediction .

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	docker run -p 8501:8501 multi-disease-prediction

# Run with pipeline
docker-run-pipeline:
	@echo "Running Docker container with pipeline..."
	docker run -p 8501:8501 -e RUN_PIPELINE=true multi-disease-pipeline

# Setup development environment
setup-dev:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Unix/MacOS"
	@echo "  venv\\Scripts\\activate     # On Windows"
	@echo "Then run: make install"

# Full setup (install + train)
setup: install train
	@echo "Setup complete! Run 'make run' to start the app."

# Quick start (install + train + run)
quickstart: setup run 