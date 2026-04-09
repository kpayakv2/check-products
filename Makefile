# Makefile for Product Similarity Checker
# Compatible with Windows, macOS, and Linux

.PHONY: help install install-dev test test-cov format lint clean run example setup-env

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  format       - Format code with black and isort"
	@echo "  lint         - Run linting checks"
	@echo "  clean        - Clean up cache and build files"
	@echo "  run          - Run the main program"
	@echo "  analyze      - Analyze output results"
	@echo "  example      - Run with example data"
	@echo "  setup-env    - Set up development environment"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# Testing
test:
	pytest tests/

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-performance:
	pytest tests/performance/ -v

test-ui:
	pytest tests/ui/ -v

test-cov:
	pytest --cov=main --cov-report=html --cov-report=term tests/

test-report:
	pytest --html=test-report.html --self-contained-html tests/

# Code formatting and linting
format:
	black .
	isort .

lint:
	black --check .
	isort --check-only .
	flake8 .

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

# Running
run:
	python main.py

analyze:
	python run_analysis.py

example:
	python main.py --old-products-csv example_old_products.csv --new-products-csv example_new_products.csv --output-dir example_output

# Development setup
setup-env:
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  Windows: venv\\Scripts\\activate"
	@echo "  macOS/Linux: source venv/bin/activate"
	@echo "Then run: make install-dev"

# Windows-specific commands
setup-env-windows:
	python -m venv venv
	venv\Scripts\activate && pip install -r requirements-dev.txt

# Build and distribution
build:
	python -m build

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*

# Documentation
docs:
	@echo "Documentation files:"
	@echo "  README.md - Main documentation"
	@echo "  API.md - API reference"
	@echo "  CONTRIBUTING.md - Contribution guide"
	@echo "  CHANGELOG.md - Version history"
