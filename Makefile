.PHONY: help install install-dev format lint type-check test clean setup-dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	python -m pip install -r requirements.txt

install-dev: ## Install development dependencies
	python -m pip install black isort flake8 mypy pytest pytest-cov "black[jupyter]" pre-commit
	@echo "For markdown linting, install with: npm install --prefix ~/.npm-global markdownlint-cli2"
	@echo "Then add ~/.npm-global/bin to your PATH"
	@echo "Pre-commit hooks will be handled by the setup script if needed"

setup-dev: install-dev ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make format' to format code"
	@echo "Run 'make lint' to check code quality"
	@echo "Run 'make test' to run tests"

format: ## Format code with black and isort
	black .
	isort .

lint: ## Run linting checks
	@echo "Running Python linting..."
	@command -v flake8 >/dev/null 2>&1 || { echo "❌ Run 'make install-dev' first to install linting tools"; exit 1; }
	flake8 .
	black --check .
	isort --check-only .
	@echo "✅ Python linting passed"

lint-fix: ## Fix linting issues
	@echo "Fixing Python code..."
	@command -v black >/dev/null 2>&1 || { echo "❌ Run 'make install-dev' first"; exit 1; }
	black .
	isort .
	@echo "✅ Python code formatted"

lint-markdown: ## Lint markdown files (requires markdownlint-cli2)
	@command -v markdownlint-cli2 >/dev/null 2>&1 || { echo "❌ Install with: npm install -g markdownlint-cli2"; exit 1; }
	markdownlint-cli2 "**/*.md"

fix-markdown: ## Fix markdown files
	@command -v markdownlint-cli2-fix >/dev/null 2>&1 || { echo "❌ Install with: npm install -g markdownlint-cli2"; exit 1; }
	markdownlint-cli2-fix "**/*.md"

type-check: ## Run type checking with mypy
	mypy code/ local_inference/ config.py

test: ## Run tests with pytest
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov-report=html --cov-report=term

clean: ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check: lint-fix lint type-check ## Run all code quality checks

# SageMaker specific commands
package-model: ## Package model for SageMaker deployment
	@echo "Packaging model artifacts..."
	tar -czf model.tar.gz code/

deploy-local: ## Run local inference test
	python local_inference/object_detection_image.py

load-test: ## Run Locust load testing (requires endpoint)
	@echo "Make sure to set SAGEMAKER_ENDPOINT_NAME environment variable"
	cd locust_scripts && locust -f locust_rfdetr.py --host=https://runtime.sagemaker.us-east-1.amazonaws.com