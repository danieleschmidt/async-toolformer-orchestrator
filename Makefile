.PHONY: help install install-dev test test-unit test-integration test-e2e lint format typecheck clean build docs serve-docs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package in development mode
	pip install -e ".[dev]"

install-dev: install ## Install development dependencies and setup pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests (requires API keys)
	pytest tests/e2e/ -v

test-coverage: ## Run tests with coverage report
	pytest --cov=async_toolformer --cov-report=html --cov-report=term

lint: ## Run linting checks
	ruff check .
	bandit -r src/ -f json || true

format: ## Format code
	ruff format .
	ruff check --fix .

typecheck: ## Run type checking
	mypy src/

check: lint typecheck test-unit ## Run all checks (lint, typecheck, unit tests)

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

docs: ## Build documentation
	cd docs && make html

serve-docs: docs ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

release-test: ## Test release to PyPI test
	python -m build
	twine check dist/*
	twine upload --repository testpypi dist/*

release: ## Release to PyPI
	python -m build
	twine check dist/*
	twine upload dist/*

docker-build: ## Build Docker image
	docker build -t async-toolformer-orchestrator .

docker-test: ## Run tests in Docker
	docker run --rm async-toolformer-orchestrator make test

benchmark: ## Run performance benchmarks
	python -m pytest benchmarks/ -v --benchmark-only

profile: ## Run profiling
	python -m cProfile -o profile.stats examples/benchmark_example.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"