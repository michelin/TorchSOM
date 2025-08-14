# TorchSOM Development Makefile: provides convenient commands for local development and testing
# Run `make help` to see all available commands

.PHONY: help install test test-quick lint format security docs clean all

# Default target
help:  ## Show this help message
	@echo "TorchSOM Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install     # Install development dependencies"
	@echo "  make test        # Run all tests with coverage"
	@echo "  make lint        # Run all code quality checks"
	@echo "  make format      # Auto-fix formatting issues"
	@echo "  make all         # Run everything (like CI)"

install:  ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	pip install -e ".[dev]"
	pip install pre-commit
	pre-commit install

TESTS ?= tests/unit/ # tests/unit/ or tests/unit/test_som.py::TestInputValidation
test:  ## Run all tests with coverage
	@echo "🧪 Running tests with coverage..."
	pytest $(TESTS) -v \
		--cov=torchsom \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-config=pyproject.toml \
		--junit-xml=junit.xml \
		-m "unit or gpu"
# -m "not unit and not gpu"
# -v verbose, -x exit on first failure, -m marker: run only tests with the corresponding markers
test-quick:  ## Run tests without coverage (faster)
	@echo "⚡ Running quick tests..."
	pytest tests/unit/ -v -x -m "unit and not gpu"

test-gpu:  ## Run GPU tests (requires CUDA)
	@echo "🖥️ Running GPU tests..."
	pytest tests/unit/ -v -m "gpu"

lint:  ## Run all code quality checks
	@echo "🔍 Running code quality checks..."
	@echo "  🎨 Checking code formatting..."
	black --check --diff torchsom/ tests/
	@echo "  📦 Checking import sorting..."
	isort --check-only --diff torchsom/ tests/
	@echo "  🔍 Running linter..."
	ruff check torchsom/ tests/
	@echo "  🎯 Type checking..."
	mypy torchsom/
	@echo "✅ All quality checks passed!"
# ruff check torchsom/ tests/ => read-only mode, report violations without modifications
# ruff check torchsom/ tests/ --fix => fix safe, non-destructive violations
# ruff check torchsom/ tests/ --fix --unsafe-fixes => fix unsafe, potentially destructive violations (might need review)
# ruff check torchsom/ tests/ --fix --unsafe-fixes --diff => fix unsafe, potentially destructive violations (might need review)
# mypy torchsom/ --ignore-missing-imports => Skip checking for modules it cannot find

format:  ## Auto-fix formatting and import issues
	@echo "🎨 Auto-fixing code formatting..."
	black torchsom/ tests/
	isort torchsom/ tests/
	ruff check --fix torchsom/ tests/
	@echo "✅ Formatting applied!"

security:  ## Run security scans
	@echo "🔒 Running security scans..."
	@echo "🛡️ Bandit security check (library code)..."
	bandit -r torchsom/ --exclude tests --skip B101,B311,B601
	@echo "🛡️ Bandit security check (tests, skip assert rule)..."
	bandit -r tests --skip B101,B311,B601
	@echo "✅ Security scans completed!"
# @echo "🔍 Pip audit..."
# pip-audit
# @echo "⚠️ Safety vulnerability check..."
# safety scan

docs:  ## Check documentation quality
	@echo "📚 Checking documentation..."
	@echo "  📝 Docstring style..."
	pydocstyle torchsom/ --convention=google
	@echo "  📊 Docstring coverage..."
	interrogate torchsom/ --verbose --ignore-init-method --ignore-magic --ignore-module --fail-under=80
	@echo "  🏗️ Building HTML documentation..."
	sphinx-build -b html docs/source/ docs/build/html_temp
	@echo "✅ Documentation checks and build complete!"

clean-docs:  ## Remove Sphinx build artifacts
	@echo "🧹 Cleaning documentation build..."
	rm -rf docs/build/html_temp/
	@echo "✅ Cleaned!"
# rm -rf build/*

# ! NEED TO CHECK IF THIS WORKS
precommit:  ## Run pre-commit hooks on all files
	@echo "🔧 Running pre-commit hooks..."
	pre-commit run --all-files

clean:  ## Clean up generated files
	@echo "🧹 Cleaning up..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	rm -f junit.xml coverage.xml
	rm -f bandit-report.json safety-report.json pip-audit-report.json
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup completed!"

all: format lint security docs test  ## Run everything (full CI simulation)
	@echo ""
	@echo "🎉 All checks passed! Ready to push to GitHub!"

# Quick commands for common tasks, defining aliases for the most common commands (alias: common_command)
fix: format
check: lint
coverage: test
