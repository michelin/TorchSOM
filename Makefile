# TorchSOM Development Makefile
# Run `make help` to see available commands

# Note on pyproject.toml integration:
#   - black, isort, ruff: read [tool.black], [tool.isort], [tool.ruff] automatically
#   - mypy: must pass --config-file pyproject.toml
#   - pytest: reads --cov-config=pyproject.toml for coverage config
#   - bandit: does NOT read TOML, uses CLI options only
#   - interrogate: can use --config if needed
#   - sphinx-build: uses conf.py, TOML not involved
#   - pip-compile: reads dependencies from pyproject.toml

.PHONY: help install test test-quick test-gpu test-integration lint format security docs clean clean-docs precommit complexity dependencies ci all publish fix check coverage install-dev install-tests install-security install-linting install-docs install-precommit install-all test-coverage check-black check-isort check-ruff check-mypy lint-all format-black format-isort format-ruff format-all check-docstrings measure-docstrings-coverage build-docs clean-build clean-test clean-lint clean-security clean-python clean-all check-cc check-mi complexity-all build-dist upload-dist

# --------------------------
# Aliases for commands
# --------------------------

install: install-all
cov: test-coverage
check: lint-all
fix: format-all
complexity: complexity-all
clean: clean-all

# --------------------------
# Default help target
# --------------------------

help:  ## Show this help message
	@echo "TorchSOM Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo ""
	@echo "  make install          # Install development dependencies"
	@echo "  make test             # Run all tests with coverage"
	@echo "  make lint             # Run all code quality checks"
	@echo "  make format           # Auto-fix formatting issues"
	@echo "  make all              # Run full CI simulation"
	@echo ""

# --------------------------
# Environment Setup
# --------------------------

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"

install-tests:  ## Install test dependencies
	pip install -e ".[tests]"

install-security:  ## Install security dependencies
	pip install -e ".[security]"

install-linting:  ## Install linting dependencies
	pip install -e ".[linting]"

install-docs:  ## Install documentation dependencies
	pip install -e ".[docs]"

install-precommit:  ## Install pre-commit dependencies
	pip install pre-commit
	pre-commit install

install-cz:  ## Install Commitizen
	pip install commitizen

install-all:  ## Install all dependencies
	@echo "📦 Installing all dependencies..."
	$(MAKE) install-dev
	$(MAKE) install-tests
	$(MAKE) install-security
	$(MAKE) install-linting
	$(MAKE) install-docs
	$(MAKE) install-precommit

# --------------------------
# Testing
# --------------------------

TESTS ?= tests/unit/  # Default test path

test-gpu:  ## Run GPU tests (requires CUDA)
	@echo "🖥️ Running GPU tests..."
	pytest $(TESTS) -v -x -m "gpu"
# -v verbose, -x exit on first failure, -m marker: run only tests with the corresponding markers

test-integration:  ## Run integration tests
	@echo "🖥️ Running integration tests..."
	pytest $(TESTS) -v -m "integration"

test-coverage:  ## Run all tests with coverage
	@echo "🧪 Running tests with coverage..."
	pytest
# pytest $(TESTS) -v \
# 	--cov=torchsom \
# 	--cov-report=term-missing \
# 	--cov-report=html \
# 	--cov-config=pyproject.toml \
# 	--junit-xml=junit.xml \
# 	-m "unit or gpu"

# --------------------------
# Code Quality
# --------------------------

check-black:  ## Run black check
	black --check --diff torchsom/ tests/

check-isort:  ## Run isort check
	isort --check-only --diff torchsom/ tests/

check-ruff:  ## Run ruff check
	ruff check torchsom/ tests/

check-mypy:  ## Run mypy check
	mypy torchsom/ --ignore-missing-imports --strict

lint-all:  ## Run all code quality checks
	@echo "🔍 Running code quality checks (formatting, sorting, linting, type checking)..."
	$(MAKE) check-black
	$(MAKE) check-isort
	$(MAKE) check-ruff
	$(MAKE) check-mypy
	@echo "✅ All quality checks passed!"

format-black:  ## Auto-fix black formatting
	black torchsom/ tests/

format-isort:  ## Auto-fix isort formatting
	isort torchsom/ tests/

format-ruff:  ## Auto-fix ruff formatting
	ruff check --fix torchsom/ tests/

format-all:  ## Auto-fix formatting and imports
	@echo "🎨 Auto-fixing code formatting, imports, and linting..."
	$(MAKE) format-black
	$(MAKE) format-isort
	$(MAKE) format-ruff
	@echo "✅ Formatting applied!"

precommit:  ## Run pre-commit hooks on all files
	@echo "🔧 Running pre-commit hooks..."
	pre-commit run --all-files

# --------------------------
# Security
# --------------------------

security:  ## Run security scans
	@echo "🔒 Running security scans..."
	bandit -r torchsom/ --exclude tests --skip B101,B311,B601
	bandit -r tests --skip B101,B311,B601
	@echo "✅ Security scans completed!"

# --------------------------
# Documentation
# --------------------------

check-docstrings:  ## Check docstrings
	pydocstyle torchsom/ --convention=google

measure-docstrings-coverage:  ## Assess docstring coverage
	interrogate torchsom/ --verbose --ignore-init-method --ignore-magic --ignore-module --fail-under=80

build-docs:  ## Build documentation
	sphinx-build -b html -W --keep-going docs/source/ docs/build/html
# sphinx-build -b html docs/source/ docs/build/html

docs:  ## Check documentation quality
	@echo "📚 Checking documentation..."
	$(MAKE) check-docstrings
	$(MAKE) measure-docstrings-coverage
	$(MAKE) build-docs
	@echo "✅ Documentation checks and build complete!"

# --------------------------
# Cleanup
# --------------------------

clean-docs:  ## Remove Sphinx build artifacts
	@echo "🧹 Cleaning documentation build..."
	rm -rf docs/build/html/
	@echo "✅ Documentation cleaned!"

clean-build:  ## Remove build and distribution artifacts
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	@echo "✅ Build artifacts cleaned!"

clean-test:  ## Remove test and coverage artifacts
	@echo "🧹 Cleaning test artifacts..."
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -f junit.xml coverage.xml
	@echo "✅ Test artifacts cleaned!"

clean-lint:  ## Remove linting and type checking cache
	@echo "🧹 Cleaning linting cache..."
	rm -rf .mypy_cache/ .ruff_cache/
	@echo "✅ Linting cache cleaned!"

clean-security:  ## Remove security scan reports
	@echo "🧹 Cleaning security reports..."
	rm -f bandit-report.json safety-report.json pip-audit-report.json
	@echo "✅ Security reports cleaned!"

clean-python:  ## Remove Python cache files
	@echo "🧹 Cleaning Python cache..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name __pycache__ -delete
	@echo "✅ Python cache cleaned!"

clean-all:  ## Clean up all generated files (runs all clean- commands)
	@echo "🧹 Cleaning up all generated files..."
	@echo ""
	$(MAKE) clean-build
	$(MAKE) clean-test
	$(MAKE) clean-lint
	$(MAKE) clean-security
	$(MAKE) clean-python
	$(MAKE) clean-docs
	@echo ""
	@echo "✅ All cleanup completed!"

# --------------------------
# Complexity Analysis
# --------------------------

check-cc:  ## Check cyclomatic complexity
	radon cc torchsom/ --show-complexity --min B

check-mi:  ## Check maintainability index
	radon mi torchsom/ --show --min B

complexity-all:  ## Run cyclomatic complexity and maintainability analysis
	@echo "🔍 Running complexity analysis..."
	$(MAKE) check-cc
	$(MAKE) check-mi
	@echo "✅ Complexity analysis completed!"

# --------------------------
# Dependencies
# --------------------------

dependencies:  ## Check for dependency conflicts: : super long with pip-compile
	@echo "🔍 Checking for dependency conflicts..."
	pip check
	@echo "✅ Dependency checks completed!"
# pip-compile pyproject.toml --dry-run --verbose

# --------------------------
# Changelog / Release Notes
# --------------------------

changelog:  ## Generate or update CHANGELOG.md based on commits
	cz changelog

bump:  ## Bump version and update changelog: should be used on the release branch (main)
	cz bump --changelog
# To preview the changes:
# cz bump --dry-run

# release-changelog: changelog  ## Generate changelog and commit it automatically
# 	git config user.name "github-actions[bot]"
# 	git config user.email "github-actions[bot]@users.noreply.github.com"
# 	git add CHANGELOG.md
# 	git commit -m "chore: update changelog for $(shell git describe --tags --abbrev=0)" || echo "No changes to commit"


# --------------------------
# CI / Full Pipeline
# --------------------------

ci:  ## Run CI pipeline (without tests)
	@echo ""
	$(MAKE) check
	$(MAKE) fix
	$(MAKE) security
	$(MAKE) complexity
	$(MAKE) dependencies
	$(MAKE) docs
	@echo "🎉 All checks passed (without tests)! Ready to push!"

all:  ## Run full CI simulation (includes tests)
	@echo ""
	$(MAKE) ci
	$(MAKE) cov
	@echo ""
	@echo "🎉 All checks passed! Ready to push!"

# --------------------------
# Publishing
# --------------------------

build-dist:  ## Build distribution
	python -m build

upload-dist:  ## Upload distribution
	twine upload dist/*

publish:  ## Build and upload to PyPI (manually but is triggered by .github/workflows/release.yml with tag modifications)
	@echo "📦 Publishing to PyPI..."
	@bash -c '\
		source .env; \
		export TWINE_USERNAME TWINE_PASSWORD; \
		$(MAKE) build-dist; \
		$(MAKE) upload-dist \
	'
	@echo "✅ Published to PyPI!"
