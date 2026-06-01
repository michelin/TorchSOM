# --------------------------
# Testing
# --------------------------

TESTS ?= tests/unit/

.PHONY: test-unit test-smoke test-gpu test-integration test-coverage

test-unit:  ## Run unit tests (fast, no coverage)
	uv run pytest $(TESTS) -v -x -m "unit" --no-cov

test-smoke:  ## Run smoke tests for basic functionality
	uv run pytest $(TESTS) -v -x -m "smoke" --no-cov

test-gpu:  ## Run GPU tests (requires CUDA)
	uv run pytest $(TESTS) -v -x -m "gpu"

test-integration:  ## Run integration tests
	uv run pytest $(TESTS) -v -m "integration"

test-coverage:  ## Run all tests with coverage
	@echo "Running tests with coverage..."
	uv run pytest
