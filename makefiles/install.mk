# --------------------------
# Environment Setup (uv)
# --------------------------

.PHONY: install-dev install-tests install-security install-linting install-docs install-precommit install-cz install-all

install-dev:  ## Sync development dependencies
	uv sync --extra dev

install-tests:  ## Sync test dependencies
	uv sync --extra tests

install-security:  ## Sync security dependencies
	uv sync --extra security

install-linting:  ## Sync linting dependencies
	uv sync --extra linting

install-docs:  ## Sync documentation dependencies
	uv sync --extra docs

install-precommit:  ## Install pre-commit hooks
	uv run pre-commit install

install-cz:  ## Install Commitizen
	uv add --dev commitizen

install-all:  ## Sync all dependencies and install hooks
	uv sync --all-extras
	$(MAKE) install-precommit
