# --------------------------
# Formatting
# --------------------------

.PHONY: format-ruff format-all precommit

format-ruff:  ## Auto-fix formatting, imports, and lint issues
	uv run ruff format torchsom/ tests/
	uv run ruff check --fix torchsom/ tests/

format-all:  ## Auto-fix all formatting issues
	@echo "Auto-fixing code formatting..."
	$(MAKE) format-ruff
	@echo "Formatting applied."

precommit:  ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files
