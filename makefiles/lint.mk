# --------------------------
# Code Quality
# --------------------------

.PHONY: check-ruff check-mypy lint-all

check-ruff:  ## Run ruff linter (style, imports, security, docstrings)
	uv run ruff check torchsom/ tests/

check-mypy:  ## Run mypy type checker
	uv run mypy torchsom/ --ignore-missing-imports

lint-all:  ## Run all code quality checks (ruff + mypy)
	@echo "Running code quality checks..."
	$(MAKE) check-ruff
	$(MAKE) check-mypy
	@echo "All quality checks passed!"
