# --------------------------
# Complexity Analysis
# --------------------------

.PHONY: check-cc check-mi complexity-all

check-cc:  ## Check cyclomatic complexity (flag B+ ratings)
	uv run radon cc torchsom/ --show-complexity --min B

check-mi:  ## Check maintainability index (flag B+ ratings)
	uv run radon mi torchsom/ --show --min B

complexity-all:  ## Run cyclomatic complexity and maintainability analysis
	@echo "Running complexity analysis..."
	$(MAKE) check-cc
	$(MAKE) check-mi
	@echo "Complexity analysis completed."
