# --------------------------
# Cleanup
# --------------------------

.PHONY: clean-docs clean-build clean-test clean-lint clean-security clean-python clean-notebooks clean-all

clean-docs:  ## Remove Sphinx build artifacts
	rm -rf docs/build/html/

clean-build:  ## Remove build and distribution artifacts
	rm -rf build/ dist/ *.egg-info/

clean-test:  ## Remove test and coverage artifacts
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -f junit.xml coverage.xml

clean-lint:  ## Remove linting and type checking cache
	rm -rf .mypy_cache/ .ruff_cache/

clean-security:  ## Remove security scan reports
	rm -f bandit-report.json safety-report.json pip-audit-report.json

clean-python:  ## Remove Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name __pycache__ -delete

clean-notebooks:  ## Clear Jupyter notebook outputs
	find notebooks/ -name "*.ipynb" -exec uv run jupyter nbconvert --clear-output --inplace {} \; 2>/dev/null || true

clean-all:  ## Clean all generated files
	@echo "Cleaning all generated files..."
	$(MAKE) clean-build
	$(MAKE) clean-test
	$(MAKE) clean-lint
	$(MAKE) clean-security
	$(MAKE) clean-python
	$(MAKE) clean-docs
	@echo "Cleanup complete."
