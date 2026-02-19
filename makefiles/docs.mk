# --------------------------
# Documentation
# --------------------------
# Docstring style is enforced by ruff's D rules (pydocstyle convention: google).
# Docstring coverage is measured by interrogate.

.PHONY: measure-docstrings-coverage build-docs open-docs serve-docs docs

measure-docstrings-coverage:  ## Assess docstring coverage
	uv run interrogate torchsom/ --verbose --ignore-init-method --ignore-magic --ignore-module --fail-under=80

build-docs:  ## Build Sphinx documentation
	uv run sphinx-build -b html --keep-going docs/source/ docs/build/html

open-docs:  ## Open built documentation in browser
	@if [ -f docs/build/html/index.html ]; then \
		python -m webbrowser docs/build/html/index.html; \
	else \
		echo "Documentation not built yet. Run 'make docs' first."; \
	fi

serve-docs:  ## Serve documentation with live-reload (requires sphinx-autobuild)
	uv run sphinx-autobuild docs/source/ docs/build/html --open-browser --watch torchsom/

docs:  ## Build documentation and measure docstring coverage
	@echo "Building documentation..."
	$(MAKE) measure-docstrings-coverage
	$(MAKE) build-docs
	@echo "Documentation complete!"
