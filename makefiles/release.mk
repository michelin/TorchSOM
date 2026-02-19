# --------------------------
# Changelog / Release Notes
# --------------------------

.PHONY: changelog bump

changelog:  ## Generate or update CHANGELOG.md based on commits
	uv run cz changelog

bump:  ## Bump version and update changelog (use on release branch: main)
	uv run cz bump --changelog

# --------------------------
# Publishing
# --------------------------

.PHONY: build-dist upload-dist publish

build-dist:  ## Build distribution packages
	uv build

upload-dist:  ## Upload distribution to PyPI
	uv run twine upload dist/*

publish:  ## Build and upload to PyPI
	@echo "Publishing to PyPI..."
	@bash -c '\
		source .env; \
		export TWINE_USERNAME TWINE_PASSWORD; \
		$(MAKE) build-dist; \
		$(MAKE) upload-dist \
	'
	@echo "Published to PyPI."
