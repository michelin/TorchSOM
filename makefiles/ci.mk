# --------------------------
# CI / Full Pipeline
# --------------------------

.PHONY: ci all

ci:  ## Run CI pipeline (lint + security + complexity + docs, no tests)
	@echo ""
	$(MAKE) check
	$(MAKE) security
	$(MAKE) complexity
	$(MAKE) docs
	@echo ""
	@echo "All CI checks passed (without tests)!"

all:  ## Run full CI simulation (CI + format + tests)
	@start=$$(date +%s); \
	$(MAKE) ci; \
	$(MAKE) fix; \
	$(MAKE) cov; \
	elapsed=$$(( $$(date +%s) - $$start )); \
	echo ""; \
	echo "All checks passed in $${elapsed}s! Ready to push."
