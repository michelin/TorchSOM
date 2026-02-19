# --------------------------
# Security
# --------------------------
# Security linting (bandit-equivalent) is handled by ruff's S rules in lint.mk.
# This target runs an explicit audit of installed dependencies.

.PHONY: security

security:  ## Audit installed packages for known vulnerabilities
	@echo "Auditing dependencies..."
	uv run pip-audit
	@echo "Security audit completed."
