# TorchSOM Development Makefile
# Run `make` or `make help` to see available commands.
# Uses `uv` (https://docs.astral.sh/uv/) for environment and dependency management.

SHELL := /bin/bash
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

# --------------------------
# Sub-makefiles
# --------------------------

include makefiles/ci.mk
include makefiles/clean.mk
include makefiles/complexity.mk
include makefiles/docs.mk
include makefiles/format.mk
include makefiles/install.mk
include makefiles/lint.mk
include makefiles/release.mk
include makefiles/security.mk
include makefiles/test.mk

# --------------------------
# Convenience Aliases
# --------------------------

.PHONY: install cov check fix complexity clean dev qa

install: install-all       ## Alias: sync all dependencies
cov: test-coverage         ## Alias: run tests with coverage
check: lint-all            ## Alias: run all linting checks
fix: format-all            ## Alias: auto-fix formatting
complexity: complexity-all ## Alias: run complexity analysis
clean: clean-all           ## Alias: clean all generated files

dev: install-all           ## Setup dev environment (install + pre-commit)

qa:                        ## Run full QA suite (check + fix + cov)
	$(MAKE) check
	$(MAKE) fix
	$(MAKE) cov

# --------------------------
# Help
# --------------------------

# Color codes
CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
BOLD   := \033[1m
RESET  := \033[0m

help:  ## Show this help message
	@printf "\n$(BOLD)TorchSOM Development Commands$(RESET) (powered by uv)\n\n"
	@printf "$(GREEN)Setup$(RESET)\n"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' makefiles/install.mk | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@printf "\n$(GREEN)Code Quality$(RESET)\n"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' makefiles/lint.mk makefiles/format.mk | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@printf "\n$(GREEN)Testing$(RESET)\n"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' makefiles/test.mk | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@printf "\n$(GREEN)Documentation$(RESET)\n"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' makefiles/docs.mk | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@printf "\n$(GREEN)Security & Complexity$(RESET)\n"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' makefiles/security.mk makefiles/complexity.mk | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@printf "\n$(GREEN)Release$(RESET)\n"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' makefiles/release.mk | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@printf "\n$(GREEN)CI / Pipelines$(RESET)\n"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' makefiles/ci.mk | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@printf "\n$(GREEN)Cleanup$(RESET)\n"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' makefiles/clean.mk | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@printf "\n$(GREEN)Aliases$(RESET)\n"
	@printf "  $(CYAN)%-28s$(RESET) %s\n" "install" "Sync all dependencies"
	@printf "  $(CYAN)%-28s$(RESET) %s\n" "cov" "Run tests with coverage"
	@printf "  $(CYAN)%-28s$(RESET) %s\n" "check" "Run all linting checks"
	@printf "  $(CYAN)%-28s$(RESET) %s\n" "fix" "Auto-fix formatting"
	@printf "  $(CYAN)%-28s$(RESET) %s\n" "dev" "Setup dev environment"
	@printf "  $(CYAN)%-28s$(RESET) %s\n" "qa" "Full QA (check + fix + cov)"
	@printf "  $(CYAN)%-28s$(RESET) %s\n" "clean" "Clean all generated files"
	@printf "\n"
