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

.PHONY: install cov check fix complexity clean dev qa workflow

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
	@printf "\n$(GREEN)Reference$(RESET)\n"
	@printf "  $(CYAN)%-28s$(RESET) %s\n" "workflow" "Print the recommended development workflow"
	@printf "\n"


workflow:                  ## Print the recommended development workflow
	@printf "\n$(BOLD)TorchSOM — Development Workflow$(RESET)\n"
	@printf "Run $(CYAN)make workflow$(RESET) any time to see this reference.\n\n"
	@printf "$(BOLD)━━━ 1. First-time setup (once per machine) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "  $(CYAN)make dev$(RESET)\n"
	@printf "  Installs all dependency groups and wires up pre-commit hooks.\n\n"
	@printf "$(BOLD)━━━ 2. Inner loop — while coding ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "  $(CYAN)make fix$(RESET)        Auto-format code (ruff format + ruff --fix)\n"
	@printf "  $(CYAN)make test-unit$(RESET)  Run unit tests fast, fail on first error (-x, no coverage)\n"
	@printf "  $(CYAN)make test-smoke$(RESET) Quick sanity check on core functionality only\n"
	@printf "  Repeat freely. Format before testing so failures are logic issues, not style.\n\n"
	@printf "$(BOLD)━━━ 3. Before every commit ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "  Pre-commit runs automatically if installed via $(CYAN)make dev$(RESET).\n"
	@printf "  To run it manually: $(CYAN)make precommit$(RESET)\n"
	@printf "  If it modifies files, stage them and commit again.\n\n"
	@printf "  Use Conventional Commits for messages — required for auto-changelog:\n"
	@printf "    $(YELLOW)feat$(RESET): new feature   $(YELLOW)fix$(RESET): bug fix   $(YELLOW)refactor$(RESET): no behaviour change\n"
	@printf "    $(YELLOW)docs$(RESET): docs only      $(YELLOW)test$(RESET): tests     $(YELLOW)chore$(RESET): tooling / deps\n\n"
	@printf "$(BOLD)━━━ 4. Before opening a Pull Request ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "  Run in order (each step gates the next):\n\n"
	@printf "  $(CYAN)make fix$(RESET)        1. Auto-fix style — clean diff before anything else\n"
	@printf "  $(CYAN)make check$(RESET)      2. Lint (ruff) + type-check (mypy) — catch errors cheaply\n"
	@printf "  $(CYAN)make cov$(RESET)        3. Full test suite with coverage report\n"
	@printf "  $(CYAN)make security$(RESET)   4. Audit dependencies for known CVEs (pip-audit)\n"
	@printf "  $(CYAN)make complexity$(RESET) 5. Flag functions with high cyclomatic complexity (radon)\n"
	@printf "  $(CYAN)make docs$(RESET)       6. Docstring coverage (≥80%%) + build Sphinx HTML\n\n"
	@printf "  Or in one shot: $(CYAN)make all$(RESET)  (ci + fix + cov)\n\n"
	@printf "$(BOLD)━━━ 5. Release (main branch only) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "  $(CYAN)make bump$(RESET)       Bump version + auto-generate CHANGELOG.md from commits\n"
	@printf "  $(CYAN)make publish$(RESET)    Build dist and upload to PyPI (reads .env for credentials)\n\n"
	@printf "$(BOLD)━━━ Quick reference table ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)\n"
	@printf "  $(BOLD)%-20s %-22s %s$(RESET)\n" "When" "Command" "Why"
	@printf "  %-20s $(CYAN)%-22s$(RESET) %s\n" "First clone"       "make dev"        "Install everything + hooks"
	@printf "  %-20s $(CYAN)%-22s$(RESET) %s\n" "While coding"      "make fix"        "Auto-format before anything"
	@printf "  %-20s $(CYAN)%-22s$(RESET) %s\n" ""                  "make test-unit"  "Fast feedback loop"
	@printf "  %-20s $(CYAN)%-22s$(RESET) %s\n" "Before commit"     "make precommit"  "Style gate (auto if hooks set)"
	@printf "  %-20s $(CYAN)%-22s$(RESET) %s\n" "Before PR"         "make all"        "Full quality gate"
	@printf "  %-20s $(CYAN)%-22s$(RESET) %s\n" "Release branch"    "make bump"       "Version bump + changelog"
	@printf "  %-20s $(CYAN)%-22s$(RESET) %s\n" ""                  "make publish"    "Ship to PyPI"
	@printf "\n"
