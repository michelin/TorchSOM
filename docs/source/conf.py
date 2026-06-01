"""Configuration for Sphinx documentation."""

import shutil
from contextlib import suppress
from pathlib import Path

from sphinx.application import Sphinx


def _read_version() -> str:
    """Read the package version from pyproject.toml."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            if line.strip().startswith('version = "'):
                return line.split('"')[1]
    return "0.0.0"


def _copy_repo_assets_to_static() -> None:
    """Copy repo-level assets/ into docs/_static/assets for reliable access."""
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "assets"
    dst = Path(__file__).parent / "_static" / "assets"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with suppress(Exception):
            shutil.copytree(src, dst, dirs_exist_ok=True)


def setup(app: Sphinx) -> None:  # noqa: ARG001
    """Setup the Sphinx application."""
    _copy_repo_assets_to_static()


# -- Project information -----------------------------------------------------

project = "TorchSOM"
copyright = "2025, Manufacture Française des Pneumatiques Michelin"
author = "Louis Berthier"
release = _read_version()

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_design",
]

autodoc_typehints = "description"
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = []

# -- MathJax configuration ---------------------------------------------------
# Define LaTeX macros used in the math so the docs match the JMLR paper's
# notation. MathJax v3 core does not provide \coloneqq (it comes from the
# mathtools package in LaTeX), so it is declared here.
mathjax3_config = {
    "tex": {
        "macros": {
            "coloneqq": "\\mathrel{:=}",
        },
    },
}

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/michelin/TorchSOM",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#B22222",
        "color-brand-content": "#B22222",
    },
    "dark_css_variables": {
        "color-brand-primary": "#E05555",
        "color-brand-content": "#E05555",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_static_path = ["_static"]
html_logo = "_static/assets/logo.png"

# -- Todo extension ----------------------------------------------------------
# Author-facing only: keep ``.. todo::`` notes out of the published HTML.
todo_include_todos = False

# -- Linkcheck ---------------------------------------------------------------
# PyTorch's documentation renders anchors client-side, so linkcheck cannot verify
# them from static HTML even though intersphinx resolves the targets at build time.
linkcheck_anchors_ignore_for_url = [r"https://docs\.pytorch\.org/.*"]
