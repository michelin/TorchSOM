# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TorchSOM"
copyright = "2025, Manufacture Française des Pneumatiques Michelin"
author = "Louis Berthier"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",  # Reports build duration per step.
    "sphinx.ext.doctest",  # Allows running doctests in code snippets embedded in the documentation.
    "sphinx.ext.autodoc",  # Automatically includes documentation from Python docstrings.
    "sphinx.ext.autosummary",  # Generates summary tables for modules/functions/classes with short descriptions.
    "sphinx.ext.intersphinx",  # Links to objects in external documentation projects (e.g., Python, NumPy).
    "sphinx.ext.viewcode",  # Adds links to highlighted source code.
    "sphinx.ext.githubpages",  # Adds .nojekyll file needed for GitHub Pages deployment.
    "sphinx_copybutton",  # Adds a "copy to clipboard" button on code blocks.
    "sphinx.ext.napoleon",  # Enables parsing of NumPy/Google-style docstrings.
]

autodoc_typehints = "description"
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # "alabaster" "sphinx_rtd_theme"
html_theme_options = {
    # "analytics_id": "G-XXXXXXXXXX",  # Google Analytics ID to enable pageview tracking, provided in the Google Analytics account.
    "analytics_anonymize_ip": False,  # If True, anonymizes user IP addresses in analytics data.
    "logo_only": False,  # If True, displays only the logo without project title text.
    "prev_next_buttons_location": "both",  # Places "Previous" and "Next" navigation buttons at the bottom of the page. Options: 'top', 'bottom', 'both'.
    "style_external_links": False,  # If True, adds an external link icon to all external hyperlinks.
    "vcs_pageview_mode": "",  # Version control system mode (e.g., 'blob' for GitHub). Empty disables this.
    "style_nav_header_background": "#B22222",  # Changes the hex color of the top left header containing the logo.
    "flyout_display": "hidden",  # Controls behavior of sidebar flyouts. 'hidden' disables them.
    "version_selector": True,  # Enables a UI component for selecting documentation versions (requires configuration).
    "language_selector": True,  # Enables a UI component for selecting languages (requires localization setup).
    "collapse_navigation": False,  # Collapses sub-sections in the sidebar for cleaner appearance. If False then expend.
    "sticky_navigation": True,  # Keeps sidebar navigation fixed while scrolling.
    "navigation_depth": 4,  # Enables deeper table-of-contents nesting in the sidebar.
    "includehidden": True,  # Includes hidden TOC entries in the sidebar.
    "titles_only": False,  # If True, shows only the page titles (no section titles) in the sidebar.
}

html_context = {
    "display_github": True,  # Enable GitHub link in the header
    "github_user": "LouisTier",
    "github_repo": "TorchSOM",
    "github_version": "dev",
    "conf_py_path": "/docs/source/",
}
html_static_path = ["_static"]
html_logo = "logo.jpg"
# html_favicon = "../../favicon.ico" # Icon shown in the browser tab
