# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(1, os.path.abspath("../../lib/sycamore/sycamore/"))

project = "Sycamore"
copyright = "2024, Aryn Inc."
author = "The Aryn Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []
smartquotes = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "Sycamore"
html_static_path = ["_static"]
html_favicon = "images/favicon.ico"
html_theme_options = {
    "source_repository": "https://github.com/aryn-ai/sycamore",
    "source_branch": "main",
    "source_directory": "docs/",
}
