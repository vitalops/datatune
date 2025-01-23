# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Datatune'
copyright = '2025, Abhijith Neil Abraham'
author = 'Abhijith Neil Abraham'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx_design'
]

# Enable MyST features
myst_enable_extensions = [
    "colon_fence",
    "attrs_inline",
    "deflist"
]

# Custom CSS
html_css_files = [
    'css/custom.css',
]

# Change the theme to Read the Docs
html_theme = 'sphinxawesome_theme'

# Add markdown support
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

# for GitHub Pages
html_baseurl = 'https://vitalops.github.io/datatune'
html_static_path = ['_static']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
