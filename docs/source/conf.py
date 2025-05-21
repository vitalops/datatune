# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Datatune'
copyright = '2025, Vitalops Technologies'
author = 'Abhijith Neil Abraham, Fariz Rahman'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',  # Important for markdown support
    'sphinx_book_theme',
    'sphinx_design',
]

# Set source parsers (explicit registration)
source_parsers = {
   '.md': 'myst_parser.sphinx_',
}

# Add markdown support - IMPORTANT: Order matters here!
source_suffix = ['.rst', '.md']

# Enable MyST features - expanded list for better Markdown support
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
]

# For code block syntax highlighting
myst_highlight_code_blocks = True

# Custom CSS
html_css_files = [
    'css/custom.css',
]

# Use the Sphinx Book Theme
html_theme = 'sphinx_book_theme'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# for GitHub Pages
html_baseurl = 'https://vitalops.github.io/datatune'
html_static_path = ['_static']

# Theme options - enhanced for better documentation appearance
html_logo = "_static/datatune_logo.png"
html_title = "Datatune"
html_favicon = "_static/datatune_logo.png"

html_theme_options = {
    "logo": {
        "image_light": "_static/datatune_logo.png",
        "image_dark": "_static/datatune_logo.png"
    },
    "repository_url": "https://github.com/vitalops/datatune",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "announcement": "Datatune: Transform data with natural language using LLMs",
    "extra_navbar": "",
    "extra_footer": "",
}

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__call__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for linkcheck builder -------------------------------------------
linkcheck_ignore = [r'https://github.com/vitalops/datatune/.*']