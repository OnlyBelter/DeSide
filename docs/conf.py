# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(r'..'))


# -- Project information -----------------------------------------------------

project = 'DeSide'
copyright = '2021, iSynBio'
author = 'Xin Xiong'

# The full version, including alpha/beta/rc tags
release = '0.9.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    'sphinx.ext.autodoc',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.autosummary',
    # 'sphinx.ext.intersphinx',
    # 'matplotlib.sphinxext.plot_directive',
    # 'gallery_generator',
    # 'numpydoc',
    # 'sphinx_issues',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# https://sphinx-book-theme.readthedocs.io/en/latest/, sphinx-book-theme-0.0.42
html_theme = 'sphinx_book_theme'
html_logo = "_static/logo.png"
html_title = "DEep-learning and SIngle-cell based DEconvolution"
html_copy_source = True
# html_sourcelink_suffix = ""
html_favicon = "_static/logo.png"
# html_last_updated_fmt = ""


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# ref to https://github.com/mwaskom/seaborn/blob/master/doc/conf.py
html_theme_options = {
    "theme_dev_mode": True,
    "path_to_docs": "docs",
    "repository_url": "https://github.com/OnlyBelter/DeSide",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}

# Add type of source files
source_suffix = ['.rst', '.md']

fontawesome_included = True
