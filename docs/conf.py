# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../bioverse/'))


# -- Project information -----------------------------------------------------

project = 'Bioverse'
copyright = '2021, Alex Bixel'
author = 'Alex Bixel'

# The full version, including alpha/beta/rc tags
release = '1.1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon', 
              'nbsphinx',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosectionlabel']

# Ensuring section labels are unique across your project
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'
#html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'rightsidebar': False,
    'stickysidebar': True,
    'collapsiblesidebar': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

nbsphinx_allow_errors = True

nbsphinx_prolog = r"""

.. raw:: html

    <div class="admonition note">
    <b>Note:</b> This example is available as an interactive Jupyter notebook in the file <a href="https://github.com/abixel/bioverse/blob/master/docs/{{env.doc2path(env.docname, base=none) }}"><code>{{ env.doc2path(env.docname, base=none) }}</code></a>
    </div>

"""

autodoc_member_order = 'bysource'

napoleon_google_docstring = False
