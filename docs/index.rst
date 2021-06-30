.. Bioverse documentation master file, created by
   sphinx-quickstart on Mon Jun 21 12:19:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bioverse
====================================

Bioverse is a Python package for simulating the results of a statistical survey of the properties of nearby terrestrial exoplanets via direct imaging or transit spectroscopy. An in-depth outline of the underlying statistical framework and examples of how it can be applied to astrophysics mission concepts is given in `Bixel & Apai (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..228B/abstract>`_. **Readers are strongly encouraged to review this paper before proceeding.** This documentation covers the Python implementation of ``Bioverse``, but does not review many of its underlying statistical assumptions.

The :doc:`Overview<overview>` section describes the code's structure and primary classes and should be reviewed first. Following that, the :doc:`Examples<Notebooks/Tutorial1>` section offers step-by-step examples for producing some of the results published in the paper, as well as ways to modify and expand upon the code. These examples are also available as interactive Jupyter notebooks in the ``Notebooks`` directory of the GitHub repository.

Installation
*****************************

Bioverse can be cloned from its GitHub repository:

.. code-block:: bash

    git clone https://www.github.com/abixel/bioverse/

To install Bioverse, navigate to the directory containing ``setup.py`` and run:

.. code-block:: bash

   pip install .

Bioverse will be added to PyPI in a future update.

Dependencies
************
Bioverse is compatible with Python 3.7+. It has the following dependencies, all of which can be installed using ``pip``:

- ``astroquery``
- ``dynesty``
- ``emcee``
- ``matplotlib``
- ``numpy``
- ``scipy``
- ``tqdm`` (optional: provides a progress bar for long processes)
- ``pandas`` (optional: used for data visualization)

.. toctree::
   :maxdepth: 2
   :caption: Overview
   :hidden:
   
   overview
   table
   module1
   module2
   module3
   module4
   
.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:
   
   Notebooks/GettingStarted1
   Notebooks/GettingStarted2
   Notebooks/Example1
   Notebooks/Example2

.. toctree::
   :caption: Full Documentation
   :hidden:
   :glob:   

   apidoc/bioverse.analysis
   apidoc/bioverse.classes
   apidoc/bioverse.constants
   apidoc/bioverse.custom
   apidoc/bioverse.functions
   apidoc/bioverse.generator
   apidoc/bioverse.hypothesis
   apidoc/bioverse.plots
   apidoc/bioverse.survey
   apidoc/bioverse.util
