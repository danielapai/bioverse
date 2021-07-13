.. Bioverse documentation master file, created by
   sphinx-quickstart on Mon Jun 21 12:19:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bioverse
========

Bioverse is a Python package for simulating the results of a statistical survey of the properties of nearby terrestrial exoplanets via direct imaging or transit spectroscopy. An in-depth outline of the underlying statistical framework and examples of how it can be applied to astrophysics mission concepts is given in `Bixel & Apai (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..228B/abstract>`_. **Readers are strongly encouraged to review this paper before proceeding.** This documentation covers the Python implementation of Bioverse, but does not review many of its underlying statistical assumptions.

The :doc:`Overview<overview>` section describes the code's structure and primary classes and should be reviewed first. Following that, the :doc:`Examples<Notebooks/Tutorial1>` section offers step-by-step examples for producing some of the results published in the paper, as well as ways to modify and expand upon the code. These examples are also available as interactive Jupyter notebooks in the ``Notebooks`` directory of the GitHub repository.

References & Acknowledgements
*****************************
Papers making use of Bioverse should reference `Bixel & Apai (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..228B/abstract>`_. You should also include references to the `emcee <https://github.com/dfm/emcee>`_ and `dynesty <https://github.com/joshspeagle/dynesty>`_ packages, which are used for hypothesis testing and parameter fitting.

Bioverse was developed with support from the following grants and collaborations:

- `Alien Earths <http://eos-nexus.org/>`_ & Earths in Other Solar Systems
- NASA Earth and Space Science Fellowship Program (grant No. 80NSSC17K0470)
- NASA's Nexus for Exoplanet System Science (NExSS) 

Installation
************

Bioverse can be cloned from its `GitHub repository <https://www.github.com/abixel/bioverse/>`_:

.. code-block:: bash

    git clone https://www.github.com/abixel/bioverse/

To install Bioverse, navigate to the directory containing ``setup.py`` and run: [#f1]_

.. code-block:: bash

   pip install .

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
- ``PyQt5`` (optional: enables configuration GUI)

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
   gui
   
.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:
   
   Notebooks/Tutorial1
   Notebooks/Tutorial2
   Notebooks/Example1
   Notebooks/Example2

.. toctree::
   :caption: API Documentation
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

.. rubric:: Footnotes

.. [#f1] Bioverse will be added to PyPI in a future update.
