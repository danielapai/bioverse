.. Bioverse documentation master file, created by
   sphinx-quickstart on Mon Jun 21 12:19:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bioverse
========

Bioverse is a Python package for simulating the results of a statistical survey of the properties of nearby terrestrial exoplanets via direct imaging or transit spectroscopy. An in-depth outline of the underlying statistical framework and examples of how it can be applied to astrophysics mission concepts is given in `Bixel & Apai (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..228B/abstract>`_. **Readers are strongly encouraged to review this paper before proceeding.** This documentation covers the Python implementation of Bioverse, but does not review many of its underlying statistical assumptions.

The :doc:`Overview<overview>` section describes the code's structure and primary classes and should be reviewed first. Following that, the :doc:`Examples<Notebooks/Tutorial1>` section offers step-by-step examples for producing some of the results published in the paper, as well as ways to modify and expand upon the code. Most of these examples are also available as interactive Jupyter notebooks in the ``Notebooks`` directory of the GitHub repository.

Installation
************

Bioverse can now be installed directly from PyPI. To install Bioverse, use the following command:

.. code-block:: bash

    pip install bioverse

Alternatively, Bioverse can be cloned from its `GitHub repository <https://github.com/danielapai/bioverse/>`_:

.. code-block:: bash

    git clone https://www.github.com/danielapai/bioverse/
    cd bioverse
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
   tutorial_tref
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

Feedback & Development
**********************
Bioverse is open source and in active development. We welcome all feedback, bug reports, or feature requests. Feel free to open a pull request if you'd like to contribute! If you think you found a bug, please raise an `issue <https://github.com/danielapai/bioverse/issues>`_.

References & Acknowledgements
*****************************
Papers making use of Bioverse should cite `Bixel & Apai (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..228B/abstract>`_, `Hardegree-Ullman et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023AJ....165..267H/abstract>`_, and `Schlecker et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024PSJ.....5....3S/abstract>`_. ::

 @ARTICLE{2021AJ....161..228B,
   author = {{Bixel}, Alex and {Apai}, D{\'a}niel},
   title = "{Bioverse: A Simulation Framework to Assess the Statistical Power of Future Biosignature Surveys}",
   journal = {\aj},
   keywords = {Astrobiology, Exoplanets, Exoplanet atmospheres, Astrostatistics, Open source software, 74, 498, 487, 1882, 1866, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
   year = 2021,
   month = may,
   volume = {161},
   number = {5},
   eid = {228},
   pages = {228},
   doi = {10.3847/1538-3881/abe042},
 archivePrefix = {arXiv},
   eprint = {2101.10393},
   primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2021AJ....161..228B},
   adsnote = {Provided by the SAO/NASA Astrophysics Data System}
 }

::

 @ARTICLE{2023AJ....165..267H,
   author = {{Hardegree-Ullman}, Kevin K. and {Apai}, D{\'a}niel and {Bergsten}, Galen J. and {Pascucci}, Ilaria and {L{\'o}pez-Morales}, Mercedes},
   title = "{Bioverse: A Comprehensive Assessment of the Capabilities of Extremely Large Telescopes to Probe Earth-like O$_{2}$ Levels in Nearby Transiting Habitable-zone Exoplanets}",
   journal = {\aj},
   keywords = {Fundamental parameters of stars, Exoplanet systems, Exoplanets, Exoplanet atmospheres, Biosignatures, 555, 484, 498, 487, 2018, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
   year = 2023,
   month = jun,
   volume = {165},
   number = {6},
   eid = {267},
   pages = {267},
   doi = {10.3847/1538-3881/acd1ec},
 archivePrefix = {arXiv},
   eprint = {2304.12490},
   primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2023AJ....165..267H},
   adsnote = {Provided by the SAO/NASA Astrophysics Data System}
 }

::

 @ARTICLE{2024PSJ.....5....3S,
   author = {{Schlecker}, Martin and {Apai}, D{\'a}niel and {Lichtenberg}, Tim and {Bergsten}, Galen and {Salvador}, Arnaud and {Hardegree-Ullman}, Kevin K.},
   title = "{Bioverse: The Habitable Zone Inner Edge Discontinuity as an Imprint of Runaway Greenhouse Climates on Exoplanet Demographics}",
   journal = {\psj},
   keywords = {Habitable zone, Habitable planets, Astrobiology, Extrasolar rocky planets, Planetary climates, Exoplanet atmospheres, Astronomical simulations, Exoplanets, Transit photometry, Radial velocity, Bayesian statistics, Parametric hypothesis tests, 696, 695, 74, 511, 2184, 487, 1857, 498, 1709, 1332, 1900, 1904, Astrophysics - Earth and Planetary Astrophysics},
   year = 2024,
   month = jan,
   volume = {5},
   number = {1},
   eid = {3},
   pages = {3},
   doi = {10.3847/PSJ/acf57f},
   archivePrefix = {arXiv},
   eprint = {2309.04518},
   primaryClass = {astro-ph.EP},
   adsurl = {https://ui.adsabs.harvard.edu/abs/2024PSJ.....5....3S},
   adsnote = {Provided by the SAO/NASA Astrophysics Data System}
 }



If you make use of the integrated hypothesis testing and parameter fitting, you should also include references to the `emcee <https://github.com/dfm/emcee>`_ and `dynesty <https://github.com/joshspeagle/dynesty>`_ packages.



Bioverse was developed with support from the following grants and collaborations:

- `Alien Earths <http://eos-nexus.org/>`_ & Earths in Other Solar Systems
- NASA Earth and Space Science Fellowship Program (grant No. 80NSSC17K0470)
- NASA's Nexus for Exoplanet System Science (NExSS)
