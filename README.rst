Bioverse
********

Bioverse is a Python package for simulating the results of a statistical survey of the properties of nearby terrestrial exoplanets via direct imaging or transit spectroscopy. An in-depth outline of the underlying statistical framework and examples of how it can be applied to astrophysics mission concepts is given in `Bixel & Apai (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..228B/abstract>`_.

For documentation, see https://bioverse.readthedocs.io/.

References
**********
Papers making use of Bioverse should reference `Bixel & Apai (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..228B/abstract>`_. You should also include references to the `emcee <https://github.com/dfm/emcee>`_ and `dynesty <https://github.com/joshspeagle/dynesty>`_ packages, which are used for hypothesis testing and parameter fitting.

Installation
************

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
