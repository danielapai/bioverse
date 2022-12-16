####################################
Generating planetary systems
####################################

The :class:`~bioverse.generator.Generator` class
************************************************

Bioverse uses the :class:`~bioverse.generator.Generator` class to generate planetary systems in the solar neighborhood. A Generator object specifies a list of functions to be performed in sequential order onto a shared :class:`~bioverse.classes.Table`. For example, a simple generator might implement this algorithm:

- Function 1: Return the `Gaia DR2 <https://www.cosmos.esa.int/web/gaia/dr2>`_ catalog of all stars within 30 parsecs with effective temperatures above 4000 K.
- Function 2: Simulate one or more planets around each star according to the occurrence rate estimates in `Bergsten et al. 2022 <https://ui.adsabs.harvard.edu/link_gateway/2022AJ....164..190B/doi:10.3847/1538-3881/ac8fea>`_.
- Function 3: Evaluate the mass of each planet based on its radius and the mass-radius relationship published by `Wolfgang et al. (2016) <https://ui.adsabs.harvard.edu/abs/2016ApJ...825...19W/abstract>`_.

The generator will feed the output of Function 1 into Function 2, then the output of Function 2 into Function 3, and finally will return the output of Function 3 (i.e. a table of planets with known masses, radii, orbital properties, and host star properties).

Bioverse "ships" with two Generators: one for transit mode, and the other for imaging mode. The primary difference between the two is that the former uses the `Chabrier (2003) <https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract>`_ stellar mass function to generate host stars, while the latter uses an optimized host star catalog for the LUVOIR direct imaging mission (see the `LUVOIR Final Report <https://arxiv.org/abs/1912.06219>`_). The following code demonstrates how to simulate a sample of planets using the imaging mode Generator:

.. code-block:: python

    from bioverse.generator import Generator
    generator = Generator('imaging')
    sample = generator.generate()

We can inspect the Generator to see which functions it implements:

.. code-block:: python

    # List the generator's steps
    generator
        
    Generator with 11 steps:
    0: Function 'read_stellar_catalog' with 5 keyword arguments.
    1: Function 'create_planets_bergsten' with 7 keyword arguments.
    2: Function 'assign_orbital_elements' with 1 keyword arguments.
    3: Function 'impact_parameter' with 1 keyword arguments.
    4: Function 'assign_mass' with no keyword arguments.
    5: Function 'compute_habitable_zone_boundaries' with no keyword arguments.
    6: Function 'classify_planets' with no keyword arguments.
    7: Function 'geometric_albedo' with 2 keyword arguments.
    8: Function 'effective_values' with no keyword arguments.
    9: Function 'Example1_water' with 3 keyword arguments.
    10: Function 'Example2_oxygen' with 2 keyword arguments.

Each of these functions is documented under the :mod:`~bioverse.functions` module.

Passing keyword arguments
*************************

Many of the functions in the Generator accept keyword arguments that affect the properties of the simulated sample. For example, the :func:`~bioverse.functions.create_planets_bergsten` function scales the planet occurrence rates uniformly in response to its keyword argument ``f_eta``. To change the value of ``f_eta``, simply pass it to :func:`~bioverse.generator.Generator.generate` as follows:

.. code-block:: python
    
    sample = generator.generate(f_eta=1.5)

Note that this value will be passed to any function in the generator for which ``f_eta`` is an argument. This can be useful for sharing arguments across multiple functions, but be careful not to accidentally use the same keywords for two different functions.

Transit mode
************

One of Bioverse's main functions is to evaluate the sample size of a transiting exoplanet survey. However, most planets do not transit their stars, so simulating their properties would be inefficient. The argument ``transit_mode`` can be used to address this:

.. code-block:: python

    sample = generator.generate(transit_mode=True)

If ``True``, then only planets that transit their stars are simulated.

.. _adding-steps:

Adding new functions
********************

You can extend a generator by writing your own functions to simulate new planetary properties. Each function must accept a :class:`~bioverse.classes.Table` as its first and only required argument, can accept any number of keyword arguments, and must return a Table as its only return value.

For example, the following function will assign a random ocean covering fraction to Earth-sized planets in the habitable zone (exo-Earth candidates or "EECs"), while non-EECs will have no oceans.

.. code-block:: python

    def make_oceans(table, f_ocean_min=0.05, f_ocean_max=0.8):
        # f_ocean=0 for all planets
        table['f_ocean'] = np.zeros(len(table))

        # f_ocean_min < f_ocean < f_ocean_max for EECs
        EECs = table['EEC']
        table['f_ocean'][EECs] = np.random.uniform(f_ocean_min, f_ocean_max, EECs.sum())

        return table

Save this function in ``custom.py`` and insert it into the Generator as follows:

.. code-block:: python
    
    generator.insert_step('make_oceans')

You can then simulate a sample of planets with oceans for arbitrary values of ``f_ocean_min`` and ``f_ocean_max``:

.. code-block:: python

    sample = generator.generate(f_ocean_min=0.3, f_ocean_max=0.7)

You might also want to replace an existing step in the Generator with your own alternative. For example, suppose we want to replace the function that assigns planet masses (step 4: :func:`~bioverse.functions.assign_mass`) with one that implements the mass-radius relationship of `Weiss & Marcy (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...783L...6W/abstract>`_. First, define a function :func:`Weiss_Marcy_2014` in ``custom.py`` that implements this relationship using the format above. Next, we can replace step 4 with the new function:

.. code-block:: python

    # Remove step 4 and replace it with the new mass-radius relationship
    del generator.steps[4]
    generator.insert_step('Weiss_Marcy_2014', 4)
    
Note that the function :func:`Weiss_Marcy_2014` should also compute the density and surface gravity of each planet as :func:`~bioverse.functions.assign_mass` currently does.

Saving and loading
******************

You can save the modified version of a Generator under a new name:

.. code-block:: python
    
    generator.save('imaging_with_oceans')

and load it as follows:

.. code-block:: python

    generator = Generator('imaging_with_oceans')
