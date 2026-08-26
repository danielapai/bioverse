####################################
Generating planetary systems
####################################

The :class:`~bioverse.generator.Generator` class
************************************************

Bioverse uses the :class:`~bioverse.generator.Generator` class to generate planetary systems in the solar neighborhood. A Generator object specifies a list of functions to be performed in sequential order onto a shared :class:`~bioverse.classes.Table`. For example, a simple generator might implement this algorithm:

- Function 1: Return the `Gaia DR3 <https://www.cosmos.esa.int/web/gaia/data-release-3>`_ catalog of all stars within 30 parsecs with effective temperatures above 4000 K.
- Function 2: Simulate one or more planets around each star according to the occurrence rate estimates in `Bergsten et al. 2022 <https://ui.adsabs.harvard.edu/link_gateway/2022AJ....164..190B/doi:10.3847/1538-3881/ac8fea>`_.
- Function 3: Evaluate the mass of each planet based on its radius and the mass-radius relationship published by `Wolfgang et al. (2016) <https://ui.adsabs.harvard.edu/abs/2016ApJ...825...19W/abstract>`_.

The generator will feed the output of Function 1 into Function 2, then the output of Function 2 into Function 3, and finally will return the output of Function 3 (i.e. a table of planets with known masses, radii, orbital properties, and host star properties).

Bioverse "ships" with two Generators: one for transit mode, and the other for imaging mode. One of the main differences between these generators is the set of planetary properties that are simulated. The default transit generator includes steps that compute properties which affect the observability of transiting exoplanets, such as the transit depth, impact parameter and transit duration. The imaging generator instead focuses on computing planetary properties that influence their observability with direct imaging, including the planet-star contrast, angular separation and illumination phase. Another difference between these default generators is the stellar target list used to generate host stars. By default the transit mode generates a population of stars consistent with Gaia DR3, while the imaging mode uses a host star catalog for the LUVOIR direct imaging mission concept (see the `LUVOIR Final Report <https://arxiv.org/abs/1912.06219>`_). The following code demonstrates how to simulate a sample of planets using the imaging mode Generator:

.. code-block:: python

    from bioverse.generator import Generator
    generator = Generator('imaging')
    sample = generator.generate()

We can inspect the Generator to see which functions it implements:

.. code-block:: python

    # List the generator's steps
    generator

    Generator with 10 steps:
    0: Function 'read_stellar_catalog' with 6 keyword arguments.
    1: Function 'create_planets_SAG13' with 10 keyword arguments.
    2: Function 'assign_orbital_elements' with 4 keyword arguments.
    3: Function 'solve_kep' with 4 keyword arguments.
    4: Function 'assign_mass' with 2 keyword arguments.
    5: Function 'compute_habitable_zone_boundaries' with 1 keyword arguments.
    6: Function 'classify_planets' with no keyword arguments.
    7: Function 'geometric_albedo' with 3 keyword arguments.
    8: Function 'compute_contrast' with 2 keyword arguments.
    9: Function 'effective_values' with no keyword arguments.

Each of these functions is documented under the :mod:`~bioverse.functions` module.

The transit Generator uses a different set of steps:

.. code-block:: python

    generator_transit = Generator('transit')
    generator_transit

    Generator with 9 steps:
    0: Function 'create_stars_Gaia' with 9 keyword arguments.
    1: Function 'create_planets_SAG13' with 10 keyword arguments.
    2: Function 'assign_orbital_elements' with 4 keyword arguments.
    3: Function 'compute_transit_params' with 2 keyword arguments.
    4: Function 'assign_mass' with 2 keyword arguments.
    5: Function 'geometric_albedo' with 3 keyword arguments.
    6: Function 'compute_habitable_zone_boundaries' with 1 keyword arguments.
    7: Function 'classify_planets' with no keyword arguments.
    8: Function 'scale_height' with no keyword arguments.

Passing keyword arguments
*************************

Many of the functions in the Generator accept keyword arguments that affect the properties of the simulated sample. For example, the :func:`~bioverse.functions.create_planets_SAG13` function scales the planet occurrence rates via its keyword argument ``eta_Earth``. There are two ways to change it:

**Method 1** — pass it directly to :func:`~bioverse.generator.Generator.generate`:

.. code-block:: python

    sample = generator.generate(eta_Earth=0.15)

**Method 2** — use :meth:`~bioverse.generator.Generator.set_arg` to update the stored default across all future calls:

.. code-block:: python

    generator.set_arg('eta_Earth', 0.15)
    sample = generator.generate()

:meth:`~bioverse.generator.Generator.set_arg` sets the value for every step in the generator that accepts that keyword argument. To set multiple arguments at once, use :meth:`~bioverse.generator.Generator.set_args`:

.. code-block:: python

    generator.set_args(eta_Earth=0.15, zero_ecc=True)
    
    # Alternatively, using a dictionary
    args = {'eta_Earth': 0.15, 'zero_ecc': True}
    generator.set_args(**args)

Note that keyword arguments are matched by name across all steps, so be careful not to accidentally use the same keyword in two different functions with conflicting meanings.

Transit mode
************

One of Bioverse's main functions is to evaluate the sample size of a transiting exoplanet survey. However, most planets do not transit their stars, so simulating their properties would be inefficient. The argument ``transit_mode`` can be used to address this:

.. code-block:: python

    generator.set_arg('transit_mode', True)
    sample = generator.generate()

If ``True``, then only planets that transit their stars are simulated. Note that ``transit_mode`` is already set to ``True`` by default in the transit Generator, so this is only necessary if you are building a custom generator. When using :class:`~bioverse.survey.TransitSurvey`, ``transit_mode`` is set automatically.

.. _adding-steps:

Adding custom functions
***********************

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

Either of the following works:

    a) Save this function in ``custom.py`` and insert it into the Generator as follows:

    .. code-block:: python

        generator.insert_step('make_oceans')

    b) Insert the function itself:

    .. code-block:: python

        generator.insert_step(make_oceans)

    c) Provide function name and path to the file containing the function:

    .. code-block:: python

        generator.insert_step('make_oceans', filename='rel/or/abs/path/to/file/myfunctions.py')

With the custom function loaded, we can simulate a sample of planets with oceans for arbitrary values of ``f_ocean_min`` and ``f_ocean_max``:

.. code-block:: python

    sample = generator.generate(f_ocean_min=0.3, f_ocean_max=0.7)

You might also want to replace an existing step in the Generator with your own alternative. For example, suppose we want to replace the function that assigns planet masses (step 4: :func:`~bioverse.functions.assign_mass`) with one that implements the mass-radius relationship of `Weiss & Marcy (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...783L...6W/abstract>`_. All we need to do is to define a custom function :func:`Weiss_Marcy_2014` that implements this relationship and load it using one of the three ways described above. For example, let's add it to ``custom.py``.  Now we can replace step 4 with the new function:

.. code-block:: python

    # Remove step 4 and replace it with the new mass-radius relationship
    generator.remove_step(idx=4)
    generator.insert_step('Weiss_Marcy_2014', 4)

    #or equivalently
    generator.replace_step('Weiss_Marcy_2014', 4)
    
Note that the function :func:`Weiss_Marcy_2014` should also compute the density and surface gravity of each planet as :func:`~bioverse.functions.assign_mass` currently does.

Saving and loading
******************

You can save the modified version of a Generator under a new name:

.. code-block:: python

    generator.save('imaging_with_oceans')

and load it as follows:

.. code-block:: python

    generator = Generator('imaging_with_oceans')

Using Pre_Generator to speed up repeated runs
**********************************************

Some generator steps — such as reading a stellar catalog — are computationally expensive but produce the same output every time. If you need to run a generator repeatedly (e.g., to scan over different values of ``eta_Earth``), re-running those steps on every iteration wastes time.

The :class:`~bioverse.generator.Pre_Generator` class addresses this. It is a :class:`~bioverse.generator.Generator` subclass intended to be run *once* to produce a base :class:`~bioverse.classes.Table`, which is then passed as the starting point for subsequent generator runs. This separates the one-time, expensive steps from the steps that vary between runs.

For example, with the transit Generator the first step (``create_stars_Gaia``) is the most expensive. We can run it once with a ``Pre_Generator`` and reuse the result:

.. code-block:: python

    from bioverse.generator import Generator, Pre_Generator

    # Run the expensive stellar catalog step once
    pre_gen = Pre_Generator()
    pre_gen.insert_step('create_stars_Gaia')
    stars = pre_gen.generate()

    # Load the transit generator and start from step 1, skipping create_stars_Gaia
    generator = Generator('transit')
    for eta in [0.1, 0.15, 0.2]:
        sample = generator.generate(d=stars, idx_start=1, eta_Earth=eta)

The ``d`` parameter of :meth:`~bioverse.generator.Generator.generate` accepts a pre-existing :class:`~bioverse.classes.Table` to use as input, and ``idx_start`` controls which step the generator begins from. Together they allow you to resume generation from any intermediate result, whether that result comes from a ``Pre_Generator`` or was saved from a previous run.
