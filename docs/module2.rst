#################################
Simulating survey datasets
#################################

The :class:`~bioverse.survey.Survey` class
*******************************************

The output of :meth:`~bioverse.generator.Generator.generate` is a :class:`~bioverse.classes.Table` containing the values of several parameters for planets within the bounds of the simulation. However, only a subset of these will be detectable by a transit or direct imaging survey. For those planets, only a subset of their properties can be directly probed, and only with a finite level of precision. Module 2 captures these details by simulating the observing limits and measurement precision of a direct imaging or transit spectroscopy survey of the planet population.

The survey simulation module is implemented by the :class:`~bioverse.survey.Survey` class [#f1]_ and its children classes :class:`~bioverse.survey.ImagingSurvey` and :class:`~bioverse.survey.TransitSurvey`. The Survey describes several key components of an exoplanet survey including:

- ``diameter``: the diameter of the telescope primary in meters (or the area-equivalent diameter for a telescope array)
- ``t_slew``: slew time between observations, in days
- :class:`~bioverse.survey.ImagingSurvey`
    - ``inner_working_angle`` and ``outer_working_angle``: IWA/OWA of the coronagraphic imager
    - ``contrast_limit``: log-contrast limit (i.e. faintest detectable planet)

- :class:`~bioverse.survey.TransitSurvey`
    - ``N_obs_max``: maximum allowable number of transit observations per target
    - ``t_max``: maximum amount of time across which to combine transit observations, in days
    
- ``T_st_ref``, ``R_st_ref``, and ``d_ref``: temperature (Kelvin), radius (:math:`R_\odot`), and distance (parsec) of the reference star (see :ref:`reference-case`)
- ``D_ref``: diameter of the reference telescope, in meters

Each type of survey "ships" with a default configuration:

.. code-block:: python

    from bioverse.survey import ImagingSurvey, TransitSurvey
    survey_imaging = ImagingSurvey('default')
    survey_transit = TransitSurvey('default')

The default imaging survey is modeled after `LUVOIR-A <https://arxiv.org/abs/1912.06219>`_, with a coronagraphic imager and 15-meter primary aperture. The default transit survey is modeled after the Nautilus Space Observatory, with a 50-meter equivalent light-collecting area.

Which planets are detectable?
*****************************

Given a simulated set of planets to observe, the Survey first determines which of these are detectable. For a :class:`~bioverse.survey.TransitSurvey`, this set consists of all transiting planets, while for an :class:`~bioverse.survey.ImagingSurvey`, it consists of all planets within the coronagraphic IWA/OWA and brighter than the limiting contrast. This can be invoked as follows


.. code-block:: python

    detected = s_imaging.compute_yield(sample)

Conducting measurements
************************************

The Survey will conduct a series of measurements on the detectable planet sample, each defined by a :class:`~bioverse.survey.Measurement` object. A Measurement's parameters include:

- ``key``: the name of the planet property to be measured
- ``precision``: the relative or absolute precision of the measurement (e.g. 10% or 0.1 AU)
- ``t_ref``: the amount of time in days required to conduct this measurement for a typical target (see below)
- ``t_total``: the amount of survey time in days allocated toward this measurement
- ``wl_eff``: the effective wavelength of observation in microns
- ``priority``: a set of rules describing how targets are prioritized (described below)

To conduct these measurements and produce a dataset:

.. code-block:: python

    data = s_imaging.observe(detected)

Quick-run
*********

In total, to produce a simulated sample of planets, determine which planets are detectable, and produce a mock dataset requires the following:

.. code-block:: python

    from bioverse.generator import Generator
    from bioverse.survey import ImagingSurvey

    generator = Generator('imaging')
    survey = ImagingSurvey('default')

    sample = generator.generate(eta_Earth=0.15)
    detected = survey.compute_yield(sample)
    data = survey.observe(detected)

The last three lines can be combined into the following:

.. code-block:: python

    sample, detected, data = survey.quickrun(generator, eta_Earth=0.15)

:meth:`~bioverse.survey.Survey.quickrun` will pass any keyword arguments to the :meth:`~bioverse.generator.Generator.generate` method, and will by default pass ``transit_mode=True`` for a :class:`~bioverse.survey.TransitSurvey`.

.. _reference-case:

Exposure time calculations
**************************

Spectroscopic observations of exoplanets are time-consuming, and for some surveys the amount of time required to conduct them will be a limiting factor on sample size. To accomodate this, Bioverse calculates the exposure time :math:`t_i` required to conduct the spectroscopic measurement for each planet, then prioritizes each planet according to :math:`t_i` as well as its weight parameter (see :ref:`target-prioritization`). In the simulated dataset, planets that could not be observed within the total allotted time ``t_total`` will have ``nan`` values for the measured value.

A Measurement's "reference time", ``t_ref``, is the exposure time required to perform the measurement for an Earth-like planet (receiving the same flux as Earth) orbiting a typical star (whose properties are defined by the Survey parameters ``T_st_ref``, ``R_st_ref``, and ``d_ref``), with a telescope of diameter ``D_ref``. For the default imaging survey, the typical target orbits a Sun-like star at a distance of 10 pc, while for the transit survey, the host star is a mid-M dwarf.

Bioverse uses ``t_ref``, along the wavelength of observation ``wl_eff``, to determine the exposure time ``t_i`` required for each individual planet with the following equation:

    
.. math::

    \frac{t_i}{t_\text{ref}} = f_i
    \left(\frac{d_i}{d_\text{ref}}\right)^2
    \left(\frac{R_*}{R_{*, \text{ref}}}\right)^{-2}
    \left(\frac{B(\lambda_\text{eff},T_{*,i})}{B(\lambda_\text{eff},T_{*, \text{ref}})}\right)^{-1}
    \left(\frac{D}{D_\text{ref}}\right)^{-2}

:math:`f_i` encompasses the different factors affecting spectroscopic signal strength in imaging and transit mode:

.. math::

    f_i^\text{imaging} &= \left(\frac{\zeta_i}{\zeta_\oplus}\right)^{-1}

    f_i^\text{transit} &= 
    \left(\frac{h_{i}}{h_\oplus}\right)^{-2}
    \left(\frac{R_{p,i}}{R_\oplus}\right)^{-2}
    \left(\frac{R_{*,i}}{R_{*, \text{ref}}}\right)^4

Importantly, this calculation is conducted for each Measurement with a different value of ``t_ref``. **Therefore, the same planet may have real values for one Measurement and ``nan`` for another.** This is particularly relevant for the transit survey, where the total number of transiting planets for which e.g. planet size and orbital period can be measured is much larger than the number that can be spectroscopically characterized. To return just the subset of detected planets that were observed for a given Measurement, use the :meth:`~bioverse.classes.Table.observed` method:

.. code-block:: python

    observed = data.observed('has_O2')

The determination of ``t_ref`` often relies on radiative transfer and instrument noise estimates that are generally not done in Bioverse. It can be accomplished by citing relevant studies in the literature or using third-party tools such as the `Planetary Spectrum Generator <https://psg.gsfc.nasa.gov/>`_. One method of calculating ``t_ref`` for the transit survey is demonstrated in :doc:`tutorial_tref`.

Bioverse can calculate ``t_ref`` given two simulated spectra files - one with and one without the targeted absorption feature - both of which contain measurements for wavelength , flux, and flux uncertainty as the first three columns. You must also specify the simulated exposure time and the minimum and maximum wavelengths for the absorption feature. The :func:`~bioverse.util.compute_t_ref` function will then determine the exposure time required for a 5-sigma detection (in the same units as the input exposure time).

.. code-block:: python

    from bioverse.util import compute_t_ref

    # Scales from simulated spectra for a combined 100 hr exposure time, targeting the O3 feature near 0.6 microns.
    t_ref = compute_t_ref(filenames=('spectrum_O3.dat', 'spectrum_noO3.dat'), t_exp=100, wl_min=0.4, wl_max=0.8)
    print("Required exposure time: {:.1f} hr".format(t_ref))

Output: ``Required exposure time: 73.9 hr``

Finally, change the ``t_ref`` and ``wl_eff`` attributes of the associated Measurement object, using units of days and microns respectively:

.. code-block:: python

    survey = TransitSurvey('default')
    survey.measurements['has_O2'].t_ref = 73.9/24
    survey.measurements['has_O2'].wl_eff = 0.6

.. _target-prioritization:

Target prioritization
*********************

For measurements where ``t_total`` is finite and ``t_ref`` is non-zero, targets must be prioritized in case there is insufficient time to characterize all of them. In Bioverse, target prioritization depends both on the target's scientific interest (quantified by the weight parameter ``w_i``) and the amount of time ``t_i`` required to properly characterize it. Each target's priority is calculated as follows:

    :math:`p_i = w_i/t_i`

Bioverse will observe targets in order of decreasing ``p_i`` until ``t_total`` has been exhausted. The resulting dataset will fill in ``nan`` values for any targets that were not observed.

By default, ``w_i = 1`` for all targets, but it can be raised or lowered for planets that meet certain criteria. For example, to assign ``w_i = 5`` for targets with radii between 1-2 :math:`R_\oplus`:

.. code-block:: python

    m = survey.measurement['has_O2']
    m.set_weight('R', weight=5, min=1, max=2)

To exclude a set of targets, set ``w_i = 0``. For example, to restrict a measurement to exo-Earth candidates only:

.. code-block:: python

    m.set_weight('EEC', weight=0, value=False)

In transit mode, targets are weighted by :math:`a/R_*` to correct the detection bias toward shorter period planets. To disable this feature:

.. code-block:: python

    m.debias = False

.. rubric:: Footnotes

.. [#f1] :class:`~bioverse.survey.Survey` should never be called directly; instead :class:`~bioverse.survey.ImagingSurvey` or :class:`~bioverse.survey.TransitSurvey` should be used.