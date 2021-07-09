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

Reference case
**************

A Measurement's "reference time", ``t_ref``, is the exposure time required to perform the measurement for an Earth-like planet orbiting a typical star (whose properties are defined under the Survey by ``T_st_ref``, ``R_st_ref``, and ``d_ref``), with a telescope of diameter ``D_ref``. Bioverse uses ``t_ref``, along the wavelength of observation ``wl_eff``, to determine the exposure time ``t_i`` required for each individual planet with the following equation:

    
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

The determination of ``t_ref`` is generally not done in Bioverse. It can be accomplished by citing relevant studies in the literature or using third-party tools such as the `Planetary Spectrum Generator <https://psg.gsfc.nasa.gov/>`_.

To change ``t_ref`` and ``wl_eff`` for a specific Measurement:

.. code-block:: python

    survey = ImagingSurvey('default')
    survey.measurements['has_H2O'].t_ref = 0.04
    survey.measurements['has_H2O'].wl_eff = 1.4 


.. Some planetary properties are either trivial to measure (i.e. host star effective temperature) or their measurement occurs concurrently with their detection - for example, planet-star contrast (in imaging mode) or planet radius (in transit mode). Other properties - especially the detection of atmospheric species - require time-intensive spectroscopic observations spanning several hours or days of integration time. This is especially relevant for a transiting exoplanet survey as the amount of SNR built up per transit observation is limited by the transit duration, and the number of transits observable within a reasonable survey lifetime is limited by the orbital period.

.. As an example, consider the amount of time required to detect H2O in a transiting exoplanet's atmosphere. One way to estimate this would be to simulate the planet's observed spectrum (with uncertainties), measure the amplitude of H2O absorption features, and compute the amount of time required to achieve a 5-sigma detection of that amplitude. However, to repeat this for every planet would be computationally intensive, and would prohibit the use Bioverse to simulate thousands of realizations of the same survey.

.. A much faster method involves estimating the amount of time required to characterize a single planet whose properties are broadly representative of the "typical" survey target, then scaling that exposure time to each planet based on the major factors affecting signal strength. The determination of this "reference time" ``t_ref`` is generally not done in Bioverse. It can be accomplished by citing relevant studies in the literature or using third-party tools such as the `Planetary Spectrum Generator <https://psg.gsfc.nasa.gov/>`_.

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