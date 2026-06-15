#################################
Simulating survey datasets
#################################

The :class:`~bioverse.survey.Survey` class
*******************************************

The output of :meth:`~bioverse.generator.Generator.generate` is a :class:`~bioverse.classes.Table` containing the values of several parameters for planets within the bounds of the simulation. However, only a subset of these will be detectable by a transit or direct imaging survey. For those planets, only a subset of their properties can be directly probed, and only with a finite level of precision. Module 2 captures these details by simulating the observing limits and measurement precision of a direct imaging or transit spectroscopy survey of the planet population.

The survey simulation module is implemented by the :class:`~bioverse.survey.Survey` class [#f1]_ and its children classes :class:`~bioverse.survey.ImagingSurvey` and :class:`~bioverse.survey.TransitSurvey`. The Survey describes several key components of an exoplanet survey including:

- ``diameter``: the diameter of the telescope primary in meters (or the area-equivalent diameter for a telescope array)
- ``t_max``: maximum survey duration in days
- ``t_slew``: slew time between observations, in days
- :class:`~bioverse.survey.ImagingSurvey`
    - ``inner_working_angle`` and ``outer_working_angle``: IWA/OWA of the coronagraphic imager
    - ``contrast_limit``: log-contrast limit (i.e. faintest detectable planet)

- :class:`~bioverse.survey.TransitSurvey`
    - ``N_obs_max``: maximum allowable number of transit observations per target
    - ``d_max``: maximum distance for planet detections, in parsecs
    - ``P_max``: maximum orbital period for planet detections, in days. Planets with longer periods transit less frequently; setting ``P_max`` reflects the practical limit imposed by survey duration and the minimum number of transits required for confirmation.
    - ``min_depth``: minimum transit depth for planet detections
    - ``m_G_max``: maximum Gaia G-band magnitude for host star detections

Each type of survey "ships" with a default configuration:

.. code-block:: python

    from bioverse.survey import ImagingSurvey, TransitSurvey
    survey_imaging = ImagingSurvey('default')
    survey_transit = TransitSurvey('default')

The default imaging survey is modeled after `LUVOIR-A <https://arxiv.org/abs/1912.06219>`_, with a coronagraphic imager and 15-meter primary aperture. The default transit survey is modeled after the Nautilus Space Observatory, with a 50-meter equivalent light-collecting area.

Which planets are detectable?
*****************************

Given a simulated set of planets to observe, the Survey first determines which of these are detectable. For a :class:`~bioverse.survey.TransitSurvey`, this set consists of all transiting planets satisfying the detection filters above, while for an :class:`~bioverse.survey.ImagingSurvey`, it consists of all planets within the coronagraphic IWA/OWA and brighter than the limiting contrast. This can be invoked as follows:

.. code-block:: python

    detected = survey_imaging.compute_yield(sample)

Conducting measurements
************************************

The Survey will conduct a series of measurements on the detectable planet sample, each defined by a :class:`~bioverse.survey.Measurement` object. A Measurement's parameters include:

- ``key``: the name of the planet property to be measured
- ``survey``: the Survey object associated with this Measurement
- ``precision``: the relative or absolute precision of the measurement (e.g. ``'10%'`` or ``0.1``)

To add a measurement to a survey:

.. code-block:: python

    survey.add_measurement('R', precision='5%')

To add multiple measurements at once:

.. code-block:: python

    survey.add_measurements(R='5%', M_st='5%', age='30%')

To conduct all measurements and produce a dataset:

.. code-block:: python

    data = survey_imaging.observe(detected)

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

:meth:`~bioverse.survey.Survey.quickrun` will pass any keyword arguments to the :meth:`~bioverse.generator.Generator.generate` method. For a :class:`~bioverse.survey.TransitSurvey`, ``transit_mode=True`` is automatically applied to the generator via ``set_arg()`` before generating.

.. _reference-case:

Exposure time calculations
**************************

Spectroscopic observations of exoplanets are time-consuming, and for some surveys the amount of time required to conduct them will be a limiting factor on sample size. To accommodate this, Bioverse calculates the exposure time :math:`t_i` required to conduct the spectroscopic measurement for each planet and schedules observations within the survey lifetime. Planets that cannot be observed within the total survey time will be excluded from the yield.

To enable exposure time calculations, pass ``method='scaling_relation'`` to :meth:`~bioverse.survey.Survey.compute_yield` and supply a dictionary of reference parameters via :meth:`~bioverse.survey.Survey.set_reference_observation`. The convenience function :func:`~bioverse.survey.read_scaling_dict` provides pre-configured reference dictionaries for the standard survey modes:

.. code-block:: python

    from bioverse.survey import TransitSurvey, read_scaling_dict

    survey = TransitSurvey('default')
    survey.set_reference_observation(**read_scaling_dict('transit_H2O'))
    survey.add_measurement('has_H2O')

    detected = survey.compute_yield(sample, method='scaling_relation')
    data = survey.observe(detected)

Available labels for :func:`~bioverse.survey.read_scaling_dict` are ``'imaging_H2O'``, ``'imaging_O2'``, ``'transit_H2O'``, and ``'transit_O2'``.

Bioverse uses the reference parameters to determine the exposure time :math:`t_i` required for each individual planet with the following equation:

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

The determination of ``t_ref`` often relies on radiative transfer and instrument noise estimates. One method of calculating ``t_ref`` for the transit survey using the Planetary Spectrum Generator is demonstrated in :doc:`tutorial_tref`.

.. _target-prioritization:

Target prioritization
*********************

When using ``method='scaling_relation'``, targets are prioritized by the ratio of scientific weight to required exposure time:

    :math:`p_i = w_i / t_i`

Bioverse observes targets in order of decreasing :math:`p_i` until the survey lifetime is exhausted.

By default, ``w_i = 1`` for all targets, but it can be raised or lowered using :meth:`~bioverse.survey.Survey.set_weight`. For example, to assign ``w_i = 5`` for targets with radii between 1–2 :math:`R_\oplus`:

.. code-block:: python

    survey.set_weight('R', weight=5, min=1, max=2)

To exclude a set of targets entirely, set ``w_i = 0``. For example, to restrict observations to exo-Earth candidates only:

.. code-block:: python

    survey.set_weight('EEC', weight=0, value=False)

The convenience function :func:`~bioverse.survey.prioritize_survey` applies the standard target weights used in the Bioverse example analyses:

.. code-block:: python

    from bioverse.survey import prioritize_survey
    prioritize_survey(survey, 'transit_H2O')

Available labels are ``'imaging_H2O'``, ``'imaging_O2'``, ``'transit_H2O'``, and ``'transit_O2'``.

In transit mode, the ``debias`` option in :meth:`~bioverse.survey.TransitSurvey.compute_yield` applies a weighting of :math:`a/R_*` to correct the detection bias toward shorter period planets:

.. code-block:: python

    detected = survey.compute_yield(sample, method='scaling_relation', debias=True)


.. rubric:: Footnotes

.. [#f1] :class:`~bioverse.survey.Survey` should never be called directly; instead :class:`~bioverse.survey.ImagingSurvey` or :class:`~bioverse.survey.TransitSurvey` should be used.
