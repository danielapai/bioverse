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

The default imaging survey is modeled after `LUVOIR-A <https://arxiv.org/abs/1912.06219>`_, with a coronagraphic imager and 15-meter primary aperture (Updated HWO configurations are also available). The default transit survey is modeled after the Nautilus Space Observatory - a satellite constellation with a 50-meter equivalent light-collecting area.

Which planets are observable?
*****************************

Given a simulated set of planets to observe, the Survey computes which of these are detectable using the ``compute_yield()`` function

.. code-block:: python

    detected = survey_imaging.compute_yield(sample, method='detectable')


There are many different methods for computing the yield of exoplanets detected by a survey which can be specified with the ``method`` keyword argument. These differ depending on the survey type and availability of planet properties in the Table object. The currently implemented methods of computing the yield are as follows:

- ``detectable``: All planets which meet threshold criteria for detectability are observed. For a :class:`~bioverse.survey.TransitSurvey`, this consists of all transiting planets satisfying the arguments used to define the survey such as the minimum transit depth, maximum magnitude, period etc. For a :class:`~bioverse.survey.ImagingSurvey`, planets are selected if they are within the Inner and Outer working angles of the coronagraph and are brighter than a limiting contrast value. This method does not consider the exposure or integration time required to observe each target or the total mission lifetime. As such, this method provides a set of all the targets that could be potentially observable, which is likely a significantly larger sample than the number of planets a mission would actually be able to observe.

- ``blind_survey`` or ``exp_time`` (Currently only for Imaging survey): Simulate a survey where planetary targets are not known a priori, and will be discovered over the course of a survey. This is the case for direct imaging missions such as HWO, which will be the first mission capable of imaging exoEarths in the habitable zone, so the planets it wants to observe have not been detected yet. This method works by allocating a set exposure time to each star based on the time it would take to characterize a hypothetical Earth-sized planet in the habitable zone. This allocation needs to be performed early in the generator or pre_generator steps using the ``schedule_DI_survey`` function. After planet generation, the exposure time required to characterize each planet is calculated. If the time allocated to observing the star exceeds the required exposure time for planet characterization, it is considered to be observed.  Exposure times are calculated using a direct imaging exposure time calculator based on that of `Stark et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...795..122S/abstract>`_.

- ``scaling_relation``: Use the scaling relation of `Bixel and Apai (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....161..228B/abstract>`_ to calculate exposure times based on a reference calculation. These scaling relations are described in more detail in the following section. Typical reference calculations use NASA's Planetary Spectrum Generator or equivalent radiative transfer codes to calculate the exposure time required to recover a given spectral feature.  For each planet exposure times are calculated using this scaling relation, and planets are selected for survey based on their assigned priority (see section on Target Prioritization) until the total survey duration is reached.


Scaling Relations
----------------------------

In yield calculations with ``method='scaling_relation'`` exposure times are approximated using a scaling relation in planet and stellar properties. These exposure times are scaled relative to a reference value, :math:`t_{\text{ref}}`, for the exposure time necessary to recover the features of interest in the spectrum a specified planet type (Such as oxygen in the atmosphere of an exoEarth). The determination of ``t_ref`` often relies on radiative transfer and instrument noise estimates. One method of calculating ``t_ref`` for the transit survey using the Planetary Spectrum Generator is demonstrated in :doc:`tutorial_tref`.

To set the reference observation for a survey, supply a dictionary of reference parameters via :meth:`~bioverse.survey.Survey.set_reference_observation`. The reference dictionary is typically in the form:

.. code-block:: python

    scaling_dict = {
            't_ref': 0.1, #exposure time required (days)
            'wl_eff': 0.7, #effective wavelength in um
            'T_st_ref': 5788., #Stellar temperature (K)
            'R_st_ref': 1.0, #stellar radius (R_sun)
            'D_ref': 15.0, # Telescope diameter (m)
            'd_ref': 10.0, #distance to star (pc)
            'R_pl_ref': 1.0, #planet radius (R_earth)
            'H_ref': 9} #atmospheric scale height (km)

See the description of :meth:`~bioverse.survey.Survey.exp_time_scaling_relation` for more details.

Bioverse uses these reference parameters to determine the exposure time :math:`t_i` required for each individual planet with the following equation:

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


The convenience function :func:`~bioverse.survey.read_scaling_dict` provides pre-configured reference dictionaries for the characterization of select spectral features in exoEarth atmospheres for the standard survey modes. Available labels for :func:`~bioverse.survey.read_scaling_dict` are ``'imaging_H2O'``, ``'imaging_O2'``, ``'transit_H2O'``, and ``'transit_O2'``.

The following is an example for how the scaling relation method can be used:

.. code-block:: python

    from bioverse.survey import TransitSurvey, read_scaling_dict
    from bioverse.generator import Generator

    #use default transit generator
    gen= Generator('transit')
    #add a step to calculate whether a planet has water (see example 1)
    gen.insert_step('Example1_water')

    #generate planets using this generator
    sample=gen.generate()

    #load the default transit survey
    survey = TransitSurvey('default')
    #read a dictionary of prebuilt scaling parameters for the case of detecting water
    #   in the atmosphere of a transiting exoEarth
    scale_dict=read_scaling_dict('transit_H2O')
    #set the survey's reference observation using this dictionary
    survey.set_reference_observation(**scale_dict)
    #add the measurement of water to the list of properties that will be measured
    survey.add_measurement('has_H2O')

    #determine which planets in the sample have been detected
    detected = survey.compute_yield(sample, method='scaling_relation')
    data = survey.observe(detected)

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

Or alternatively

.. code-block:: python

    measure_dict={'R':'5%','M_st':'5%','age':'30%'}
    survey.add_measurements(**measure_dict)

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


.. rubric:: Footnotes

.. [#f1] :class:`~bioverse.survey.Survey` should never be called directly; instead :class:`~bioverse.survey.ImagingSurvey` or :class:`~bioverse.survey.TransitSurvey` should be used.
