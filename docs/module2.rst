#################################
Simulated survey datasets
#################################

The :class:`~bioverse.survey.Survey` class
*******************************************

The output of :meth:`~bioverse.generator.Generator.generate` is a :class:`~bioverse.classes.Table` containing the values of several parameters for planets within the bounds of the simulation. However, only a subset of these will be detectable by a transit or direct imaging survey. For those planets, only a subset of their properties can be directly probed, and only with a finite level of precision. Module 2 captures these details by simulating the observing limits and measurement precision of a direct imaging or transit spectroscopy survey of the planet population.

The survey simulation module is implemented by the :class:`~bioverse.survey.Survey` class and its children classes :class:`~bioverse.survey.ImagingSurvey` and :class:`~bioverse.survey.TransitSurvey`. The Survey describes several key components of an exoplanet survey including:

- the type of survey (either 'imaging' or 'transit')
- the diameter of the telescope primary (or effective diameter for a telescope array)
- :class:`~bioverse.survey.ImagingSurvey`
    - IWA/OWA of the coronagraphic imager
    - contrast limit (i.e. faintest detectable planet)
    - slew time between observations, in days

- :class:`~bioverse.survey.TransitSurvey`
    - maximum allowable number of transit observations per target
    
- properties of the reference star for exposure time scaling (described below)

Each type of survey "ships" with a default configuration:

.. code-block:: python

    from bioverse.survey import ImagingSurvey, TransitSurvey
    survey_imaging = ImagingSurvey('default')
    survey_transit = TransitSurvey('default')

The default imaging survey is modeled after LUVOIR-A, with a coronagraphic imager and 15-meter primary aperture. The default transit survey is modeled after the Nautilus Space Observatory, with a 50-meter equivalent light-collecting area.

Which planets are detectable?
*****************************

Given a simulated set of planets to observe, the Survey first determines which of these are detectable. For a :class:`~bioverse.survey.TransitSurvey`, this set consists of all transiting planets, while for an :class:`~bioverse.survey.ImagingSurvey`, it consists of all planets within the coronagraphic IWA/OWA and brighter than the limiting contrast. This can be invoked as follows


.. code-block:: python

    detected = s_imaging.compute_yield(sample)

Conducting measurements
************************************

The Survey will conduct a series of measurements on the detectable planet sample, each defined by a :class:`~bioverse.survey.Measurement` object. A Measurement describes:

- the parameter to be measured (example: 'a' to measure semi-major axis)
- the relative or absolute precision with which the parameter is measured (e.g. 10% or 0.1 AU)
- the conditions defining the subset of targets for which to apply this measurement (e.g. 'd < 20' for targets within 20 parsecs)
- the amount of survey time allocated toward this measurement (in days, or infinite to apply to all valid targets)
- the amount of time required to conduct this measurement for the reference target (below; only if t_total is finite)

To conduct these measurements and produce a dataset is simple:

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

Reference case
**************

Some planetary properties are either trivial to measure (i.e. host star effective temperature) or their measurement occurs concurrently with their detection - for example, planet-star contrast (in imaging mode) or planet radius (in transit mode). Other properties - especially the detection of atmospheric species - require time-intensive spectroscopic observations spanning several hours or days of integration time. This is especially relevant for a transiting exoplanet survey as the amount of SNR built up per transit observation is limited by the transit duration, and the number of transits observable within a reasonable survey lifetime depends on the orbital period.

As an example, consider the amount of time required to detect H2O in a transiting exoplanet's atmosphere. One way to estimate this would be to simulate the planet's observed spectrum (with uncertainties), measure the amplitude of H2O absorption features, and compute the amount of time required to achieve a 5-sigma detection of that amplitude. However, to repeat this for every planet would be computationally intensive, and would prohibit the use Bioverse to simulate thousands of realizations of the same survey.

A much faster method involves estimating the amount of time required to characterize a single planet whose properties are broadly representative of the "typical" survey target, then scaling that exposure time to each planet based on the major factors affecting signal strength.

Target prioritization
*********************

It is not always feasible to characterize all targets within a finite survey duration (e.g., 10 years). Therefore, targets must be prioritized. In Bioverse, target prioritization depends both on the target's scientific interest (or weight w_i) and the amount of time required to properly characterize it (t_i as computed above)
