#################################
Simulating survey datasets
#################################

The :class:`~bioverse.survey.Survey` class
******************************************

The output of :meth:`~bioverse.generator.Generator.generate` is a :class:`~bioverse.classes.Table` containing the values of several parameters for planets within the bounds of the simulation. The survey module determines which planets are *detected* by a mission and what *measurements* are recorded, including exposure time limits and target prioritization.

The survey simulation layer is implemented by the :class:`~bioverse.survey.Survey` class [#f1]_ and its subclasses :class:`~bioverse.survey.ImagingSurvey` and :class:`~bioverse.survey.TransitSurvey`. Each class is a Python ``dataclass``; constructor arguments (``diameter``, ``t_max``, coronagraph limits for imaging, ``N_obs_max`` for transits, and so on) can be passed as keyword arguments or collected in a dictionary and unpacked:

.. code-block:: python

    survey_kwargs = {"diameter": 15.0, "t_max": 10 * 365.25}
    survey_imaging = ImagingSurvey(label=None, **survey_kwargs)

See :doc:`apidoc/bioverse.survey` for field definitions and methods.

Each type of survey ships with a default configuration:

.. code-block:: python

    from bioverse.survey import ImagingSurvey, TransitSurvey
    survey_imaging = ImagingSurvey('default')
    survey_transit = TransitSurvey('default')

The default imaging survey is modeled after `LUVOIR-A <https://arxiv.org/abs/1912.06219>`_, with a coronagraphic imager and 15-meter primary aperture. The default transit survey is modeled after the Nautilus Space Observatory, with a 50-meter equivalent light-collecting area.

Adding measurements to a survey
*******************************

Measurements are stored on the survey as :class:`~bioverse.survey.Measurement` objects. Add them with:

- :meth:`~bioverse.survey.Survey.add_measurement` — one measurement at a time; optional ``idx`` controls ordering.
- :meth:`~bioverse.survey.Survey.add_measurements` — convenience wrapper that takes measurement names and a single precision value each (see that method’s docstring).

.. code-block:: python

    survey = ImagingSurvey('default')
    survey.add_measurement('my_param', precision='10%')

``add_measurement`` forwards additional keyword arguments to :class:`~bioverse.survey.Measurement` (for example ``precision``).

Which planets are detected? ``compute_yield``
**********************************************

Given a simulated planet sample, the survey determines which planets count toward the *yield* using :meth:`~bioverse.survey.ImagingSurvey.compute_yield` or :meth:`~bioverse.survey.TransitSurvey.compute_yield`. The ``method`` keyword selects the algorithm:

``method='detectable'`` (default)
    Returns planets that are *in principle* detectable (e.g. transiting, or within IWA/OWA and brighter than the contrast floor for imaging) **without** applying exposure times, per-target budgets, or mission duration limits.

``method='scaling_relation'``
    Uses a reference observation and a scaling law to assign exposure times and overhead, then schedules targets at the **survey** level until the mission time budget is exhausted. You must define the reference observation with :meth:`~bioverse.survey.Survey.set_reference_observation` so that all required reference keys are present (see :ref:`scaling-relation-yield`).

``method='exp_time'`` (direct imaging only)
    Intended for *blind* direct-imaging surveys where targets are not known in advance. Requires a generator step that runs :func:`~bioverse.functions.schedule_DI_survey` so host stars carry an allocated time ``t_req``; exposure times for planets use the direct-imaging calculator (see :ref:`di-blind-yield`). In code, only ``exp_time`` is accepted today; a ``blind_survey`` alias may be added later.

Typical usage:

.. code-block:: python

    detected = survey_imaging.compute_yield(sample, method='detectable')

:meth:`~bioverse.survey.Survey.quickrun` forwards ``method`` to ``compute_yield`` (default ``'detectable'``).

.. _scaling-relation-yield:

Scaling relation method (``method='scaling_relation'``)
*******************************************************

For ``method='scaling_relation'``, call :meth:`~bioverse.survey.Survey.set_reference_observation` with (at minimum) the keys expected by :meth:`~bioverse.survey.Survey.exp_time_scaling_relation`: ``t_ref``, ``wl_eff``, ``T_st_ref``, ``R_st_ref``, ``D_ref``, ``d_ref``, ``R_pl_ref``, and ``H_ref``. These describe a *reference* exposure time and observing configuration; per-planet exposure times are scaled from that reference.

For spectroscopic observations, exposure time :math:`t_i` for target :math:`i` scales with the reference time :math:`t_\mathrm{ref}` as:

.. math::

    \frac{t_i}{t_\text{ref}} = f_i
    \left(\frac{d_i}{d_\text{ref}}\right)^2
    \left(\frac{R_*}{R_{*, \text{ref}}}\right)^{-2}
    \left(\frac{B(\lambda_\text{eff},T_{*,i})}{B(\lambda_\text{eff},T_{*, \text{ref}})}\right)^{-1}
    \left(\frac{D}{D_\text{ref}}\right)^{-2}

:math:`f_i` collects factors that depend on observing mode:

.. math::

    f_i^\text{imaging} &= \left(\frac{\zeta_i}{\zeta_\oplus}\right)^{-1}

    f_i^\text{transit} &=
    \left(\frac{h_{i}}{h_\oplus}\right)^{-2}
    \left(\frac{R_{p,i}}{R_\oplus}\right)^{-2}
    \left(\frac{R_{*,i}}{R_{*, \text{ref}}}\right)^4

After exposure times are computed, :meth:`~bioverse.survey.Survey.schedule_observations` applies **survey-level** target prioritization (see :ref:`target-prioritization-survey`) and returns the subset of planets that fit within ``t_max``.

If you derive a reference exposure time from external tools (e.g. the Planetary Spectrum Generator), you can plug the resulting numeric values into ``set_reference_observation``. A worked PSG-focused walkthrough is still available in :doc:`tutorial_tref`, but configuring the **survey** (rather than individual ``Measurement`` objects) is now the supported path for yield calculations that use scaling relations.

.. _di-blind-yield:

Direct imaging ``exp_time`` (blind survey) method
**************************************************

For ``method='exp_time'`` on an :class:`~bioverse.survey.ImagingSurvey`, the code compares per-planet exposure times from :func:`~bioverse.util.DI_exposure_time_calculator` — based on `Stark et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJ...795..122S/abstract>`_ — to the time allocated to each host star. **You must run** :func:`~bioverse.functions.schedule_DI_survey` **as a generator step** before planets are generated so that the star table includes ``t_req`` (time allocated per star). After planet properties are simulated, ``compute_yield`` marks planets as detected when the required exposure is less than the host’s allocated time.

See ``docs/examples/debug_changes_for_DI.py`` for a side-by-side comparison of ``method='detectable'`` and ``method='scaling_relation'`` on the same sample, with comments describing where ``method='exp_time'`` fits in.

Conducting measurements
************************

After yields are known, :meth:`~bioverse.survey.Survey.observe` simulates the measurement sequence defined on the survey and returns a data table (with ``data.error`` for uncertainties). Prefer inspecting the columns you need directly; :meth:`~bioverse.classes.Table.observed` is a legacy helper and may not reflect how yields and measurements interact in the current workflow.

.. code-block:: python

    data = survey_imaging.observe(detected)

Quick-run
*********

To generate a sample, compute yields, and produce mock data in one call:

.. code-block:: python

    from bioverse.generator import Generator
    from bioverse.survey import ImagingSurvey

    generator = Generator('imaging')
    survey = ImagingSurvey('default')

    sample = generator.generate(eta_Earth=0.15)
    detected = survey.compute_yield(sample)
    data = survey.observe(detected)

or equivalently:

.. code-block:: python

    sample, detected, data = survey.quickrun(generator, eta_Earth=0.15)

:meth:`~bioverse.survey.Survey.quickrun` passes extra keyword arguments to :meth:`~bioverse.generator.Generator.generate` and, for :class:`~bioverse.survey.TransitSurvey`, defaults to ``transit_mode=True`` unless you override it.

.. _target-prioritization-survey:

Target prioritization and scheduling
**************************************

Target weights and time allocation are handled when yields use ``method='scaling_relation'`` (and within :meth:`~bioverse.survey.Survey.schedule_observations`). Configure weights on the **survey** with :meth:`~bioverse.survey.Survey.set_weight`, not on individual measurements. For example, to favor planets with radii between 1 and 2 :math:`R_\oplus`:

.. code-block:: python

    survey.set_weight('R', weight=5, min=1, max=2)

To exclude a class of targets, set ``weight=0`` for the corresponding rule. For transit surveys, debiasing (weighting by :math:`a/R_*`) can be toggled via the ``debias`` argument to :meth:`~bioverse.survey.TransitSurvey.compute_yield`.

.. rubric:: Footnotes

.. [#f1] :class:`~bioverse.survey.Survey` should never be called directly; instead :class:`~bioverse.survey.ImagingSurvey` or :class:`~bioverse.survey.TransitSurvey` should be used.
