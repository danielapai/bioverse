"""
Example: :meth:`~bioverse.survey.ImagingSurvey.compute_yield` with different
``method`` values for a direct imaging (DI) survey.

- ``method='detectable'`` — all planets that satisfy contrast and separation
  limits, ignoring exposure time and mission duration.

- ``method='scaling_relation'`` — uses exposure times from a reference
  observation (:meth:`~bioverse.survey.Survey.set_reference_observation`) and
  schedules targets via the survey (see documentation for equations).

- ``method='exp_time'`` — blind survey mode: requires a prior
  :func:`~bioverse.functions.schedule_DI_survey` step in the generator so that
  each host star has an allocated observing budget ``t_req``; exposure times for
  planets use :func:`~bioverse.util.DI_exposure_time_calculator` (based on
  `Stark et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014ApJ...795..122S/abstract>`_).
  A planet counts as detected if its required exposure is less than the time
  allocated to its star.

This script only illustrates the *API* for the first two methods; wiring
``schedule_DI_survey`` into a full generator is mission-specific.
"""
from bioverse.generator import Generator
from bioverse.survey import ImagingSurvey


def main():
    generator = Generator("imaging")
    survey = ImagingSurvey("default")

    sample = generator.generate(d_max=30.0)

    # (1) Detectability only — no exposure / duration limits.
    detected_detectable = survey.compute_yield(sample, method="detectable")

    # (2) Scaling-relation yield — define a reference observation, then compute.
    survey.set_reference_observation(
        t_ref=3.1 / 24.0,
        wl_eff=0.6,
        T_st_ref=5788.0,
        R_st_ref=1.0,
        D_ref=15.0,
        d_ref=10.0,
        R_pl_ref=1.0,
        H_ref=9.0,
    )
    detected_scaling = survey.compute_yield(sample, method="scaling_relation")

    # (3) exp_time / blind survey — uncomment when your Table includes `t_req`
    # from :func:`~bioverse.functions.schedule_DI_survey` in the generator:
    # detected_blind = survey.compute_yield(sample, method="exp_time")

    print(
        len(detected_detectable),
        "detectable;",
        len(detected_scaling),
        "scaling_relation yield",
    )
    return detected_detectable, detected_scaling


if __name__ == "__main__":
    main()
