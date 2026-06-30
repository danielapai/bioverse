# Import numpy
import numpy as np

# Import the relevant modules
from bioverse.survey import ImagingSurvey, read_scaling_dict, prioritize_survey
from bioverse.generator import Generator
from bioverse.hypothesis import Hypothesis
from bioverse import analysis
import time


def habitable_zone_water(d, f_water_habitable=0.75, f_water_nonhabitable=0.01):
    d['has_H2O'] = np.zeros(len(d), dtype=bool)
    m1 = d['R'] > 0.8*d['S']**0.25
    d['has_H2O'][m1] = np.random.uniform(0, 1, size=m1.sum()) < f_water_nonhabitable
    m2 = d['EEC']
    d['has_H2O'][m2] = np.random.uniform(0, 1, size=m2.sum()) < f_water_habitable
    return d


def f(theta, X):
    a_inner, delta_a, f_HZ, df_notHZ = theta
    in_HZ = (X > a_inner) & (X < (a_inner + delta_a))
    return in_HZ * f_HZ + (~in_HZ) * f_HZ*df_notHZ


def f_null(theta, X):
    shape = (np.shape(X)[0], 1)
    return np.full(shape, theta)


if __name__ == '__main__':

    generator = Generator('imaging')

    survey = ImagingSurvey('default')
    # start from the default Imaging survey and add an H2O observation
    template_name = 'imaging_H2O'
    # import dictionary with reference times
    scaling_dict = read_scaling_dict(template_name)
    # set those reference times to be used in the exposure time calculation
    survey.set_reference_observation(**scaling_dict)
    # use a saved target prioritization scheme for the given observation
    prioritize_survey(survey, template_name)

    # add H2O measurement to survey measurements
    survey.add_measurement('has_H2O')
    # also add scaled semimajor axis
    survey.add_measurement('a_eff')

    generator.insert_step(habitable_zone_water)

    # Specify the names of the parameters (theta), features (X), and labels (Y)
    params = ('a_inner', 'delta_a', 'f_HZ', 'df_notHZ')
    features = ('a_eff',)
    labels = ('has_H2O',)

    bounds = np.array([[0.1, 2], [0.01, 10], [0.001, 1.0], [0.001, 1.0]])
    h_HZ = Hypothesis(f, bounds, params=params, features=features, labels=labels, log=(True, True, True, True))

    bounds_null = np.array([[0.001, 1.0]])
    h_HZ.h_null = Hypothesis(f_null, bounds_null, params=('f_H2O',), features=features, labels=labels, log=(True,))

    # Reload `generator` and `h_HZ`
    generator = Generator('imaging')
    from bioverse.hypothesis import h_HZ   # <- will error, see note above

    f_water_habitable = np.logspace(-2, 0, 10)
    start = time.time()

    results = analysis.test_hypothesis_grid(h_HZ, generator, survey, f_water_habitable=f_water_habitable, t_total=10*365.25, processes=8, N=20)

    elapsed = time.time() - start

    print(f"Runtime: {elapsed:.2f}s")