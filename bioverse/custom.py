""" Define new functions for planet simulation here. Funnction arguments should be provided a default value."""

# Add import statements as necessary
import numpy as np


def M_R(observed):
    """Just for this test, add a silly mass measurement"""
    observed['M'] = observed['R']*2
    return observed


def label_lateM(table, Mst_threshold=0.4):
    # is_lateM=False for all planets
    table['is_late'] = np.zeros(len(table))

    table['is_late'] = table['M_st'] < Mst_threshold
    return table


def occurrence_hypo(theta, X):
    """Define the hypothesis that the planet occurrence is lower around late Ms than around early Ms.
    """
    return None


def magma_ocean(d, f_magma=0.1, a_max=0.1):
    """Assign a fraction of planets global magma oceans that change the planet's radius.

    Parameters:
    -----------
    d: Table
        The population of planets.
    f_magma: float
        The fraction of planets that have global magma oceans.
    a_max: float
        The maximum semimajor-axis of the planet's orbit to have a magma ocean.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.

    """

    # randomly assign planets to have a magma ocean
    d['has_magmaocean'] = np.random.binomial(1, f_magma, len(d)).astype(bool)

    # but only for planets with a semimajor-axis less than a_max
    mask = d['a'] > a_max
    d['has_magmaocean'][mask] = False

    # reduce the radius of the planets with magma oceans
    mask = d['has_magmaocean']
    d['R'][mask] *= .8         # HAS TO BE REPLACED WITH MODEL OUTPUT FOR MAGMA OCEAN PLANETS

    return d
