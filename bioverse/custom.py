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


def magma_ocean(d, f_magma=0.1, lambda_a=2., a_cut=0.1, radius_factor=0.8):
    """Assign a fraction of planets global magma oceans that change the planet's radius.

    Parameters:
    -----------
    d: Table
        The population of planets.
    f_magma: float
        The fraction of planets that have global magma oceans.
    lambda_a: float
        Decay parameter for the semi-major axis dependence of having a global magma ocean.
    a_cut: float
        cutoff sma for magma oceans. Defines position of the exponential decay.
    radius_factor: float
        The fraction of a planet's original radius that is reduced due to a global magma ocean.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets with new columns 'has_magmaocean' and 'is_small'.

    """
    # randomly assign planets to have a magma ocean, depending on the semi-major axes
    P_magma = f_magma * np.exp(-(d['a']/a_cut)**lambda_a) # HAS TO BE REPLACED WITH MODEL OUTPUT FOR MAGMA OCEAN PLANETS
    d['has_magmaocean'] = np.random.uniform(0, 1, len(d)) < P_magma

    # reduce the radius of the planets with magma oceans
    mask = d['has_magmaocean']
    d['R'][mask] *= radius_factor # HAS TO BE REPLACED WITH MODEL OUTPUT FOR MAGMA OCEAN PLANETS

    # define planets with smaller radius than expected. THIS SHOULD BE REPLACED WITH SOMETHING MORE REALISTIC
    d['is_small'] = d['R'] < np.mean(d['R'])

    return d
