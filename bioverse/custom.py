""" Define new functions for planet simulation here. Function arguments should be provided a default value."""

# Add import statements as necessary
import numpy as np

# Bioverse modules and constants
from .classes import Table
from . import util
from .util import CATALOG
from .constants import CONST, ROOT_DIR, DATA_DIR



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


def magma_ocean(d, funform='exp_decay', f_magma=0.1, lambda_a=2., a_cut=0.1, radius_reduction=0.2):
    """Assign a fraction of planets global magma oceans that change the planet's radius.

    Parameters:
    -----------
    d: Table
        The population of planets.
    funform: str
        The functional form of the magma ocean probability as a function of effective semi-major axis (sma).
        Can be 'exp_decay' or 'step'.
    f_magma: float
        The fraction of planets that have global magma oceans.
    lambda_a: float
        Decay parameter for the semi-major axis dependence of having a global magma ocean.
    a_cut: float
        cutoff effective sma for magma oceans. Defines position of the exponential decay or step.
    radius_reduction: float
        The fraction by which a planet's radius is reduced due to a global magma ocean.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets with new columns 'has_magmaocean' and 'is_small'.

    """
    # randomly assign planets to have a magma ocean, depending on the semi-major axes
    if funform == 'exp_decay':
        # assign magma ocean with a probability following a exponential decay function
        P_magma = f_magma * np.exp(-(d['a_eff']/a_cut)**lambda_a) # HAS TO BE REPLACED WITH MODEL OUTPUT FOR MAGMA OCEAN PLANETS
    elif funform == 'step':
        # assign a magma ocean with probability f_magma, but only to planets with a_eff < a_cut
        P_magma = f_magma * (d['a_eff'] < a_cut)
    else:
        raise ValueError('funform must be either "exp_decay" or "step"')

    d['has_magmaocean'] = np.random.uniform(0, 1, len(d)) < P_magma

    # reduce the radius of the planets with magma oceans
    mask = d['has_magmaocean']
    try:
        d.loc[mask,'R'] *= (1 - radius_reduction) # HAS TO BE REPLACED WITH MODEL OUTPUT FOR MAGMA OCEAN PLANETS
    except AttributeError:
        d['R'][mask] *= (1 - radius_reduction) # HAS TO BE REPLACED WITH MODEL OUTPUT FOR MAGMA OCEAN PLANETS

    # define planets with smaller radius than expected. THIS SHOULD BE REPLACED WITH SOMETHING MORE REALISTIC
    d['is_small'] = d['R'] < np.mean(d['R'])

    return d
