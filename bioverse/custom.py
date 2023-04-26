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
