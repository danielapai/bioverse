""" Define new functions for planet simulation here. """

# Add import statements as necessary
import numpy as np

from .constants import ROOT_DIR
from .constants import CONST

def map_biogeo_results(d, biogeo_file='biogeo_output.csv'):
    """ Randomly applies output states from the biogeo model to Earth-like planets. """

    # Placeholders
    d['pH2'], d['pCO2'], d['pCH4'] = np.zeros((3, len(d)), dtype=float)
    d['scenario'] = np.full(len(d), 'uninhabitable')
    EEC = d['EEC']

    # Load the output table and convert partial pressure -> mixing ratio assuming 1 atm surface pressure
    c = 1/CONST['P_Earth']
    pH2, pCO2, pCH4, scenario = np.loadtxt(ROOT_DIR+'/'+biogeo_file, unpack=True, skiprows=1, delimiter=',', dtype=str)
    pH2, pCO2, pCH4 = pH2.astype(float)*c, pCO2.astype(float)*c, pCH4.astype(float)*c

    # Assign a random row in the output table to each simulated planet
    idx = np.random.choice(np.arange(len(pH2)), replace=True, size=EEC.sum())
    d['pH2'][EEC], d['pCO2'][EEC], d['pCH4'][EEC], d['scenario'][EEC] = pH2[idx], pCO2[idx], pCH4[idx], scenario[idx]

    return d

    