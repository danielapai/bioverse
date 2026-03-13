""" Defines constant values used elsewhere in the code. """

# Imports
import os
import numpy as np

# Top-level code directory and sub-directories
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
UI_DIR = ROOT_DIR+'/UI/'
DATA_DIR = ROOT_DIR+'/Data/'
ATMOSPHERE_TEMPLATES_DIR = ROOT_DIR+'/Templates/Atmospheres/'
MODELS_DIR = ROOT_DIR+'/Objects/Models/'
INSTRUMENTS_DIR = ROOT_DIR+'/Instruments/'
OBJECTS_DIR = ROOT_DIR+'/Objects/'
GENERATORS_DIR = OBJECTS_DIR+'/Generators/'
SURVEYS_DIR = OBJECTS_DIR+'/Surveys/'
PLOTS_DIR = ROOT_DIR+'/Plots/'
RESULTS_DIR = ROOT_DIR+'/Results/'
CATALOG_FILE = DATA_DIR+'Gaia.csv'
FUNCTIONS_DIR = ROOT_DIR+'/functions/'

# Physical constants (in cgs where applicable)
CONST = {}
CONST['T_eff_sol'] = 5777.
CONST['yr_to_day'] = 365.2422
CONST['AU_to_solRad'] = 215.03215567054767
CONST['rho_Earth'] = 5.51
CONST['g_Earth'] = 980.7
CONST['amu_to_kg'] = 1.66054e-27
CONST['R_Earth'] = 6.371e8
CONST['R_Sun'] = 6.9634e10
CONST['h_Earth'] = 8.5e5
CONST['P_Earth'] = 101325.
CONST['S_Earth'] = 238.         # global-averaged, top-of-atmosphere insolation of Earth in W/m2. Scaled by an assumed albedo of 0.3.

# Data types
ARRAY_TYPES = (np.ndarray,list,tuple)
LIST_TYPES = ARRAY_TYPES
FLOAT_TYPES = (float,np.double,np.float64)
INT_TYPES = (int,np.int_,np.int64,np.integer,np.int8)
STR_TYPES = (str,np.str_)
BOOL_TYPES = (bool,np.bool_)

#Constants for HZ polynomials
#Seff_sun, a,b,c,d
HZ_CONST={'recent_venus': [1.7763,1.4335e-4,3.3954e-9,-7.6364e-12,-1.1950e-15], #From Kopparapu 2013
          'runaway_greenhouse': [1.0385,1.2456e-4,1.4612e-8,-7.6345e-12,-1.1950e-15],
          'moist_greenhouse':[1.0146,8.1884e-5,1.9394e-9,-4.3618e-12, -6.8260e-16],
          'max_greenhouse': [0.3507, 5.9578e-5,1.6707e-9,-3.0058e-12,-5.1925e-16],
          'early_mars': [0.3207, 5.4471e-5,1.5275e-9,-2.1709e-12,-3.8282e-16],
          'leconte': [1.105, 1.1921e-4, 9.5932e-9, -2.6189e-12, 1.3710e-16], #From Ramirez 2018
          'CO2_max': [0.3587,5.8087e-5,1.5393e-9,-8.3547e-13,1.0319e-16]
}