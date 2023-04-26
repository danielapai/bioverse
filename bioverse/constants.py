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

# Program version
VERSION = "1.0"

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
FLOAT_TYPES = (float,np.float,np.float_,np.float64)
INT_TYPES = (int,np.int_,np.int64,np.integer,np.int8)
STR_TYPES = (str,np.str_)
BOOL_TYPES = (bool,np.bool_)