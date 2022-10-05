""" Contains all functions currently used to simulate planetary systems. To define new functions, add them to custom.py. """

# Python imports
import numpy as np
import os
import sys

# Bioverse modules and constants
from .classes import Table
from . import util
from .util import CATALOG
from .constants import CONST, ROOT_DIR, DATA_DIR

def create_stars_Gaia(d, d_max=150, M_st_min=0.075, M_st_max=2.0, T_min=0., T_max=10., T_eff_split=4500., seed=42):
    """ Reads temperatures and coordinates for high-mass stars from Gaia DR2. Simulates low-mass stars from the
    Chabrier+2003 PDMF.  Ages are drawn from a uniform distribution, by default from 0 - 10 Gyr. All other
    stellar properties are calculated using the scaling relations of Pecaut+2013.
    
    Parameters
    ----------
    d : Table
        An empty Table object.
    d_max : float, optional
        Maximum distance to which to simulate stars, in parsecs.
    M_st_min : float, optional
        Minimum stellar mass, in solar units.
    M_st_max : float, optional
        Maximum stellar mass, in solar units.
    T_min : float, optional
        Minimum stellar age, in Gyr.
    T_max : float, optional
        Maximum stellar age, in Gyr.
    T_eff_split : float, optional
        Effective temperature (in Kelvin) below which to simulate stars from a PDMF instead of using Gaia data.
    seed : int or 1-d array_like, optional
        Seed for numpy's RandomState. Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    d : Table
        Table containing the sample of simulated stars.  
    """

    np.random.seed(seed)

    # Create a stellar property conversion table from Pecaut+2013, sorted by ascending temperature
    # Note: the table from Pecaut+2013 was filtered down to A0V - L2V
    table = np.genfromtxt(DATA_DIR+'Pecaut2013.dat', dtype=None, encoding=None, names=True)
    table = table[np.argsort(table['Teff'])]
    cvt = {}
    cvt['subSpT'], cvt['SpT'], cvt['L_st'] = table['SpT'], np.array([s[0] for s in table['SpT']]), 10**table['logL']
    cvt['T_eff_st'], cvt['M_st'], cvt['R_st'], cvt['M_G'] = table['Teff'], table['Msun'], table['R_Rsun'], table['M_G']
    
    # Translate min/max stellar masses into temperatures, and T_eff_split into mass
    T_eff_min, T_eff_max = np.interp([M_st_min, M_st_max], cvt['M_st'], cvt['T_eff_st'])
    M_st_split = np.interp(T_eff_split, cvt['T_eff_st'], cvt['M_st'])

    # Step 1: High-mass stars
    d_Gaia = Table()

    # Load Gaia DR2 coordinates and temperatures, filtered by temperature and distance
    mask = np.isnan(CATALOG['teff_val']) | (CATALOG['teff_val'] < T_eff_split) | (CATALOG['parallax'] < 1000/d_max) 
    t = CATALOG[~mask]
    d_Gaia['d'], d_Gaia['ra'], d_Gaia['dec'], d_Gaia['T_eff_st'] = 1000/t['parallax'], t['ra'], t['dec'], t['teff_val']
    d_Gaia['M_G'] = t['phot_g_mean_mag'] - (5*np.log10(d_Gaia['d']) - 5)

    # Convert effective temperature to mass
    d_Gaia['M_st'] = np.interp(d_Gaia['T_eff_st'], cvt['T_eff_st'], cvt['M_st'])
    
    # Step 2: Low-mass stars
    d_IMF = Table()

    # Use the split PDFM from Chabrier 2003 (Table 1) to calculate the number of systems
    x_IMF = np.linspace(np.log10(M_st_min), np.log10(M_st_max), 100)
    y_IMF = np.zeros(len(x_IMF), dtype=float)
    y_IMF[x_IMF<=0] = 0.158 * np.exp(-(x_IMF[x_IMF<=0]-np.log10(0.079))/(2*0.69**2))
    y_IMF[x_IMF>0] = 4.43e-2*(10**x_IMF[x_IMF>0])**(-1.3)
    eta = (y_IMF*np.median(x_IMF[1:]-x_IMF[:-1])).sum()
    N_IMF = int(4 * np.pi/3 * d_max**3 * eta)
    
    # Assign a total stellar mass to each system, drawn from the PDMF
    draw = np.random.uniform(0, 1, size=N_IMF)
    d_IMF['M_st'] = 10**np.interp(draw, np.cumsum(y_IMF)/np.sum(y_IMF), x_IMF)

    # Convert mass to effective temperature and G-band absolute magnitude
    d_IMF['T_eff_st'] = np.interp(d_IMF['M_st'], cvt['M_st'], cvt['T_eff_st'])
    d_IMF['M_G'] = np.interp(d_IMF['M_st'], cvt['M_st'], cvt['M_G'])

    # Discard high-mass (i.e. high T_eff) simulated stars
    d_IMF = d_IMF[d_IMF['T_eff_st'] < T_eff_split]
    N_IMF = len(d_IMF)

    # Assign random coordinates (distance, RA, and Dec)
    d_IMF['d'] = np.cbrt(np.random.uniform(0, d_max**3, N_IMF))
    d_IMF['ra'] = np.random.uniform(0, 360, N_IMF)
    d_IMF['dec'] = np.arccos(np.random.uniform(0, 1, N_IMF))*180/np.pi - 90

    # Step 3: Combine the high- and low-mass stars and randomize their order
    d_Gaia['simulated'], d_IMF['simulated'] = np.zeros(len(d_Gaia), dtype=bool), np.ones(len(d_IMF), dtype=bool)
    d = d_Gaia.append(d_IMF, inplace=False)
    d.shuffle(inplace=True)

    # Calculate radius, luminosity, and spectral type
    for key in ['R_st', 'L_st']:
        d[key] = np.interp(d['T_eff_st'], cvt['T_eff_st'], cvt[key])

    d['SpT'] = np.full(len(d), 'none', dtype='U10')
    d['subSpT'] = np.full(len(d), 'none', dtype='U10')
    for i in range(len(cvt['T_eff_st'])-1):
        mask = (d['T_eff_st'] >= cvt['T_eff_st'][i]) & (d['T_eff_st'] < cvt['T_eff_st'][i+1])
        d['SpT'][mask] = cvt['SpT'][i]
        d['subSpT'][mask] = cvt['subSpT'][i]

    # Draw a random age for each system
    d['age'] = np.random.uniform(T_min, T_max, len(d))

    # Assign a starID to each system
    d['starID'] = np.arange(len(d), dtype=int)

    return d

def read_stellar_catalog(d, filename=DATA_DIR+'LUVOIR_targets.dat', d_max=30., T_min=0., T_max=10., mult=1, seed=42):
    """ Reads a list of stellar properties from the LUVOIR target catalog and fills in missing values.

    Parameters
    ----------
    d : Table
        An empty Table object.
    filename : str, optional
        Filename containing the LUVOIR target catalog.
    d_max : float, optional
        Maximum distance to which to simulate stars, in parsecs.
    T_min : float, optional
        Minimum stellar age, in Gyr.
    T_max : float, optional
        Maximum stellar age, in Gyr.
    mult : float, optional
        Multiple on the total number of stars simulated. If > 1, duplicates some entries from the LUVOIR catalog.
    seed : int or 1-d array_like, optional
        Seed for numpy's RandomState. Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    d : Table
        Table containing the sample of simulated stars.  
    """

    np.random.seed(seed)

    # Read the catalog with column names
    path = filename if os.path.exists(filename) else ROOT_DIR+'/'+filename
    catalog = np.genfromtxt(path,unpack=False,names=True,dtype=None,encoding=None)
    for name in catalog.dtype.names:
        d[name.strip()] = list(catalog[name])*int(mult)

    # Missing values (TODO: this part is ironically incomplete)
    if 'd' not in d.keys():
        d['d'] = np.cbrt(np.random.uniform(0, d_max**3, len(d)))
    if 'x' not in d.keys():
        cost,phi = np.random.uniform(-1, 1, len(d)), np.random.uniform(0, 2*np.pi, len(d))
        r = d['d']*np.sin(np.arccos(cost))
        d['x'], d['y'], d['z'] = r*np.cos(phi), r*np.sin(phi), d['d']*cost
    if 'age' not in d.keys():
        d['age'] = np.random.uniform(T_min, T_max, size=len(d))
    if 'logL' not in d.keys(): 
        d['logL'] = np.log10(d['L_st'])
    if 'binary' not in d.keys():
        d['binary'] = np.random.uniform(0, 1, len(d))<0.5

    # Enforce a maximum distance
    d = d[d['d']<d_max]

    # Assign stellar IDs and names
    d['starID'] = np.arange(len(d), dtype=int)
    d['star_name'] = np.char.array(np.full(len(d), 'SIM-'))+np.char.array(np.arange(len(d)).astype(str))
    d['simulated'] = np.zeros(len(d), dtype=bool)

    return d

def create_planets_bergsten(d, R_min=1.0, R_max=3.5, P_min=2, P_max=100., seed=42):
    """ Generates planets with periods and radii according to Bergsten+2022 occurrence rate estimates.

    Parameters
    ----------
    d : Table
        Table containing simulated host stars.
    R_min : float, optional
        Minimum planet radius, in Earth units.
    R_max : float, optional
        Maximum planet radius, in Earth units.
    P_min : float, optional
        Minimum orbital period, in days.
    P_max : float, optional
        Maximum orbital period, in days.
    seed : int or 1-d array_like, optional
        Seed for numpy's RandomState. Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets. Replaces the input Table.
    """

    np.random.seed(seed)
    
    # Bins, Parameters, and Values from Bergsten et al. 2022
    # Stellar Mass Bins
    massbins = [[0.556,0.815],[0.815,0.909],[0.909,1.008],[1.008,1.16],[1.16,1.629]]
    # Mass scalings of the Radius Valley
    Rsplit = [1.8198343592849173,1.92711136271344,1.978919030506143,2.0407383044237983,2.173374790737642]

    # Best-Fit Free Parameters
    F_0 = [0.8856361259972422,0.6996159608485483,0.6327526676490182,0.6129575597286655,0.5009869054654867]
    P_break = [14.288973856114884,6.133363167804966,6.918644494007631,12.017636043391624,6.958618711705457]
    beta1 = [0.15196765221136338,1.1888123910972717,0.911590303062237,0.43848485219532984,1.9016215517122808]
    beta2 = [-1.153131110847535,-0.6819146491558035,-0.8437029161066527,-1.101432111115821,-0.6907313978319928]
    P_central = [7.543177275679916,11.257915734447987,12.976947091193168,17.568566667378498,16.567186596597324]
    s = [1.4622073535853128,2.0225542991906944,2.532054278425091,2.1627543721548346,2.154972607211169]
    chi1 = [0.7528696587569685,0.73341694903406,0.8304039843638047,0.8254924295852643,0.8725919841057193]
    chi2 = [0.32550731709760333,0.35832058563162017,0.2560695336448793,0.3084550279549865,0.3926113907097983]
    # Uncertainty in avg. number of planets per star
    sigma_F0 = [0.05293837, 0.04864412, 0.04215205, 0.04083521, 0.04121296]

    # pre-computed normalization parameters
    Cn = [0.04381, 0.05183, 0.06002, 0.05155, 0.05177]
    
    ### Need a way to modify for transit mode...?
#     if transit_mode:
        
    # Some empty arrays to temporarily store planet parameters from different stellar mass bins
    num_planets = np.empty(len(d))
    master_P, master_R = [],[]

    # Set up probability grid in R and P
    P,dP = np.linspace(P_min,P_max,1000,retstep=True)
    R,dR = np.linspace(R_min,R_max,1000,retstep=True)
    xv, yv = np.meshgrid(P, R)
    
    # Generate probability grids in bins of stellar mass
    for i,_ in enumerate(F_0):

        # Broken power law (overall occurrence)
        g1 = (xv / P_break[i])**np.where(xv < P_break[i], beta1[i], beta2[i]) 
        # Hyperbolic tangent
        sx = 0.5 + 0.5*np.tanh((np.log10(xv)-np.log10(P_central[i]))/np.log10(s[i]))
        X = chi1[i]*(1-sx) + sx*(chi2[i])
        # Fractional occurrence
        a,b,c = np.log(1.0), np.log(Rsplit[i]), np.log(3.5)
        g2 = X*(c-b) / ((b-a) + X*(c-b) - X*(b-a))
        g_split = np.where(yv<Rsplit[i], g2, 1-g2)
        # Shape function (combined)
        g = g_split/yv * g1 

        # differential occurrence rate
        dN = F_0[i] * Cn[i] * g
        # Calculate number of planets per star (eta ~= F_0, by definition)
        eta = dN.sum() * (dP) * (dR)
        
        ### ignore modulation and transit mode for now
        N_pl = eta.astype(int)

        # Identify which stars are in the current stellar mass bin
        ### Only works if M_st_min and M_st_max are within the bin ranges for now...
        in_mass_bin = (d['M_st'] >= massbins[i][0]) & (d['M_st'] < massbins[i][1])
        # Give each star N_pl planets, and allow some to have an extra planet 
        # (e.g. if eta = 2.2 then 20% of those stars will have 3 planets)
        N_pl += np.random.uniform(0, 1, len(num_planets[in_mass_bin])) < (eta - eta.astype(int))
        num_planets[in_mass_bin] = N_pl.astype(int).copy()

        # Draw a (radius, period) for each planet in proportion to dN
        pflat, Pflat, Rflat = dN.flatten()/dN.sum(), xv.flatten(), yv.flatten()
        idx = np.random.choice(np.arange(len(pflat)), p=pflat, size=N_pl.sum())
        drawnR, drawnP = Rflat[idx], Pflat[idx]

        # "Smooth" the drawn values to prevent aliasing on the grid values
        drawnR = np.random.uniform(drawnR-dR/2., drawnR+dR/2.)
        drawnP = np.random.uniform(drawnP-dP/2., drawnP+dP/2.)
        master_P.append(drawnP)
        master_R.append(drawnR)
        
    # Expand the current table to match the number of planets, keeping the host star properties
    num_planets = [int(n) for n in num_planets]
    d = d[np.repeat(d['starID'], num_planets).astype(int)]
    d['planetID'] = np.arange(len(d))
    d['N_pl'] = np.repeat(num_planets, num_planets).astype(int)

    # create empty period and radius columns to be filled in shortly
    d['P'] = np.empty(len(d))
    d['R'] = np.empty(len(d))

    # Determine the order of the planet in the system (not necessarily by period)
    d['order'] = util.get_order(np.array(num_planets))

    # With the table expanded, now slot the (P,R) from correct distributions
    for i,_ in enumerate(F_0):
        in_mass_bin = (d['M_st'] >= massbins[i][0]) & (d['M_st'] < massbins[i][1])
        # Radius (R_Earth), period (d)
        d['R'][in_mass_bin] = master_R[i]
        d['P'][in_mass_bin] = master_P[i]

    ### No need to modulate by SpTy since the above is mass-dependent.

    # Compute semi-major axis and insolation
    d.compute('a')
    d.compute('S')

    return d

def create_planets_SAG13(d, eta_Earth=0.075, R_min=0.5, R_max=14.3, P_min=0.01, P_max=10., normalize_SpT=True,
                         transit_mode=False, optimistic=False, optimistic_factor=5, seed=42):
    """ Generates planets with periods and radii according to SAG13 occurrence rate estimates, but incorporating
    the dependence of occurrence rates on spectral type from Mulders+2015.

    Parameters
    ----------
    d : Table
        Table containing simulated host stars.
    eta_Earth : float, optional
        The number of Earth-sized planets in the habitable zones of Sun-like stars. All occurrence
        rates are uniformly scaled to produce this estimate.
    R_min : float, optional
        Minimum planet radius, in Earth units.
    R_max : float, optional
        Maximum planet radius, in Earth units.
    P_min : float, optional
        Minimum orbital period, in years.
    P_max : float, optional
        Maximum orbital period, in years.
    normalize_SpT : bool, optional
        If True, modulate occurrence rates by stellar mass according to Mulders+2015. Otherwise, assume no
        dependency on stellar mass.
    transit_mode : bool, optional
        If True, only transiting planets are simulated. Occurrence rates are modified to reflect the R_*/a transit probability.
    optimistic : bool, optional
        If True, extrapolate the results of Mulders+2015 by assuming rocky planets are much more common around late-type M dwarfs. If False,
        assume that occurrence rates plateau with stellar mass for stars cooler than ~M3.
    optimistic_factor : float, optional
        If optimistic = True, defines how many times more common rocky planets are around late-type M dwarfs compared to Sun-like stars.
    seed : int or 1-d array_like, optional
        Seed for numpy's RandomState. Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets. Replaces the input Table.
    """

    np.random.seed(seed)

    # SAG13 power law parameters
    R_break = 3.4
    gamma = [0.38, 0.73]
    alpha = [-0.19, -1.18]
    beta = [0.26, 0.59]

    # Compute the scaling factors for planet count and period based on Mulders+2015
    x_M, y_N, y_P = util.compute_occurrence_multiplier(optimistic, optimistic_factor)

    if transit_mode:
        beta = np.array(beta) - (2/3)
        gamma = np.array(gamma) / 215

    # Set up the probability grid in R and P
    lnx = np.linspace(np.log(R_min), np.log(R_max), 100)
    lny = np.linspace(np.log(P_min), np.log(P_max), 100)
    lnxv, lnyv = np.meshgrid(lnx, lny)
    dN = gamma[0]*(np.exp(lnxv)**alpha[0])*(np.exp(lnyv)**beta[0])
    dN2 = gamma[1]*(np.exp(lnxv)**alpha[1])*(np.exp(lnyv)**beta[1])
    dN[lnxv>np.log(R_break)] = dN2[lnxv>np.log(R_break)]

    # Multiplicative factor to increase/reduce eta Earth versus the SAG13 value
    eta_Earth_0 = 0.24
    dilution = 1 if eta_Earth is None else eta_Earth_0 / eta_Earth

    # Determine the number of planets for each star by integrating the
    # occurrence rates and modulating for spectral type
    eta = dN.sum() * (lnx[1]-lnx[0]) * (lny[1]-lny[0]) / dilution
    if normalize_SpT:
        eta = eta * np.interp(d['M_st'], x_M, y_N)
    if transit_mode:
        eta = eta * d['R_st'] * d['M_st']**(-1/3)
    N_pl = eta.astype(int)

    # Allow some stars an extra planet (e.g. if eta = 2.2 then 20% of those stars will have 3 planets)
    N_pl += np.random.uniform(0, 1, len(d)) < (eta - eta.astype(int))

    # Draw a (radius, period) for each planet in proportion to dN
    Pflat, lnRflat, lnPflat = dN.flatten()/dN.sum(), lnxv.flatten(), lnyv.flatten()
    idx = np.random.choice(np.arange(len(Pflat)), p=Pflat, size=N_pl.sum())
    lnR, lnP = lnRflat[idx], lnPflat[idx]

    # "Smooth" the drawn values to prevent aliasing on the grid values
    dlnR, dlnP = lnx[1]-lnx[0], lny[1]-lny[0]
    lnR = np.random.uniform(lnR-dlnR/2., lnR+dlnR/2.)
    lnP = np.random.uniform(lnP-dlnP/2., lnP+dlnP/2.)

    # Expand the current table to match the number of planets, keeping the host star properties
    d = d[np.repeat(d['starID'], N_pl).astype(int)]
    d['planetID'] = np.arange(len(d))
    d['N_pl'] = np.repeat(N_pl, N_pl).astype(int)

    # Determine the order of the planet in the system (not necessarily by period)
    d['order'] = util.get_order(N_pl)

    # Radius (R_Earth), period (d)
    P_yr = np.exp(lnP)
    d['R'] = np.exp(lnR)
    d['P'] = P_yr*CONST['yr_to_day']

    # Modulate the period based on spectral type
    if normalize_SpT:
        d['P'] *= np.interp(d['M_st'], x_M, y_P)

    # Compute semi-major axis and insolation
    d.compute('a')
    d.compute('S')

    return d

def name_planets(d):
    """ Assign a name to each star and each planet based on its order in the system.
    
    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.

    """
    alph = np.char.array(list('bcdefghijklmnopqrstuvwxyz'*3))
    d['star_name'] = np.char.array('SIM-')+d['starID'].astype(str)
    d['planet_name'] = d['star_name'] + alph[d['order']]
    return d
    
def assign_orbital_elements(d, transit_mode=False, seed=42):
    """ Draws values for any remaining Keplerian orbital elements. Eccentricities
    are drawn from a beta distribution following Kipping et al. (2013).

    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.
    transit_mode : bool, optional
        If True, only transiting planets are simulated, so cos(i) < R_*/a for all planets.
    seed : int or 1-d array_like, optional
        Seed for numpy's RandomState. Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.

    """

    np.random.seed(seed)

    for key in ['e','cos(i)','M0','w_LAN','w_AP']:
        # Skip keys which have already been assigned
        if key in d:
            continue
        # Draw eccentricity from a beta distribution, truncated at e > 0.8
        if key == 'e':
            d['e'] = np.random.beta(0.867,3.03,size=len(d))
            d['e'][d['e']>0.8] = np.random.uniform(0,0.8,(d['e']>0.8).sum())
        # Draw angular elements from isotropic distributions
        elif key == 'cos(i)':
            if transit_mode:
                cosi_max = d['R_st']/d['a']/CONST['AU_to_solRad']
                d['cos(i)'] = np.random.uniform(-cosi_max, cosi_max, size=len(d))
            else:
                d['cos(i)'] = np.random.uniform(-1,1,size=len(d))
        else:
            d['M0'],d['w_LAN'],d['w_AP'] = np.random.uniform(0,2*np.pi,(3,len(d)))

    return d

def impact_parameter(d, transit_mode=False):
    """ Calculates the impact parameter/transit duration.
    
    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.
    transit_mode : bool, optional
        If True, only transiting planets are simulated, so planets with b > 1 are discarded.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.
    
    """
    # Transit impact parameter (> 1 means not transiting)
    # TODO: currently only valid for circular orbits
    a = d['a']*CONST['AU_to_solRad']
    d['b'] = (a*d['cos(i)']/d['R_st'])

    # Discard non-transiting planets?
    d['transiting'] = np.abs(d['b']) <= 1
    if transit_mode:
        d = d[d['transiting']]

    # Calculate transit duration (d)
    tr = d['transiting']
    a, R_pl, R_st = d['a'][tr]*215.032, d['R'][tr]/109.2, d['R_st'][tr]
    d['T_dur'] = np.full(len(d), np.nan)
    d['T_dur'][tr] = (d['P'][tr]/np.pi) * np.arcsin(((R_st+R_pl)**2 - (d['b'][tr]*R_st)**2)**0.5 / a)
    
    return d

def assign_mass(d):
    """ Determines planet masses using a probabilistic mass-radius relationship,
    following Wolfgang et al. (2016). Also calculates density and surface gravity.

    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.
    """
    # Extract the radius of each planet
    R = d['R']
    M = np.zeros(R.shape)
    
    # Determine which are small, large planets
    mask1 = R>=1.6
    mask2 = (R>0.8)&(R<1.6)
    mask3 = R<=0.8
    
    # Draw masses for larger planets, with a spread of 1.9 M_E, with a minimum of 0.01 M_E
    M[mask1] = util.normal(2.7*R[mask1]**1.3,1.9,0.01,10000,mask1.sum())
    
    # Compute the maximum mass for each planet (Wolfgang 2016, Equation 5) where R > 0.2
    a,b,c = 0.0975,0.4938,0.7932
    M_max = 10**((-b+(b**2-4*a*(c-R[mask2]))**0.5)/(2*a))
    
    # Draw masses for the small planets from a truncated normal distribution (minimum: 0.1 Earth density)
    mu = 1.4*R[mask2]**2.3
    M[mask2] = util.normal(mu,0.3*mu,0.1*R[mask2]**3,M_max,mask2.sum())
    
    # For planets smaller than R < 0.2, assume Earth density
    M[mask3] = R[mask3]**3

    # Store the calculated masses in the database
    d['M'] = M

    # Density in g/cm3
    d['rho'] = CONST['rho_Earth']*d['M']/d['R']**3

    # Surface gravity in cm/s2
    d['g'] = CONST['g_Earth']*d['M']/d['R']**2

    return d

def classify_planets(d):
    """ Classifies planets by size and insolation following Kopparapu et al. (2018).
    
    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.
    
    """
    R,S = d['R'],d['S']

    class1 = np.full(len(d),'NONE',dtype='<U30')
    class2 = np.full(len(d),'NONE',dtype='<U30')

    # Boundary radii
    R0 = np.array([0.5,1.0,1.75,3.5,6.0,14.3])

    # Condensation points (Table 1)
    S0 = np.array([[182,187,188,220,220,220],
                   [1.0,1.12,1.15,1.65,1.65,1.7],
                   [0.28,0.30,0.32,0.45,0.40,0.45],
                   [0.0035,0.0030,0.0030,0.0030,0.0025,0.0025]])
    
    # Interpolate to find the hot/warm and warm/cold boundaries for each object
    S_inner = np.interp(R,R0,S0[1,:])
    S_outer = np.interp(R,R0,S0[2,:])

    # Insolation-based classification
    class1[S>S_inner] = 'hot'
    class1[(S<S_inner)&(S>S_outer)] = 'warm'
    class1[S<S_outer] = 'cold'

    # Size-based classification
    class2 = np.full(len(d),'NONE',dtype='<U30')
    class2[R<1.00] = 'rocky'
    class2[(R>1.00)&(R<1.75)] = 'super-Earth'
    class2[(R>1.75)&(R<3.50)] = 'sub-Neptune'
    class2[(R>3.50)&(R<6.00)] = 'sub-Jovian'
    class2[R>6.00] = 'Jovian'

    d['class1'] = class1
    d['class2'] = class2

    # Determine which planets are "exo-Earth candidates"
    # The lower limit on planet size depends on insolation
    lim = 0.8*d['S']**0.25
    d['EEC'] = (d['R'] > lim) & (d['R'] < 1.4) & (d['a'] > d['a_inner']) & (d['a'] < d['a_outer'])

    return d

def compute_habitable_zone_boundaries(d):
    """ Computes the habitable zone boundaries from Kopparapu et al. (2014), including
    dependence on planet mass.

    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.
    """
    L,T_eff = d['L_st'],d['T_eff_st']
    M_pl = d['M']
    
    # Parameters for each planet mass and boundary (Table 1)
    M_ref = np.array([0.1,1.,5.])
    S_eff_sol = np.array([[1.776,0.99,0.356,0.32],
                          [1.776,1.107,0.356,0.32],
                          [1.776,1.188,0.356,0.32]])
    a = np.array([[2.136e-4,1.209e-4,6.171e-5,5.547e-5],
                  [2.136e-4,1.332e-4,6.171e-5,5.547e-5],
                  [2.136e-4,1.433e-4,6.171e-5,5.547e-5]])
    b = np.array([[2.533e-5,1.404e-8,1.698e-9,1.526e-9],
                  [2.533e-5,1.58e-8,1.698e-9,1.526e-9],
                  [2.533e-5,1.707e-8,1.698e-9,1.526e-9]])
    c = np.array([[-1.332e-11,-7.418e-12,-3.198e-12,-2.874e-12],
                  [-1.332e-11,-8.308e-12,-3.198e-12,-2.874e-12],
                  [-1.332e-11,-8.968e-12,-3.198e-12,-2.874e-12]])
    d2 = np.array([[-3.097e-15,-1.713e-15,-5.575e-16,-5.011e-16],
                  [-3.097e-15,-1.931e-15,-5.575e-16,-5.011e-16],
                  [-3.097e-15,-2.084e-15,-5.575e-16,-5.011e-16]])
    
    # Interpolate in mass to estimate the constant values for each planet
    S_eff_sol0 = np.array([np.interp(M_pl,M_ref,S_eff_sol[:,i]) for i in range(len(S_eff_sol[0]))])
    a = np.array([np.interp(M_pl,M_ref,a[:,i]) for i in range(len(a[0]))])
    b = np.array([np.interp(M_pl,M_ref,b[:,i]) for i in range(len(b[0]))])
    c = np.array([np.interp(M_pl,M_ref,c[:,i]) for i in range(len(c[0]))])
    d2 = np.array([np.interp(M_pl,M_ref,d2[:,i]) for i in range(len(d2[0]))])
    
    # Calculate the effective stellar flux at each boundary (Equation 4)
    T_st = T_eff-5780.
    S_eff = S_eff_sol0+a*T_st+b*T_st**2+c*T_st**3+d2*T_st**4
    
    # The corresponding distances in AU (stars with T_eff > 7200 K are not habitable so d = inf)
    Tm = T_eff<7200
    dist = (L/S_eff)**0.5
    dist[:,~Tm] = np.inf

    # Inner, outer HZ bounds for each planet
    d_in,d_out = dist[1],dist[2]
    d['a_inner'],d['a_outer'] = d_in,d_out
    d['S_inner'],d['S_outer'] = S_eff[1],S_eff[2]

    # Compute the mean semi-major axis
    d['a0'] = d['a']/(1-d['e']**2)**0.5

    # By default planets are in the 'None' zone
    zones = np.full(len(d),'None',dtype='<U20')
    
    # Kopparapu+2014 limits
    zones[d['a0']<=d['a_inner']] = 'runaway'
    zones[d['a0']>=d['a_outer']] = 'maximum'
    zones[(d['a0']>d['a_inner'])&(d['a0']<d['a_outer'])] = 'temperate'
    
    # Kane+2014 "Venus zone" inner edge (also e.g. Zahnle+2013)
    #zones[d['S']>25] = 'hot'

    #d['zone'] = zones

    return d

def scale_height(d):
    """ Computes the equilibrium temperature and isothermal scale height
    by assigning a mean molecular weight based on size.
    
    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.

    """
    # Assign the mean molecular weight based on planet size
    rocky = np.in1d(d['class2'], ['rocky', 'super-Earth'])

    d['mu'], d['pCO2'], d['pN2'] = np.zeros((3, len(d)))

    # For hot rocky planets, assume CO2 atmospheres
    m = rocky & (d['a0'] <= d['a_inner'])
    d['pCO2'][m] = 1.0
    d['mu'][m] = 44.01

    # For rocky HZ planets, assume 1 bar N2/CO2 atmospheres and estimate pCO2 (bars) versus insolation
    # following Lehmer et al. (2020) (Figure 1).
    x0, y0 = 1.05, 0.01
    m = rocky & (d['a0'] > d['a_inner']) & (d['a0'] < d['a_outer'])
    d['pCO2'][m] = 10**(np.log10(y0) + 3.92*(x0-d['S'][m]))

    # For planets with pCO2 < 1 bar, make up the difference with N2
    diff = 1 - d['pCO2'][m]
    d['pN2'][m] = (diff>0)*diff

    # Calculate MMW from pCO2 and pN2
    P = d['pCO2'][m] + d['pN2'][m]
    d['mu'][m] = (d['pCO2'][m]/P)*44.01 + (d['pN2'][m]/P)*28.02
    
    # For cold rocky planets, assume 1 bar N2 atmospheres
    m = rocky & (d['a0'] >= d['a_outer'])
    d['pN2'][m] = 1.
    d['mu'][m] = 44.01

    # For ice giants, assume H/He atmospheres
    d['mu'][~rocky] = 2.5

    # Compute the equilibrium temperature (using Earth as the basis)
    # Estimate the bond albedo as the geometric albedo
    d['T_eq'] = 255.*(d['S']*(1-d['A_g']))**0.25/(1-0.29)**0.25

    # Compute the average atmospheric temperature (= T_eq for all but EECs)
    d['T_atm'] = d['T_eq']
    d['T_atm'][d['EEC']] = 250.

    # Compute the scale height (km)
    m = d['mu']*CONST['amu_to_kg']
    d['H'] = 1.38e-23*d['T_atm']/m/(d['g']/100)/1000

    return d

def geometric_albedo(d, A_g_min=0.1, A_g_max=0.7, seed=42):
    """ Assigns each planet a random geometric albedo from 0.1 -- 0.7, and computes the contrast ratio when viewed at quadrature.
    
    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.
    A_g_min : float, optional
        Minimum geometric albedo.
    A_g_max : float, optional
        Maximum geometric albedo.
    seed : int or 1-d array_like, optional
        Seed for numpy's RandomState. Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.
    """

    np.random.seed(seed)

    # Random albedo
    d['A_g'] = np.random.uniform(A_g_min, A_g_max, len(d))

    # Contrast ratio
    d['contrast'] = d['A_g'] * (4.258756e-5 * d['R'] / d['a'])**2 / np.pi

    return d

def effective_values(d):
    """ Computes the "effective" radius and semi-major axis (i.e. assuming an Earth-like planet).
    
    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.
    """
    # Effective semi-major axis = sma corresponding to the same insolation in the Solar System
    d['a_eff'] = d['a'] * d['L_st']**-0.5

    # Effective radius = radius if A_g = 0.29
    d['R_eff'] = (d['contrast'] * np.pi / 0.29)**0.5 / (4.258756e-5 / d['a'])

    return d

def compute_transit_params(d):
    """ Computes the transit depth of each planet.

    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.
    """
    Rp, Rs = d['R']*CONST['R_Earth'], d['R_st']*CONST['R_Sun']
    d['depth'] = (Rp/Rs)**2

    return d

def Example1_water(d, f_water_habitable=0.75, f_water_nonhabitable=0.1, minimum_size=True, seed=42):
    """ Determines which planets have water, according to the following model:

    f(S,R)  = f_water_habitable     if S_inner < S < S_outer and 0.8 S^0.25 < R < 1.4
            = f_water_nonhabitable  if R > 0.8 S^0.25
    
    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.
    f_water_habitable : float, optional
        Fraction of potentially habitable planets ("exo-Earth candidates") with atmospheric water vapor.
    f_water_nonhabitable : float, optional
        Fraction of non-habitable planets with atmospheric water vapor.
    minimum_size : bool, optional
        Whether or not to enforce a minimum size for non-habitable planets to have H2O atmospheres.
    seed : int or 1-d array_like, optional
        Seed for numpy's RandomState. Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.
    """

    np.random.seed(seed)

    d['has_H2O'] = np.zeros(len(d),dtype=bool)

    # Non-habitable planets with atmospheres
    m1 = d['R'] > 0.8*d['S']**0.25 if minimum_size else np.ones(len(d), dtype=bool)
    d['has_H2O'][m1] = np.random.uniform(0,1,size=m1.sum()) < f_water_nonhabitable

    # Potentially habitable planets
    m2 = d['EEC']
    d['has_H2O'][m2] = np.random.uniform(0,1,size=m2.sum()) < f_water_habitable

    return d

def Example2_oxygen(d, f_life=0.7, t_half=2.3, seed=42):
    """ Applies the age-oxygen correlation from Example 2.

    Parameters
    ----------
    d : Table
        Table containing the sample of simulated planets.
    f_life : float, optional
        Fraction of EECs (Earth-sized planets in the habitable zone) with life.
    tau : float, optional
        Timescale of atmospheric oxygenation (in Gyr), i.e. the age by which 63% of inhabited planets have oxygen.
    seed : int or 1-d array_like, optional
        Seed for numpy's RandomState. Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    d : Table
        Table containing the sample of simulated planets.
    """

    np.random.seed(seed)

    # Determine which planets have life
    d['life'] = d['EEC'] & (np.random.uniform(0, 1, len(d)) < f_life)
    
    # Determine which planets have oxygenated atmospheres
    f_oxy = 1 - 0.5**(d['age']/t_half)
    d['has_O2'] = d['life'] & (np.random.uniform(0, 1, len(d)) < f_oxy)

    return d
