""" Miscellanous functions used elsewhere in the code. """

# Python imports
import importlib.util
import numpy as np
from warnings import warn

# Bioverse modules and constants
from .constants import LIST_TYPES, CATALOG_FILE, INT_TYPES, FLOAT_TYPES
from .import truncnorm_hack

# Load the Gaia stellar target catalog into memory for fast access
try:
    CATALOG = np.genfromtxt(CATALOG_FILE, delimiter=',', names=True)
except FileNotFoundError:
    warn("could not load {:s} - try running util.update_stellar_catalog")

# Progress bar if tqdm is installed, else a dummy function
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    
def bar(arg, do_bar=True):
    """ Given an iterable, returns a progress bar if tqdm is installed. Otherwise, returns the iterable.
    
    Parameters
    ----------
    arg : iterable
        Iterable for which to return a progress bar.
    do_bar : bool
        If False, return `arg` and don't display a progress bar.
    
    Returns
    -------
    tqdm : iterable
        If tqdm is installed, return a progress bar formed from `arg`. Otherwise, just return `arg`.
    
    """
    if tqdm is not None and do_bar:
        return tqdm(arg)
    else:
        return arg

# Function to get the input value types (try: int, float, bool, str)
def get_type(x):
    try:
        int(x)
        return int
    except ValueError:
        try:
            float(x)
            return float
        except ValueError:
            if x.strip().upper() in ['TRUE','FALSE']:
                return bool
            else:
                return str

# Determines whether `a` is a binary array (including float arrays e.g. [0., 1., 1., 0., ...]))
def is_bool(a):
    unique = np.unique(a[~np.isnan(a)])
    if len(unique) == 2 and 0 in unique and 1 in unique:
        return True
    elif len(unique) == 1 and (0 in unique or 1 in unique):
        return True
    else:
        return False

def as_tuple(x):
    """ Returns the parameter as a tuple. """
    if isinstance(x, tuple):
        return x
    elif np.ndim(x) == 0:
        return (x,)
    elif np.ndim(x) ==1:
        return tuple(x)

# Imports a function given the filename and the name of the function
def import_function_from_file(function_name,filename):
        # Get package and module names
        package_name = filename.strip('/').split('/')[-2] # should be "bioverse"
        module_name = '.'.join(filename.strip('/').split('/')[-1].split('.')[:-1])

        # Import the module
        spec = importlib.util.spec_from_file_location(package_name+'.'+module_name, filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Return the function
        return mod.__dict__[function_name]

# Returns the "colors" of a planet based on its class and orbit
def get_planet_colors(d):
    c_or = 'blue' if d['class1'] == 'cold' else 'green' if d['class1'] == 'warm' else 'red'
    c_pl = 'red' if d['class2'] == 'rocky' else 'green' if d['class2'] == 'super-Earth' else 'blue'
    return c_pl,c_or

# Cycles through the values in a list
def cycle_index(vals,val,delta):
    N = len(vals)
    idx = list(vals).index(val)
    idx_new = idx + delta
    if idx_new < 0: idx_new = N + idx_new
    if idx_new >= N: idx_new = idx_new - N
    return vals[idx_new]

# Fills an array with the appropriate "nan" value for its type
def nan_fill(a,dtype=None):
    return np.full(a.shape,np.nan,dtype=a.dtype if dtype is None else dtype)

# Translates an array of object counts (e.g. [1,2,2,1,3]) into a longer list
# of the individual objects' order of creation (e.g., [1,1,2,1,2,1,1,2,3])
# Uses a numpy trick from: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
def get_order(N):
    start = np.repeat(np.zeros(len(N)),N)
    counter = np.arange(N.sum())-np.repeat(N.cumsum()-N,N)
    return (start+counter).astype(int)

# Translates the 'subset' specified for a model into a mask
def mask_from_model_subset(pl,subset):
    mask = True
    for key in subset.keys():
        val = subset[key]
        if isinstance(val, LIST_TYPES):
            mask = mask & ((pl[key]>val[0])&(pl[key]<val[1]))
        else:
            mask = mask & (pl[key]==val)
    return mask

def compute_bin_centers(bins):
    """ Given a set of N bin edges, returns N-1 bin centers and half-widths. """
    return (bins[...,1:]+bins[...,:-1])/2., (bins[...,1:]-bins[...,:-1])/2.

def compute_eta_Earth(d, by_type=True):
    """ Computes the value of eta Earth for a simulated sample of planets. Note this could be inaccurate if
    there are stars without planets which are usually not listed in the simulated sample, although the algorithm
    does attempt to correct for this.

    Parameters
    ----------
    d : Table
        Simulated sample of planets.
    by_type : bool, optional
        If True, calculate eta Earth separately for each spectral type.
    """
    # Determine the fraction of stars with planets
    f_pl = len(np.unique(d['starID']))/max(d['starID'])

    # Calculate eta Earth for all stars
    N_st = max(d['starID'])
    eta = d['EEC'].sum()/N_st
    print("eta Earth = {:.2f} per star".format(eta))

    # Calculate eta Earth for each spectral type
    if by_type:
        for SpT in ['F','G','K','M']:
            # Determine the number of stars for this spectral type, accounting for stars without planets
            mask = d['SpT'] == SpT
            if mask.sum() == 0:
                continue
            N_st = len(np.unique(d['starID'][mask]))/f_pl
            eta = d['EEC'][mask].sum()/N_st
            print("          = {:.3f} per {:s} star (+- {:.5f})".format(eta, SpT, d['EEC'][mask].sum()**0.5/N_st))

        # Also calculate for M1-M6 and M7-L0 stars
        mask = (d['T_eff_st'] > 3050) & (d['T_eff_st'] <= 3900)
        N_st = len(np.unique(d['starID'][mask]))/f_pl
        eta = d['EEC'][mask].sum()/N_st
        print("          = {:.3f} per M1-M4 star (+- {:.5f})".format(eta, d['EEC'][mask].sum()**0.5/N_st))

        mask = (d['T_eff_st'] > 1950) & (d['T_eff_st'] <= 3050)
        N_st = len(np.unique(d['starID'][mask]))/f_pl
        eta = d['EEC'][mask].sum()/N_st
        print("          = {:.3f} per M5-L2 star (+- {:.5f})".format(eta, d['EEC'][mask].sum()**0.5/N_st))

def compute_occurrence_multiplier(optimistic=False, optimistic_factor=3, N_pts=30):
    """ Determines the multiplier for occurrence rates and planet periods as a function of stellar mass. """
    # Scaling factors for spectral type (Mulders+2015, Table 1 and Figure 4)
    M_st_0 = [0.17, 0.69, 0.97, 1.43]
    f_N = 1/np.array([0.35, 0.55, 0.75, 1.0])
    f_a = 1/np.array([1.6, 1.4, 1.2, 1.0])


    # Edit the M star point to be 3.5x solar
    f_N[0] = 3.5 * f_N[2]

    # In the "optimistic" case, assume TRAPPIST-1 analogs (M = 0.08) have 5x as many planets as the typical Kepler star
    if optimistic:
        # Add a 0.8 solar mass point with 10x as many planets as solar type stars
        M_st_0 = np.append([0.08], M_st_0)
        f_N = np.append([optimistic_factor*f_N[0]], f_N)
        f_a = np.append([f_a[0]], f_a)

    # Fit a second-order polynomial to f_N and f_a
    p_N = np.polyfit(M_st_0, f_N, deg=2)
    p_a = np.polyfit(M_st_0, f_a, deg=2)

    # Evaluate over a grid of masses and convert semi-major axis to period
    x = np.logspace(np.log10(0.05), np.log10(2), N_pts)
    #y_N = np.polyval(p_N, x)
    #y_P = np.polyval(p_a, x)**1.5
    y_N = np.interp(x, M_st_0, f_N)
    y_P = np.interp(x, M_st_0, f_a)**1.5

    # Normalize these factors so that y = 1 for the typical Kepler planet host
    # (M ~ 0.965 per Kopparapu+2018)
    M_st_typ = 0.965
    #y_N /= np.polyval(p_N, M_st_typ)
    #y_P /= np.polyval(p_a, M_st_typ)**1.5
    y_N /= np.interp(M_st_typ, M_st_0, f_N)
    y_P /= np.interp(M_st_typ, M_st_0, f_a)**1.5

    return x, y_N, y_P

def update_stellar_catalog(d_max=100, filename=CATALOG_FILE):
    """ Updates the catalog of nearby sources from Gaia DR2 and saves it to a file. Requires astroquery. """
    from astroquery.gaia import Gaia
    
    Nmax = 100000
    query = "SELECT TOP {:d} source_id, parallax, ra, dec, teff_val, phot_g_mean_mag".format(Nmax)+\
            " FROM gaiadr2.gaia_source"+\
            " WHERE (parallax>={:f}) AND (teff_val>=4000)".format(1000/d_max)+\
            " ORDER BY parallax DESC"

    job = Gaia.launch_job(query)
    table = job.get_results()
    table.write(filename, overwrite=True)

    if len(table) == Nmax:
        print("Warning! Gaia query exceeds row limit!")
    print("Catalog updated. Don't forget to restart!")

def get_xyz(pl,t=0,M=None,n=3):
    # Computes the x/y/z separation in AU at time(s) t, assuming the mean longitude at t=0 is M0 and the unit is days 
    # n determines the number of iterations of Newton's method for solving Kepler's equation
    
    # Determine the mean longitude at time(s) t
    if M is None:
        M = pl['M0']+(2*np.pi*t)/pl['P']
    
    # Eccentric orbit
    if np.any(pl['e'] != 0):
        # Increment M by 2pi (so the solver doesn't break near M = 0)
        M += 2 * np.pi
        
        # Eccentric anomaly (Kepler's equation solved w/ Newton's method with one iteration)
        E = M
        for i in range(n):
            E = E - (E-pl['e']*np.sin(E)-M)/(1-pl['e']*np.cos(E))
            
        # Check that the equation is properly solved
        sinE = np.sin(E)
        if (np.abs(M-(E-pl['e']*sinE)) > (0.002*np.pi)).any():
            print("Kepler's equation failed to solve! (e = {:.5f})".format(pl['e']))
        
        # Distance
        cosE = np.cos(E)
        r = pl['a']*(1-pl['e']*cosE)
        
        # True anomaly
        cos_nu = (cosE-pl['e'])/(1-pl['e']*cosE)
        sin_nu = ((1-pl['e']**2)**0.5*sinE)/(1-pl['e']*cosE)
        
    # Circular orbit
    else:
        nu = M
        r = pl['a']
        sin_nu = np.sin(M)
        cos_nu = np.cos(M)
    
    # Compute some intermediate terms
    cos_w,sin_w = np.cos(pl['w_AP']),np.sin(pl['w_AP'])
    cos_w_nu = cos_nu*cos_w-sin_nu*sin_w
    sin_w_nu = sin_nu*cos_w+sin_w*cos_nu
        
    # Compute x,y,z. z is the direction towards/from the observer.
    x = r*(np.cos(pl['w_LAN'])*cos_w_nu-np.sin(pl['w_LAN'])*sin_w_nu*pl['cos(i)'])
    y = r*(np.sin(pl['w_LAN'])*cos_w_nu+np.cos(pl['w_LAN'])*sin_w_nu*pl['cos(i)'])
    z = r*(np.sin(np.arccos(pl['cos(i)']))*sin_w_nu)
    
    return x,y,z

# Draws N samples from a normal PDF with mean value a and standard deviation b
# (optional) bounded to xmin < x < xmax
def normal(a,b,xmin=None,xmax=None,size=1):
    if b is None or np.sum(b) == 0: return np.full(size,a)    
    else:
        aa = -np.inf if xmin is None else (xmin-a)/b
        bb = np.inf if xmax is None else (xmax-a)/b
        # Deal with nan values bug
        if xmin is not None and np.size(aa)>1:
            aa[np.isnan(aa)] = -np.inf
        if xmax is not None and np.size(bb)>1:
            bb[np.isnan(bb)] = np.inf
        
        # truncnorm.rvs is extremely slow in newer SciPy versions; this line can be uncommented once that issue is fixed
        #return scipy.stats.truncnorm.rvs(a=aa,b=bb,loc=a,scale=b,size=size)
        
        # This module reproduces truncnorm as of SciPy v1.3.3
        return truncnorm_hack.truncnorm.rvs(a=aa,b=bb,loc=a,scale=b,size=size)

def binned_average(x, y, bins=10, match_counts=True):
    """ Computes the average value of a variable in bins of another variable.

    Parameters
    ----------
    x : float array
        Array of independent values along which to perform the binning.
    y : float array
        Array of dependent values to be averaged.
    bins : int or float array, optional
        Number of bins or array of bin edges.
    match_counts : bool, optional
        If True, adjust the bin sizes so that an equal number of data points fall in each bin. Passing
        an array of bin edges for `bins` will override this setting.

    Returns
    -------
    bins : float array
        Array of bin edges.
    values : float array
        Average value of `y` in each bin. 
    errors : float array
        Uncertainty on `values` in each bin, i.e. the standard error on the mean.
    """
    # Unless given, compute the bin edges
    if isinstance(bins, (INT_TYPES, FLOAT_TYPES)):
        bins = int(bins)
        if match_counts:
            bins = np.percentile(x,np.linspace(0,100,bins+1))
        else:
            binsize = np.ptp(x)/bins
            bins = np.arange(np.amin(x),np.amax(x)+binsize,binsize)
    
    # Compute the average value of `y` in each bin
    values, errors = np.full((2,len(bins)-1), np.nan)
    for i in range(len(bins)-1):
        inbin = (x>=bins[i])&(x<=bins[i+1])
        if inbin.sum() > 0:
            values[i], errors[i] = np.mean(y[inbin]), np.std(y[inbin])/(inbin.sum())**0.5

    return bins,values,errors

def compute_t_ref(filenames, t_exp, wl_min, wl_max, threshold=5, usecols=(0, 1, 2)):
    """ Computes t_ref for the detection of a spectroscopic feature. User must first use PSG or
    other tools to simulate spectra of the reference target with and without the feature of interest.

    Parameters
    ----------
    filenames : (str, str)
        Points to two PSG output spectra files - one where the atmosphere contains the species of interest,
        and one where it does not (the order does not matter).
    t_exp : float
        Exposure time for the PSG simulations - must be identical for both.
    wl_min : float
        Minimum wavelength of the absorption feature, same units as the PSG output.
    wl_max : float
        Maximum wavelength of the absorption feature.
    threshold : float, optional
        SNR threshold for a confident detection.
    usecols : (int, int, int), optional
        Specifies the column numbers corresponding to (wavelength, radiance, uncertainty) in the input files.

    Returns
    -------
    t_ref : float
        Exposure time required to reach the targeted detection SNR, same units as `t_exp`.
    """

    # Check input
    if np.size(filenames) != 2:
        raise ValueError("`filenames` must be a 2-element tuple or list")

    # Loads the spectrum and measurement uncertainties for both files
    x1, y1, yerr = np.loadtxt(filenames[0], usecols=usecols, unpack=True)
    x2, y2, _ = np.loadtxt(filenames[1], usecols=usecols, unpack=True)

    # Ensure the wavelength values match within the specified range
    idx1 = (x1 >= wl_min) & (x1 <= wl_max)
    idx2 = (x2 >= wl_min) & (x2 <= wl_max)
    if not (x1[idx1] == x2[idx2]).all():
        raise ValueError("mismatch between wavelength values in {:s} and {:s}".format(*filenames))

    # Calculate the detection SNR and determine the required exposure time for a detection
    terms = np.abs(y1[idx1] - y2[idx2]) / yerr[idx1]
    snr = np.sqrt(np.sum(terms**2))
    t_ref = t_exp * (threshold/snr)**2

    return t_ref


def compute_logbins(binWidth_dex, Range):
    """ Compute the bin edges for a logarithmic grid.

    Parameters
    ----------
    binWidth_dex : float
        width of bins in log space (dex)
    Range : Tuple
        range for parameter

    Returns
    -------
    bins : array
        bins for one dimension

    Example
    -------
    >>> binWidth_dex = 1.0
    >>> Range = (10., 1000.)
    >>> compute_logbins(binWidth_dex, Range)
    array([   10.,   100.,  1000.])
    """
    # add binWidth_dex to logrange to include last bin edge
    logRange = (np.log10(Range[0]), np.log10(Range[1]) + binWidth_dex)
    return 10**np.arange(logRange[0], logRange[1], binWidth_dex)