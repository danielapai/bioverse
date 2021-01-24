from utils import *

# Functions for interacting with the Planetary Spectrum Generator


# Produces the reflected light spectrum of an Earth-like planet with an isothermal atmosphere
# as viewed from 5 pc with LUVOIR-A
def get_isothermal_spectrum(abundances,unit='ppmv',res=70):
    # abundances is a dict where keys are molecules and values are mixing ratios (ppmv)
    
    # Load the template
    t = PSGTemplate('templates/LUVOIR_5pc.cfg')
    
    # Set the abundances
    keys = abundances.keys()
    vals = [str(abundances[key]) for key in keys]
    t['ATMOSPHERE-NGAS'] = N = len(keys)    
    t['ATMOSPHERE-GAS'] = ','.join(keys)
    t['ATMOSPHERE-ABUN'] = ','.join(vals)
    t['ATMOSPHERE-TYPE'] = ','.join(['HIT[{:d}]'.format(i) for i in range(N)])
    t['ATMOSPHERE-UNIT'] = ','.join([unit]*N)
    t['ATMOSPHERE-TAU'] = ','.join(['1']*N)
    
    # Set the spectral resolution (default 70)
    t['GENERATOR-RESOLUTION'] = res
    
    # Run the model and return the results
    x,y,dy = t.run()

    return x,y,dy

# Reads the output of the biogeo model (abundances) and adds 80% N2, then
# generates a spectrum for each
def run_PSG_models(filename,pN2=800000.,N_max=None):
    # Reads the header to get each column
    keys = open(filename).readlines()[0].strip('#').split(',')
    vals = np.loadtxt(filename,dtype=float,delimiter=',',unpack=True)
    
    # Get the values for each abundance
    abundances = nestedDict()
    for i in range(len(keys)):
        key = keys[i].strip()
        if key[0] == 'p':
            abundances[key.strip('p')] = vals[i]
    
    # Add a flat N2 abundance
    abundances['N2'] = np.full(len(abundances),pN2)
    
    # Run the PSG individually for each model
    y,dy=None,None
    N = len(abundances) if N_max is None else min(N_max,len(abundances))
    for i in bar(range(N)):
        x,y0,dy0 = get_isothermal_spectrum(abundances[i])
        if y is None: y,dy = [y0],[dy0]
        else: y,dy = np.append(y,[y0],axis=0),np.append(dy,[dy0],axis=0)
    
    return x,np.swapaxes(y,0,1),np.swapaxes(dy,0,1)
