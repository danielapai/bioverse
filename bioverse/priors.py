# System modules
import numpy as np
from .constants import ARRAY_TYPES, INT_TYPES, ROOT_DIR

# PDMF
def Chabrier_2003_PDMF(M_st_min=0.07,M_st_max=5.0):
    # The split PDMF for single objects defined by Chabrier 2003, Table 1
    # x is log(m); y is dn/dlog(m)
    x = np.linspace(np.log10(M_st_min),np.log10(M_st_max),100)
    y = np.zeros(len(x),dtype=float)
    y[x<=0] = 0.158 * np.exp(-(x[x<=0]-np.log10(0.079))/(2*0.69**2))
    y[x>0] = 4.43e-2*(10**x[x>0])**(-1.3)
    
    return x,y

def Chabrier_2003(s,st,p):
    # Number of stars to generate
    N = st.length()
    
    # Get the PDMF (dn/dlog(m) vs log(m))
    x,y1 = Chabrier_2003_PDMF(p)
  
    # Integrate and normalize
    y2 = np.cumsum(y1)/np.sum(y1)
    
    # Draw N values from 0 to 1 and match these to the appropriate stellar mass
    draw = np.random.uniform(0,1,size=N)
    
    return 10**np.interp(draw,y2,x)

# HABITABLE ZONE
def Kopparapu_2014(d,pl):
    # Calculates distances corresponding to (Recent Venus, Runaway GH, Max GH, Early Mars)
    i1, i2 = 1,2
    
    L,T_eff = 10**d['logL'],d['Star1']['T_eff_st']
    M_pl = pl['M']
    
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
    d = np.array([[-3.097e-15,-1.713e-15,-5.575e-16,-5.011e-16],
                  [-3.097e-15,-1.931e-15,-5.575e-16,-5.011e-16],
                  [-3.097e-15,-2.084e-15,-5.575e-16,-5.011e-16]])
    
    # Interpolate to estimate the constant values for the given planet
    S_eff_sol0 = np.array([np.interp(M_pl,M_ref,S_eff_sol[:,i]) for i in range(len(S_eff_sol[0]))])
    a = np.array([np.interp(M_pl,M_ref,a[:,i]) for i in range(len(a[0]))])
    b = np.array([np.interp(M_pl,M_ref,b[:,i]) for i in range(len(b[0]))])
    c = np.array([np.interp(M_pl,M_ref,c[:,i]) for i in range(len(c[0]))])
    d = np.array([np.interp(M_pl,M_ref,d[:,i]) for i in range(len(d[0]))])
    
    # Calculate the effective stellar flux at each boundary (Equation 4)
    T_st = T_eff-5780.
    S_eff = S_eff_sol0+a*T_st+b*T_st**2+c*T_st**3+d*T_st**4
    
    # The corresponding distances in AU (stars with T_eff > 7200 K are not habitable so d = inf)
    m = T_eff<7200
    d = np.full((4,len(T_eff)),np.inf)
    d[:,m] = (L[m]/S_eff[:,m])**0.5

    # Return d(i1) and d(i2) corresponding to the inner, outer HZ bounds
    return d[i1], d[i2]

def assign_zones(s,pl):
    # Assign a 'zone' based on the average distance and HZ boundaries

    # By default planets are 'normal'
    zones = np.full(pl.length(),'none',dtype='<U20')
    
    # Kopparapu+2014 limits
    zones[pl['a0']<=pl['a_in']] = 'runaway'
    zones[pl['a0']>=pl['a_out']] = 'maximum'
    zones[(pl['a0']>pl['a_in'])&(pl['a0']<pl['a_out'])] = 'temperate'
    
    # Kane+2014 "Venus zone" inner edge (also e.g. Zahnle+2013)
    zones[pl['S']>25] = 'hot'

    return zones

# CLASSIFICATION
def probabilistic_classification(s,pl):
    # Assigns composition classes probabilistically based on radius. Specifically,
    # the probability that a planet is a non habitable sub-Neptune increases linearly
    # from 0 at 1.4 R_E, to 1 at 1.7 R_E.
    
    # Extract the planet radii
    R_pl = pl['R']
    if type(R_pl) in ARRAY_TYPES:
        rp = R_pl
    else:
        rp = np.array([R_pl])
    
    # Assign the probability that a planet is of "ice giant" class
    P = np.interp(rp,[1.4,1.7],[0,1])
    
    # Draw the classes from "terrestrial" and "ice giant"
    draw = np.random.uniform(0,1,len(rp))
    classes = np.full(len(rp),'none',dtype='<U30')
    classes[draw<=P] = 'ice giant'
    classes[draw>=P] = 'terrestrial'
    
    # Assign the probability that a planet is of "super-Earth" class
    P = np.interp(rp,[0.5,0.8],[0,1])
    
    # Draw the classes from "sub-terrestrial" and "terrestrial"
    draw = np.random.uniform(0,1,len(rp))
    classes[draw>=P] = 'sub-terrestrial'
    
    
    # Divide sub-terrestrial/terrestrial by an escape velocity vs insolation criteria
    #vesc_min = 6. # km/s, minimum escape velocity to maintain an atmosphere at Earth insolation
    
    #vesc = planet['v_esc']
    #ins = planet['ins']
    #classes[vesc<(vesc_min*ins**0.25)] = 'sub-terrestrial'
    #classes[R_pl<0.5] = 'sub-terrestrial'
    

    if type(R_pl) in ARRAY_TYPES:
        return classes
    else:
        return classes[0]

# TYPES
def assign_atmosphere_types(classes,zones):
    # Assigns an atmosphere type to each planet based on its class and zone
    # The types for each combination of zone & planet class are supplied in the input file
    
    N_planets = len(classes)
    
    # Open the input file
    type0,class0,zone0,p0 = np.loadtxt('input/types.dat',unpack=True,dtype=str,delimiter=',')
    p0 = np.array(p0,dtype=float)
    
    # Fix some stuff
    type0 = np.array([t0.replace('\t','') for t0 in type0])
    class0 = np.array([c0.replace('\t','') for c0 in class0])
    zone0 = np.array([r0.replace('\t','') for r0 in zone0])

    # Get unique classes, zones in the input file
    uclass,uzone = np.unique(class0), np.unique(zone0)

    # Loop through composition and class
    types = np.full(N_planets,None,dtype='<U10')
    cmasks = [np.in1d(classes,uclass[i]) for i in range(len(uclass))]
    rmasks = [np.in1d(zones,uzone[i]) for i in range(len(uzone))]
    for i in range(len(uclass)):
        for j in range(len(uzone)):
            # If no possible combinations of this sort exists, continue
            mask0 = (class0==uclass[i])&(zone0==uzone[j])
            if np.sum(mask0) == 0:
                continue
            
            # Find all of the simulated planets which satisfy this combination
            mask = cmasks[i]&rmasks[j]
            
            # Assign types to these planets based on the relative abundances in the input file
            if len(type0[mask0])==1:
                types[mask] = type0[mask0]
            else:
                types[mask] = np.random.choice(type0[mask0],np.sum(mask),p=p0[mask0])
            
    # Check that every planet has a type; possibly a combination of class+zone has been missed in the input file
    if np.sum((types=='None')&(zones!='none'))>0:
        print("Error: not all class+zone combinations have a type! Check the input file for:")
        print("Class = {0}; zone = {1}".format(np.unique(classes[types=='None']),np.unique(zones[types=='None'])))
    
    return types
