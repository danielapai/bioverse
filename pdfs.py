# System modules
import numpy as np
import scipy

# Draws N samples from the following PDF:
# dP/dX ~ (ax)^b1 if xmin < x =< x0, (ax)^b2 if xmax > x > x0
# dP/dX = 0 if x < xmin or x > xmax
# dX = dlog(x) if log = True else dx
def power_law_broken(a,b1,b2,x0,xmin,xmax,size=1,log=True):
    # Calculate the normalization factor so the CDF = 1 when integrated from xmin to xmax
    if log: A = (b1**(-1)*(1-(xmin/x0)**b1)-b2**(-1)*(1-(xmax/x0)**b2))**(-1)
    else: A = (a**b1/(b1+1)*(x0**(b1+1)-xmin**(b1+1))+a**b2/(b2+1)*(xmax**(b2+1)-x0**(b2+1)))**(-1)
    
    # Calculate the CDF at x = x0
    if log: y0 = (A/b1)*(1-(xmin/x0)**b1)
    else: y0 = A*a**b1/(b1+1)*(x0**(b1+1)-xmin**(b1+1))
    
    # Draw N values uniformly from 0 to 1
    y = np.random.uniform(0,1,size)
    
    # Solve for x where CDF = y
    m,X = y<=y0,np.full(size,None,dtype=float)
    if log:
        X[m] = (b1*x0**b1/A*y[m]+xmin**b1)**(1./b1)
        X[~m] = (b2*x0**b2/A*(y[~m]-y0)+x0**b2)**(1./b2)
    else:
        X[m] = (y[m]*(b1+1)/(A*a**b1)+xmin**(b1+1))**(1/(b1+1))
        X[~m] = ((y[~m]-y0)*(b2+1)/(A*a**b2)+x0**(b2+1))**(1/(b2+1))
        
    if size == 1:
        return X[0]
    else:
        return X

# Draws N samples from the following PDF:
# dP/dX ~ (ax)^b if xmin < x < xmax
# dP/dX = 0 if x < xmin or x > xmax
# dX = dlog(x) if log = True else dx
def power_law(a,b,xmin,xmax,N=1,log=True):
    return power_law_broken(a,b,1.,xmax,xmin,xmax,N=N,log=log)

def power_law_split(gamma,alpha,beta,x_split,xmin,xmax,ymin,ymax,size=1,log=True):
    """ Draws N samples from the following PDF:

    d^2P/dXdY = gamma_i * X^alpha_i * Y^beta_i

    where gamma_i = gamma[0] for X <= X_split, gamma[1] for X > X_split.
    """
    # Calculate the normalization constant
    g,a,b = np.array(gamma),np.array(alpha)+2,np.array(beta)+2
    A = g[0] * a[0]**-1 * b[0]**-1 * (x_split**a[0]-xmin**a[0]) * (ymax**b[0]-ymin**b[0])
    A += g[1] * a[1]**-1 * b[1]**-1 * (xmax**a[1]-x_split**a[1]) * (ymax**b[1]-ymin**b[1])
    A = A**-1

# Draws N samples from a normal PDF with mean value a and standard deviation b
# (optional) bounded to xmin < x < xmax
def normal(a,b,xmin=None,xmax=None,size=1):
    if b is None: return np.full(size,a)    
    else:
        aa = -100 if xmin is None else (xmin-a)/b
        bb = 100 if xmax is None else (xmax-a)/b
        # Deal with nan values bug
        if xmin is not None and np.size(aa)>1:
            aa[np.isnan(aa)] = -100
        if xmax is not None and np.size(bb)>1:
            bb[np.isnan(bb)] = 100
        return scipy.stats.truncnorm.rvs(a=aa,b=bb,loc=a,scale=b,size=size)

"""
# Draw from a bounded normal distribution
def bounded_gaussian_sample(N,mu,sig,xmin,xmax,dx=None):
    # Returns a sample selected from a normal distribution but limited to [xmin, xmax]
    
    if sig is None:
        return np.full(N,mu)
    
    a,b = (xmin-mu)/sig,(xmax-mu)/sig
    return stats.truncnorm.rvs(a=a,b=b,loc=mu,scale=sig,size=N)
"""
