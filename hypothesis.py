""" Defines the Hypothesis class as well as two hypotheses used in Bixel & Apai (2021). """

import dynesty
import emcee
import numpy as np
from scipy.stats import mannwhitneyu

# Bioverse modules
from util import is_bool

class Hypothesis():
    """ Describes a Bayesian hypothesis. The hypothesis function `f` should be defined with the following annotations:

    def f(theta:params, X:features) -> labels
 
    where `params`, `features`, and `labels` are tuples naming the parameters, independent, and dependent variables.
    An example:

    def h_HZ_func(theta:('a_inner', 'a_outer', 'f_HZ', 'f_notHZ'), X:('a_eff',)) -> ('has_H2O',):
    
    Parameters
    ----------
    f : function
        Function describing the hypothesis. Must be annotated as described above.
    bounds : array
        Nx2 array describing the [min, max] limits of each parameter. These are enforced even if a different prior
        distribution is defined by `prior_function`.
    prior_function : function, optional
        Used by emcee. Function which returns ln(P_prior), must be defined as prior(theta). If None, assume a 
        uniform distribution.
    guess_function : function, optional
        Used by emcee. Function which guesses valid sets of parameters. Must be defined as guess_function(n), and 
        should return an n x m set of parameter guesses. If None, draw parameters randomly within `bounds`.
    tfprior_function : function, optional
        Used by dynesty. Function which transforms (0, 1) into (min, max) with the appropriate prior probability.
        If None, assume a uniform prior distribution.
    log : bool array, optional
        Array of length N specifying which parameters should be sampled by a log-uniform distribution.
    """
    def __init__(self, f, bounds, prior_function=None, guess_function=None, tfprior_function=None, log=None, h_null=None):
        self.f, self.bounds = f, np.array(bounds)
        self.prior_function, self.guess_function, self.tfprior_function = prior_function, guess_function, tfprior_function
        self.log = np.zeros(len(self.bounds), dtype=bool) if log is None else np.array(log)
        self.h_null = h_null
        self.lnlike = None

        # Save the parameter, feature, and label names
        ann = self.f.__annotations__.copy()
        keys = list(ann.keys())
        for key, val in ann.items():
            if isinstance(val, str):
                ann[key] = (val,)
        self.params, self.features, self.labels = ann[keys[0]], ann[keys[1]], ann['return']
        self.nparams, self.nfeatures, self.nlabels = len(self.params), len(self.features), len(self.labels)
    
    def __call__(self, *args, **kwargs):
        """ Returns h(theta, x). Alias for self.f. """
        return self.f(*args, **kwargs)

    def guess_uniform(self, n, bounds):
        """ Default guess function. Guesses uniformly within self.bounds. """
        return np.random.uniform(bounds[:,0], bounds[:,1], size=(n, len(bounds)))
    
    def guess(self, n):
        """ Guesses a set of values for theta, preferably where P(theta) > -inf. """
        if self.guess_function is not None:
            return self.guess_function(n, self.bounds)
        return self.guess_uniform(n, self.bounds) 

    def lnprior_uniform(self, theta):
        """ Default (log-)uniform prior distribution, checks that all values are within bounds. """
        valid = (theta > self.bounds[:,0]).all() & (theta < self.bounds[:,1]).all()
        if not valid:
            return -np.inf
        
        return 0 - np.sum(self.log*np.log(theta))
    
    def lnprior(self, theta):
        """ Returns P(theta) (for emcee). """
        lnp = self.lnprior_uniform(theta)
        if self.prior_function is not None:
            lnp += self.prior_function(theta)
        return lnp

    def tfprior(self, u):
        if self.tfprior_function is None:
            return self.tfprior_uniform(u)
        else:
            return self.tfprior_function(u)

    def tfprior_uniform(self, u):
        """ Transforms the unit cube `u` into parameters drawn from (log-)uniform prior distributions. """
        theta, log = np.copy(u), self.log

        try:
            # Uniform priors
            if (~log).sum() > 0:
                bnd = self.bounds[~log]
                theta[~log] = u[~log] * np.ptp(bnd, axis=1) + np.amin(bnd, axis=1)

            # Log-uniform priors 
            if log.sum() > 0:
                bnd = np.log10(self.bounds[log])
                theta[log] = 10**(u[log] * np.ptp(bnd, axis=1) + np.amin(bnd, axis=1))
        except IndexError:
            print("ERROR")
            print(u)
            return

        return theta

    def lnlike_binary(self, theta, x, y):
        """ Likelihood function L(y | x, theta) if y is binary. """
        yh = self.f(theta, x)
        terms = np.log(y*yh + (1-y)*(1-yh))
        return np.sum(terms) 

    def lnlike_multivariate(self, theta, x, y):
        """ Likelihood function L(y | x, theta) if y is continuous. """
        # WIP: Assume 10% std dev on Y
        std = 0.1*np.median(y, axis=0)
        yh = self.f(theta, x)
        terms = -(y-yh)**2 / (2*std**2)
        return np.sum(terms)

    def lnprob(self, theta, x, y):
        """ Posterior probability function P(theta | x, y). """
        lnpr = self.lnprior(theta)
        if theta.ndim == 1 and np.isinf(lnpr):
            return lnpr, None
        lnlk = self.lnlike(theta, x, y)
        if np.isnan(lnlk):
            print(x, y)
        return lnlk + lnpr, lnlk
    
    def sample_posterior_dynesty(self, X, Y, nlive=100, nburn=None, verbose=False):
        """ Uses dynesty to sample the parameter posterior distributions and compute the log-evidence."""
        # If not explicitly set, nburn=10
        nburn = 10 if nburn is None else nburn

        # Sample the posterior distribution
        sampler = dynesty.NestedSampler(self.lnlike, self.tfprior, self.nparams, logl_args=(X, Y), nlive=nlive)
        sampler.run_nested(print_progress=verbose, dlogz=0.2)

        # Sample the null hypothesis
        sampler_null = dynesty.NestedSampler(self.h_null.lnlike, self.h_null.tfprior, Y.shape[1], logl_args=(X, Y), nlive=300)
        sampler_null.run_nested(print_progress=verbose, dlogz=0.1)

        # Return the posterior distribution samples and logZ difference
        lnZ = sampler.results.logz[-1] - sampler_null.results.logz[-1]
        return sampler.results.samples[nburn:, :], sampler.results.logl[nburn:], lnZ

    def sample_posterior_emcee(self, x, y, nsteps=500, nwalkers=32, nburn=100, autocorr=False):
        """ Uses emcee to sample the parameter posterior distributions. """
        sampler = emcee.EnsembleSampler(nwalkers, self.nparams, self.lnprob, args=(x, y))
        pos = self.guess(nwalkers)
        sampler.run_mcmc(pos, nsteps)
        if autocorr:
            print(sampler.get_autocorr_time())

        return sampler.get_chain(discard=nburn, flat=True), sampler.get_blobs(discard=nburn, flat=True)
 
    def compute_AIC(self, theta_opt, x, y):
        """ Computes the Akaike information criterion for optimal parameter set `theta_opt`. """
        return 2 * len(theta_opt) - 2 * self.lnlike(theta_opt, x, y)

    def compute_BIC(self, theta_opt, x, y):
        """ Computes the Bayesian information criterion for optimal parameter set `theta_opt`. """
        return len(theta_opt) * np.log(len(x)) - 2 * self.lnlike(theta_opt, x, y)

    def get_observed(self, data):
        """ Identifies which planets in the data set have measurements of the relevant features/labels. """
        observed = np.ones(len(data), dtype=bool)
        for key in np.append(self.features, self.labels):
            observed = observed & ~np.isnan(data[key])
        return observed
    
    def get_XY(self, data):
        """ Returns the X (features) and Y (labels) matrices for valid planets. Computes values as needed. """
        for key in np.append(self.features, self.labels):
            data.compute(key)
        observed = self.get_observed(data)
        X = np.array([data[feature][observed] for feature in self.features]).T
        Y = np.array([data[label][observed] for label in self.labels]).T
        return X, Y

    def fit(self, data, nsteps=500, nwalkers=16, nburn=100, nlive=100, return_chains=False, return_sampler=False,
            verbose=False, method='dynesty', mw_alternative='greater', return_data=False):
        """ Uses MCMC to sample the posterior distribution of h(theta | x, y) using a simulated data set, and compare
        to the null hypothesis via a model comparison metric.

        Parameters
        ----------
        data : Table
            Simulated data set containing the features and labels.
        nsteps : int, optional
            Number of steps per MCMC walker.
        return_chains : bool, optional
            If True, return the full MCMC posterior distribution samples.
        
        Returns
        -------
        results : dict
            Dictionary containing the results of the model fit:
                'h' : this Hypothesis object
                'means' : mean value of each parameter's posterior distribution
                'stds' : std dev of each parameter's posterior distribution
                'medians' : median value of each parameter's posterior distribution
                'UCIs' : 1-sigma confidence interval above the median
                'LCIs' : 1-sigma confidence interval below the median
                'CIs' : width of the +- 1 sigma confidence interval about the median
                'AIC' : Akaike information criterion compared to the null hypothesis (i.e. AIC_null - AIC_alt)
                'BIC' : Bayesian information criterion compared to the null hypothesis
                'chains' : full chain of MCMC samples (if `return_chains` is True)
        """
        # Extract the features and labels from the simulated data set
        X, Y = self.get_XY(data)

        # Determine which likelihood function to use
        if Y.shape[1] == 1 and is_bool(Y):
            self.lnlike = self.lnlike_binary
            self.h_null.lnlike = self.h_null.lnlike_binary
        else:
            print("using multivariate")
            self.lnlike = self.lnlike_multivariate
            self.h_null.lnlike = self.h_null.lnlike_multivariate

        # One test method or multiple?
        results = {}
        if np.ndim(method) == 0:
            method = (method,)
        if return_data:
            results['X'], results['Y'] = X, Y

        # Sample the posterior distribution (dynesty)
        if 'dynesty' in method:
            chains, loglikes, dlnZ = self.sample_posterior_dynesty(X, Y, nlive=nlive, nburn=nburn, verbose=verbose)

        # Sample the posterior distribution (emcee)
        # If both emcee and dynesty are used, then emcee will override the posterior distribution
        if 'emcee' in method:
            chains, loglikes = self.sample_posterior_emcee(X, Y, nsteps=nsteps, nwalkers=nwalkers)

        # Perform a Mann-Whitney test to compare X[Y] and X[~Y], assuming X and Y are 1-D
        # By default, tests whether X[Y] > X[~Y]
        if 'mannwhitney' in method:
            X0, Y0 = X[:, 0], Y[:, 0].astype(bool)
            if Y0.sum() > 0 and (~Y0).sum() > 0:
                U, p = mannwhitneyu(X0[Y0], X0[~Y0], alternative=mw_alternative)
            else:
                p = 0.5
            results['p'] = p

        if 'emcee' in method or 'dynesty' in method:
            # Compute the AIC and BIC relative to the null hypothesis (i.e. AIC_null - AIC)
            theta_opt, theta_opt_null = chains[np.argmax(loglikes)], np.mean(Y, axis=0)
            dAIC = self.h_null.compute_AIC(theta_opt_null, X, Y) - self.compute_AIC(theta_opt, X, Y)
            dBIC = self.h_null.compute_BIC(theta_opt_null, X, Y) - self.compute_BIC(theta_opt, X, Y)

            # Summary statistics
            conf = 95
            means, stds, medians = np.mean(chains, axis=0), np.std(chains, axis=0), np.median(chains, axis=0)
            LCIs, UCIs = np.abs(np.percentile(chains, [(100-conf)/2, 100-(100-conf/2)], axis=0) - medians)
            CIs = UCIs+LCIs

            # Return the results in a dict
            results.update({'means':means, 'stds':stds, 'medians':medians, 'LCIs':LCIs, 'UCIs':UCIs, 'CIs':UCIs+LCIs,
                            'dAIC':dAIC, 'dBIC':dBIC, 'dlnZ': dlnZ if 'dynesty' in method else None,
                            'chains':chains if return_chains else None, 'niter':chains.shape[0]})

        return results

# NULL HYPOTHESIS
# Returns (theta1, theta2, ...) for each element in X
def f_null(theta:(), X:()) -> ():
    shape = (np.shape(X)[0], np.shape(theta)[0])
    return np.full(shape, theta)

# HABITABLE ZONE HYPOTHESIS (4-parameter, alternative)
def f_HZ(theta:('a_inner', 'delta_a', 'f_HZ', 'df_notHZ'), X:('a_eff',)) -> ('has_H2O',):
    a_inner, delta_a, f_HZ, df_notHZ = theta
    in_HZ = (X > a_inner) & (X < (a_inner + delta_a))
    return in_HZ * f_HZ + (~in_HZ) * f_HZ*df_notHZ
bounds_HZ = np.array([[0.1, 2], [0.01, 10], [0.001, 1.0], [0.001, 1.0]])

# Null hypothesis: log-uniform from 0.001 to 1
bounds_HZ_null = np.array([[0.001, 1.0]]) 
h_HZ_null = Hypothesis(f_null, bounds_HZ_null, log=(True,))

h_HZ = Hypothesis(f_HZ, bounds_HZ, log=(True, True, True, True), h_null=h_HZ_null)

# AGE-OXYGEN CORRELATION HYPOTHESIS (2-parameter)
# f(x) = f_life * (1 - exp(-x/tau))
def f_age_oxygen(theta:('f_life', 't_half'), X:('age',)) -> ('has_O2',):
    f_life, t_half = theta
    return f_life * (1 - 0.5**(X/t_half))
bounds_age_oxygen = np.array([[0.01, 1], [0.1, 100]])

# Null hypothesis: log-uniform from 0.001 to 1
bounds_age_oxygen_null = np.array([[0.001, 1.0]])
h_age_oxygen_null = Hypothesis(f_null, bounds_age_oxygen_null, log=(True,))

h_age_oxygen = Hypothesis(f_age_oxygen, bounds_age_oxygen, log=(True, True), h_null=h_age_oxygen_null)

