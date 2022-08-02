""" Defines the Hypothesis class as well as two hypotheses used in Bixel & Apai (2021). """

import dynesty
import emcee
import numpy as np
from scipy.stats import mannwhitneyu
from warnings import warn

# Bioverse modules
from .util import is_bool, as_tuple

class Hypothesis():
    """ Describes a Bayesian hypothesis.
    
    Parameters
    ----------
    f : function
        Function describing the hypothesis. Must be defined as f(theta, X) where theta is a tuple of parameter values
        and X is a set of independent variables. Returns the calculated values of Y, the set of dependent variables for
        each entry in X.
    bounds : array
        Nx2 array describing the [min, max] limits of each parameter. These are enforced even if a different prior
        distribution is defined.
    params : tuple of str, optional
        Names of the parameter(s) of the hypothesis.
    features : tuple of str, optional
        Names of the feature(s) or independent variables.
    labels : tuple of str, optional
        Names of the label(s) or dependent variables.
    lnprior_function : function, optional
        Used by emcee. Function which returns ln(P_prior), must be defined as prior(theta). If None, assume a (log-)uniform distribution.
    guess_function : function, optional
        Used by emcee. Function which guesses valid sets of parameters. Must be defined as guess_function(n), and 
        should return an n x m set of parameter guesses. If None, draw parameters randomly within `bounds`.
    tfprior_function : function, optional
        Used by dynesty. Function which transforms (0, 1) into (min, max) with the appropriate prior probability.
        If None, assume a (log-)uniform distribution.
    log : bool array, optional
        Array of length N specifying which parameters should be sampled by a log-uniform distribution.
    """
    def __init__(self, f, bounds, params=(), features=(), labels=(), lnprior_function=None, guess_function=None, tfprior_function=None, log=None, h_null=None):
        self.f, self.bounds = f, np.array(bounds)
        self.lnprior_function, self.guess_function, self.tfprior_function = lnprior_function, guess_function, tfprior_function
        self.log = np.zeros(len(self.bounds), dtype=bool) if log is None else np.array(log)
        self.h_null = h_null
        self.lnlike = None

        # Warn if only one prior function is defined
        if self.lnprior_function is None and self.tfprior_function is not None:
            warn("prior function is defined for dynesty but not emcee!")
        if self.lnprior_function is not None and self.tfprior_function is None:
            warn("prior function is defined for emcee but not dynesty!")

        # Warn if log = True despite a custom prior distribution
        if np.any(self.log) and not (self.lnprior_function is None and self.tfprior_function is None):
            warn("should not pass log=True with a user-defined prior function!")

        # Save the parameter, feature, and label names 
        self.params, self.features, self.labels = as_tuple(params), as_tuple(features), as_tuple(labels)
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
        if self.lnprior_function is not None:
            lnp += self.lnprior_function(theta)
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

    def lnlike_binary(self, theta, x, y, _):
        """ Likelihood function L(y | x, theta) if y is binary. The last argument is a placeholder. """
        yh = self.f(theta, x)
        terms = np.log(y*yh + (1-y)*(1-yh))
        return np.sum(terms) 

    def lnlike_multivariate(self, theta, x, y, sigma):
        """ Likelihood function L(y | x, theta) if y is continuous and has `sigma` uncertainty. """
        yh = self.f(theta, x)
        terms = -(y-yh)**2 / (2*sigma**2)
        return np.sum(terms)

    def lnprob(self, theta, x, y, sigma):
        """ Posterior probability function P(theta | x, y). """
        lnpr = self.lnprior(theta)
        if theta.ndim == 1 and np.isinf(lnpr):
            return lnpr, None
        lnlk = self.lnlike(theta, x, y, sigma)
        if np.isnan(lnlk):
            print(x, y)
        return lnlk + lnpr, lnlk
    
    def sample_posterior_dynesty(self, X, Y, sigma, nlive=100, nburn=None, verbose=False):
        """ Uses dynesty to sample the parameter posterior distributions and compute the log-evidence."""
        # If not explicitly set, nburn=10
        nburn = 10 if nburn is None else nburn

        # Sample the posterior distribution
        sampler = dynesty.NestedSampler(self.lnlike, self.tfprior, self.nparams, logl_args=(X, Y, sigma), nlive=nlive)
        sampler.run_nested(print_progress=verbose, dlogz=0.2)

        # Sample the null hypothesis
        sampler_null = dynesty.NestedSampler(self.h_null.lnlike, self.h_null.tfprior, Y.shape[1], logl_args=(X, Y, sigma), nlive=300)
        sampler_null.run_nested(print_progress=verbose, dlogz=0.1)

        # Return the posterior distribution samples and logZ difference
        lnZ = sampler.results.logz[-1] - sampler_null.results.logz[-1]
        return sampler.results.samples[nburn:, :], sampler.results.logl[nburn:], lnZ

    def sample_posterior_emcee(self, x, y, sigma, nsteps=500, nwalkers=32, nburn=100, autocorr=False):
        """ Uses emcee to sample the parameter posterior distributions. """
        sampler = emcee.EnsembleSampler(nwalkers, self.nparams, self.lnprob, args=(x, y, sigma))
        pos = self.guess(nwalkers)
        sampler.run_mcmc(pos, nsteps)
        if autocorr:
            print(sampler.get_autocorr_time())

        return sampler.get_chain(discard=nburn, flat=True), sampler.get_blobs(discard=nburn, flat=True)
 
    def compute_AIC(self, theta_opt, x, y, sigma):
        """ Computes the Akaike information criterion for optimal parameter set `theta_opt`. """
        return 2 * len(theta_opt) - 2 * self.lnlike(theta_opt, x, y, sigma)

    def compute_BIC(self, theta_opt, x, y, sigma):
        """ Computes the Bayesian information criterion for optimal parameter set `theta_opt`. """
        return len(theta_opt) * np.log(len(x)) - 2 * self.lnlike(theta_opt, x, y, sigma)

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
        sigma = np.array([data.error[label][observed] for label in self.labels]).T
        return X, Y, sigma

    def fit(self, data, nsteps=500, nwalkers=16, nburn=100, nlive=100, return_chains=False,
            verbose=False, method='dynesty', mw_alternative='greater', return_data=False):
        """
        Sample the posterior distribution of h(theta | x, y) using a simulated data set, and compare
        to the null hypothesis via a model comparison metric.

        Parameters
        ----------
        data : Table
            Simulated data set containing the features and labels.
        nsteps : int, optional
            Number of steps per MCMC walker.
        nburn : int, optional
            Number of burn-in steps for the Monte Carlo walk.
        nlive : int, optional
            Number of live points for the nested sampler.
        return_chains : bool, optional
            Wether or not to return the Monte Carlo chains.
        verbose :
            Wether or not to generate extra output during the run.
        method : str, optional
            Which sampling method to use. Options: dynesty (default), emcee, mannwhitney,
        mw_alternative : str, {'two-sided', 'less', 'greater'}, optional
            Defines the alternative hypothesis. Default is 'two-sided'.
            Let *F(u)* and *G(u)* be the cumulative distribution functions of the
            distributions underlying `x` and `y`, respectively. Then the following
            alternative hypotheses are available:

            * 'two-sided': the distributions are not equal, i.e. *F(u) ≠ G(u)* for
              at least one *u*.
            * 'less': the distribution underlying `x` is stochastically less
              than the distribution underlying `y`, i.e. *F(u) > G(u)* for all *u*.
            * 'greater': the distribution underlying `x` is stochastically greater
              than the distribution underlying `y`, i.e. *F(u) < G(u)* for all *u*.
        return_data : bool
            Wether or not to return the data

        Returns
        -------
        results : dict
            Dictionary containing the results of the model fit:
                'means' : mean value of each parameter's posterior distribution
                'stds' : std dev of each parameter's posterior distribution
                'medians' : median value of each parameter's posterior distribution
                'UCIs' : 2-sigma confidence interval above the median
                'LCIs' : 2-sigma confidence interval below the median
                'CIs' : width of the +- 2 sigma confidence interval about the median
                'AIC' : Akaike information criterion compared to the null hypothesis (i.e. AIC_null - AIC_alt)
                'BIC' : Bayesian information criterion compared to the null hypothesis
                'chains' : full chain of MCMC samples (if `return_chains` is True)
        """
        # Extract the features and labels from the simulated data set
        X, Y, sigma = self.get_XY(data)

        # Determine which likelihood function to use
        if Y.shape[1] == 1 and is_bool(Y):
            self.lnlike = self.lnlike_binary
            self.h_null.lnlike = self.h_null.lnlike_binary
        else:
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
            chains, loglikes, dlnZ = self.sample_posterior_dynesty(X, Y, sigma, nlive=nlive, nburn=nburn, verbose=verbose)

        # Sample the posterior distribution (emcee)
        # If both emcee and dynesty are used, then emcee will override the posterior distribution
        if 'emcee' in method:
            chains, loglikes = self.sample_posterior_emcee(X, Y, sigma, nsteps=nsteps, nwalkers=nwalkers)

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
            dAIC = self.h_null.compute_AIC(theta_opt_null, X, Y, sigma) - self.compute_AIC(theta_opt, X, Y, sigma)
            dBIC = self.h_null.compute_BIC(theta_opt_null, X, Y, sigma) - self.compute_BIC(theta_opt, X, Y, sigma)

            # Summary statistics
            conf = 95
            means, stds, medians = np.mean(chains, axis=0), np.std(chains, axis=0), np.median(chains, axis=0)
            LCIs, UCIs = np.abs(np.percentile(chains, [(100-conf)/2, 100-(100-conf/2)], axis=0) - medians)

            # Return the results in a dict
            results.update({'means':means, 'stds':stds, 'medians':medians, 'LCIs':LCIs, 'UCIs':UCIs, 'CIs':UCIs+LCIs,
                            'dAIC':dAIC, 'dBIC':dBIC, 'dlnZ': dlnZ if 'dynesty' in method else None,
                            'chains':chains if return_chains else None, 'niter':chains.shape[0]})

        return results

# NULL HYPOTHESIS (N-parameter)
def f_null(theta, X):
    """ Function for a generic null hypothesis. Returns (theta1, theta2, ...) for each element in X. """
    shape = (np.shape(X)[0], np.shape(theta)[0])
    return np.full(shape, theta)

# HABITABLE ZONE HYPOTHESIS (4-parameter)
def f_HZ(theta, X):
    """ Function for the habitable zone hypothesis. """
    a_inner, delta_a, f_HZ, df_notHZ = theta
    in_HZ = (X > a_inner) & (X < (a_inner + delta_a))
    return in_HZ * f_HZ + (~in_HZ) * f_HZ*df_notHZ

params_HZ = ('a_inner', 'delta_a', 'f_HZ', 'df_notHZ')
features_HZ = ('a_eff',)
labels_HZ = ('has_H2O',)
bounds_HZ = np.array([[0.1, 2], [0.01, 10], [0.001, 1.0], [0.001, 1.0]])

# Null hypothesis: log-uniform from 0.001 to 1
bounds_HZ_null = np.array([[0.001, 1.0]]) 
h_HZ_null = Hypothesis(f_null, bounds_HZ_null, log=(True,))

h_HZ = Hypothesis(f_HZ, bounds_HZ, params=params_HZ, features=features_HZ, labels=labels_HZ,
                  log=(True, True, True, True), h_null=h_HZ_null)

# AGE-OXYGEN CORRELATION HYPOTHESIS (2-parameter)
def f_age_oxygen(theta, X): 
    """ Function for the age-oxygen correlation hypothesis. """
    f_life, t_half = theta
    return f_life * (1 - 0.5**(X/t_half))

params_age_oxygen = ('f_life', 't_half')
features_age_oxygen = ('age',)
labels_age_oxygen = ('has_O2',)
bounds_age_oxygen = np.array([[0.01, 1], [0.1, 100]])

# Null hypothesis: log-uniform from 0.001 to 1
bounds_age_oxygen_null = np.array([[0.001, 1.0]])
h_age_oxygen_null = Hypothesis(f_null, bounds_age_oxygen_null, log=(True,))

h_age_oxygen = Hypothesis(f_age_oxygen, bounds_age_oxygen, params=params_age_oxygen, features=features_age_oxygen,
                          labels=labels_age_oxygen, log=(True, True), h_null=h_age_oxygen_null)


def magma_ocean_hypo_exp(theta, X):
    """ Define a hypothesis for a magma ocean-adapted radius-sma distribution that follows an exponential decay.

    Parameters
    ----------
    theta : array_like
        Array of parameters for the hypothesis.
        f_magma : float
            fraction of planets having a magma ocean
        a_cut: float
            cutoff effective sma for magma oceans. Defines position of the exponential decay.
        lambda_a: float
            Decay parameter for the semi-major axis dependence of having a global magma ocean.
    X : array_like
        Independent variable. Includes semimajor axis a.

    Returns
    -------
    array_like
        Functional form of hypothesis
    """
    f_magma, a_cut, lambda_a = theta
    a_eff = X
    return f_magma * np.exp(-(a_eff/a_cut)**lambda_a)

def magma_ocean_hypo_step(theta, X):
    """ Define a hypothesis for a magma ocean-adapted radius-sma distribution following a step function. Tests the
    hypothesis that the average planet size is smaller within the cutoff effective radius.

    Parameters
    ----------
    theta : array_like
        Array of parameters for the hypothesis.
        f_magma : float
            fraction of planets having a magma ocean
        a_cut: float
            cutoff effective sma for magma oceans. Defines where the step occurs.
        radius_factor: float
            Factor by which the radius is multiplied for magma ocean-bearing planets.
        R_avg : float
            Average radius of the planets _without_ magma oceans.
    X : array_like
        Independent variable. Includes semimajor axis a.

    Returns
    -------
    array_like
        Functional form of hypothesis
    """
    f_magma, a_cut, radius_factor, R_avg = theta
    a_eff = X

    # R_avg for a_eff >= a_cut, reduced, f_magma-weighted average radius otherwise
    return (R_avg - f_magma * (1 - radius_factor)) * (a_eff < a_cut) + R_avg * (a_eff >= a_cut)


# params = ('f_magma', 'a_cut', 'lambda_a')
# features = ('a_eff',)
# labels = ('is_small',)

# define priors for the parameters in theta (uniform for f_magma, lambda_a; log-uniform for a_cut)
# bounds = np.array([[0.00, 1.0], [0.01, 10.], [0.1, 100.]])
# h_magmaocean = Hypothesis(magma_ocean_hypo, bounds, params=params, features=features, labels=labels, log=(False, True, False))


def magma_ocean_f0(theta, X):
    """ Define the null hypothesis that the radius distribution is random and independent of sma.
    """
    return np.full(np.shape(X), theta)

