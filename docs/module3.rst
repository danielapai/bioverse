##########################
Hypothesis testing
##########################

The :class:`~bioverse.hypothesis.Hypothesis` class
**************************************************

The combined result of the first two modules is a simulated dataset representing the output of the exoplanet survey. The third module addresses the power of that dataset for testing statistical hypotheses. The first step in this exercise involves defining the hypotheses you want to test, and in ``Bioverse`` this is done via a :class:`~bioverse.hypothesis.Hypothesis` object. This object includes:

- a set of dependent variable(s) ``X``, called features
- a set of independent variable(s) ``Y``, called labels
- a set of parameters ``theta``
- a Python function describing the quantitative relationship between ``X`` and ``Y`` in terms of ``theta``
- the prior distribution of values of ``theta``
- an alternative (or null) hypothesis against which to test

For example, consider the hypothesis that planet mass and radius can be related by a simple power law: :math:`M(R|M_0, \alpha) = M_0 R^{\alpha}`. In this case, X = R, Y = M, and theta = (M_0, alpha).

The first step in defining this hypothesis is to write out the function ``Y = f(X | theta)``:

.. code-block:: python

    def f(theta:('M_0', 'alpha'), X:('R',)) -> ('M',):
        M_0, alpha = theta
        R, = X
        return M_0 * R ** alpha

The function annotations tell the code the number and names of the parameters and variables being related. In this case, planet radius (``R``) and mass (``M``) will be extracted from the simulated dataset as the features and labels.

We must define the bounds on values for M_0 and alpha - conservative constraints might be 0.1 < M_0 < 10 and 2 < alpha < 5. We will also choose a log-uniform distribution for M_0, as its bounds span a few orders of magnitude.

.. code-block:: python

    bounds = np.array([[0.1, 10], [2, 5]])
    log = (True, False)

Next, we can initialize the Hypothesis:

.. code-block:: python

    from bio.hypothesis import Hypothesis
    h_mass_radius = Hypothesis(f, bounds, log=log)
    
The null hypothesis
*******************

In order to test the evidence in favor of ``h_mass_radius``, we must define an alternative (or "null") hypothesis. In this case, the hypothesis states that planetary mass is independent of radius, and ranges from 0.01 and 100 M_Earth:

.. code-block:: python

    def f_null(theta:('M_random',), X:('R',)) -> ('M',):
         shape = (np.shape(X)[0], 1)
         return np.full(shape, theta)
    
    bounds_null = np.array([[0.01, 100]])
    h_mass_radius.h_null = Hypothesis(f_null, bounds, log=(True,))
    
Note that :func:`~bioverse.hypothesis.f_null` (defined in the :mod:`~bioverse.hypothesis` module) provides the same function as above for an arbitrary number of parameters, features, and labels.

Testing the hypothesis
**********************
Next, we can test ``h_mass_radius`` using a dataset from the previous examples:

.. code-block:: python

    results = h_mass_radius.fit(data)

The :meth:`~bioverse.hypothesis.Hypothesis.fit()` method will pull the measured values of 'R' and 'M' and test them using one or more of the following methods (set by the `method` keyword):

- ``method = dynesty`` (default) Uses nested sampling to sample the parameter space of ``theta`` and compute the Bayesian evidence for both the Hypothesis and the null hypothesis. Implemented by ``dynesty``.
- ``method = emcee`` Uses Markov Chain Monte Carlo to sample the parameter space of ``theta``. Implemented by ``emcee``.
- ``method = mannwhitney`` Assuming ``X`` to be a single continuous variable and ``Y`` a single boolean, reports the probability that ``X[Y]`` and ``X[~Y]`` are drawn from the same parent distribution. Implemented by ``scipy``.

By default, nested sampling is used to estimate the Bayesian evidence in favor of the Hypothesis in comparison to the null hypothesis. 

Likelihood functions
********************

Both ``dynesty`` and ``emcee`` require a Bayesian likelihood function to be defined. The likelihood function is proportional to the probability that `Y` would be drawn given `X` and a set of values for `theta`. Currently, two likelihood functions are supported:

- binomial: If `Y` is a single boolean parameter (e.g., 'has_H2O') then ``f`` is interpreted as the likelihood that ``Y == 1`` given ``X``. In this case the likelihood function is:

    :math:`\ln\mathcal{L} = \sum_i \ln \left( Y_i f(X|\theta) + (1-Y_i)f(X|\theta) \right)`

- multivariate: If `Y` is one or more continuous variables then ``f`` is interpreted as the expectation values of ``Y`` given ``X``. In this case the likelihood function is the multivariate Gaussian:

    :math:`\ln\mathcal{L} = \sum_i \left[ -(Y_i-f(X|\theta))^2/(2\sigma_i^2) \right]`



Prior distributions
*******************

The prior distributions of the parameters ``theta`` can be set to either uniform or log-uniform functions *or* defined by the user. For uniform and log-uniform, only the boundaries of these distributions must be given:

.. code-block:: python

    # For theta = (M_0, alpha)
    bounds = np.array([[0.1, 10], [2, 5]])
    
    # Log-uniform distribution for M_0, uniform distribution for alpha
    h_mass_radius = Hypothesis(f, bounds, log=(True, False))
    
User-defined priors: TODO

Posterior distributions
***********************

When using ``dynesty`` or ``emcee``, the ``results`` object will contain summary statistics of the posterior distributions for the values of ``theta``, including the mean, median, and lower and upper 95% confidence intervals. Alternatively, by passing ``return_chains = True`` to the ``fit()`` method, the entire chain of sampled values will be return. Given enough time, the distribution of these values will converge onto the posterior distribution. In general, ``emcee`` converges much more efficiently and should be used to estimate (for example) the precision with which model parameters can be constrained.





        



