""" Provides functions for iterating over several simulations to compute statistical power and more. """

# Python modules
import multiprocessing as mp
import numpy as np
import signal
import time
import traceback

# Bioverse modules and constants
from . import util

# Prevents a crash due to matplotlib when multiprocessing
mp.set_start_method('spawn', force=True)

def test_hypothesis_grid(h, generator, survey, N=10, processes=1, do_bar=True, bins=15, return_chains=False,
                         mw_alternative='greater', method='dynesty', nlive=100, **kwargs):
    """ Runs simulated surveys over a grid of survey and astrophysical parameters. Each time, uses the simulated
    data set to fit the hypothesis parameters and computes the model evidence versus the null hypothesis. """

    # Split `kwargs` into `grid` (list values + N) and `fixed` (scalar values) keyword arguments
    grid = {key:np.array(val) for key, val in kwargs.items() if np.ndim(val) == 1}
    grid['N'] = N if np.ndim(N) == 1 else np.arange(N)
    fixed = {key:val for key, val in kwargs.items() if np.ndim(val) == 0}

    # Initialize the process pool (+ some lines to enable ctrl+c interrupt; https://stackoverflow.com/a/35134329)
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(processes)
    signal.signal(signal.SIGINT, original_sigint_handler)

    # Run each grid cell as a separate process with a random RNG seed
    grid_shape = tuple(np.size(v) for v in grid.values())
    N_iter = int(np.prod(grid_shape))
    procs, seeds = [], np.random.randint(0, 1e9, N_iter)
    for idx in range(N_iter):
        # Determine the grid values (+ fixed arguments) for this iteration, excepting 'N'
        idxes = np.unravel_index(idx, grid_shape)
        iter_kwargs = fixed.copy()
        iter_kwargs.update({key:val[idxes[i]] for i, (key, val) in enumerate(grid.items()) if key != 'N'})

        # Start each iteration as a separate process
        args = (h, generator, survey, bins, return_chains, 
                mw_alternative, method, seeds[idx], nlive, iter_kwargs)
        proc = pool.apply_async(test_hypothesis_grid_iter,args)
        procs.append(proc)
        
    # Collect the results from each process into the appropriate grid cell
    results = {}
    try:
        for idx in util.bar(range(N_iter), do_bar):
            res = procs[idx].get()
            for key in res:
                val = res[key]
                if val is None:
                    results[key] = None
                    continue
                if key not in results:
                    results[key] = np.full((N_iter, *np.shape(val)), np.nan)
                elif np.ndim(val) > 0:
                    # If results[key] is longer/shorter than previous iterations then adjust the array sizes
                    diff = results[key][idx].shape[0] - val.shape[0]
                    if diff > 0:
                        pad = np.full((diff, *val.shape[1:]), np.nan)
                        val = np.concatenate([val, pad], axis=0)
                    elif diff < 0:
                        pad = np.full((N_iter, -diff, *val.shape[1:]), np.nan)
                        results[key] = np.concatenate([results[key], pad], axis=1)
                results[key][idx] = val
    except:
        traceback.print_exc()
        print("\nInterrupted, terminating remaining processes")
        pool.terminate()
    else:
        pool.close()

    # Reshape the output into a grid
    def reshape(arr):
        if arr is None:
            return None
        return arr.reshape(*grid_shape, *arr.shape[1:])
    for key in results:
        results[key] = reshape(results[key])

    # Save the hypothesis, generator, survey, grid values, and fixed arguments
    results['h'], results['generator'], results['survey'], results['grid'], results['fixed'] =\
        h, generator, survey, grid, fixed

    return results

def test_hypothesis_grid_iter(h, generator, survey, bins, return_chains,
                              mw_alternative, method, seed, nlive, kwargs):
    """ Runs a single iteration for test_hypothesis_grid (separated for multiprocessing). """
    # Prevents duplicate results when multiprocessing
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = 42
    
    # Format the `t_total` argument so that it only affects measurements relevant to the hypothesis
    if 't_total' in kwargs and not isinstance(kwargs['t_total'], dict):
        t_total = kwargs.pop('t_total')
        kwargs['t_total'] = {key: t_total for key in (*h.features, *h.labels)}

    # Simulate a data set and fit the hypothesis with it
    # Also time each step for future reference
    t_start= time.time()
    sample, detected, data = survey.quickrun(generator, seed=seed, **kwargs)
    t_sim = time.time() - t_start
    results = h.fit(data, return_chains=return_chains, method=method, nlive=nlive, mw_alternative=mw_alternative)
    t_fit = time.time() - t_start - t_sim

    # Count the number of hot/warm/cold planets and EECs which were characterized
    obs = detected[h.get_observed(data)]
    try:
        N_hot, N_warm, N_cold = [int((obs['class1']==typ).sum()) for typ in ['hot', 'warm', 'cold']]
        N_EEC, N_pl = int(obs['EEC'].sum()), len(obs)
        results['N_hot'], results['N_warm'], results['N_cold'] = N_hot, N_warm, N_cold
        results['N_EEC'], results['N_pl'] = N_EEC, N_pl
    except KeyError:
        results['N_pl'] = len(obs)
    finally:
        pass

    # Compute the average value of labels versus features
    obs = data[h.get_observed(data)]
    if len(obs) > 0:
        bins, values, errors = util.binned_average(obs[h.features[0]], obs[h.labels[0]], bins=bins)
    else:
        bins, values, errors = np.full((3, bins), np.nan)

    results['bins'], results['values'], results['errors'] = bins, values, errors
    results['t_sim'], results['t_fit'] = t_sim, t_fit
    
    return results

def compute_statistical_power(results, threshold=None, method='dlnZ'):
    """ Computes the statistical power of a hypothesis test, i.e. the fraction of simulated tests which
    pass a model comparison significance threshold.
    
    Parameters
    ----------
    results : dict
        Output of test_hypothesis_grid.
    threshold: float, optional
        Significance threshold to enforce.
    method: ('dAIC', 'dBIC', 'dlnZ', 'p', 'logp')
        Specifies which method to use.

    Returns
    -------
    power : float array
        Array of statistical power for each test in the results grid. Shape is shape(results[method])[:-1].
    """

    # Determine which grid axis corresponds to 'N'
    axis = list(results['grid'].keys()).index('N')

    # Determine the upper/lower threshold
    thresholds = {'dAIC': 6, 'dBIC': 6, 'dlnZ': 3, 'p': 0.05, 'logp': -1.3}
    if method not in thresholds:
        raise ValueError("'method' must be one of "+str(tuple(thresholds)))
    threshold = thresholds[method] if threshold is None else threshold

    comparisons = {'dAIC': np.greater, 'dBIC': np.greater, 'dlnZ': np.greater, 'p': np.less, 'logp': np.less}
    test_result = comparisons[method](results[method], threshold)
    return test_result.mean(axis=axis)

def random_simulation(results, generator, survey, bins=15, mw_test=False, mw_alternative='greater', method='dynesty',
                      nlive=100, return_chains=True, **grid_kwargs):
    # Determine the grid values for this simulation
    grid, iter_kwargs = results['grid'], {}
    for key, val in grid.items():
        if key in grid_kwargs:
            iter_kwargs[key] = grid_kwargs[key]
        else:
            iter_kwargs[key] = np.random.choice(val)

    # Add 'fixed' arguments from `results` and `grid_kwargs`
    iter_kwargs.update(results['fixed'])
    for key, val in grid_kwargs.items():
        if key not in grid:
            iter_kwargs[key] = val
            
    # Add fixed arguments and run the simulation
    args = (results['h'], generator, survey, bins, return_chains, mw_test, 
            mw_alternative, method, None, nlive, iter_kwargs)
    results_out = test_hypothesis_grid_iter(*args)
    results_out['h'] = results['h']
    results_out['fixed'] = iter_kwargs
    return results_out

def compare_methods(h, data, methods=['dynesty', 'emcee'], **kwargs):
    results = {}
    for method in methods:
        res = h.fit(data, method=method, return_chains=True, **kwargs)
        results[method] = res
        results[method]['h'] = h
    return results

### Helper Methods
def _count_by_class(observed):
    """ Counts detected planets by class for one simulated observation. Shared by the
    number_vs_* worker functions below. """
    row = {'all': len(observed), 'EEC': int(observed['EEC'].sum())}
    for key in ['hot', 'warm', 'cold']:
        row[key] = int((observed['class1'] == key).sum())
    return row

def _number_vs_time_worker(generator, survey, h, t_total, seed, kwargs):
    """ Runs one simulation (fixed seed) and sweeps it over every value of t_total.
    Defined at module level so it can be sent to a worker process. """
    np.random.seed(seed)
    sample, detected, _ = survey.quickrun(generator, seed=seed, **kwargs)
    categories = ['all', 'EEC', 'hot', 'warm', 'cold']
    row = {key: np.zeros(len(t_total), dtype=int) for key in categories}
    for i, t in enumerate(t_total):
        data = survey.observe(detected, t_total={label: t for label in h.labels})
        observed = detected[h.get_observed(data)]
        counts = _count_by_class(observed)
        for key in categories:
            row[key][i] = counts[key]
    return row

def number_vs_time(h, generator, survey, t_total, N=30, average=True, processes=1, **kwargs):
    """ Determines how many planets are characterized by the simulated survey versus time budget.

    Parameters
    ----------
    processes : int, optional
        Number of worker processes. Each of the N simulations is independent, so they run
        concurrently across `processes` workers; the sweep over `t_total` within each
        simulation stays serial since it reuses that simulation's result. Default is 1
        (serial, original behavior).
    """
    categories = ['all', 'EEC', 'hot', 'warm', 'cold']
    N_pl = {key:np.zeros((len(t_total), N), dtype=int) for key in categories}

    seeds = np.random.randint(0, 1e9, size=N)
    args = [(generator, survey, h, t_total, seeds[j], kwargs) for j in range(N)]

    if processes > 1:
        pool = mp.Pool(processes)
        procs = [pool.apply_async(_number_vs_time_worker, a) for a in args]
        rows = [procs[j].get() for j in util.bar(range(N))]
        pool.close()
    else:
        rows = [_number_vs_time_worker(*a) for a in util.bar(args)]

    for j, row in enumerate(rows):
        for key in categories:
            N_pl[key][:, j] = row[key]

    if average:
        for key, val in N_pl.items():
            N_pl[key] = np.mean(val, axis=1)

    N_pl['t_total'] = t_total

    return N_pl

def _number_vs_param_worker(generator, survey, h, param_name, param_val, seed, kwargs):
    """ Runs one simulation with `param_name` set to `param_val` (e.g. eta_Earth or d_max).
    Defined at module level so it can be sent to a worker process. """
    np.random.seed(seed)
    call_kwargs = dict(kwargs)
    call_kwargs[param_name] = param_val
    sample, detected, data = survey.quickrun(generator, seed=seed, **call_kwargs)
    observed = detected[h.get_observed(data)]
    return _count_by_class(observed)

def _number_vs_param(h, generator, survey, param_name, param_grid, N, average, processes, kwargs):
    """ Shared implementation for number_vs_eta and number_vs_distance: both sweep N
    independent simulations over a 1-D parameter grid, and every (i, j) combination is
    independent of every other. """
    categories = ['all', 'EEC', 'hot', 'warm', 'cold']
    N_pl = {key: np.zeros((len(param_grid), N), dtype=int) for key in categories}

    seeds = np.random.randint(0, 1e9, size=N * len(param_grid))
    args = [
        (generator, survey, h, param_name, val, seeds[j * len(param_grid) + i], kwargs)
        for j in range(N) for i, val in enumerate(param_grid)
    ]

    if processes > 1:
        pool = mp.Pool(processes)
        procs = [pool.apply_async(_number_vs_param_worker, a) for a in args]
        rows = [procs[idx].get() for idx in util.bar(range(len(args)))]
        pool.close()
    else:
        rows = [_number_vs_param_worker(*a) for a in util.bar(args)]

    idx = 0
    for j in range(N):
        for i in range(len(param_grid)):
            counts = rows[idx]
            idx += 1
            for key in categories:
                N_pl[key][i, j] = counts[key]

    if average:
        for key, val in N_pl.items():
            N_pl[key] = np.mean(val, axis=1)

    return N_pl

def number_vs_eta(h, generator, survey, eta_Earth, N=30, average=True, processes=1, **kwargs):
    """ Determines how many planets are characterized by the simulated survey versus eta Earth.

    Parameters
    ----------
    processes : int, optional
        Number of worker processes. All N*len(eta_Earth) simulations are independent, so they
        run concurrently across `processes` workers. Default is 1 (serial, original behavior).
    """
    N_pl = _number_vs_param(h, generator, survey, 'eta_Earth', eta_Earth, N, average, processes, kwargs)
    N_pl['eta_Earth'] = eta_Earth
    return N_pl

def number_vs_distance(h, generator, survey, d_max, N=30, average=True, processes=1, **kwargs):
    """ Determines how many planets are characterized by the simulated survey versus d_max.

    Parameters
    ----------
    processes : int, optional
        Number of worker processes. All N*len(d_max) simulations are independent, so they
        run concurrently across `processes` workers. Default is 1 (serial, original behavior).
    """
    N_pl = _number_vs_param(h, generator, survey, 'd_max', d_max, N, average, processes, kwargs)
    N_pl['d_max'] = d_max
    return N_pl