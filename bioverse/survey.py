from dataclasses import asdict, dataclass
import numpy as np

from .classes import Object, Table
from . import util
from .constants import STR_TYPES, BOOL_TYPES

@dataclass(repr=False)
class Survey(dict, Object):
    """
    Describes an exoplanet survey, including methods for creating simulated datasets.
    This class should not be called directly; instead use ImagingSurvey or TransitSurvey.
    """
    
    label: str = None
    diameter: float = 15.0
    t_max: float = 10*365.25
    t_slew: float = 0.1
    T_st_ref: float = 5788.
    R_st_ref: float = 1.0
    D_ref: float = 15.0
    d_ref: float = 10.0

    def __post_init__(self):
        if type(self) == Survey:
            raise ValueError("don't call Survey directly - instead use ImagingSurvey or TransitSurvey")
        self.measurements = {}
        Object.__init__(self, self.label)

    def __repr__(self):
        s = "{:s} with the following parameters:".format(type(self).__name__)
        for key, val in asdict(self).items():
            s += "\n  {0}: {1}".format(key, val)

        s += "\n\nConducts the following measurements"
        for i,k in enumerate(self.measurements):
            s += "\n({:d}) {:s}".format(i, self.measurements[k].__repr__())
        return s

    def add_measurement(self, key, **kwargs):
        """ Adds a measurement to the survey. """
        self.measurements[key] = Measurement(key, self, **kwargs)

    def quickrun(self, generator, t_total=None, N_sim=1, **kwargs):
        """ Convenience function which generates a sample, computes the detection yield, and returns a simulated data set.
        
        Parameters
        ----------
        generator : Generator
            Generator object to be used to generate the planet population.
        t_total : float, optional
            Total amount of observing time for any measurements with a limited observing time.
        N_sim : int, optional
            If greater than 1, simulate the survey this many times and return the combined result.
        **kwargs
            Keyword arguments to be passed to the Generator.

        Returns
        -------
        sample : Table
            Table of all simulated planets.
        detected : Table
            Table of planets detected by the survey.
        data : Table
            Simulated data set produced by the survey.
        """
        
        # For transit surveys, set transit_mode=True unless otherwise specified
        if isinstance(self, TransitSurvey) and 'transit_mode' not in kwargs:
            kwargs['transit_mode'] = True

        if N_sim > 1:
            sample, detected, data = Table(), Table(), Table()
            for i in range(int(N_sim)):
                res = self.quickrun(generator, t_total=t_total, N_sim=1, **kwargs)
                sample.append(res[0])
                detected.append(res[1])
                data.append(res[2])
        else:
            sample = generator.generate(**kwargs)
            detected = self.compute_yield(sample)
            data = self.observe(detected, t_total=t_total)

        return sample, detected, data
        
    def observe(self, y, t_total=None, data=None):
        """ Returns a simulated data set for a table of simulated planets. Each measurement specified in the
        survey configuration file is performed on every planet in the table.
        
        Parameters
        ----------
        y : Table
            Table containing the set of planets to be observed, usually the detection yield of the survey.
        t_total : float, optional
            Total observing time for any measurements with limited observing time.
        data : Table, optional
            Table in which to store the measurements.

        Returns
        -------
        data : Table
            Table of measurements made by the survey, with one row for each planet observed.
        """
        # Create an output Table to store measurements
        data = data if data is not None else Table()

        # Perform each measurement in order
        for _, m in self.measurements.items():
            if m.key not in y.keys():
                print("Could not measure property: {:s}".format(m.key))
                continue

            # Determine the amount of time for this measurement (if it has any exposure time constraints)
            t = np.inf
            if t_total is not None and m.t_ref is not None:
                if isinstance(t_total, dict) and m.key in t_total:
                    t = t_total[m.key]
                elif not isinstance(t_total, dict):
                    t = t_total
                    
            data = m.measure(y, data, t_total=t)
        return data

@dataclass(repr=False)
class ImagingSurvey(Survey):
    inner_working_angle: float = 3.5
    outer_working_angle: float = 64
    contrast_limit: float = -10.6
    mode: str = 'imaging'

    def compute_yield(self, d, wl_eff=0.5, A_g=0.3):
        """ Computes a simple estimate of the detection yield for an imaging survey. Compares the contrast ratio and
        projected separation of each planet when observed at quadrature to the contrast limit and inner/outer working
        angles of the survey. Planets that satisfy these criteria are considered to be detected.
        
        Parameters
        ----------
        d : Table
            Table of all simulated planets which the survey could attempt to observe.
        wl_eff : float, optional
            Effective wavelength of observation in microns (used for calculating the IWA/OWA).
        A_g : float, optional
            Geometric albedo of each planet, ignored if 'A_g' is already assigned.

        Returns
        -------
        yield : Table
            Copy of the input table containing only planets which were detected by the survey.
        """

        # Compute the angular separation at quadrature (milli-arcseconds)
        separation = d['a'] / d['d'] * 1000

        # If no albedo is given, assume one
        if 'A_g' not in d:
            d['A_g'] = np.full(len(d), A_g)

        # Determine which planets are brighter than the contrast limit
        if 'contrast' not in d:
            d['contrast'] = d['A_g'] * (4.258756e-5*d['R']/d['a'])**2 / np.pi
        mask1 = d['contrast'] > 10**self.contrast_limit

        # Determine which planets are within the field of view
        iwa = self.inner_working_angle * (1e-6*wl_eff) / self.diameter*206265*1000
        owa = self.outer_working_angle * (1e-6*wl_eff) / self.diameter*206265*1000
        mask2 = (separation > iwa) & (separation < owa)

        # Return the output table
        return d[mask1 & mask2]

    def compute_scaling_factor(self, d):
        return (d['contrast']/1e-10)**-1

@dataclass(repr=False)
class TransitSurvey(Survey):
    N_obs_max: float = 1000
    mode: str = 'transit'

    def compute_yield(self, d):
        """ Computes a simple estimate of the detection yield for a transit survey. All transiting planets
        are considered to be detected.

        Parameters
        ----------
        d : Table
            Table of all simulated planets which the survey could attempt to observe.

        Returns
        -------
        yield : Table
            Copy of the input table containing only planets which were detected by the survey.
        """
        # Determine which planets transit their stars
        mask1 = d['transiting']

        # Return the output table
        return d[mask1]

    def compute_scaling_factor(self, d):
        """ Computes the scaling factor for the reference exposure time in transit mode. """
        d.compute('S')
        return (d['H']/9)**-2 * d['R']**-2 * (d['R_st']/self.R_st_ref)**4

class Measurement():
    """
    Class describing a simple measurement to be applied to a table of planets detected by a Survey.

    Parameters
    ----------
    key : str
        Name of the parameter will be measured (e.g., 'M' for planet mass)
    precision : str or float, optional
        Precision of measurement, e.g. '10%' or 0.10 units. If zero, a perfect measureent is made.
    condition : str array, optional
        Conditional statements specifying which planets the measurement applies to (e.g., '0.5 < R < 1.5').
        Defaults to None, in which case the measurement applies to all planets.
    t : float, optional
        Amount of time required per measurement, arbitrary unit. If `t_scale` is specified, then `t` is the time
        required for a reference case i.e. where all of the scale factors are 1.
    t_total : float, optional
        Total amount of time allocated for this measurement, same unit as `t`. The measurement will be made for as 
        many targets as possible within this time allocation.
    priority : dict, optional
        Each entry describes how targets should be prioritized based on a planet property. For example,
        {'S':(0.3,1.,2)} indicates that planets with 0.3 < S < 1 should have 2x priority (i.e. the same priority
        as a target requiring half as much observing time).
    """
    def __init__(self, key, survey, precision=0., condition=[], t_total=None, t_ref=None, t_min=0., priority={},
                 wl_eff=0.5, debias=True, bounds=None):
        super().__init__()

        # Save the keyword values
        self.key = key
        self.survey = survey
        self.precision = precision
        self.conditions = condition
        self.t_total = t_total
        self.t_ref = t_ref
        self.t_min = t_min
        self.priority = {key:np.array(val) for key, val in priority.items()}
        self.wl_eff = wl_eff
        self.debias = debias
        self.bounds = bounds if bounds is not None else [-np.inf, np.inf]

    def __repr__(self):
        s = "Measures parameter '{:s}'".format(self.key)
        if self.precision != 0.:
            s += " with {:s} precision".format(str(self.precision))
        for i,cdtn in enumerate(self.conditions):
            s += "\n    Conditions: {:s}".format(cdtn) if i == 0 else ' AND {:s}'.format(cdtn)
        if self.t_ref is not None:
            s += "\n    Average time required: {:.1f} d".format(self.t_ref)
            
        return s

    def measure(self, detected, data=None, t_total=None):
        """ Produces the measurement for planets in a table and places them into a data table.
        
        Parameters
        ----------
        detected : Table
            Table containing detected planets.
        data : Table, optional
            Table in which to store the measured values for each planet. If not given,
            then a new table is created.
        
        Returns
        -------
        data : Table
            Table containing the measured values for each planet.
        """
        # Create a new output table if necessary
        if data is None:
            data = Table()
        data['planetID'] = detected['planetID']
        data['starID'] = detected['starID']
        
        # Determine which planets are valid targets, and which of those can be observed in the allotted time
        valid = self.compute_valid_targets(data)
        observable = self.compute_observable_targets(data, valid, t_total)

        # Simulate a measurement for each observable planet
        x = self.perform_measurement(detected[self.key][observable])

        # Place this measurement into the measurements database
        if self.key not in data.keys():
            data[self.key] = np.full(len(detected), np.nan)
        data[self.key][observable] = x

        return data
    
    def compute_valid_targets(self, data):
        """ Determines which planets are valid targets based on whether their previously measured parameters
        meet the measurement conditions.

        Parameters
        ----------
        data : Table
            Table of data values already measured for these planets.
        
        Returns
        -------
        valid : bool array
            Specifies which planets in the table are valid targets.
        """
        valid = np.ones(len(data), dtype=bool)
        
        # Supported comparison strings
        comparisons = {'<': np.less, '>': np.greater, '==': np.in1d}
        
        cdtns = self.conditions
        if cdtns is not None:
            if np.ndim(cdtns) == 0: cdtns = [cdtns]
            for cdn in cdtns:
                # Determine which parameter is being compared to which value, and which comparison to perform
                cmp_str = [cs for cs in comparisons if cs in cdn][0]
                key, val = cdn.split(cmp_str)
                key, val = key.strip(), val.strip()

                # Ensure that the parameter has already been measured, else try to compute it
                if key not in data:
                    data.compute(key)
                
                # Make the comparison, assuming the comparison value to be the same type as in `d`
                try:
                    dtype = type(data[key][~np.isnan(data[key])][0])
                    valid = valid & comparisons[cmp_str](data[key], dtype(val))
                except IndexError:
                    valid = valid & np.zeros(len(valid), dtype=bool)
        
        return valid
        
    def compute_observable_targets(self, data, valid, t_total=None):
        """ Determines which planets are observable based on the total allotted observing time.

        Parameters
        ----------
        data : Table
            Table of data values already measured for these planets.
        valid : bool array
            Mask specifying which targets in `data` are valid targets.
        t_total : float, optional
            Total observing time for this measurement. If None, use self.t_total.
        
        Returns
        -------
        observable : bool array
            Specifies which planets in the table are observable within the allotted time.
        """

        # If t_total is infinite, then all valid targets are observable
        if t_total is None:
            t_total = self.t_total
        if np.isinf(t_total):
            return valid

        # Compute the required exposure and overhead time for each target
        t_exp, N_obs = self.compute_exposure_time(data)
        t_over = self.compute_overhead_time(data)

        # (Transit mode) Targets are invalid if they cannot be observed within 10 yr
        if self.survey.mode == 'transit':
            valid[(N_obs*data['P']) > self.survey.t_max] = False
            valid[N_obs > self.survey.N_obs_max] = False

        # Compute the priority of each target; invalid targets have zero priority
        weights = self.compute_weights(data)
        weights *= self.compute_debias(data) 
        priority = weights/(t_exp+t_over)
        priority[~valid] = 0.

        # Observe planets in order of priority, until we run out of time or valid targets
        t_sum, observable = 0., np.zeros(len(data), dtype=bool)
        for i in range(valid.sum()):
            # Determine the current highest priority planet and how much time is required to observe it
            t_obs = t_exp + t_over
            idx = np.argmax(priority)

            # Add this amount of time to the budget - if it's too much, then stop observing immediately
            t_sum += t_obs[idx]
            if t_sum > t_total:
                break

            # (Imaging mode) Observe the target along with other planets in the same system
            if self.survey.mode == 'imaging':
                t_exp[data['starID']==data['starID'][idx]] -= t_exp[idx]
                
            # (Transit mode) Only observe the target
            elif self.survey.mode == 'transit':
                t_exp[idx] -= t_exp[idx]

            # Mark planets with t_exp <= 0 as "observable" and remove them from the line-up
            finished = t_exp <= 0
            observable[finished&valid], priority[finished] = True, 0.

        return observable

    def compute_exposure_time(self, d):
        """ Computes the exposure time and number of observations required to characterize each planet in `d`. """
        wl = self.wl_eff / 10000 # convert from microns -> cm
        h, c, k, T_eff_sol = 6.6261e-27, 2.9979e10, 1.3807e-16, 5780.

        # Reference case parameters
        T_ref, R_ref, D_ref, d_ref = self.survey.T_st_ref, self.survey.R_st_ref, self.survey.D_ref, self.survey.d_ref

        # Scaling factors for each survey mode
        f = self.survey.compute_scaling_factor(d)

        # Calculate the exposure time required for each target, based on the reference time
        flux_ratio = (np.exp(h*c/(wl*k*T_ref)) - 1) / (np.exp(h*c/(wl*k*d['T_eff_st'])) - 1)
        t_exp = f * self.t_ref * (self.survey.diameter/D_ref)**-2 * (d['d']/d_ref)**2 * (d['R_st']/R_ref)**-2 * flux_ratio**-1

        # Number of observations (1 for imaging mode, integer multiple of transit duration for transit mode)
        if self.survey.mode == 'imaging':
            N_obs = np.ones(len(d))
        if self.survey.mode == 'transit':
            N_obs = np.ceil(t_exp/d['T_dur'])
            N_act = t_exp/d['T_dur']
            t_exp = d['T_dur']*N_obs

        return t_exp, N_obs

    def compute_overhead_time(self, d, N_obs=1):
        """ Computes the overheads associated with each observation. """

        # Imaging mode: 1x overhead for each target
        if self.survey.mode == 'imaging':
            t_over = np.full(len(d), self.survey.t_slew)

        # Transit mode: N_obs x (overhead + T_dur) for each target
        if self.survey.mode == 'transit':
            t_over = N_obs * (self.survey.t_slew + d['T_dur'])

        return t_over

    def compute_weights(self, d):
        """ Computes the priority weight of each planet in `d`. """
        weights = np.ones(len(d))
        for key, val1 in self.priority.items():
            # Try to compute d[key] if it hasn't already been measured
            d.compute(key)
            for val2 in val1:
                weights[(d[key] >= val2[0]) & (d[key] <= val2[1])] *= val2[2]
        return weights

    def compute_debias(self, d):
        """ Removes detection biases from the data set (transit mode only). """
        weights = np.ones(len(d))
        if not self.debias:
            return weights

        d.compute('a')
        d.compute('a_eff')
        if 'a' in d and self.survey.mode == 'transit':
            weights = d['a']/d['R_st']
        return weights

    def compute_priority(self, data):
        """ Computes the target priority based on its weight and required exposure time. """
        # Determine which targets are valid targets; if no targets are valid, return zeros
        valid = self.compute_valid_targets(data)
        if valid.sum() == 0:
            return np.zeros(len(valid), dtype=bool)

        # Compute the exposure times and overheads for each target
        t_exp, N_obs = self.compute_exposure_time(data)
        t_over = self.compute_overhead_time(data, N_obs)

        # (Transit mode) Targets are invalid if N_obs * P > 10 yr
        if self.survey.mode == 'transit':
            valid[(N_obs*data['P']) > self.survey.t_max] = False
            valid[N_obs > self.survey.N_obs_max] = False

        # Compute target weights (invalid targets have no weight)
        #weights = self.compute_weights(data)
        weights = self.compute_weights_new(data)
        weights[~valid] = 0.

        return weights/(t_exp+t_over)

    def perform_measurement(self, x):
        """ Simulates measurements of the parameter from a set of true values.
        
        Parameters
        ----------
        x : array
            Array of true values on which to perform the measurement.

        Returns
        -------
        xm : array
            Array of measured values with the same length and type as `x`.
        """
        # Skip empty values
        if len(x) == 0:
            return x

        # If the value is type str or bool then the measurement has to be "exact"
        if type(x[0]) in list(np.append(STR_TYPES, BOOL_TYPES)):
            return x

        # Percentage-based precision
        if type(self.precision) is str and '%' in self.precision:
            sig = x*float(self.precision.strip('%'))/100.
            
        # Absolute precision
        else:
            sig = float(self.precision)

        # Return draw from bounded normal distribution
        return util.normal(x, sig, xmin=self.bounds[0], xmax=self.bounds[1], size=len(x))

def reset_imaging_survey():
    """ Re-creates the default imaging survey. """
    s_imaging = ImagingSurvey(label=None,
                            diameter = 15.0,
                            inner_working_angle = 3.5,
                            outer_working_angle = 64.0,
                            t_slew = 0.1,
                            T_st_ref = 5788.,
                            R_st_ref = 1.0,
                            D_ref = 15.0,
                            d_ref = 10.0)

    # Define the measurements to conduct
    margs = {}
    mkeys = ['L_st', 'R_st', 'T_eff_st', 'd', 'contrast', 'a', 'has_H2O', 'age', 'EEC', 'has_O2']

    margs['precision'] = {'contrast': '10%',
                        'a': '10%',
                        'age': '10%',}

    margs['condition'] = {'has_H2O': ('R_eff>0.5', 'R_eff<2.0', 'a_eff<10', 'a_eff>0.1'),
                        'has_O2': ('EEC==1',)}

    margs['priority'] = {}
    margs['priority']['has_H2O'] = {'a_eff': [[0.2, 1.0, 1.0],
                                            [1.0, 2.0, 5.0],
                                            [2.0, 10.0, 2.0]]}
    margs['priority']['has_O2'] = {'age': [[0, 1, 10],
                                        [1, 2, 5],
                                        [2, 10, 1]]}

    margs['t_ref'] = {'has_H2O': 0.035,
                    'has_O2': 0.1}

    # Add the measurements to s_imaging
    for mkey in mkeys:
        kwargs = {}
        for key, vals in margs.items():
            if mkey in vals:
                kwargs[key] = vals[mkey]
        s_imaging.add_measurement(mkey, **kwargs)
    
    s_imaging.save(label='default')

def reset_transit_survey():
    """ Re-creates the default transit survey. """
    s_transit = TransitSurvey(label=None,
                            diameter = 50.0,
                            N_obs_max = 1000,
                            t_slew = 0.0208,
                            T_st_ref = 3300,
                            R_st_ref = 0.315,
                            D_ref = 50.0,
                            d_ref = 50.0)

    # Define the measurements to conduct
    margs = {}
    mkeys = ['L_st', 'R_st', 'M_st', 'T_eff_st', 'd', 'H', 'age', 'depth', 'T_dur', 'P', 'has_H2O', 'EEC', 'has_O2']

    margs['precision'] = {'T_eff_st': 25.,
                        'R_st': '5%',
                        'M_st': '5%',
                        'age': '30%',
                        'P': 0.001}

    margs['condition'] = {'has_H2O': ('R>0.7', 'R<1.5', 'a_eff>0.1', 'a_eff<10.0'),
                        'has_O2': ('EEC==1',)}

    margs['t_ref'] = {'has_H2O': 7.5,
                    'has_O2': 3.1}

    margs['priority'] = {}
    margs['priority']['has_H2O'] = {'a_eff': [[ 0.3  ,  0.816,  2.   ],
                                            [ 0.816,  1.414,  3.   ],
                                            [ 1.414, 10.   ,  6.   ]]}
    margs['priority']['has_O2'] = {'age': [[ 0,  2,  3],
                                        [ 2,  4,  2],
                                        [ 4,  6,  1],
                                        [ 6,  8,  2],
                                        [ 8, 12,  3]]}

    # Add the measurements to s_transit
    for mkey in mkeys:
        kwargs = {}
        for key, vals in margs.items():
            if mkey in vals:
                kwargs[key] = vals[mkey]
        s_transit.add_measurement(mkey, **kwargs)
    
    s_transit.save(label='default')