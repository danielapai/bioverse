""" Contains class definitions. """

# System modules
from copy import deepcopy
import numpy as np
import pickle
import time

# Bioverse modules and constants
from .constants import DATA_DIR, OBJECTS_DIR
from .constants import STR_TYPES, INT_TYPES
from .constants import CONST
from .util import interpolate_luminosity, interpolate_nuv, hz_evolution, normal

# Imports pandas.DataFrame if installed
try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None

class Object():
    """ This class allows the Generator and Survey classes to be saved as .pkl files under the Objects/ directory.

    Parameters
    ----------
    label : str, optional
        Name of the Generator or Survey. Default is to create a new object.
    """
    def __init__(self, label=None):
        # Save the label
        self.label = label
        self.filename = ''

        # If a label is given, load that object
        if label is not None:
            filename = self.get_filename_from_label(label)

            pkl = pickle.load(open(filename, 'rb'))
            for key in pkl.__dict__.keys():
                self.__dict__[key] = pkl.__dict__[key]
                # If the sub-class inherits from dict, then add the dict keys as well
                if isinstance(pkl, dict):
                    for key in pkl.keys():
                        self[key] = pkl[key]
        else:
            print("Created a new {:s}".format(type(self).__name__))

    def save(self, label=None):
        """ Saves the Object as a template in a .pkl file under ./<object type>s/. """
        if label is None:
            label = self.label
            if label is None:
                raise ValueError("no label specified")
        else:
            self.label = label

        filename = self.get_filename_from_label(label)
        pickle.dump(self,open(filename,'wb'))

    def get_filename_from_label(self, label):
        clas = type(self).__name__
        return OBJECTS_DIR+'/{:s}s/{:s}.pkl'.format(clas, label)

class Table(dict):
    """ Class for storing numpy arrays in a table-like format. Inherits dict. Each key is treated as a separate table column. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Uncertainties
        self.error = None

    def __repr__(self):
        s1 = 'Table of {:d} objects with {:d} parameters'.format(len(self), len(self.keys()))
        return s1

    def __str__(self):
        return self.pdshow()

    def pdshow(self):
        """ If pandas is installed, show the Table represented as a DataFrame. Otherwise, return an error. """
        if DataFrame is None:
            raise ModuleNotFoundError("Package `pandas` is required to display the Table")
        else:
            df_rep = DataFrame(self)
            display(df_rep)
            return ''

    def __len__(self):
        """ Returns the number of rows in the table rather than the number of keys (default dict behavior). """
        if len(self.keys()) == 0:
            return 0
        else:
            for key, val in self.items():
                if isinstance(val, (dict, Table)):
                    continue
                else:
                    return 1 if np.isscalar(val) else len(val)
        return 0

    def __getitem__(self, key):
        """ Expands on dict.__getitem__. Passing a string key will return values for the corresponding column. Passing an
        int, int array, bool array, or slice will return values for the corresponding table rows.
        """
        # If `key` is a string then return the column it refers to
        if isinstance(key, STR_TYPES):
            # If a semicolon is used, retrieve the column from a sub-table, e.g. Planets:Atmosphere:O2
            if ':' in key:
                key1, key2 = self.split_key(key)
                return self[key1][key2]
            return super().__getitem__(key)

        # If `key` is an int, int array, bool array, or slice then return the corresponding row(s)
        if isinstance(key, (np.ndarray, slice)) or isinstance(key, INT_TYPES):
            out = type(self)()
            for k in self.keys():
                out[k] = self[k][key]
            return out

        return super().__getitem__(key)

    def __setitem__(self, key, value, force=False):
        """ Expands on dict.__setitem__. Adds a new column to the table.

        Parameters
        ----------
        force : bool, optional
            If True, ignores the requirement that table column lengths match. Used for Table.append.
        """
        # Ensure that the new entry matches the size of existing entries
        if len(self) > 0 and not force:
            if (np.isscalar(value) or isinstance(value, (dict, type(None)))) and len(self) == 1:
                pass
            elif ~np.isscalar(value) and len(self) == len(value):
                pass
            else:
                raise ValueError("size of new column does not match other columns")

        # If a semicolon is used, set the value inside a nested dict, e.g. Planets:Atmosphere:O2
        if ':' in key:
            key1,key2 = self.split_key(key)
            self[key1].__setitem__(key2, value)
        else:
            # Convert lists to NumPy arrays before adding the column))
            super().__setitem__(key, value if (isinstance(value, np.ndarray) or np.size(value)==1) else np.array(value))

    def split_key(self, key):
        """ Splits a key such as 'Planets:Atmosphere:O2' into ('Planets','Atmosphere:O2') and ensures that the first
        key refers to a dict-like object. """
        split = key.split(':')
        key1,key2 = split[0],':'.join(split[1:])
        if isinstance(self[key1], dict):
            return key1, key2
        else:
            raise KeyError("invalid key {:s}; '{:s}' does not refer to a dict-like object".format(key,key1))

    def keys(self):
        """ Returns an array of keys instead of a dict_keys object, because I prefer it this way. """
        return np.array(list(super().keys()))

    def sort_by(self, key, inplace=False, ascending=True):
        """ Sorts the table by the values in one column.

        Parameters
        ----------
        key : str
            Name of the column by which to sort the table.
        inplace : bool, optional
            If True, sort the table in-place. Otherwise, return a new sorted Table.
        ascending : bool, optional
            If True, sort from least to greatest.

        Returns
        -------
        sortd : Table
            A sorted copy of this table. Only returned if `inplace` is True.
        """
        order = np.argsort(self[key])[::1 if ascending else -1]
        sortd = self if inplace else self.copy()
        for key in self.keys():
            sortd[key] = self[key][order]
        return None if inplace else sortd

    def get_stars(self):
        """ Returns just the first entry for each star in the Table. """
        idxes = np.array([np.where(self['starID']==idx)[0][0] for idx in np.unique(self['starID'])])
        return self[idxes]

    def legend(self, keys=None, filename=DATA_DIR+'legend.dat'):
        """ Prints the description of parameter(s) in the Table.

        Parameters
        ----------
        keys : str or str array, optional
            Key or list of keys to describe. If not specified, every key is described.
        filename : str, optional
            CSV file containing the list of parameter descriptions. Default is ./legend.dat.
        """
        # Loop through each line in the legend file and read the key, data type, and description into a dictionary
        leg = {}
        for line in open(filename, 'r').readlines():
            if line.strip() == ''  or line.strip()[0] == '#':
                continue
            key,dtype = line.split('#')[0].split(',')[:2]
            description = ','.join(line.split('#')[0].split(',')[2:])
            # Print a warning if duplicate keys are found
            if key.strip() in leg.keys():
                print("WARNING: Duplicate entries found in the legend for parameter {:s}!".format(key))
            leg[key.strip()] = [dtype.strip(), description.strip()]

        # Loop through each key and print the data type and description (if available)
        if keys is None:
            keys = self.keys()
        elif np.ndim(keys) == 0:
            keys = [keys]
        for key in keys:
            if key not in self.keys():
                continue
            if key in leg.keys():
                print("{:20s} {:10s} {:s}".format(key, '('+leg[key][0]+')', leg[key][1]))
            else:
                print("{:20s} (not found in legend)".format(key))

    def copy(self):
        """ Returns a deep copy of the Table instead of a shallow copy (as in dict.copy). This way, if a column
        is filled by objects (such as Atmosphere objects), a copy of those is returned instead of a reference. """
        return deepcopy(self)

    def append(self, table, inplace=True):
        """ Appends another table onto this table in-place. The second table must have the same
        columns, unless this table is empty, in which case the columns are copied over.
        
        Parameters
        ----------
        table : Table
            Table to be appended onto this one.
        inplace : bool, optional
            If True, append inplace and return None. If False, return a new Table.
        """
        # Append in-place?
        if not inplace:
            out_table = self.copy()
            out_table.append(table, inplace=True)
            return out_table

        # If this table has no columns, then just copy over the columns from `table`
        if len(self.keys()) == 0:
            super().__init__(table)

        # Otherwise ensure the columns match then append the rows to the end
        elif np.array_equal(np.sort(self.keys()), np.sort(table.keys())):
            for key in self.keys():
                self.__setitem__(key, np.append(self[key], table[key]), force=True)
        else:
            raise ValueError("attempting to append a Table with different columns")

    def compute(self, key, force=False):
        """ Computes the value of `key` using other values in the dictionary and a pre-defined formula. Useful for
        translating measured values (e.g. 'a', 'L_st') into secondary data products (e.g., 'S'). Will also propagate
        uncertainties contained in self.error. """
        if key in self and not force:
            # Unless force == True, don't re-compute existing values
            return

        if key[:4] == 'log(':
            k = key[4:-1]
            self.compute(k)
            self[key] = np.log10(self[k])
            # Approximate error on log(x) as (log(x+dx)-log(x-dx))/2
            if self.error:
                self.error[key] = 0.5 * (np.log10(self[k]+self.error[k])-np.log10(self[k]-self.error[k]))

        elif key[:3] == 'ln(':
            k = key[3:-1]
            self.compute(k)
            self[key] = np.log(self[k])
            # Approximate error on ln(x) as (ln(x+dx)-ln(x-dx))/2
            if self.error:
                self.error[key] = 0.5 * (np.log(self[k]+self.error[k])-np.log(self[k]-self.error[k]))

        elif key == 'S':
            self.compute('a')
            self['S'] = self['L_st']/self['a']**2
            if self.error:
                self.error['S'] = self['S'] * np.sqrt((self.error['L_st']/self['L_st'])**2 + (2*self.error['a']/self['a'])**2)

            # add absolute instellation in W/m2 (assuming dayside-average)
            self['S_abs'] = self['S']*CONST['S_Earth']

        elif key == 'R':
            self['R'] = (109.2**2)*self['depth']*self['R_st']**2
            if self.error:
                self.error['R'] = self['R'] * np.sqrt((self.error['depth']/self['depth'])**2 + (2*self.error['R_st']/self['R_st'])**2)

        elif key == 'rho':
            self['rho'] = self['M']/self['R']**3
            if self.error:
                self.error['rho'] = self['rho'] * np.sqrt((self.error['M']/self['M'])**2 + (3*self.error['R']/self['R'])**2)

        elif key == 'a':
            self['a'] = (self['M_st']*(self['P']/CONST['yr_to_day'])**2)**(1/3)
            if self.error:
                self.error['a'] = self['a'] * np.sqrt((1/3*self.error['M_st']/self['M_st'])**2 + (2/3*self.error['P']/self['P'])**2)

        elif key == 'a_eff':
            self.compute('S')
            self['a_eff'] = self['S']**-0.5
            if self.error:
                self.error['a_eff'] = self['a_eff'] * (0.5*self.error['S']/self['S'])

        elif key == 'R_eff':
            self['R_eff'] = (self['contrast'] * np.pi / 0.29)**0.5 / (4.258756e-5 / self['a'])
            if self.error:
                self.error['R_eff'] = self['R_eff'] * np.sqrt((1/2*self.error['contrast']/self['contrast'])**2 + (self.error['a']/self['a'])**2)

        elif key == 'h_eff':
            self.compute('S')
            self['h_eff'] = CONST['h_Earth']*self['S']/self['R']
            if self.error:
                self.error['h_eff'] = self['h_eff'] * np.sqrt((self.error['S']/self['S'])**2 + (self.error['R']/self['R'])**2)

        elif key == 'max_nuv':

            # self.compute('a')
            if not hasattr(self, 'evolution'):
                self.evolve(errors=True)

            # find maximum NUV flux occurring after 1 Myr
            self['max_nuv'] = np.full(len(self), np.nan)
            for i, p in self.evolution.items():
                try:
                    # get maximum 'nuv' for times > 1e-3 and only where 'in_hz' is True
                    max_nuv = np.max(p["nuv"][(p["time"] > 1e-3) & p["in_hz"]])
                    self["max_nuv"][self["planetID"] == i] = max_nuv
                except (ValueError, IndexError) as e:
                    # planet is either too young or estimated to never be in the HZ. Set max_nuv to NUV flux of final time step
                    self["max_nuv"][self["planetID"] == i] = p["nuv"][-1]

            if self.error:
                # approximate error using square root of sum of squares
                self.error['max_nuv'] = self['max_nuv'] * np.sqrt((self.error['age'] / self['age']) ** 2 +
                                                                  (self.error['M_st'] / self['M_st']) ** 2 + (
                                                                              self.error['a'] / self['a']) ** 2)

                # approximate error according to Richey-Yowell et al. 2023 Table 1 (~0.1 dex)
                self.error['max_nuv'] = self['max_nuv'] * 0.1



        # elif key == 'hz_and_uv':
        #
        #
        #     if not hasattr(self, 'evolution'):
        #         # evolve planets with errors
        #         self.evolve(errors=True)
        #
        #     df = self.to_pandas()
        #     df['hz_and_uv'] = np.full(len(self), False, dtype=bool)
        #
        #     for id, planet in self.evolution.items():
        #         dt = (max(planet['time']) - min(planet['time'])) / len(planet['time'])
        #         t = planet['time']
        #         in_hz = planet['in_hz']
        #         nuv = planet['nuv']
        #
        #         # check if planet ever was in the HZ and had NUV fluxes above the threshold value
        #         hz_and_uv = in_hz & (nuv > NUV_thresh)
        #
        #         t_consec_overlaps = np.diff(np.where(np.concatenate(([hz_and_uv[0]],
        #                                                              hz_and_uv[:-1] != hz_and_uv[1:],
        #                                                              [True])))[0])[::2] * dt
        #         df.loc[df['planetID'] == id, 'hz_and_uv'] = (t_consec_overlaps > deltaT_min / 1000.).any()
        #     self['hz_and_uv'] = df['hz_and_uv']


        else:
            raise ValueError("no formula defined for {:s}".format(key))

    def evolve(self, eec_only=True, sigma_nuv_dex=0.1, errors=False, seed=42, **kwargs):
        """ Add time evolution of habitable zones and NUV flux for each planet.

        This adds a new attribute `evolution` to the Table. `Table.evolution`
        is a dictionary with the planet IDs as keys. Each entry is a dictionary
        with the following keys:
            - 'time' : time grid in Gyr
            - 'lum' : luminosity evolution in L_sun
            - 'nuv' : NUV flux evolution in erg/s/cm^2
            - 'in_hz' : boolean array indicating whether the planet is in the HZ at each time step

        Parameters
        ----------
        eec_only : bool, optional
            If True, consider only planets that are "exo-Earth candidates" at observation time.
            Otherwise, consider all planets.
        sigma_nuv_dex : float, optional
            The intrinsic, typical standard deviation of the NUV data in Richey-Yowell et al. (2023), in dex.
        errors : bool, optional
            If True, treat as observed survey data and consider measurement errors.
        seed : int, optional
            Seed for random number generators.

        Returns
        -------
        None

        Example
        -------
        >>> g = Generator()
        >>> planets = g.generate()
        >>> planets.evolve()
        >>> plt.plot(planets.evolution[1]['time'], planets.evolution[1]['lum'])
        """

        np.random.seed(seed)
        self.evolution = {}

        if eec_only:
            planets = self[self['EEC'] == True]
            planets.error = self.error[self['EEC'] == True] if self.error is not None else None
        else:
            planets = self

        # Load the luminosity and nuv interpolation functions
        interp_lum, extrap_nn = interpolate_luminosity()
        interp_nuv = interpolate_nuv()

        dd = planets.to_pandas()
        # # updated_series = my_series.where(my_series > 30, 0)
        # dd.loc[dd.subSpT.str.contains('K.*'), 'nuv_class'] = 'K'
        # dd.loc[dd.subSpT.str.contains('M[1-3].*'), 'nuv_class'] = 'earlyM'
        # dd.loc[dd.subSpT.str.contains('M[4-9].*'), 'nuv_class'] = 'lateM'
        #
        # # for earlier spectral types, use spectral class K. TODO: implement for more massive stars
        # dd.loc[dd.nuv_class.isnull(), 'nuv_class'] = 'K'

        t0 = 16.5e-3  # t0 in Richey-Yowell et al. 2023
        dt = 0.01  # time step in Gyr

        if (errors and planets.error is not None):
            # this seems to be observed survey data. Consider measurement errors.
            errors = planets.error.to_pandas()
            for (index, planet), (erridx, error) in zip(dd.iterrows(), errors.iterrows()):

                # grid in Gyr, sample ~every 0.01 Gyr from 16.5 Myr to the age of the system.
                age = normal(planet['age'], error['age'], xmin=1e-6)
                T = np.arange(t0, age, step=dt) if age > t0 + dt else np.linspace(t0, age,
                                                                                  num=3)  # avoid single-element arrays in too young systems.

                # Compute the time evolution of the habitable zone
                lum_evo = interp_lum(normal(planet['M_st'], error['M_st'], xmin=0.08), T)

                # outside the bounds of the CT interpolator, extrapolate using a nearest neighbor approach
                if np.isnan(lum_evo).any():
                    lum_evo = extrap_nn(normal(planet['M_st'], error['M_st'], xmin=0.08), T)

                a_inner, a_outer = hz_evolution(planet, lum_evo)
                a = normal(planet['a'], error['a'], xmin=1e-6)
                in_hz = (a >= a_inner) & (a <= a_outer)

                # interpolate in NUV table, varying the mass within the error
                nuv_evo = interp_nuv(normal(planet['M_st'], error['M_st'], xmin=0.08), T)

                # add the instrinsic, typical error of the NUV data in Richey-Yowell et al. (2023) by applying a random bias
                np.random.seed(seed)
                bias = 10 ** np.random.normal(0, sigma_nuv_dex)
                nuv_evo = nuv_evo * bias

                self.evolution[planet["planetID"]] = {
                    "time": T,
                    "lum": lum_evo,
                    "nuv": nuv_evo,
                    "in_hz": in_hz,
                }

        else:
            for index, planet in dd.iterrows():
                # use face values without measurement errors

                # a time grid in Gyr, sample ~every 0.01 Gyr
                T = (
                    np.arange(t0, planet["age"], step=dt)
                    if planet["age"] > t0 + dt
                    else np.linspace(t0, planet["age"], num=3)
                )  # avoid single-element arrays in too young systems.

                # Compute the time evolution of the habitable zone
                lum_evo = interp_lum(planet["M_st"], T)

                # outside the bounds of the CT interpolator, extrapolate using a nearest neighbor approach
                if np.isnan(lum_evo).any():
                    lum_evo = extrap_nn(planet["M_st"], T)

                a_inner, a_outer = hz_evolution(planet, lum_evo)
                in_hz = (planet["a"] >= a_inner) & (planet["a"] <= a_outer)

                # Compute the time evolution of the NUV flux
                # nuv_evo = interp_nuv[planet['nuv_class']](T)

                # add the instrinsic, typical error of the NUV data in Richey-Yowell et al. (2023) by applying a random bias
                np.random.seed(seed)
                bias = 10 ** np.random.normal(0, sigma_nuv_dex)
                # nuv_evo = interp_nuv(planet["M_st"], T)
                nuv_evo = interp_nuv(planet["M_st"], T) * bias

                self.evolution[planet["planetID"]] = {
                    "time": T,
                    "lum": lum_evo,
                    "nuv": nuv_evo,
                    "in_hz": in_hz,
                }

        if eec_only:
            # remove planets where in_hz is not really True in the final time step
            for id, planet in self.evolution.copy().items():
                if not planet["in_hz"][-1]:
                    del self.evolution[id]

                    if self.error is not None:
                        self.error.update({key: np.delete(vals, np.where(self['planetID'] == id)) for key, vals in
                                           self.error.items()})

                    self.update({key: np.delete(vals, np.where(self['planetID'] == id)) for key, vals in self.items()})


    def shuffle(self, inplace=True):
        """ Re-orders rows in the Table. If `inplace` is False, return a new re-ordered Table instead. """
        if not inplace:
            out_table = self.copy()
            out_table.shuffle(inplace=True)
            return out_table

        order = np.random.choice(range(len(self)), replace=False, size=len(self))
        for key, val in self.items():
            self[key] = val[order]


    def to_pandas(self):
        """export Table into a pandas DataFrame"""
        df = DataFrame(self)
        return df

    def observed(self, key):
        """ Returns the subset of rows for which self[key] is not nan. """
        if key not in self:
            raise ValueError("parameter '{:s}' not found".format(key))
        return self[~np.isnan(self[key])]

class Stopwatch():
    """ This class uses the time module to profile chunks of code. Use the start() and stop() methods to start and
    stop the Stopwatch, and the mark() methods to record a time. stop() and read() will report the total time elapsed
    as well as the time between each step.
    """
    def __init__(self):
        self.clear()
        self.t0 = time.time()

    def clear(self):
        self.t0 = None
        self.t = []
        self.flags = []

    def mark(self, flag=None):
        self.t.append(time.time()-self.t0)
        self.flags.append(str(len(self.t)-1) if flag is None else flag)

    def stop(self, flag=None):
        self.mark('stop' if flag is None else flag)
        self.read()

    def read(self):
        print("{:40s}\t{:20s}\t{:20s}".format("Mark","Time step (s)","Total elapsed time (s)"))
        for i in range(len(self.t)):
            print("({:3d}) {:30s}\t{:20.8f}\t{:20.8f}".format(i, self.flags[i], self.t[i]-(self.t[i-1] if i>0 else 0.), self.t[i]))
        idx = np.argmax(np.array(self.t[1:])-np.array(self.t[:-1]))
        print("\nLongest step: ({0}) {1}".format(idx, self.flags[idx]))
