""" Contains class definitions. """

# System modules
from copy import deepcopy
import numpy as np
import pickle
import time

# Bioverse modules and constants
from .constants import ROOT_DIR, OBJECTS_DIR
from .constants import STR_TYPES, INT_TYPES
from .constants import CONST

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
        return self.pdshow()

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
            # If a semi-colon is used, retrieve the column from a sub-table, e.g. Planets:Atmosphere:O2
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

        # If a semi-colon is used, set the value inside a nested dict, e.g. Planets:Atmosphere:O2
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
            If True, sort the table in-place. Otherwise return a new sorted Table.
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

    def legend(self, keys=None, filename=ROOT_DIR+'/legend.dat'):
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

        elif key == 'R':
            self['R'] = (109.2**2)*self['depth']*self['R_st']**2
            if self.error:
                self.error['R'] = self['R'] * np.sqrt((self.error['depth']/self['depth'])**2 + (2*self.error['R_st']/self['R_st'])**2)

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

        else:
            raise ValueError("no formula defined for {:s}".format(key))
    
    def shuffle(self, inplace=True):
        """ Re-orders rows in the Table. If `inplace` is False, return a new re-ordered Table instead. """
        if not inplace:
            out_table = self.copy()
            out_table.shuffle(inplace=True)
            return out_table

        order = np.random.choice(range(len(self)), replace=False, size=len(self))
        for key, val in self.items():
            self[key] = val[order]

    def pdshow(self):
        """ If pandas is installed, show the Table represented as a DataFrame. Otherwise return an error. """
        if DataFrame is None:
            raise ModuleNotFoundError("Package `pandas` is required to display the Table")
        else:
            df_rep = DataFrame(self).__repr__()
            print(df_rep)
            return df_rep
    
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
        